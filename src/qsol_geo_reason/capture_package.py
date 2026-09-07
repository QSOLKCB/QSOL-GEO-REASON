"""Content-bound receipts for imported serving implementation packages."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .canonical import sha256_json
from .capture_common import CaptureContractError

_IGNORED_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _hash_regular_file(path: Path, where: str) -> str:
    try:
        if not path.is_file():
            raise CaptureContractError(f"{where} contains a non-regular package artifact: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise CaptureContractError(f"unable to hash {where} package artifact {path}: {exc}") from exc


def _python_package_provenance(module: Any, where: str) -> dict[str, Any]:
    """Hash the imported package tree, excluding disposable interpreter bytecode.

    The receipt intentionally follows the package actually imported by this process
    rather than distribution metadata alone. That binds editable installs, patched
    wheels and custom redistributions even when they retain the same version string.
    """
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise CaptureContractError(f"canonical capture cannot locate the imported {where} package")
    try:
        root = Path(module_file).resolve(strict=True).parent
    except OSError as exc:
        raise CaptureContractError(f"canonical capture cannot resolve the imported {where} package") from exc
    if not root.is_dir():
        raise CaptureContractError(f"canonical capture {where} package root is not a directory")

    hashes: dict[str, str] = {}
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CaptureContractError(f"unable to enumerate the imported {where} package") from exc
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise CaptureContractError(f"{where} package artifact escapes its package root") from exc
        if "__pycache__" in relative.parts or path.suffix in _IGNORED_BYTECODE_SUFFIXES:
            continue
        if path.is_dir():
            continue
        rel = relative.as_posix()
        if not rel or rel.startswith("/") or "\\" in rel or any(part in {"", ".", ".."} for part in relative.parts):
            raise CaptureContractError(f"{where} package contains a noncanonical artifact path: {rel!r}")
        hashes[rel] = _hash_regular_file(path, where)

    if not hashes:
        raise CaptureContractError(f"canonical capture found no files in the imported {where} package")
    return {
        "file_count": len(hashes),
        "receipt_sha256": sha256_json(hashes),
    }


def _validate_python_package_provenance(
    observed: dict[str, Any] | Any, *, count_field: str, receipt_field: str, where: str
) -> None:
    count = observed.get(count_field) if hasattr(observed, "get") else None
    receipt = observed.get(receipt_field) if hasattr(observed, "get") else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError(f"{where} package file count must be a positive integer")
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(character not in "0123456789abcdef" for character in receipt)
    ):
        raise CaptureContractError(f"{where} package receipt must be a lowercase SHA-256 digest")


__all__ = ["_python_package_provenance", "_validate_python_package_provenance"]

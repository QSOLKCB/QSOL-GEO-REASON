"""Content-bound receipts for imported serving implementation packages."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import marshal
import types
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .provenance import (
    SourceIdentityError,
    _bytecode_optimization_level,
    _source_path_for_bytecode,
    _stable_code_identity,
)

_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})


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


def _authenticate_package_bytecode(
    root: Path, cache_path: Path, hashes: Mapping[str, str], where: str
) -> None:
    """Require cached executable code to match the exact source bytes in the receipt.

    Ordinary CPython caches do not change package identity, but they are executable
    inputs, not disposable evidence. Compare their complete code structure with a
    fresh compilation in the cache's optimization lane. Compilation does not execute
    the package. Filenames are excluded from code identity so relocated wheel caches
    can match; the source path and source bytes are independently receipt-bound.
    """
    try:
        source_path = _source_path_for_bytecode(cache_path).resolve(strict=True)
        source_relative = source_path.relative_to(root).as_posix()
        if source_path.suffix != ".py" or source_relative not in hashes:
            raise ValueError("bytecode is not backed by receipt-bound package source")
        optimization = _bytecode_optimization_level(cache_path)
        if cache_path.parent.name == "__pycache__":
            expected_cache = Path(importlib.util.cache_from_source(
                str(source_path), optimization=str(optimization) if optimization else ""
            ))
            if cache_path.name != expected_cache.name:
                raise ValueError("noncanonical or foreign-interpreter cache filename")
        if not cache_path.is_file():
            raise ValueError("bytecode cache is not a regular file")
        source = source_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != hashes[source_relative]:
            raise ValueError("package source changed after its receipt was computed")
        raw = cache_path.read_bytes()
        if len(raw) < 16 or raw[:4] != importlib.util.MAGIC_NUMBER:
            raise ValueError("invalid or foreign-interpreter bytecode header")
        if int.from_bytes(raw[4:8], "little") & ~3:
            raise ValueError("invalid bytecode header flags")
        payload = io.BytesIO(raw[16:])
        observed = marshal.load(payload)
        if not isinstance(observed, types.CodeType) or payload.read(1):
            raise ValueError("bytecode payload must contain exactly one module code object")
        expected = compile(
            source, str(source_path), "exec", dont_inherit=True, optimize=optimization
        )
        if _stable_code_identity(observed) != _stable_code_identity(expected):
            raise ValueError("cached executable code differs from receipt-bound source")
    except (
        SourceIdentityError, OSError, EOFError, ValueError, TypeError,
        SyntaxError, RuntimeError,
    ) as exc:
        raise CaptureContractError(
            f"unable to authenticate {where} package bytecode against receipt-bound source: {cache_path}"
        ) from exc


def _python_package_provenance(module: Any, where: str) -> dict[str, Any]:
    """Hash the imported package tree and authenticate its executable bytecode.

    The receipt intentionally follows the package actually imported by this process
    rather than distribution metadata alone. That binds editable installs, patched
    wheels and custom redistributions even when they retain the same version string.
    Source-equivalent caches are verified, then excluded from the aggregate receipt
    so normal cache creation, invalidation mode and installation relocation do not
    create different source identities. Unverifiable or mismatching caches fail closed.
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
    bytecode: list[Path] = []
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CaptureContractError(f"unable to enumerate the imported {where} package") from exc
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise CaptureContractError(f"{where} package artifact escapes its package root") from exc
        if path.is_dir():
            if path.is_symlink():
                raise CaptureContractError(
                    f"{where} package contains an unenumerated symlinked directory: {path}"
                )
            continue
        rel = relative.as_posix()
        if not rel or rel.startswith("/") or "\\" in rel or any(part in {"", ".", ".."} for part in relative.parts):
            raise CaptureContractError(f"{where} package contains a noncanonical artifact path: {rel!r}")
        if path.suffix in _BYTECODE_SUFFIXES:
            bytecode.append(path)
        else:
            hashes[rel] = _hash_regular_file(path, where)

    if not hashes:
        raise CaptureContractError(f"canonical capture found no files in the imported {where} package")
    for cache_path in bytecode:
        _authenticate_package_bytecode(root, cache_path, hashes, where)
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

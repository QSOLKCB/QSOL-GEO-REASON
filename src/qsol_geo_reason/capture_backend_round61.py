"""Round-61 pre-load PyTorch package stability hardening.

Round 58 retained a package stat/change-time baseline after authenticated model loading.
That detects transient changes during the later observation window, but a lazy import
could still execute temporarily replaced PyTorch bytes while the inherited constructor
was loading a model and have those bytes restored before the post-load baseline was
created.  This layer fingerprints the package tree before inherited loading begins and
requires exact stat identity after construction.
"""
from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from .capture_backend_round56 import HuggingFacePyTorchBackend as _Round56Backend
from .capture_common import CaptureContractError


_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _package_file_stat_fingerprint_round61(path: Path) -> tuple[int, ...]:
    try:
        link = os.lstat(path)
        target = os.stat(path)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to stat pre-load PyTorch package artifact: {path}"
        ) from exc
    if stat.S_ISDIR(target.st_mode):
        raise CaptureContractError(
            f"pre-load PyTorch package stability expected a file, found directory: {path}"
        )
    return (
        int(link.st_mode),
        int(link.st_dev),
        int(link.st_ino),
        int(link.st_size),
        int(link.st_mtime_ns),
        int(link.st_ctime_ns),
        int(target.st_mode),
        int(target.st_dev),
        int(target.st_ino),
        int(target.st_size),
        int(target.st_mtime_ns),
        int(target.st_ctime_ns),
    )


def _torch_package_stability_from_root_round61(
    root: Path,
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    try:
        canonical_root = root.resolve(strict=True)
        paths = sorted(canonical_root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as exc:
        raise CaptureContractError(
            "unable to enumerate PyTorch package before authenticated model loading"
        ) from exc
    if not canonical_root.is_dir():
        raise CaptureContractError("pre-load PyTorch package root is not a directory")

    entries: list[tuple[str, tuple[int, ...]]] = []
    for path in paths:
        try:
            relative = path.relative_to(canonical_root)
        except ValueError as exc:
            raise CaptureContractError(
                "pre-load PyTorch package artifact escapes its package root"
            ) from exc
        if path.is_dir():
            if path.is_symlink():
                raise CaptureContractError(
                    f"PyTorch package contains an unenumerated symlinked directory: {path}"
                )
            continue
        if path.suffix in _BYTECODE_SUFFIXES:
            continue
        rel = relative.as_posix()
        if (
            not rel
            or rel.startswith("/")
            or "\\" in rel
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise CaptureContractError(
                f"PyTorch package contains a noncanonical pre-load stability path: {rel!r}"
            )
        entries.append((rel, _package_file_stat_fingerprint_round61(path)))
    if not entries:
        raise CaptureContractError(
            "canonical capture found no PyTorch files for pre-load stability receipt"
        )
    return str(canonical_root), tuple(entries)


def _preimport_torch_package_stability_round61() -> (
    tuple[str, tuple[tuple[str, tuple[int, ...]], ...]] | None
):
    """Fingerprint the importable torch package without importing torch itself."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not isinstance(spec.origin, str) or not spec.origin.strip():
        return None
    try:
        origin = Path(spec.origin).resolve(strict=True)
    except OSError as exc:
        raise CaptureContractError(
            "unable to resolve PyTorch package before authenticated model loading"
        ) from exc
    return _torch_package_stability_from_root_round61(origin.parent)


def _loaded_torch_package_stability_round61(torch_module: Any) -> (
    tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]
):
    module_file = getattr(torch_module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise CaptureContractError(
            "canonical capture cannot locate loaded PyTorch for pre-load stability verification"
        )
    try:
        origin = Path(module_file).resolve(strict=True)
    except OSError as exc:
        raise CaptureContractError(
            "unable to resolve loaded PyTorch package after authenticated model loading"
        ) from exc
    return _torch_package_stability_from_root_round61(origin.parent)


def _make_round61_constructor(original_init: Any):
    """Closure-bind the inherited constructor so the preload guard is not rebindable."""

    def __init__(self: Any, request: Mapping[str, Any]) -> None:
        before = _preimport_torch_package_stability_round61()
        original_init(self, request)
        if self._real_round56_torch_runtime():
            if before is None:
                raise CaptureContractError(
                    "canonical OBSERVATION could not establish a pre-load PyTorch package baseline"
                )
            after = _loaded_torch_package_stability_round61(self._torch)
            if after != before:
                raise CaptureContractError(
                    "imported PyTorch package changed transiently during authenticated model loading"
                )

    __init__.__name__ = "__init__"
    __init__.__qualname__ = f"{_Round56Backend.__qualname__}.__init__"
    __init__.__doc__ = original_init.__doc__
    return __init__


_Round56Backend.__init__ = _make_round61_constructor(_Round56Backend.__init__)
del _make_round61_constructor

HuggingFacePyTorchBackend = _Round56Backend

__all__ = ["HuggingFacePyTorchBackend"]

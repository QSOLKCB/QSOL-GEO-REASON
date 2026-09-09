"""Mapped native Metal/MPS runtime provenance for canonical MPS capture."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .capture_common import CaptureContractError
from . import capture_cuda_runtime as _runtime


_MPS_PATH_MARKERS = (
    "/metal.framework/",
    "/metalperformanceshaders.framework/",
    "/metalperformanceshadersgraph.framework/",
    "/mps.framework/",
    "/mpsgraph.framework/",
    "/libmetal",
    "agxmetal",
)
_MPS_BASENAMES = frozenset(
    {
        "metal",
        "metalperformanceshaders",
        "metalperformanceshadersgraph",
        "mps",
        "mpsgraph",
    }
)


def _is_mps_runtime_library(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    return name in _MPS_BASENAMES or any(marker in normalized for marker in _MPS_PATH_MARKERS)


def _loaded_mps_library_entries():
    if sys.platform != "darwin" or os.name == "nt":
        raise CaptureContractError(
            "canonical MPS observation requires Darwin mapped-runtime enumeration"
        )
    # Resolve through capture_cuda_runtime at call time. Round 58 replaces this
    # entry point with its dyld mapped executable-segment implementation, so MPS
    # receives the same cryptographic mapped-image identity as CPU/CUDA on Darwin.
    return list(_runtime._darwin_loaded_library_paths(_is_mps_runtime_library))


def loaded_mps_runtime_library_snapshot(
    *, require_nonempty: bool,
) -> tuple[dict[str, int | str], tuple[tuple[object, ...], ...]]:
    count, receipt, stability = _runtime._runtime_library_snapshot(
        _loaded_mps_library_entries(),
        predicate=_is_mps_runtime_library,
        label="MPS",
        require_nonempty=require_nonempty,
    )
    return (
        {
            "mps_runtime_library_file_count": count,
            "mps_runtime_library_receipt_sha256": receipt,
        },
        stability,
    )


def loaded_mps_runtime_library_provenance() -> dict[str, int | str]:
    provenance, _stability = loaded_mps_runtime_library_snapshot(require_nonempty=True)
    return provenance


__all__ = ["loaded_mps_runtime_library_provenance", "loaded_mps_runtime_library_snapshot"]

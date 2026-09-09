"""QSOL-GEO-REASON deterministic geometry and capture instrumentation."""
from __future__ import annotations

import importlib
from typing import Any

from .geometry import (
    align_pair,
    cosine_alignment,
    finite_difference,
    menger_curvature_sequence,
    path_length,
    resample_arclength,
)
from .simulation import run_recipe, validate_recipe

__version__ = "0.3.0.dev0"

_CAPTURE_EXPORTS = frozenset(
    {
        "CAPTURE_PROTOCOL_ID",
        "CaptureBackendUnavailable",
        "CaptureContractError",
        "execute_capture",
        "validate_capture_request",
        "write_capture_bundle",
    }
)


def __getattr__(name: str) -> Any:
    """Load the capture stack only when a capture API symbol is requested.

    Geometry/simulation imports no longer trigger production-backend composition as a
    package-import side effect.  The canonical CLI and worker import the capture stack
    explicitly at their documented boundaries.
    """
    if name in _CAPTURE_EXPORTS:
        module = importlib.import_module(".capture", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAPTURE_PROTOCOL_ID",
    "CaptureBackendUnavailable",
    "CaptureContractError",
    "align_pair",
    "cosine_alignment",
    "execute_capture",
    "finite_difference",
    "menger_curvature_sequence",
    "path_length",
    "resample_arclength",
    "run_recipe",
    "validate_capture_request",
    "validate_recipe",
    "write_capture_bundle",
]

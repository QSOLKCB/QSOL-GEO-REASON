"""QSOL-GEO-REASON deterministic geometry and capture instrumentation."""

from .capture import (
    CAPTURE_PROTOCOL_ID,
    CaptureBackendUnavailable,
    CaptureContractError,
    execute_capture,
    validate_capture_request,
    write_capture_bundle,
)
from .geometry import (
    align_pair,
    cosine_alignment,
    finite_difference,
    menger_curvature_sequence,
    path_length,
    resample_arclength,
)
from .simulation import run_recipe, validate_recipe

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

__version__ = "0.1.0"

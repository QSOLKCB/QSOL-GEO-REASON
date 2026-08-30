"""QSOL-GEO-REASON deterministic reference instrumentation."""

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
    "align_pair",
    "cosine_alignment",
    "finite_difference",
    "menger_curvature_sequence",
    "path_length",
    "resample_arclength",
    "run_recipe",
    "validate_recipe",
]

__version__ = "0.1.0"

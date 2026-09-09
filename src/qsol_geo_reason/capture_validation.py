"""Canonical request/checkpoint validation facade for GEO-CAP-001."""
from __future__ import annotations

from typing import Any

from .capture_common import CaptureContractError, _LOADING_INFO_KEYS
from .capture_validation_core import _quantization_reasons, validate_capture_request


def _validate_loading_info(loading_info: Any) -> None:
    """Require the complete Transformers loading diagnostic contract."""
    if not isinstance(loading_info, dict):
        raise CaptureContractError("Transformers did not return checkpoint loading information")

    missing = [key for key in _LOADING_INFO_KEYS if key not in loading_info]
    if missing:
        raise CaptureContractError(
            "checkpoint loading information is incomplete; missing diagnostics: "
            + ", ".join(sorted(missing))
        )

    problems: dict[str, list[Any]] = {}
    for key in _LOADING_INFO_KEYS:
        value = loading_info[key]
        if not isinstance(value, (list, tuple)):
            raise CaptureContractError(
                f"checkpoint loading info {key!r} has invalid shape; expected a sequence"
            )
        if value:
            problems[key] = list(value)
    if problems:
        summary = "; ".join(f"{key}={value!r}" for key, value in problems.items())
        raise CaptureContractError(
            "checkpoint load was not exact; canonical observation rejected: " + summary
        )


__all__ = ["validate_capture_request"]

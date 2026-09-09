"""Round-64 sealing for Hugging Face Hub package provenance.

The core loader now content-binds Hugging Face Hub before its first import and checks
critical Transformers/Safetensors loader bindings across every Hub snapshot lookup.
This layer keeps that package receipt in a closure-owned vault, re-authenticates the
live package before metadata publication, and extends the exact production metadata
contract/verifier with the Hub package receipt fields.
"""
from __future__ import annotations

import sys
import types
import weakref
from typing import Any, Mapping

from .capture_backend_round61 import HuggingFacePyTorchBackend as _Round61Backend
from .capture_common import CaptureContractError, _PRODUCTION_BACKEND_KEYS
from .capture_package import (
    _python_package_provenance,
    _validate_python_package_provenance,
)
from . import capture_provenance as _capture_provenance


_HUB_PACKAGE_FIELDS = frozenset(
    {
        "huggingface_hub_package_file_count",
        "huggingface_hub_package_receipt_sha256",
    }
)
_PRODUCTION_BACKEND_KEYS.update(_HUB_PACKAGE_FIELDS)


def _hub_receipt_from_mapping(value: Any) -> tuple[int, str]:
    if not isinstance(value, Mapping):
        raise CaptureContractError("Hugging Face Hub package provenance baseline is missing")
    count = value.get("file_count")
    receipt = value.get("receipt_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError("Hugging Face Hub package file count is malformed")
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(character not in "0123456789abcdef" for character in receipt)
    ):
        raise CaptureContractError("Hugging Face Hub package receipt is malformed")
    return count, receipt


def _live_hub_receipt() -> tuple[int, str]:
    module = sys.modules.get("huggingface_hub")
    if not isinstance(module, types.ModuleType):
        raise CaptureContractError(
            "canonical OBSERVATION lost the imported Hugging Face Hub package"
        )
    return _hub_receipt_from_mapping(
        _python_package_provenance(module, "Hugging Face Hub")
    )


def _make_round64_hub_vault():
    baselines: weakref.WeakKeyDictionary[Any, tuple[int, str]] = weakref.WeakKeyDictionary()

    def remember(instance: Any, receipt: tuple[int, str]) -> None:
        if instance in baselines:
            raise CaptureContractError(
                "Hugging Face Hub package provenance baseline was already initialized"
            )
        baselines[instance] = receipt

    def recall(instance: Any) -> tuple[int, str] | None:
        return baselines.get(instance)

    return remember, recall


_remember_round64_hub_receipt, _recall_round64_hub_receipt = _make_round64_hub_vault()
del _make_round64_hub_vault


def _make_round64_constructor(original_init: Any):
    def __init__(self: Any, request: Mapping[str, Any]) -> None:
        original_init(self, request)
        recorded = _hub_receipt_from_mapping(
            getattr(self, "_huggingface_hub_package_provenance", None)
        )
        live = _live_hub_receipt()
        if live != recorded:
            raise CaptureContractError(
                "Hugging Face Hub package changed during authenticated backend construction"
            )
        _remember_round64_hub_receipt(self, recorded)

    __init__.__name__ = "__init__"
    __init__.__qualname__ = f"{_Round61Backend.__qualname__}.__init__"
    __init__.__doc__ = original_init.__doc__
    return __init__


def _make_round64_metadata(original_metadata: Any):
    def metadata(self: Any) -> Mapping[str, Any]:
        baseline = _recall_round64_hub_receipt(self)
        if baseline is None:
            # Preserve dependency-free object-level fixtures that never entered the
            # real production constructor. Exact OBSERVATION verification still
            # requires the fields below.
            return original_metadata(self)
        live = _live_hub_receipt()
        if live != baseline:
            raise CaptureContractError(
                "Hugging Face Hub package changed after authenticated construction"
            )
        data = dict(original_metadata(self))
        data["huggingface_hub_package_file_count"] = baseline[0]
        data["huggingface_hub_package_receipt_sha256"] = baseline[1]
        return data

    metadata.__name__ = "metadata"
    metadata.__qualname__ = f"{_Round61Backend.__qualname__}.metadata"
    metadata.__doc__ = original_metadata.__doc__
    return metadata


_Round61Backend.__init__ = _make_round64_constructor(_Round61Backend.__init__)
_Round61Backend.metadata = _make_round64_metadata(_Round61Backend.metadata)
del _make_round64_constructor
del _make_round64_metadata


_ORIGINAL_VALIDATE_PRODUCTION_METADATA_SHAPE = (
    _capture_provenance._validate_production_metadata_shape
)


def _validate_production_metadata_shape_round64(
    observed: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    _ORIGINAL_VALIDATE_PRODUCTION_METADATA_SHAPE(observed, request)
    _validate_python_package_provenance(
        observed,
        count_field="huggingface_hub_package_file_count",
        receipt_field="huggingface_hub_package_receipt_sha256",
        where="Hugging Face Hub",
    )


_capture_provenance._validate_production_metadata_shape = (
    _validate_production_metadata_shape_round64
)

_ORIGINAL_VALIDATE_BACKEND_METADATA = _capture_provenance._validate_backend_metadata


def _validate_backend_metadata_round64(
    observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str
) -> None:
    _ORIGINAL_VALIDATE_BACKEND_METADATA(observed, request, evidence_class)
    if evidence_class != "OBSERVATION":
        return
    _validate_python_package_provenance(
        observed,
        count_field="huggingface_hub_package_file_count",
        receipt_field="huggingface_hub_package_receipt_sha256",
        where="Hugging Face Hub",
    )


_capture_provenance._validate_backend_metadata = _validate_backend_metadata_round64

HuggingFacePyTorchBackend = _Round61Backend

__all__ = ["HuggingFacePyTorchBackend"]
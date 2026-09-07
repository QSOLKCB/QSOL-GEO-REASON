"""Final audit hardening layer for GEO-CAP-001 production observations.

This layer closes audit findings that need to run around the existing production
constructor while deliberately leaving that constructor as the canonical implementation.
That keeps the long-established construction/restoration invariants source-visible and
adds only narrowly scoped pre-construction and provenance guards.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping

from .capture_common import CaptureContractError
from .capture_package import _python_package_provenance
from .capture_validation import validate_capture_request
from . import capture_backend_production as _production


_BaseProductionBackend = _production.HuggingFacePyTorchBackend


class HuggingFacePyTorchBackend(_BaseProductionBackend):
    """Audit-complete production backend exported at the canonical boundary."""

    _CPU_MATH_ENVIRONMENT_FIELDS = {
        "onednn_max_cpu_isa": "ONEDNN_MAX_CPU_ISA",
        "dnnl_max_cpu_isa": "DNNL_MAX_CPU_ISA",
        "mkl_cbwr": "MKL_CBWR",
    }

    def __new__(cls, request: Mapping[str, Any]):
        """Freeze initialization-time audit inputs before the inherited constructor."""
        validated = validate_capture_request(request)
        instance = super().__new__(cls)
        device = validated["backend"]["device"]

        # oneDNN/DNNL/MKL dispatch controls are initialization-time execution inputs.
        # Capture them before the inherited production constructor can import or
        # initialize PyTorch. If PyTorch already exists, historical values cannot be
        # reconstructed and the manifest records that fact explicitly.
        cpu_active = device == "cpu"
        known = cpu_active and "torch" not in sys.modules
        environment = {
            field: os.environ.get(variable)
            for field, variable in cls._CPU_MATH_ENVIRONMENT_FIELDS.items()
        }
        instance._canonical_cpu_math_environment_known = known if cpu_active else None
        instance._canonical_cpu_math_environment = environment if known else None

        # The native Tokenizers receipt is initialized only after the inherited
        # constructor has authenticated and loaded the tokenizer.
        instance._tokenizers_native_backend_active = False
        instance._tokenizers_package_provenance = None
        instance._tokenizers_package_provenance_initialized = False
        return instance

    def _assert_cpu_math_environment_policy(self) -> None:
        state = vars(self)
        # Dependency-free source-audit fixtures may deliberately bypass __new__.
        # Real construction always installs this boundary state.
        if "_canonical_cpu_math_environment_known" not in state:
            return
        if getattr(self, "_device_type", None) != "cpu":
            return
        known = state.get("_canonical_cpu_math_environment_known")
        expected = state.get("_canonical_cpu_math_environment")
        if known is True:
            observed = {
                field: os.environ.get(variable)
                for field, variable in self._CPU_MATH_ENVIRONMENT_FIELDS.items()
            }
            if observed != expected:
                raise CaptureContractError(
                    "CPU math-library dispatch override drifted after initialization"
                )
        elif known is not False:
            raise CaptureContractError("CPU math-library dispatch provenance is missing")

    def _initialize_tokenizers_package_provenance(self) -> None:
        state = vars(self)
        # Synthetic fixtures built with object.__new__ intentionally do not acquire
        # production provenance. Real construction installs this marker in __new__.
        if "_tokenizers_package_provenance_initialized" not in state:
            return
        if state.get("_tokenizers_package_provenance_initialized") is True:
            return
        tokenizer = state.get("_tokenizer")
        if tokenizer is None:
            return

        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        active = backend_tokenizer is not None
        state["_tokenizers_native_backend_active"] = active
        state["_tokenizers_package_provenance"] = None
        if active:
            process_tokenizers = sys.modules.get("tokenizers")
            if process_tokenizers is None:
                raise CaptureContractError(
                    "active fast tokenizer does not expose its imported native tokenizers package"
                )
            state["_tokenizers_package_provenance"] = _python_package_provenance(
                process_tokenizers, "Tokenizers"
            )
        state["_tokenizers_package_provenance_initialized"] = True

    def _assert_cpu_dispatch_policy(self) -> None:
        # Production __init__ invokes this after the inherited model/tokenizer load,
        # making it a construction-end hook without replacing the canonical __init__.
        super()._assert_cpu_dispatch_policy()
        self._assert_cpu_math_environment_policy()
        self._initialize_tokenizers_package_provenance()

    def metadata(self) -> Mapping[str, Any]:
        # super().metadata() dynamically reaches the override above, rechecking CPU
        # dispatch policy and ensuring the native-tokenizer baseline is initialized.
        data = dict(super().metadata())
        state = vars(self)

        if "_tokenizers_package_provenance_initialized" in state:
            if state.get("_tokenizers_package_provenance_initialized") is not True:
                raise CaptureContractError("native Tokenizers package provenance was not initialized")
            native_active = state.get("_tokenizers_native_backend_active")
            if type(native_active) is not bool:
                raise CaptureContractError("native Tokenizers backend activation state is invalid")
            tokenizers_provenance = state.get("_tokenizers_package_provenance")
            if native_active:
                process_tokenizers = sys.modules.get("tokenizers")
                if process_tokenizers is None or not isinstance(tokenizers_provenance, Mapping):
                    raise CaptureContractError("native Tokenizers package provenance is missing")
                if _python_package_provenance(
                    process_tokenizers, "Tokenizers"
                ) != dict(tokenizers_provenance):
                    raise CaptureContractError(
                        "imported Tokenizers package changed after authenticated backend construction"
                    )
            elif tokenizers_provenance is not None:
                raise CaptureContractError(
                    "slow tokenizer cannot carry native Tokenizers package provenance"
                )

            data["tokenizers_native_backend_active"] = native_active
            data["tokenizers_package_file_count"] = (
                tokenizers_provenance["file_count"]
                if isinstance(tokenizers_provenance, Mapping)
                else None
            )
            data["tokenizers_package_receipt_sha256"] = (
                tokenizers_provenance["receipt_sha256"]
                if isinstance(tokenizers_provenance, Mapping)
                else None
            )

        if "_canonical_cpu_math_environment_known" in state:
            known = state.get("_canonical_cpu_math_environment_known")
            environment = state.get("_canonical_cpu_math_environment")
            data["cpu_math_dispatch_env_known"] = known
            for field in self._CPU_MATH_ENVIRONMENT_FIELDS:
                data[field] = environment[field] if environment is not None else None
        return data


# capture.py imports this module before capture_execute imports the production module.
# Rebinding therefore preserves the exact-type OBSERVATION boundary while the inherited
# production constructor remains the single canonical construction implementation.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

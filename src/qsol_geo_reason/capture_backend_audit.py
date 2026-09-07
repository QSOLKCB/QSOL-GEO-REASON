"""Final audit hardening layer for GEO-CAP-001 production observations.

This layer is intentionally small: it closes the PR #4 audit findings that need a
construction-time boundary around the existing production backend without duplicating
the large fail-closed implementation below it.
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

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        device = validated["backend"]["device"]

        # These environment variables are initialization-time execution inputs for
        # oneDNN/DNNL/MKL. Freeze them before the inherited production constructor
        # can import/initialize PyTorch. If torch was already imported, their
        # historical values are unknowable and must remain explicitly unknown.
        cpu_active = device == "cpu"
        known = cpu_active and "torch" not in sys.modules
        environment = {
            field: os.environ.get(variable)
            for field, variable in self._CPU_MATH_ENVIRONMENT_FIELDS.items()
        }
        self._canonical_cpu_math_environment_known = known if cpu_active else None
        self._canonical_cpu_math_environment = environment if known else None
        self._tokenizers_native_backend_active = False
        self._tokenizers_package_provenance: dict[str, Any] | None = None

        super().__init__(validated)
        self._assert_cpu_math_environment_policy()

        # A fast Transformers tokenizer delegates tokenization to the imported
        # native `tokenizers` implementation. Persist a content receipt for that
        # executable dependency just as the production boundary already does for
        # Transformers and PyTorch.
        backend_tokenizer = getattr(getattr(self, "_tokenizer", None), "backend_tokenizer", None)
        self._tokenizers_native_backend_active = backend_tokenizer is not None
        if self._tokenizers_native_backend_active:
            process_tokenizers = sys.modules.get("tokenizers")
            if process_tokenizers is None:
                raise CaptureContractError(
                    "active fast tokenizer does not expose its imported native tokenizers package"
                )
            self._tokenizers_package_provenance = _python_package_provenance(
                process_tokenizers, "Tokenizers"
            )

    def _assert_cpu_math_environment_policy(self) -> None:
        if getattr(self, "_device_type", None) != "cpu":
            return
        known = getattr(self, "_canonical_cpu_math_environment_known", None)
        expected = getattr(self, "_canonical_cpu_math_environment", None)
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

    def _assert_cpu_dispatch_policy(self) -> None:
        super()._assert_cpu_dispatch_policy()
        self._assert_cpu_math_environment_policy()

    def metadata(self) -> Mapping[str, Any]:
        self._assert_cpu_math_environment_policy()

        native_active = getattr(self, "_tokenizers_native_backend_active", False)
        tokenizers_provenance = getattr(self, "_tokenizers_package_provenance", None)
        if native_active:
            process_tokenizers = sys.modules.get("tokenizers")
            if process_tokenizers is None or not isinstance(tokenizers_provenance, Mapping):
                raise CaptureContractError("native Tokenizers package provenance is missing")
            if _python_package_provenance(process_tokenizers, "Tokenizers") != dict(tokenizers_provenance):
                raise CaptureContractError(
                    "imported Tokenizers package changed after authenticated backend construction"
                )

        data = dict(super().metadata())
        data["tokenizers_native_backend_active"] = bool(native_active)
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

        known = getattr(self, "_canonical_cpu_math_environment_known", None)
        environment = getattr(self, "_canonical_cpu_math_environment", None)
        data["cpu_math_dispatch_env_known"] = known
        for field in self._CPU_MATH_ENVIRONMENT_FIELDS:
            data[field] = environment[field] if environment is not None else None
        return data


# Rebind the production module export before capture_execute imports it. This keeps
# the existing exact-type OBSERVATION boundary intact and means direct imports of
# capture_backend_production also receive the audited final concrete class after
# package initialization.
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

"""Round-39 production boundary: asynchronous-signal and CUDA-library provenance."""
from __future__ import annotations

from typing import Any, Mapping

from .capture_backend_final import HuggingFacePyTorchBackend as _FinalHuggingFacePyTorchBackend
from .capture_cuda_runtime import loaded_cuda_runtime_library_provenance
from .capture_signals import _assert_no_async_signal_instrumentation
from . import capture_backend_production as _production


class HuggingFacePyTorchBackend(_FinalHuggingFacePyTorchBackend):
    """Canonical backend with signal exclusion and mapped CUDA library receipts."""

    def _assert_no_active_torch_override_modes(self) -> None:
        # production.begin_observation() dynamically dispatches this immediately
        # after acquiring exclusive Python-thread execution and before capture-state
        # mutation. Signals are a second asynchronous execution source, so bind them
        # at exactly that boundary as well as in later live-state checks.
        super()._assert_no_active_torch_override_modes()
        _assert_no_async_signal_instrumentation()

    def metadata(self) -> Mapping[str, Any]:
        observed = dict(super().metadata())
        device = observed.get("device")
        if isinstance(device, str) and device.startswith("cuda:"):
            observed.update(loaded_cuda_runtime_library_provenance())
        else:
            observed["cuda_runtime_library_file_count"] = None
            observed["cuda_runtime_library_receipt_sha256"] = None
        return observed


# Keep the exact-type OBSERVATION gate pointed at the newest concrete boundary before
# capture_execute imports the production module.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

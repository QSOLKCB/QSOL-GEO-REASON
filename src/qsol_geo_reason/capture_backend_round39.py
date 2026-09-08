"""Round-39 production boundary: signal and mapped runtime-library provenance."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .capture_backend_final import HuggingFacePyTorchBackend as _FinalHuggingFacePyTorchBackend
from .capture_common import CaptureContractError
from .capture_cpu_runtime import loaded_cpu_runtime_library_provenance
from .capture_cuda_runtime import loaded_cuda_runtime_library_provenance
from .capture_signals import _assert_no_async_signal_instrumentation
from . import capture_backend_production as _production


def _append_runtime_record(
    observed: dict[str, Any], *, prefix: str, key: str, libraries: Mapping[str, Any]
) -> None:
    extension = json.dumps(
        {key: dict(libraries)},
        sort_keys=True,
        separators=(",", ":"),
    )
    config = observed.get("torch_build_config")
    if not isinstance(config, str) or not config.strip():
        raise CaptureContractError("canonical PyTorch build configuration is missing")
    config = config.rstrip("\n") + "\n" + prefix + extension + "\n"
    observed["torch_build_config"] = config
    observed["torch_build_config_sha256"] = hashlib.sha256(
        config.encode("utf-8")
    ).hexdigest()


class HuggingFacePyTorchBackend(_FinalHuggingFacePyTorchBackend):
    """Canonical backend with signal exclusion and mapped runtime-library receipts."""

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
            # Extend the authenticated build receipt with the actual mapped
            # CUDA/NVIDIA shared-object set observed after capture.
            _append_runtime_record(
                observed,
                prefix="QSOL_GEO_CUDA_RUNTIME=",
                key="loaded_cuda_runtime_libraries",
                libraries=loaded_cuda_runtime_library_provenance(),
            )
        elif device == "cpu":
            # Conda/system PyTorch builds can dispatch through MKL, oneDNN,
            # OpenMP, OpenBLAS, Accelerate, or related libraries outside the torch
            # package tree. Bind the mapped external CPU runtime set as part of the
            # same authenticated build-config receipt.
            _append_runtime_record(
                observed,
                prefix="QSOL_GEO_CPU_RUNTIME=",
                key="loaded_cpu_runtime_libraries",
                libraries=loaded_cpu_runtime_library_provenance(),
            )
        return observed


# Keep the exact-type OBSERVATION gate pointed at the newest concrete boundary before
# capture_execute imports the production module.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

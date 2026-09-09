"""Round-39 production boundary: signal and mapped runtime-library provenance."""
from __future__ import annotations

import hashlib
import json
import time
import weakref
from typing import Any, Mapping

from .capture_backend_final import HuggingFacePyTorchBackend as _FinalHuggingFacePyTorchBackend
from .capture_common import CaptureContractError
from .capture_cpu_runtime import (
    loaded_cpu_runtime_library_provenance,
    loaded_cpu_runtime_library_snapshot,
)
from .capture_cuda_runtime import (
    assert_runtime_library_state_stable,
    loaded_cuda_runtime_library_provenance,
    loaded_cuda_runtime_library_snapshot,
)
from .capture_mps_runtime import loaded_mps_runtime_library_snapshot
from .capture_signals import _assert_no_async_signal_instrumentation
from . import capture_backend_production as _production


def _make_runtime_library_baseline_vault():
    baselines: weakref.WeakKeyDictionary[Any, tuple[Any, ...]] = weakref.WeakKeyDictionary()

    def remember(instance: Any, baseline: tuple[Any, ...]) -> None:
        if instance in baselines:
            raise CaptureContractError("runtime-library stability baseline was already initialized")
        baselines[instance] = baseline

    def recall(instance: Any) -> tuple[Any, ...] | None:
        return baselines.get(instance)

    def forget(instance: Any) -> None:
        baselines.pop(instance, None)

    return remember, recall, forget


(
    _remember_runtime_library_baseline,
    _recall_runtime_library_baseline,
    _forget_runtime_library_baseline,
) = _make_runtime_library_baseline_vault()
del _make_runtime_library_baseline_vault


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

    def _begin_runtime_library_stability_window(self) -> None:
        super()._begin_runtime_library_stability_window()
        observation_started_ns = time.time_ns()
        device = getattr(self, "_device", None)
        cuda_active = isinstance(device, str) and device.startswith("cuda:")
        mps_active = device == "mps"
        _cpu_provenance, cpu_state = loaded_cpu_runtime_library_snapshot()
        cuda_state = None
        mps_state = None
        if cuda_active:
            _cuda_provenance, cuda_state = loaded_cuda_runtime_library_snapshot()
        if mps_active:
            # Metal/MPS frameworks may be loaded lazily by the first model forward.
            _mps_provenance, mps_state = loaded_mps_runtime_library_snapshot(
                require_nonempty=False
            )
        _remember_runtime_library_baseline(
            self,
            (observation_started_ns, cpu_state, cuda_state, mps_state),
        )

    def _assert_no_active_torch_override_modes(self) -> None:
        super()._assert_no_active_torch_override_modes()
        _assert_no_async_signal_instrumentation()

    def metadata(self) -> Mapping[str, Any]:
        observed = dict(super().metadata())
        device = observed.get("device")
        cuda_active = isinstance(device, str) and device.startswith("cuda:")
        mps_active = device == "mps"
        production_device = device in {"cpu", "mps"} or cuda_active
        baseline = _recall_runtime_library_baseline(self)
        if production_device and getattr(self, "_observation_active", False) and baseline is None:
            raise CaptureContractError(
                "canonical observation is missing its pre-execution runtime-library stability receipt"
            )

        cpu_libraries: Mapping[str, Any] | None = None
        cuda_libraries: Mapping[str, Any] | None = None
        mps_libraries: Mapping[str, Any] | None = None
        if production_device:
            cpu_libraries, cpu_state = loaded_cpu_runtime_library_snapshot()
            if baseline is not None:
                observation_started_ns, cpu_before, cuda_before, mps_before = baseline
                assert_runtime_library_state_stable(
                    cpu_before,
                    cpu_state,
                    observation_started_ns=observation_started_ns,
                    label="CPU",
                )
                if cuda_active:
                    if cuda_before is None:
                        raise CaptureContractError(
                            "canonical CUDA observation is missing its pre-execution runtime receipt"
                        )
                    cuda_libraries, cuda_state = loaded_cuda_runtime_library_snapshot()
                    assert_runtime_library_state_stable(
                        cuda_before,
                        cuda_state,
                        observation_started_ns=observation_started_ns,
                        label="CUDA",
                    )
                if mps_active:
                    if mps_before is None:
                        raise CaptureContractError(
                            "canonical MPS observation is missing its pre-execution runtime receipt"
                        )
                    mps_libraries, mps_state = loaded_mps_runtime_library_snapshot(
                        require_nonempty=True
                    )
                    assert_runtime_library_state_stable(
                        mps_before,
                        mps_state,
                        observation_started_ns=observation_started_ns,
                        label="MPS",
                    )
            elif cuda_active:
                cuda_libraries, _cuda_state = loaded_cuda_runtime_library_snapshot()
            elif mps_active:
                mps_libraries, _mps_state = loaded_mps_runtime_library_snapshot(
                    require_nonempty=True
                )

            # MPS records precede the CPU-pooling record so the existing canonical
            # CPU receipt remains terminal on non-CUDA lanes. The semantic MPS
            # validator requires this exact adjacency.
            if mps_active:
                if mps_libraries is None:
                    mps_libraries, _mps_state = loaded_mps_runtime_library_snapshot(
                        require_nonempty=True
                    )
                _append_runtime_record(
                    observed,
                    prefix="QSOL_GEO_MPS_RUNTIME=",
                    key="loaded_mps_runtime_libraries",
                    libraries=mps_libraries,
                )

            # Every canonical lane performs selected-span pooling on CPU in float64.
            _append_runtime_record(
                observed,
                prefix="QSOL_GEO_CPU_RUNTIME=",
                key="loaded_cpu_runtime_libraries",
                libraries=cpu_libraries,
            )
        if cuda_active:
            if cuda_libraries is None:
                cuda_libraries, _cuda_state = loaded_cuda_runtime_library_snapshot()
            _append_runtime_record(
                observed,
                prefix="QSOL_GEO_CUDA_RUNTIME=",
                key="loaded_cuda_runtime_libraries",
                libraries=cuda_libraries,
            )
        return observed


HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]
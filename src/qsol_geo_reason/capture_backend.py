"""Canonical execution-policy facade over the audited Hugging Face/PyTorch backend core."""
from __future__ import annotations

import os
from typing import Any, Mapping

from .capture_common import CaptureContractError
from .capture_validation import validate_capture_request
from .capture_backend_core import HuggingFacePyTorchBackend as _CoreHuggingFacePyTorchBackend


class _ExplicitNoAttentionBaseModel:
    """Delegate to the resolved base model while forcing attention outputs off."""

    def __init__(self, module: Any):
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["output_attentions"] = False
        return self._module(*args, **kwargs)


class HuggingFacePyTorchBackend(_CoreHuggingFacePyTorchBackend):
    """Canonical backend with process-global execution policies frozen per capture."""

    _CUDA_ENVIRONMENT_FIELDS = {
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
        "nvidia_tf32_override": "NVIDIA_TF32_OVERRIDE",
        "torch_allow_tf32_cublas_override": "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG",
    }

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        device = validated["backend"]["device"]
        self._canonical_cuda_environment: dict[str, str | None] | None = None
        if device.startswith("cuda:"):
            # Snapshot before the core imports/initializes CUDA or cuBLAS state.
            self._canonical_cuda_environment = self._cuda_environment_state()
        self._canonical_deterministic_algorithms_enabled: bool | None = None

        super().__init__(validated)

        # CPU float64 pooling is used for every canonical device, not only CPU models.
        if getattr(self, "_canonical_cpu_thread_policy", None) is None:
            self._canonical_cpu_thread_policy = self._cpu_thread_policy_state()
        self._last_cpu_thread_policy = dict(self._canonical_cpu_thread_policy)

        deterministic_state = self._deterministic_algorithms_state()
        if self._determinism_mode == "required" and deterministic_state is not True:
            raise CaptureContractError(
                "required determinism requires deterministic algorithms enabled at construction"
            )
        self._canonical_deterministic_algorithms_enabled = deterministic_state
        self._last_deterministic_algorithms_enabled = deterministic_state

        if self._canonical_cuda_environment is not None:
            self._assert_cuda_environment_policy()

        # The core forward remains unchanged; this proxy injects an explicit
        # output_attentions=False into every resolved base-model call.
        self._base_model = _ExplicitNoAttentionBaseModel(self._base_model)

    def _cuda_environment_state(self) -> dict[str, str | None]:
        return {
            field: os.environ.get(environment)
            for field, environment in self._CUDA_ENVIRONMENT_FIELDS.items()
        }

    def _assert_cuda_environment_policy(self) -> dict[str, str | None]:
        expected = getattr(self, "_canonical_cuda_environment", None)
        if expected is None:
            return {}
        observed = self._cuda_environment_state()
        if observed != expected:
            raise CaptureContractError(
                "canonical CUDA environment policy drifted after runtime initialization: "
                f"expected={expected!r} observed={observed!r}"
            )
        return observed

    def _assert_canonical_determinism_policy(self) -> bool:
        enabled = self._deterministic_algorithms_state()
        expected = getattr(self, "_canonical_deterministic_algorithms_enabled", None)
        if expected is None:
            if self._determinism_mode == "required" and enabled is not True:
                raise CaptureContractError(
                    "required determinism policy drifted: deterministic algorithms are disabled"
                )
        elif enabled is not expected:
            raise CaptureContractError(
                "canonical deterministic-algorithm policy drifted from construction state: "
                f"expected={expected!r} observed={enabled!r}"
            )
        self._last_deterministic_algorithms_enabled = enabled
        return enabled

    def _force_canonical_determinism_policy(self) -> None:
        expected = getattr(self, "_canonical_deterministic_algorithms_enabled", None)
        if expected is None:
            expected = True if self._determinism_mode == "required" else self._deterministic_algorithms_state()
        setter = getattr(self._torch, "use_deterministic_algorithms", None)
        if not callable(setter):
            raise CaptureContractError(
                "canonical capture requires torch.use_deterministic_algorithms to restore execution policy"
            )
        if self._deterministic_algorithms_state() is not expected:
            setter(expected)
        self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()

    # Preserve the historical method names used by the audited core and tests,
    # but make them type the entire canonical determinism policy, including best_effort.
    def _assert_required_determinism_policy(self) -> bool:
        return self._assert_canonical_determinism_policy()

    def _force_required_determinism_policy(self) -> None:
        self._force_canonical_determinism_policy()

    def _assert_attention_implementation(self) -> None:
        super()._assert_attention_implementation()
        # This method is called immediately before every core forward.
        if getattr(self, "_canonical_deterministic_algorithms_enabled", None) is not None:
            self._force_canonical_determinism_policy()
        if getattr(self, "_canonical_cuda_environment", None) is not None:
            self._assert_cuda_environment_policy()

    def _pool_tensor_record(self, tensor: Any, **kwargs: Any) -> Mapping[str, Any]:
        # Every canonical pooling path reduces on CPU float64. Freeze CPU thread
        # policy even when the source tensor came from CUDA or MPS.
        if getattr(self, "_canonical_cpu_thread_policy", None) is not None:
            self._force_cpu_thread_policy()
        if getattr(self, "_canonical_deterministic_algorithms_enabled", None) is not None:
            self._assert_canonical_determinism_policy()
        if getattr(self, "_canonical_cuda_environment", None) is not None:
            self._assert_cuda_environment_policy()

        record = super()._pool_tensor_record(tensor, **kwargs)

        if getattr(self, "_canonical_cpu_thread_policy", None) is not None:
            self._last_cpu_thread_policy = self._assert_cpu_thread_policy()
        if getattr(self, "_canonical_deterministic_algorithms_enabled", None) is not None:
            self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()
        if getattr(self, "_canonical_cuda_environment", None) is not None:
            self._assert_cuda_environment_policy()
        return record

    def metadata(self) -> Mapping[str, Any]:
        if getattr(self, "_canonical_cuda_environment", None) is not None:
            self._assert_cuda_environment_policy()
        if getattr(self, "_canonical_deterministic_algorithms_enabled", None) is not None:
            self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()
        if getattr(self, "_canonical_cpu_thread_policy", None) is not None:
            self._last_cpu_thread_policy = self._assert_cpu_thread_policy()

        data = dict(super().metadata())

        # Preserve the verified per-pooling thread policy for every device rather
        # than a late ambient sample from _cpu_hardware_metadata.
        cpu_threads = self._last_cpu_thread_policy
        cpu_hardware = {
            "torch_num_threads": data.get("torch_num_threads"),
            "torch_num_interop_threads": data.get("torch_num_interop_threads"),
        }
        if cpu_threads is not None:
            cpu_hardware["torch_num_threads"] = cpu_threads["torch_num_threads"]
            cpu_hardware["torch_num_interop_threads"] = cpu_threads["torch_num_interop_threads"]
            data.update(cpu_hardware)

        # Keep prior provenance-source guarantees visible in this facade.
        _ = self._last_cuda_float32_policy
        data["cpu_mkldnn_enabled"] = data.get("cpu_mkldnn_enabled")
        data["cpu_mkldnn_matmul_fp32_precision"] = data.get("cpu_mkldnn_matmul_fp32_precision")

        if self._canonical_cuda_environment is not None:
            for field, value in self._canonical_cuda_environment.items():
                data[field] = value
        return data


__all__ = ["HuggingFacePyTorchBackend"]

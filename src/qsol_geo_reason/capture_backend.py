"""Canonical execution-policy facade over the audited Hugging Face/PyTorch backend core."""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

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
        self._canonical_deterministic_warn_only_enabled: bool | None = None
        self._canonical_evaluation_mode = False

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
        # Warn-only would allow a required deterministic violation to continue.
        # Canonical capture therefore fixes this process-global switch to false
        # in both required and best-effort modes.
        self._canonical_deterministic_warn_only_enabled = False
        self._force_canonical_determinism_policy()

        if self._canonical_cuda_environment is not None:
            self._assert_cuda_environment_policy()

        # The construction-time core already called model.eval(); make the
        # evaluation lane an explicit reusable-backend invariant as well.
        self._canonical_evaluation_mode = True
        self._force_model_eval_policy()

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

    def _deterministic_warn_only_state(self) -> bool:
        getter = getattr(self._torch, "is_deterministic_algorithms_warn_only_enabled", None)
        if not callable(getter):
            raise CaptureContractError(
                "canonical capture requires torch.is_deterministic_algorithms_warn_only_enabled"
            )
        value = getter()
        if not isinstance(value, bool):
            raise CaptureContractError("deterministic warn-only state must be boolean")
        return value

    def _assert_canonical_determinism_policy(self) -> bool:
        enabled = self._deterministic_algorithms_state()
        expected = getattr(self, "_canonical_deterministic_algorithms_enabled", None)
        if expected is None:
            if getattr(self, "_determinism_mode", "best_effort") == "required" and enabled is not True:
                raise CaptureContractError(
                    "required determinism policy drifted: deterministic algorithms are disabled"
                )
        elif enabled is not expected:
            raise CaptureContractError(
                "canonical deterministic-algorithm policy drifted from construction state: "
                f"expected={expected!r} observed={enabled!r}"
            )

        warn_expected = getattr(self, "_canonical_deterministic_warn_only_enabled", None)
        if warn_expected is not None:
            warn_only = self._deterministic_warn_only_state()
            if warn_only is not warn_expected:
                raise CaptureContractError(
                    "canonical deterministic warn-only policy drifted from the required false state: "
                    f"observed={warn_only!r}"
                )
        self._last_deterministic_algorithms_enabled = enabled
        return enabled

    def _force_canonical_determinism_policy(self) -> None:
        expected = getattr(self, "_canonical_deterministic_algorithms_enabled", None)
        if expected is None:
            mode = getattr(self, "_determinism_mode", None)
            if mode is None:
                return
            expected = True if mode == "required" else self._deterministic_algorithms_state()
        setter = getattr(self._torch, "use_deterministic_algorithms", None)
        if not callable(setter):
            raise CaptureContractError(
                "canonical capture requires torch.use_deterministic_algorithms to restore execution policy"
            )

        warn_expected = getattr(self, "_canonical_deterministic_warn_only_enabled", None)
        enabled_now = self._deterministic_algorithms_state()
        if warn_expected is None:
            if enabled_now is not expected:
                setter(expected)
        else:
            warn_now = self._deterministic_warn_only_state()
            if enabled_now is not expected or warn_now is not warn_expected:
                try:
                    setter(expected, warn_only=warn_expected)
                except TypeError as exc:
                    raise CaptureContractError(
                        "canonical capture requires warn_only control on torch.use_deterministic_algorithms"
                    ) from exc
        self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()

    # Preserve historical helper names used by the audited core, but enforce
    # the complete frozen state even when determinism.mode is best_effort.
    def _assert_required_determinism_policy(self) -> bool:
        return self._assert_canonical_determinism_policy()

    def _force_required_determinism_policy(self) -> None:
        self._force_canonical_determinism_policy()

    def _assert_model_eval_policy(self) -> None:
        if not getattr(self, "_canonical_evaluation_mode", False):
            return
        training = getattr(self._model, "training", None)
        if training is not False:
            raise CaptureContractError(
                "canonical capture requires the model to remain in evaluation mode"
            )

    def _force_model_eval_policy(self) -> None:
        if not getattr(self, "_canonical_evaluation_mode", False):
            return
        evaluator = getattr(self._model, "eval", None)
        if not callable(evaluator):
            raise CaptureContractError("canonical capture requires model.eval()")
        evaluator()
        self._assert_model_eval_policy()

    def _sdpa_policy_state(self) -> dict[str, bool | None]:
        state = dict(super()._sdpa_policy_state())
        cuda_backend = getattr(self._torch.backends, "cuda", None)
        getter = getattr(cuda_backend, "fp16_bf16_reduction_math_sdp_allowed", None)
        if not callable(getter):
            raise CaptureContractError(
                "canonical CUDA SDPA policy requires torch.backends.cuda."
                "fp16_bf16_reduction_math_sdp_allowed"
            )
        reduced_math = getter()
        if not isinstance(reduced_math, bool):
            raise CaptureContractError("math SDPA reduced-precision reduction state must be boolean")
        state["fp16_bf16_math_reduction"] = reduced_math
        return state

    def _assert_sdpa_math_policy(self) -> dict[str, bool | None]:
        state = super()._assert_sdpa_math_policy()
        if state.get("fp16_bf16_math_reduction") is not False:
            raise CaptureContractError(
                "canonical CUDA math SDPA requires FP16/BF16 reduced-precision reductions disabled"
            )
        return state

    def _force_sdpa_math_policy(self) -> None:
        cuda_backend = getattr(self._torch.backends, "cuda", None)
        toggle = getattr(cuda_backend, "allow_fp16_bf16_reduction_math_sdp", None)
        if not callable(toggle):
            raise CaptureContractError(
                "canonical CUDA SDPA policy requires torch.backends.cuda."
                "allow_fp16_bf16_reduction_math_sdp"
            )
        toggle(False)
        super()._force_sdpa_math_policy()

    def _assert_attention_implementation(self) -> None:
        super()._assert_attention_implementation()
        # The core invokes this immediately before every forward.
        if getattr(self, "_canonical_evaluation_mode", False):
            self._force_model_eval_policy()
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

    def hidden_states(
        self,
        input_ids: Sequence[int],
        layer_indices: Sequence[int],
        *,
        pool_span: tuple[int, int],
    ) -> Mapping[int, Mapping[str, Any]]:
        # Synthetic unit-test backends can deliberately bypass __init__. New
        # policy guards apply only when the corresponding construction snapshot
        # exists; every real backend created through __init__ has all snapshots.
        determinism_frozen = getattr(self, "_canonical_deterministic_algorithms_enabled", None) is not None
        threads_frozen = getattr(self, "_canonical_cpu_thread_policy", None) is not None
        cuda_environment_frozen = getattr(self, "_canonical_cuda_environment", None) is not None
        evaluation_frozen = getattr(self, "_canonical_evaluation_mode", False)

        if determinism_frozen:
            self._force_canonical_determinism_policy()
        if threads_frozen:
            self._force_cpu_thread_policy()
        if cuda_environment_frozen:
            self._assert_cuda_environment_policy()
        if evaluation_frozen:
            self._force_model_eval_policy()

        result = super().hidden_states(input_ids, layer_indices, pool_span=pool_span)

        # Verify again after the entire base-model forward, not merely after an
        # intermediate hook, so process-global and module-mode state cannot drift
        # without invalidating the canonical capture.
        if determinism_frozen:
            self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()
        if threads_frozen:
            self._last_cpu_thread_policy = self._assert_cpu_thread_policy()
        if cuda_environment_frozen:
            self._assert_cuda_environment_policy()
        if evaluation_frozen:
            self._assert_model_eval_policy()
        return result

    def metadata(self) -> Mapping[str, Any]:
        if getattr(self, "_canonical_cuda_environment", None) is not None:
            self._assert_cuda_environment_policy()
        if getattr(self, "_canonical_deterministic_algorithms_enabled", None) is not None:
            self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()
        if getattr(self, "_canonical_cpu_thread_policy", None) is not None:
            self._last_cpu_thread_policy = self._assert_cpu_thread_policy()
        if getattr(self, "_canonical_evaluation_mode", False):
            self._assert_model_eval_policy()

        data = dict(super().metadata())

        # Preserve the verified per-pooling thread policy for every device rather
        # than a late ambient sample from _cpu_hardware_metadata.
        cpu_threads = getattr(self, "_last_cpu_thread_policy", None)
        if cpu_threads is not None:
            data["torch_num_threads"] = cpu_threads["torch_num_threads"]
            data["torch_num_interop_threads"] = cpu_threads["torch_num_interop_threads"]

        if self._canonical_cuda_environment is not None:
            for field, value in self._canonical_cuda_environment.items():
                data[field] = value
        return data


__all__ = ["HuggingFacePyTorchBackend"]

"""Canonical execution-policy facade over the audited Hugging Face/PyTorch backend core."""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .capture_validation import validate_capture_request
from .capture_provenance import _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS
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
        self._validate_pre_cuda_environment(validated)
        self._canonical_cuda_environment: dict[str, str | None] | None = None
        if device.startswith("cuda:"):
            # Snapshot before the core imports/initializes CUDA or cuBLAS state.
            self._canonical_cuda_environment = self._cuda_environment_state()
        self._canonical_deterministic_algorithms_enabled: bool | None = None
        self._canonical_deterministic_warn_only_enabled: bool | None = None
        self._canonical_evaluation_mode = False
        self._canonical_model_live_state: tuple[tuple[Any, ...], ...] | None = None
        self._canonical_tokenizer_live_state: str | None = None
        self._live_state_seal_initialized = False

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

        # Seal the live objects after authenticated loading, device placement,
        # and canonical evaluation-mode setup. Snapshot receipts authenticate the
        # source files; these seals prevent later in-memory mutation from being
        # misattributed to those immutable snapshot identities.
        self._canonical_model_live_state = self._model_live_state_seal()
        self._canonical_tokenizer_live_state = self._tokenizer_live_state_seal()
        self._live_state_seal_initialized = True

        # The core forward remains unchanged; this proxy injects an explicit
        # output_attentions=False into every resolved base-model call.
        self._base_model = _ExplicitNoAttentionBaseModel(self._base_model)

    @classmethod
    def _validate_pre_cuda_environment(cls, request: Mapping[str, Any]) -> None:
        device = request["backend"]["device"]
        if not device.startswith("cuda:") or request["determinism"]["mode"] != "required":
            return
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
            raise CaptureContractError(
                "required-determinism CUDA capture requires CUBLAS_WORKSPACE_CONFIG to be "
                "':4096:8' or ':16:8' before CUDA initialization"
            )

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

    @staticmethod
    def _runtime_json_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): HuggingFacePyTorchBackend._runtime_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [HuggingFacePyTorchBackend._runtime_json_value(item) for item in value]
        return repr(value)

    def _model_live_state_seal(self) -> tuple[tuple[Any, ...], ...]:
        entries: list[tuple[Any, ...]] = []
        for kind, getter_name in (
            ("parameter", "named_parameters"),
            ("buffer", "named_buffers"),
        ):
            getter = getattr(self._model, getter_name, None)
            if not callable(getter):
                raise CaptureContractError(
                    f"canonical live-state authentication requires model.{getter_name}()"
                )
            for name, tensor in getter():
                if not isinstance(name, str) or not name:
                    raise CaptureContractError("canonical model state contains an invalid tensor name")
                version = getattr(tensor, "_version", None)
                if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                    raise CaptureContractError(
                        f"canonical model tensor {name!r} does not expose a valid mutation version"
                    )
                shape_value = getattr(tensor, "shape", None)
                if shape_value is None:
                    raise CaptureContractError(f"canonical model tensor {name!r} has no shape")
                try:
                    shape = tuple(int(dimension) for dimension in shape_value)
                except (TypeError, ValueError) as exc:
                    raise CaptureContractError(
                        f"canonical model tensor {name!r} has an invalid shape"
                    ) from exc
                dtype = str(getattr(tensor, "dtype", ""))
                device = str(getattr(tensor, "device", ""))
                if not dtype or not device:
                    raise CaptureContractError(
                        f"canonical model tensor {name!r} lacks dtype/device identity"
                    )
                entries.append((kind, name, id(tensor), version, shape, dtype, device))
        if not entries:
            raise CaptureContractError("canonical model exposes no parameter or buffer state to authenticate")
        return tuple(entries)

    def _tokenizer_live_state_seal(self) -> str:
        tokenizer = self._tokenizer
        vocab_getter = getattr(tokenizer, "get_vocab", None)
        if not callable(vocab_getter):
            raise CaptureContractError("canonical tokenizer live-state authentication requires get_vocab()")
        vocab = vocab_getter()
        if not isinstance(vocab, Mapping) or not vocab:
            raise CaptureContractError("canonical tokenizer vocabulary is missing")
        normalized_vocab: dict[str, int] = {}
        for token, index in vocab.items():
            if not isinstance(token, str) or isinstance(index, bool) or not isinstance(index, int):
                raise CaptureContractError("canonical tokenizer vocabulary is malformed")
            normalized_vocab[token] = index

        backend_state = None
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        serializer = getattr(backend_tokenizer, "to_str", None)
        if callable(serializer):
            backend_state = serializer()
            if not isinstance(backend_state, str) or not backend_state:
                raise CaptureContractError("canonical tokenizer backend serialization is invalid")

        payload = {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab": dict(sorted(normalized_vocab.items())),
            "backend_tokenizer": backend_state,
            "added_tokens_encoder": self._runtime_json_value(
                getattr(tokenizer, "added_tokens_encoder", {})
            ),
            "special_tokens_map": self._runtime_json_value(
                getattr(tokenizer, "special_tokens_map", {})
            ),
            "all_special_tokens": self._runtime_json_value(
                getattr(tokenizer, "all_special_tokens", [])
            ),
            "all_special_ids": self._runtime_json_value(
                getattr(tokenizer, "all_special_ids", [])
            ),
            "init_kwargs": self._runtime_json_value(getattr(tokenizer, "init_kwargs", {})),
        }
        return sha256_json(payload)

    def _assert_live_state_authentication(self) -> None:
        if not getattr(self, "_live_state_seal_initialized", False):
            return
        expected_model = getattr(self, "_canonical_model_live_state", None)
        expected_tokenizer = getattr(self, "_canonical_tokenizer_live_state", None)
        if expected_model is None or expected_tokenizer is None:
            raise CaptureContractError("canonical live-state authentication seal is missing")
        if self._model_live_state_seal() != expected_model:
            raise CaptureContractError(
                "live model state changed after authenticated checkpoint loading"
            )
        if self._tokenizer_live_state_seal() != expected_tokenizer:
            raise CaptureContractError(
                "live tokenizer state changed after authenticated snapshot loading"
            )

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
        super().assert_execution_request(request)
        self._assert_live_state_authentication()

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
        live_state_frozen = getattr(self, "_live_state_seal_initialized", False)

        if live_state_frozen:
            self._assert_live_state_authentication()
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
        # intermediate hook, so process-global, module-mode, and live object state
        # cannot drift without invalidating the canonical capture.
        if determinism_frozen:
            self._last_deterministic_algorithms_enabled = self._assert_canonical_determinism_policy()
        if threads_frozen:
            self._last_cpu_thread_policy = self._assert_cpu_thread_policy()
        if cuda_environment_frozen:
            self._assert_cuda_environment_policy()
        if evaluation_frozen:
            self._assert_model_eval_policy()
        if live_state_frozen:
            self._assert_live_state_authentication()
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
        if getattr(self, "_live_state_seal_initialized", False):
            self._assert_live_state_authentication()

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

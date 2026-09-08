"""State-restoring, single-use canonical OBSERVATION backend.

This layer restores host-process execution policy and the requested device's RNG.
It also seals executable module-forward identity and mutable tokenizer behavior.
The production layer supplies the observation's Python-thread exclusion boundary.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_backend import HuggingFacePyTorchBackend as _PolicyHuggingFacePyTorchBackend
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_runtime import _restore_accelerator_rng, _seed_capture_generators, _snapshot_accelerator_rng
from .capture_validation import validate_capture_request


class HuggingFacePyTorchBackend(_PolicyHuggingFacePyTorchBackend):
    """Canonical backend with scoped process-state restoration and single-use execution."""

    _TOKENIZER_BEHAVIOR_FIELDS = (
        "add_prefix_space",
        "do_lower_case",
        "strip_accents",
        "tokenize_chinese_chars",
        "clean_up_tokenization_spaces",
        "split_special_tokens",
        "padding_side",
        "truncation_side",
        "model_max_length",
        "legacy",
        "use_default_system_prompt",
        "spaces_between_special_tokens",
        "chat_template",
    )

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        # Preserve the pre-CUDA fail-closed ordering required by GEO-CAP-001.
        self._validate_pre_cuda_environment(validated)
        device = validated["backend"]["device"]
        self._canonical_cuda_environment = None
        if device.startswith("cuda:"):
            self._canonical_cuda_environment = self._cuda_environment_state()

        try:
            import torch as process_torch
        except ImportError:
            # Delegate to the existing constructor for the canonical dependency error.
            super().__init__(validated)
            raise CaptureBackendUnavailable("canonical capture requires PyTorch")

        ambient = self._snapshot_torch_process_state(process_torch, device)
        try:
            super().__init__(validated)
        finally:
            # Restore construction-time changes, including partial initialization.
            # CPU capture never snapshots or seeds an unrelated accelerator.
            self._restore_torch_process_state(process_torch, ambient)

        self._canonical_model_executable_state = self._model_executable_state_seal()
        self._observation_consumed = False
        self._observation_active = False
        self._observation_ambient_process_state: dict[str, Any] | None = None

    @staticmethod
    def _clone_rng_state(state: Any) -> Any:
        clone = getattr(state, "clone", None)
        return clone() if callable(clone) else state

    @classmethod
    def _snapshot_torch_execution_policy_state(cls, torch: Any) -> dict[str, Any]:
        """Snapshot mutable PyTorch execution policies without requiring a specific device."""
        state: dict[str, Any] = {}

        for key, getter_name in (
            ("torch_num_threads", "get_num_threads"),
            ("torch_num_interop_threads", "get_num_interop_threads"),
        ):
            getter = getattr(torch, getter_name, None)
            if callable(getter):
                value = getter()
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise CaptureContractError(
                        f"ambient PyTorch execution policy field {key} must be a positive integer"
                    )
                state[key] = value

        backends = getattr(torch, "backends", None)
        mkldnn = getattr(backends, "mkldnn", None) if backends is not None else None
        if mkldnn is not None:
            enabled = getattr(mkldnn, "enabled", None)
            if enabled is not None:
                if not isinstance(enabled, bool):
                    raise CaptureContractError("ambient torch.backends.mkldnn.enabled must be boolean")
                state["cpu_mkldnn_enabled"] = enabled
            matmul = getattr(mkldnn, "matmul", None)
            precision = getattr(matmul, "fp32_precision", None) if matmul is not None else None
            if precision is not None:
                if not isinstance(precision, str) or not precision.strip():
                    raise CaptureContractError(
                        "ambient torch.backends.mkldnn.matmul.fp32_precision must be non-empty"
                    )
                state["cpu_mkldnn_matmul_fp32_precision"] = precision

        float32_getter = getattr(torch, "get_float32_matmul_precision", None)
        if callable(float32_getter):
            precision = float32_getter()
            if precision not in {"highest", "high", "medium"}:
                raise CaptureContractError(
                    f"ambient torch float32 matmul precision is invalid: {precision!r}"
                )
            state["float32_matmul_precision"] = precision

        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        cuda_matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
        if cuda_matmul is not None:
            for key, name in (
                ("cuda_matmul_allow_tf32", "allow_tf32"),
                ("cuda_allow_fp16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"),
                ("cuda_allow_bf16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"),
            ):
                value = getattr(cuda_matmul, name, None)
                if value is not None:
                    if not isinstance(value, bool):
                        raise CaptureContractError(
                            f"ambient torch.backends.cuda.matmul.{name} must be boolean"
                        )
                    state[key] = value

        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        cudnn_tf32 = getattr(cudnn, "allow_tf32", None) if cudnn is not None else None
        if cudnn_tf32 is not None:
            if not isinstance(cudnn_tf32, bool):
                raise CaptureContractError("ambient torch.backends.cudnn.allow_tf32 must be boolean")
            state["cudnn_allow_tf32"] = cudnn_tf32

        if cuda_backend is not None:
            for key, getter_name in (
                ("cuda_flash_sdp_enabled", "flash_sdp_enabled"),
                ("cuda_mem_efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
                ("cuda_math_sdp_enabled", "math_sdp_enabled"),
                ("cuda_cudnn_sdp_enabled", "cudnn_sdp_enabled"),
                (
                    "cuda_fp16_bf16_reduction_math_sdp_allowed",
                    "fp16_bf16_reduction_math_sdp_allowed",
                ),
            ):
                getter = getattr(cuda_backend, getter_name, None)
                if callable(getter):
                    value = getter()
                    if not isinstance(value, bool):
                        raise CaptureContractError(
                            f"ambient torch.backends.cuda.{getter_name} must return boolean"
                        )
                    state[key] = value
        return state

    @classmethod
    def _restore_torch_execution_policy_state(
        cls, torch: Any, state: Mapping[str, Any]
    ) -> None:
        """Restore exactly the process-global policy fields that were observable at snapshot time."""
        for key, setter_name in (
            ("torch_num_threads", "set_num_threads"),
            ("torch_num_interop_threads", "set_num_interop_threads"),
        ):
            if key not in state:
                continue
            getter_name = setter_name.replace("set_", "get_", 1)
            getter = getattr(torch, getter_name, None)
            setter = getattr(torch, setter_name, None)
            if not callable(getter) or not callable(setter):
                raise CaptureContractError(
                    f"canonical process isolation cannot restore torch.{setter_name}"
                )
            if getter() != state[key]:
                setter(state[key])

        backends = getattr(torch, "backends", None)
        mkldnn = getattr(backends, "mkldnn", None) if backends is not None else None
        if "cpu_mkldnn_enabled" in state:
            if mkldnn is None or not hasattr(mkldnn, "enabled"):
                raise CaptureContractError(
                    "canonical process isolation cannot restore torch.backends.mkldnn.enabled"
                )
            mkldnn.enabled = state["cpu_mkldnn_enabled"]
        if "cpu_mkldnn_matmul_fp32_precision" in state:
            matmul = getattr(mkldnn, "matmul", None) if mkldnn is not None else None
            if matmul is None or not hasattr(matmul, "fp32_precision"):
                raise CaptureContractError(
                    "canonical process isolation cannot restore MKLDNN matmul precision"
                )
            matmul.fp32_precision = state["cpu_mkldnn_matmul_fp32_precision"]

        if "float32_matmul_precision" in state:
            setter = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(setter):
                raise CaptureContractError(
                    "canonical process isolation cannot restore float32 matmul precision"
                )
            setter(state["float32_matmul_precision"])

        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        cuda_matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
        for key, name in (
            ("cuda_matmul_allow_tf32", "allow_tf32"),
            ("cuda_allow_fp16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"),
            ("cuda_allow_bf16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"),
        ):
            if key not in state:
                continue
            if cuda_matmul is None or not hasattr(cuda_matmul, name):
                raise CaptureContractError(
                    f"canonical process isolation cannot restore torch.backends.cuda.matmul.{name}"
                )
            setattr(cuda_matmul, name, state[key])

        if "cudnn_allow_tf32" in state:
            cudnn = getattr(backends, "cudnn", None) if backends is not None else None
            if cudnn is None or not hasattr(cudnn, "allow_tf32"):
                raise CaptureContractError(
                    "canonical process isolation cannot restore torch.backends.cudnn.allow_tf32"
                )
            cudnn.allow_tf32 = state["cudnn_allow_tf32"]

        if cuda_backend is not None:
            for key, setter_name in (
                ("cuda_flash_sdp_enabled", "enable_flash_sdp"),
                ("cuda_mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
                ("cuda_math_sdp_enabled", "enable_math_sdp"),
                ("cuda_cudnn_sdp_enabled", "enable_cudnn_sdp"),
                (
                    "cuda_fp16_bf16_reduction_math_sdp_allowed",
                    "allow_fp16_bf16_reduction_math_sdp",
                ),
            ):
                if key not in state:
                    continue
                setter = getattr(cuda_backend, setter_name, None)
                if not callable(setter):
                    raise CaptureContractError(
                        f"canonical process isolation cannot restore torch.backends.cuda.{setter_name}"
                    )
                setter(state[key])
        elif any(key.startswith("cuda_") for key in state):
            raise CaptureContractError("canonical process isolation cannot restore CUDA backend policy")

        observed = cls._snapshot_torch_execution_policy_state(torch)
        if observed != dict(state):
            raise CaptureContractError(
                "unable to restore ambient PyTorch execution policies exactly: "
                f"expected={dict(state)!r} observed={observed!r}"
            )

    @classmethod
    def _snapshot_torch_process_state(cls, torch: Any, device: str = "cpu") -> dict[str, Any]:
        get_cpu = getattr(torch, "get_rng_state", None)
        set_cpu = getattr(torch, "set_rng_state", None)
        deterministic = getattr(torch, "are_deterministic_algorithms_enabled", None)
        warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
        setter = getattr(torch, "use_deterministic_algorithms", None)
        if not all(callable(value) for value in (get_cpu, set_cpu, deterministic, warn_only, setter)):
            raise CaptureContractError(
                "canonical process isolation requires PyTorch RNG and deterministic-policy state APIs"
            )
        enabled = deterministic()
        warned = warn_only()
        if not isinstance(enabled, bool) or not isinstance(warned, bool):
            raise CaptureContractError("ambient deterministic policy must be boolean")

        return {
            "cpu_rng": cls._clone_rng_state(get_cpu()),
            "deterministic_algorithms_enabled": enabled,
            "deterministic_warn_only_enabled": warned,
            "execution_policies": cls._snapshot_torch_execution_policy_state(torch),
            # Reading an uninitialized accelerator's RNG creates its runtime.
            # Only the explicit capture device may be touched here.
            "accelerator_rng": _snapshot_accelerator_rng(torch, device),
        }

    @classmethod
    def _restore_torch_process_state(cls, torch: Any, state: Mapping[str, Any]) -> None:
        try:
            policies = state.get("execution_policies", {})
            if not isinstance(policies, Mapping):
                raise CaptureContractError("ambient PyTorch execution-policy snapshot is malformed")
            cls._restore_torch_execution_policy_state(torch, policies)
            torch.set_rng_state(state["cpu_rng"])
            _restore_accelerator_rng(torch, state["accelerator_rng"])
            try:
                torch.use_deterministic_algorithms(
                    state["deterministic_algorithms_enabled"],
                    warn_only=state["deterministic_warn_only_enabled"],
                )
            except TypeError as exc:
                raise CaptureContractError(
                    "canonical process isolation requires warn_only restoration support"
                ) from exc
        except CaptureContractError:
            raise
        except Exception as exc:
            raise CaptureContractError("unable to restore ambient PyTorch process state") from exc

    def _model_executable_state_seal(self) -> str:
        named_modules = getattr(self._model, "named_modules", None)
        if not callable(named_modules):
            raise CaptureContractError(
                "canonical executable-state authentication requires model.named_modules()"
            )
        modules: list[dict[str, Any]] = []
        for name, module in named_modules():
            if not isinstance(name, str):
                raise CaptureContractError("canonical model graph contains an invalid module name")
            forward = getattr(module, "forward", None)
            if not callable(forward):
                raise CaptureContractError(
                    f"canonical model module {name!r} does not expose a callable forward"
                )
            target = getattr(forward, "__func__", forward)
            code = getattr(target, "__code__", None)
            code_sha = hashlib.sha256(code.co_code).hexdigest() if code is not None else None
            modules.append(
                {
                    "name": name,
                    "module_object_id": id(module),
                    "module_class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "forward_object_id": id(target),
                    "forward_module": getattr(target, "__module__", None),
                    "forward_qualname": getattr(target, "__qualname__", None),
                    "forward_code_sha256": code_sha,
                }
            )
        if not modules:
            raise CaptureContractError("canonical model graph exposes no executable modules")

        config = getattr(self._model, "config", None)
        config_to_dict = getattr(config, "to_dict", None)
        config_state = config_to_dict() if callable(config_to_dict) else repr(config)
        return sha256_json(
            {
                "modules": modules,
                "config": self._runtime_json_value(config_state),
            }
        )

    def _tokenizer_live_state_seal(self) -> str:
        base = super()._tokenizer_live_state_seal()
        behavior: dict[str, Any] = {}
        tokenizer = self._tokenizer
        for field in self._TOKENIZER_BEHAVIOR_FIELDS:
            if not hasattr(tokenizer, field):
                continue
            try:
                behavior[field] = self._runtime_json_value(getattr(tokenizer, field))
            except Exception as exc:
                raise CaptureContractError(
                    f"unable to authenticate tokenizer behavior field {field}"
                ) from exc
        return sha256_json({"base_state": base, "behavior": behavior})

    def _assert_live_state_authentication(self) -> None:
        # The policy layer performs the single tensor-content check. This layer
        # adds executable identity without copying/hashing all weights again.
        super()._assert_live_state_authentication()
        expected_graph = getattr(self, "_canonical_model_executable_state", None)
        if expected_graph is not None and self._model_executable_state_seal() != expected_graph:
            raise CaptureContractError(
                "live model executable graph changed after authenticated checkpoint loading"
            )

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
        # The inherited request guard dispatches the complete live-state check once.
        super().assert_execution_request(request)
        if getattr(self, "_observation_consumed", False):
            raise CaptureContractError(
                "canonical OBSERVATION backends are single-use and cannot be reused"
            )

    def begin_observation(self) -> None:
        if getattr(self, "_observation_consumed", False) or getattr(self, "_observation_active", False):
            raise CaptureContractError(
                "canonical OBSERVATION backends are single-use and cannot be reused"
            )
        self._observation_consumed = True
        ambient = self._snapshot_torch_process_state(self._torch, self._device)
        self._observation_active = True
        self._observation_ambient_process_state = ambient
        try:
            # Establish the frozen run seed only inside the observation session.
            _seed_capture_generators(self._torch, self._device, self._applied_seed)
            self._force_canonical_determinism_policy()
            self._assert_live_state_authentication()
        except BaseException:
            # Startup cleanup must use the same retryable ownership semantics as
            # normal session shutdown.  Keep the ambient receipt live until the
            # restore is complete; deterministic restore failures remain retryable
            # by execute_capture() instead of discarding the caller's only receipt.
            HuggingFacePyTorchBackend.end_observation(self)
            raise

    def end_observation(self) -> None:
        if not getattr(self, "_observation_active", False):
            return
        ambient = self._observation_ambient_process_state
        if ambient is None:
            raise CaptureContractError("canonical OBSERVATION ambient process-state snapshot is missing")

        interrupted: BaseException | None = None
        while True:
            try:
                self._restore_torch_process_state(self._torch, ambient)
            except (KeyboardInterrupt, SystemExit) as exc:
                if interrupted is None:
                    interrupted = exc
                # Restoration is idempotent: finish it before propagating the
                # asynchronous interruption so the caller is never left half-reset.
                continue
            break

        # Keep the only ambient-state receipt live until restoration fully succeeds.
        # Ordinary restore failures therefore leave this session retryable.
        self._observation_active = False
        self._observation_ambient_process_state = None
        if interrupted is not None:
            raise interrupted

    def _assert_tokenizer_state_authentication(self) -> None:
        if not getattr(self, "_live_state_seal_initialized", False):
            return
        expected = getattr(self, "_canonical_tokenizer_live_state", None)
        if expected is None:
            raise CaptureContractError("canonical tokenizer authentication seal is missing")
        if self._tokenizer_live_state_seal() != expected:
            raise CaptureContractError(
                "live tokenizer state changed after authenticated snapshot loading"
            )

    def tokenize(self, text: str) -> list[int]:
        # Tokenization never reads model weights. Authenticate the tokenizer here;
        # the full model still has content checks before and after every forward.
        self._assert_tokenizer_state_authentication()
        tokens = super().tokenize(text)
        self._assert_tokenizer_state_authentication()
        return tokens


__all__ = ["HuggingFacePyTorchBackend"]

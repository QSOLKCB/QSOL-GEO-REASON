"""Process-isolated, single-use canonical OBSERVATION backend.

This layer wraps the audited capture backend with host-process state restoration and
an explicit single-use evidence boundary. It also seals executable module-forward
identity and mutable tokenizer behavior that are not represented by tensor bytes.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_backend import HuggingFacePyTorchBackend as _PolicyHuggingFacePyTorchBackend
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_validation import validate_capture_request


class HuggingFacePyTorchBackend(_PolicyHuggingFacePyTorchBackend):
    """Canonical production backend with process isolation and single-use execution."""

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

        ambient = self._snapshot_torch_process_state(process_torch)
        try:
            super().__init__(validated)
        finally:
            # Construction seeds RNGs and may enable deterministic algorithms. A
            # backend object must not leave those process-global settings behind.
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
    def _snapshot_torch_process_state(cls, torch: Any) -> dict[str, Any]:
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

        state: dict[str, Any] = {
            "cpu_rng": cls._clone_rng_state(get_cpu()),
            "deterministic_algorithms_enabled": enabled,
            "deterministic_warn_only_enabled": warned,
            "cuda_rng": None,
            "mps_rng": None,
            "xpu_rng": None,
        }

        cuda = getattr(torch, "cuda", None)
        cuda_available = getattr(cuda, "is_available", None)
        if cuda is not None and callable(cuda_available) and bool(cuda_available()):
            get_all = getattr(cuda, "get_rng_state_all", None)
            set_all = getattr(cuda, "set_rng_state_all", None)
            if not callable(get_all) or not callable(set_all):
                raise CaptureContractError("canonical process isolation requires CUDA RNG state APIs")
            state["cuda_rng"] = [cls._clone_rng_state(item) for item in get_all()]

        mps = getattr(torch, "mps", None)
        mps_available = getattr(mps, "is_available", None)
        if mps is not None and callable(mps_available) and bool(mps_available()):
            get_mps = getattr(mps, "get_rng_state", None)
            set_mps = getattr(mps, "set_rng_state", None)
            if not callable(get_mps) or not callable(set_mps):
                raise CaptureContractError("canonical process isolation requires MPS RNG state APIs")
            state["mps_rng"] = cls._clone_rng_state(get_mps())

        xpu = getattr(torch, "xpu", None)
        xpu_available = getattr(xpu, "is_available", None)
        if xpu is not None and callable(xpu_available) and bool(xpu_available()):
            get_all = getattr(xpu, "get_rng_state_all", None)
            set_all = getattr(xpu, "set_rng_state_all", None)
            if not callable(get_all) or not callable(set_all):
                raise CaptureContractError("canonical process isolation requires XPU RNG state APIs")
            state["xpu_rng"] = [cls._clone_rng_state(item) for item in get_all()]
        return state

    @classmethod
    def _restore_torch_process_state(cls, torch: Any, state: Mapping[str, Any]) -> None:
        try:
            torch.set_rng_state(state["cpu_rng"])
            if state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if state.get("mps_rng") is not None:
                torch.mps.set_rng_state(state["mps_rng"])
            if state.get("xpu_rng") is not None:
                torch.xpu.set_rng_state_all(state["xpu_rng"])
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
        super()._assert_live_state_authentication()
        expected_graph = getattr(self, "_canonical_model_executable_state", None)
        if expected_graph is not None and self._model_executable_state_seal() != expected_graph:
            raise CaptureContractError(
                "live model executable graph changed after authenticated checkpoint loading"
            )

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
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
        self._observation_active = True
        ambient = self._snapshot_torch_process_state(self._torch)
        self._observation_ambient_process_state = ambient
        try:
            # Establish the frozen run seed only inside the observation session.
            self._torch.manual_seed(self._applied_seed)
            cuda = getattr(self._torch, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
                manual_seed_all = getattr(cuda, "manual_seed_all", None)
                if not callable(manual_seed_all):
                    raise CaptureContractError("canonical capture requires CUDA manual_seed_all")
                manual_seed_all(self._applied_seed)
            self._force_canonical_determinism_policy()
            self._assert_live_state_authentication()
        except Exception:
            try:
                self._restore_torch_process_state(self._torch, ambient)
            finally:
                self._observation_active = False
                self._observation_ambient_process_state = None
            raise

    def end_observation(self) -> None:
        if not getattr(self, "_observation_active", False):
            return
        ambient = self._observation_ambient_process_state
        self._observation_active = False
        self._observation_ambient_process_state = None
        if ambient is None:
            raise CaptureContractError("canonical OBSERVATION ambient process-state snapshot is missing")
        self._restore_torch_process_state(self._torch, ambient)

    def tokenize(self, text: str) -> list[int]:
        if getattr(self, "_live_state_seal_initialized", False):
            self._assert_live_state_authentication()
        tokens = super().tokenize(text)
        if getattr(self, "_live_state_seal_initialized", False):
            self._assert_live_state_authentication()
        return tokens


__all__ = ["HuggingFacePyTorchBackend"]

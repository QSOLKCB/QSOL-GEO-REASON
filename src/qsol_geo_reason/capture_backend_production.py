"""Final fail-closed production boundary for canonical GEO-CAP-001 observations.

This wrapper sits above the audited isolated backend and closes host/runtime mutation
windows that cannot be represented truthfully by ordinary manifest metadata.
"""
from __future__ import annotations

import _thread
import os
import sys
import threading
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_backend_isolated import HuggingFacePyTorchBackend as _IsolatedHuggingFacePyTorchBackend
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_validation import validate_capture_request
from .capture_execution_state import (
    _assert_exclusive_interpreter_thread,
    _callable_execution_identity,
    _cudnn_algorithm_policy_state,
)
from .capture_dispatch import (
    _MathSDPABaseModel,
    _effective_cpu_capability,
    _execution_dependencies_sha256,
    _fp16_accumulation_state,
    _model_execution_dependency_roots,
)
from .capture_hardware import _concrete_cpu_identity
from .capture_package import _python_package_provenance


class HuggingFacePyTorchBackend(_IsolatedHuggingFacePyTorchBackend):
    """Exact concrete backend permitted to emit canonical ``OBSERVATION`` evidence."""

    _EXECUTION_HOOK_REGISTRIES = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
    )
    _GLOBAL_EXECUTION_HOOK_REGISTRIES = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
        "_global_forward_pre_hooks_with_kwargs",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_hooks_always_called",
    )
    _MODULE_RUNTIME_ATTRIBUTE_EXCLUDES = frozenset(
        {
            "_parameters",
            "_buffers",
            "_non_persistent_buffers_set",
            "_modules",
            "_backward_pre_hooks",
            "_backward_hooks",
            "_is_full_backward_hook",
            "_forward_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        }
    )

    @classmethod
    def _assert_pristine_cuda_runtime(cls, torch: Any, device: str) -> None:
        """Reject CUDA observations whose runtime provenance is already ambiguous."""
        if not device.startswith("cuda:"):
            return

        version = getattr(torch, "version", None)
        hip_version = getattr(version, "hip", None) if version is not None else None
        if hip_version not in (None, ""):
            raise CaptureContractError(
                "canonical CUDA capture does not support ROCm/HIP PyTorch builds; "
                "use an NVIDIA CUDA build or an explicitly defined future ROCm lane"
            )

        cuda = getattr(torch, "cuda", None)
        initialized = getattr(cuda, "is_initialized", None) if cuda is not None else None
        if not callable(initialized):
            raise CaptureContractError(
                "canonical CUDA capture requires torch.cuda.is_initialized provenance support"
            )
        try:
            already_initialized = initialized()
        except Exception as exc:
            raise CaptureContractError(
                "unable to determine whether the CUDA runtime was initialized before capture"
            ) from exc
        if not isinstance(already_initialized, bool):
            raise CaptureContractError("torch.cuda.is_initialized must return a boolean")
        if already_initialized:
            raise CaptureContractError(
                "canonical CUDA capture requires an uninitialized CUDA runtime so frozen "
                "environment controls govern device mapping and cuBLAS/TF32 initialization"
            )

    @classmethod
    def _validate_pre_mps_environment(cls, request: Mapping[str, Any]) -> None:
        """Reject noncanonical MPS environment policy before importing PyTorch."""
        if request["backend"]["device"] != "mps":
            return
        for variable in (
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "PYTORCH_MPS_FAST_MATH",
            "PYTORCH_MPS_PREFER_METAL",
        ):
            if cls._env_flag_enabled(os.environ.get(variable)):
                raise CaptureContractError(
                    f"canonical MPS capture forbids {variable} before PyTorch import"
                )

    @staticmethod
    def _assert_pristine_mps_import_state(
        device: str, modules: Mapping[str, Any] | None = None
    ) -> None:
        """Require MPS capture to begin before any PyTorch import in this process."""
        if device != "mps":
            return
        module_table = sys.modules if modules is None else modules
        if "torch" in module_table:
            raise CaptureContractError(
                "canonical MPS capture requires a fresh PyTorch import boundary so frozen "
                "MPS environment controls govern first runtime/kernel initialization"
            )

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        device = validated["backend"]["device"]
        self._exclusive_thread_boundary_state: dict[str, Any] | None = None
        self._canonical_cudnn_algorithm_policy: dict[str, bool] | None = None
        self._last_cudnn_algorithm_policy: dict[str, bool] | None = None
        self._canonical_cpu_dispatch: dict[str, Any] | None = None
        self._canonical_cpu_environment: str | None = None
        self._canonical_cpu_processor: str | None = None
        self._canonical_fp16_accumulation_supported: bool | None = None
        self._last_fp16_accumulation: bool | None = None
        self._transformers_package_provenance: dict[str, Any] | None = None
        # Construction mutates the same process-global state as observation.
        # Own the thread boundary before import, snapshot, initialization or cleanup.
        self._enter_exclusive_python_thread_boundary()
        try:
            self._validate_pre_cuda_environment(validated)
            self._validate_pre_mps_environment(validated)
            self._canonical_cuda_environment = None
            if device.startswith("cuda:"):
                self._canonical_cuda_environment = self._cuda_environment_state()

            self._assert_pristine_mps_import_state(device)
            cpu_override_known = "torch" not in sys.modules
            cpu_override = os.environ.get("ATEN_CPU_CAPABILITY")
            try:
                import torch as process_torch
            except ImportError as exc:
                raise CaptureBackendUnavailable("canonical capture requires PyTorch") from exc
            # Retain the actual construction-time runtime, not a later import or a
            # duck-typed policy proxy. This identity never enters persisted JSON.
            self._canonical_torch_module = process_torch
            try:
                import transformers as process_transformers
            except ImportError:
                # The inherited real backend also imports Transformers and fails closed
                # when it is absent. Keeping this probe optional preserves dependency-free
                # software fixtures that deliberately replace the inherited constructor.
                process_transformers = None

            transformers_before = (
                _python_package_provenance(process_transformers, "Transformers")
                if process_transformers is not None
                else None
            )
            self._assert_pristine_cuda_runtime(process_torch, device)

            # CPU float64 pooling is part of every canonical device lane. Bind the
            # effective ATen CPU dispatch identity and the initialization-time override
            # whenever it can be known, even when model execution itself is CUDA/MPS.
            self._canonical_cpu_dispatch = {
                "cpu_aten_capability": _effective_cpu_capability(process_torch),
                "aten_cpu_capability_env": cpu_override if cpu_override_known else None,
                "aten_cpu_capability_env_known": cpu_override_known,
            }
            self._canonical_cpu_environment = cpu_override
            if os.environ.get("ATEN_CPU_CAPABILITY") != cpu_override:
                raise CaptureContractError("ATen CPU override changed during initialization")
            if device == "cpu":
                self._canonical_cpu_processor = _concrete_cpu_identity()

            if device.startswith("cuda:"):
                self._canonical_cudnn_algorithm_policy = _cudnn_algorithm_policy_state(process_torch)
                self._canonical_fp16_accumulation_supported = _fp16_accumulation_state(process_torch) is not None
            ambient = self._snapshot_torch_process_state(process_torch, device)
            try:
                super().__init__(validated)
                self._assert_torch_runtime_identity()
            finally:
                # Keep exclusion active through both inherited and final restoration,
                # including constructor failures and interrupted initialization.
                self._restore_torch_process_state(process_torch, ambient)

            loaded_transformers = getattr(self, "_transformers", None)
            if loaded_transformers is not None:
                transformers_after = _python_package_provenance(
                    loaded_transformers, "Transformers"
                )
                if (
                    transformers_before is not None
                    and transformers_after != transformers_before
                ):
                    raise CaptureContractError(
                        "imported Transformers package changed while canonical backend was loading"
                    )
                self._transformers_package_provenance = transformers_after
            elif transformers_before is not None:
                # This branch is reachable only for software fixtures that replace the
                # inherited constructor. A real inherited backend always sets
                # self._transformers before returning successfully.
                self._transformers_package_provenance = transformers_before

            # The core already brackets CUDA SDPA. Its CPU/MPS execution needs the
            # identical math-only boundary at the delegated base-model call.
            if device in {"cpu", "mps"} and self._attention_implementation == "sdpa":
                self._base_model = _MathSDPABaseModel(self._base_model, self)
            self._assert_cpu_dispatch_policy()
            self._canonical_model_runtime_attributes = self._model_runtime_attributes_seal()
            self._assert_no_registered_module_hooks()
        finally:
            self._leave_exclusive_python_thread_boundary()

    def _assert_torch_runtime_identity(self) -> None:
        """Reject replacement of the construction-bound PyTorch execution object."""
        state = vars(self)
        expected = state.get("_canonical_torch_module")
        if expected is None or state.get("_torch") is not expected:
            raise CaptureContractError(
                "canonical OBSERVATION PyTorch runtime object changed or lacks its construction binding"
            )

    @classmethod
    def _snapshot_torch_execution_policy_state(cls, torch: Any) -> dict[str, Any]:
        state = super()._snapshot_torch_execution_policy_state(torch)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        for name in ("benchmark", "deterministic"):
            value = getattr(cudnn, name, None)
            if value is not None:
                if not isinstance(value, bool):
                    raise CaptureContractError(f"ambient torch.backends.cudnn.{name} must be boolean")
                state[f"cudnn_{name}"] = value
        accumulation = _fp16_accumulation_state(torch)
        if accumulation is not None:
            state["cuda_matmul_allow_fp16_accumulation"] = accumulation
        return state

    @classmethod
    def _restore_torch_execution_policy_state(cls, torch: Any, state: Mapping[str, Any]) -> None:
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        for name in ("benchmark", "deterministic"):
            key = f"cudnn_{name}"
            if key not in state:
                continue
            if cudnn is None or not hasattr(cudnn, name) or not isinstance(state[key], bool):
                raise CaptureContractError(f"unable to restore ambient cuDNN {name} policy")
            setattr(cudnn, name, state[key])
        key = "cuda_matmul_allow_fp16_accumulation"
        if key in state:
            if type(state[key]) is not bool or _fp16_accumulation_state(torch) is None:
                raise CaptureContractError("unable to restore ambient CUDA FP16 accumulation")
            torch.backends.cuda.matmul.allow_fp16_accumulation = state[key]
        # The inherited exact-equality check dynamically calls our full snapshot.
        super()._restore_torch_execution_policy_state(torch, state)

    def _assert_cpu_dispatch_policy(self) -> None:
        expected = getattr(self, "_canonical_cpu_dispatch", None)
        if expected is None:
            return
        if _effective_cpu_capability(self._torch) != expected["cpu_aten_capability"]:
            raise CaptureContractError("effective ATen CPU dispatch policy drifted")
        if os.environ.get("ATEN_CPU_CAPABILITY") != self._canonical_cpu_environment:
            raise CaptureContractError("ATen CPU override drifted after construction")

    def _assert_fp16_accumulation_policy(self) -> bool | None:
        value = _fp16_accumulation_state(self._torch)
        supported = getattr(self, "_canonical_fp16_accumulation_supported", None)
        if supported is not None and (value is not None) is not supported:
            raise CaptureContractError("CUDA FP16 accumulation control availability changed")
        if value is True:
            raise CaptureContractError("canonical CUDA capture forbids FP16 accumulation")
        self._last_fp16_accumulation = value
        return value

    def _force_cuda_reduced_precision_policy(self) -> None:
        value = _fp16_accumulation_state(self._torch)
        if value is not None:
            self._torch.backends.cuda.matmul.allow_fp16_accumulation = False
        self._assert_fp16_accumulation_policy()
        super()._force_cuda_reduced_precision_policy()

    def _assert_cuda_reduced_precision_policy(self) -> dict[str, bool]:
        self._assert_fp16_accumulation_policy()
        return super()._assert_cuda_reduced_precision_policy()

    def _assert_cudnn_algorithm_policy(self) -> None:
        expected = getattr(self, "_canonical_cudnn_algorithm_policy", None)
        if expected is None:
            return
        observed = _cudnn_algorithm_policy_state(self._torch)
        if observed != expected:
            raise CaptureContractError("canonical cuDNN algorithm-selection policy drifted")
        self._last_cudnn_algorithm_policy = dict(observed)

    def _force_cuda_float32_policy(self) -> None:
        # The core invokes this before every CUDA forward, for both determinism modes.
        expected = getattr(self, "_canonical_cudnn_algorithm_policy", None)
        if expected is not None:
            _cudnn_algorithm_policy_state(self._torch)
            for name in ("benchmark", "deterministic"):
                setattr(self._torch.backends.cudnn, name, expected[f"cudnn_{name}"])
            self._assert_cudnn_algorithm_policy()
        super()._force_cuda_float32_policy()

    def _assert_cuda_float32_policy(self) -> dict[str, str | bool]:
        # The same core dispatch verifies algorithm selection after the full forward.
        self._assert_cudnn_algorithm_policy()
        return super()._assert_cuda_float32_policy()

    def metadata(self) -> Mapping[str, Any]:
        self._assert_cudnn_algorithm_policy()
        self._assert_cpu_dispatch_policy()
        if getattr(self, "_device_type", None) == "cuda":
            self._assert_fp16_accumulation_policy()
        if getattr(self, "_attention_implementation", None) == "sdpa":
            self._last_sdpa_policy = self._assert_sdpa_math_policy()
        package_provenance = getattr(self, "_transformers_package_provenance", None)
        transformers_module = getattr(self, "_transformers", None)
        if transformers_module is not None:
            if not isinstance(package_provenance, Mapping):
                raise CaptureContractError(
                    "canonical Transformers package provenance was not recorded at construction"
                )
            if _python_package_provenance(
                transformers_module, "Transformers"
            ) != dict(package_provenance):
                raise CaptureContractError(
                    "imported Transformers package changed after authenticated backend construction"
                )
        data = dict(super().metadata())
        policy = getattr(self, "_last_cudnn_algorithm_policy", None)
        for field in ("cudnn_benchmark", "cudnn_deterministic"):
            data[field] = policy[field] if policy is not None else None
        data["cuda_matmul_allow_fp16_accumulation"] = (
            getattr(self, "_last_fp16_accumulation", None)
            if getattr(self, "_device_type", None) == "cuda" else None
        )
        dispatch = getattr(self, "_canonical_cpu_dispatch", None)
        for field in ("cpu_aten_capability", "aten_cpu_capability_env", "aten_cpu_capability_env_known"):
            data[field] = dispatch[field] if dispatch is not None else None
        if getattr(self, "_device_type", None) == "cpu":
            processor = getattr(self, "_canonical_cpu_processor", None)
            if processor is not None:
                if not isinstance(processor, str) or not processor.strip():
                    raise CaptureContractError(
                        "canonical CPU processor identity was not recorded at construction"
                    )
                data["cpu_processor"] = processor
        if isinstance(package_provenance, Mapping):
            data["transformers_package_file_count"] = package_provenance["file_count"]
            data["transformers_package_receipt_sha256"] = package_provenance["receipt_sha256"]
        return data

    def _model_executable_state_seal(self) -> str:
        base = super()._model_executable_state_seal()
        roots = _model_execution_dependency_roots(self._model)
        return sha256_json({
            "module_graph": base,
            "execution_dependencies": _execution_dependencies_sha256(roots),
        })

    def _assert_no_global_module_hooks(self) -> None:
        """Reject PyTorch process-global module hooks that can rewrite any model execution."""
        torch = getattr(self, "_torch", None)
        if torch is None:
            return
        nn = getattr(torch, "nn", None)
        # Source-audit/unit-test tensor doubles intentionally implement only the tiny
        # PyTorch surface they exercise. A real production PyTorch module always has
        # torch.nn; absence here is therefore a synthetic-fixture escape hatch only.
        if nn is None:
            return
        modules = getattr(nn, "modules", None)
        module_api = getattr(modules, "module", None) if modules is not None else None
        if module_api is None:
            raise CaptureContractError(
                "canonical OBSERVATION requires access to PyTorch global module-hook registries"
            )

        found_registry = False
        for registry_name in self._GLOBAL_EXECUTION_HOOK_REGISTRIES:
            registry = getattr(module_api, registry_name, None)
            if registry is None:
                continue
            found_registry = True
            try:
                populated = bool(registry)
            except Exception as exc:
                raise CaptureContractError(
                    f"unable to authenticate process-global hook registry {registry_name}"
                ) from exc
            if populated:
                raise CaptureContractError(
                    "canonical OBSERVATION forbids process-global PyTorch execution hooks; "
                    f"registry={registry_name!r}"
                )
        if not found_registry:
            raise CaptureContractError(
                "canonical OBSERVATION cannot authenticate PyTorch global module-hook state"
            )

    def _assert_no_registered_module_hooks(self) -> None:
        """Reject hooks and compiled call substitutions on the authenticated graph."""
        self._assert_no_global_module_hooks()
        model = getattr(self, "_model", None)
        if model is None:
            return
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            # Synthetic source-audit fixtures may not implement torch.nn.Module. The
            # real production model always does, and the inherited graph seal checks it.
            return

        for module_name, module in named_modules():
            if not isinstance(module_name, str):
                raise CaptureContractError("canonical model graph contains an invalid module name")
            # nn.Module.compile() changes __call__ dispatch without changing forward.
            # Reject it independently of the forward/config/tensor seals, including
            # before the single observation and at every pre/post-forward guard.
            if getattr(module, "_compiled_call_impl", None) is not None:
                raise CaptureContractError(
                    "canonical OBSERVATION forbids compiled module call implementations; "
                    f"module={module_name or '<root>'!r}"
                )
            for registry_name in self._EXECUTION_HOOK_REGISTRIES:
                registry = getattr(module, registry_name, None)
                if registry is None:
                    continue
                try:
                    populated = bool(registry)
                except Exception as exc:
                    raise CaptureContractError(
                        f"unable to authenticate module hook registry {registry_name}"
                    ) from exc
                if populated:
                    display_name = module_name or "<root>"
                    raise CaptureContractError(
                        "canonical OBSERVATION forbids registered model execution hooks; "
                        f"module={display_name!r} registry={registry_name!r}"
                    )

    def _runtime_attribute_value(self, value: Any, where: str) -> Any:
        is_tensor = getattr(getattr(self, "_torch", None), "is_tensor", None)
        if callable(is_tensor):
            try:
                tensor = bool(is_tensor(value))
            except Exception as exc:
                raise CaptureContractError(
                    f"unable to classify model runtime execution attribute {where}"
                ) from exc
            if tensor:
                shape = tuple(int(item) for item in getattr(value, "shape", ()))
                return {
                    "kind": "tensor",
                    "shape": list(shape),
                    "dtype": str(getattr(value, "dtype", "")),
                    "device": str(getattr(value, "device", "")),
                    "sha256": self._tensor_content_sha256(value, where),
                }
        if callable(value):
            return _callable_execution_identity(value)
        return self._runtime_json_value(value)

    def _model_runtime_attributes_seal(self) -> str:
        """Authenticate mutable non-structural module attributes before first execution."""
        named_modules = getattr(self._model, "named_modules", None)
        if not callable(named_modules):
            raise CaptureContractError(
                "canonical runtime-attribute authentication requires model.named_modules()"
            )
        modules: list[dict[str, Any]] = []
        for module_name, module in named_modules():
            if not isinstance(module_name, str):
                raise CaptureContractError("canonical model graph contains an invalid module name")
            state = getattr(module, "__dict__", None)
            if not isinstance(state, Mapping):
                raise CaptureContractError(
                    f"canonical model module {module_name!r} has no inspectable runtime state"
                )
            attributes: dict[str, Any] = {}
            for attribute_name, value in sorted(state.items(), key=lambda pair: pair[0]):
                if attribute_name in self._MODULE_RUNTIME_ATTRIBUTE_EXCLUDES:
                    continue
                attributes[attribute_name] = self._runtime_attribute_value(
                    value, f"{module_name or '<root>'}.{attribute_name}"
                )
            modules.append({"name": module_name, "attributes": attributes})
        if not modules:
            raise CaptureContractError("canonical model graph exposes no module runtime state")
        return sha256_json(modules)

    def _assert_model_runtime_attributes(self) -> None:
        expected = getattr(self, "_canonical_model_runtime_attributes", None)
        if expected is None:
            return
        if self._model_runtime_attributes_seal() != expected:
            raise CaptureContractError(
                "live model runtime execution attributes changed after authenticated loading"
            )

    def _assert_no_active_torch_override_modes(self) -> None:
        """Reject thread-local TorchDispatchMode and TorchFunctionMode overrides."""
        torch = getattr(self, "_torch", None)
        if torch is None:
            return
        # Minimal software doubles used by source-audit tests do not model torch.nn
        # or the dispatcher C API. A real canonical PyTorch runtime always does.
        if getattr(torch, "nn", None) is None:
            return
        c_api = getattr(torch, "_C", None)
        dispatch_len = getattr(c_api, "_len_torch_dispatch_stack", None)
        function_len = getattr(c_api, "_len_torch_function_stack", None)
        if not callable(dispatch_len) or not callable(function_len):
            raise CaptureContractError(
                "canonical OBSERVATION cannot authenticate PyTorch dispatch mode stacks"
            )
        try:
            dispatch_count = dispatch_len()
            function_count = function_len()
        except Exception as exc:
            raise CaptureContractError(
                "unable to inspect PyTorch dispatch mode stacks"
            ) from exc
        for label, count in (
            ("__torch_dispatch__", dispatch_count),
            ("__torch_function__", function_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise CaptureContractError(
                    f"invalid PyTorch {label} mode-stack length"
                )
            if count:
                raise CaptureContractError(
                    f"canonical OBSERVATION forbids active PyTorch {label} modes"
                )

    @staticmethod
    def _blocked_thread_start(*_args: Any, **_kwargs: Any) -> None:
        raise CaptureContractError(
            "canonical OBSERVATION requires exclusive Python-thread execution; "
            "starting another Python thread during capture is forbidden"
        )

    def _enter_exclusive_python_thread_boundary(self) -> None:
        """Fail closed unless the observation owns the process's Python execution thread."""
        if getattr(self, "_exclusive_thread_boundary_state", None) is not None:
            raise CaptureContractError("canonical OBSERVATION thread boundary is already active")
        lock = getattr(threading, "_active_limbo_lock", None)
        active = getattr(threading, "_active", None)
        limbo = getattr(threading, "_limbo", None)
        if lock is None or not isinstance(active, Mapping) or not isinstance(limbo, Mapping):
            raise CaptureContractError(
                "canonical OBSERVATION requires inspectable Python thread active/limbo registries"
            )

        patches: list[tuple[Any, str, Any]] = []
        with lock:
            targets = [
                (threading.Thread, "start"),
                (threading, "_start_new_thread"),
                (threading, "_start_joinable_thread"),
                (_thread, "start_new_thread"),
                (_thread, "start_new"),
                (_thread, "start_joinable_thread"),
            ]
            try:
                for owner, name in targets:
                    original = getattr(owner, name, None)
                    if not callable(original):
                        continue
                    patches.append((owner, name, original))
                    setattr(owner, name, self._blocked_thread_start)
                # Raw _thread workers need not register in threading._active.
                # Inspect interpreter frames and worker count as well, while starts
                # remain blocked, rather than treating that registry as exhaustive.
                _assert_exclusive_interpreter_thread()
                current_ident = threading.get_ident()
                other_active = [ident for ident in active if ident != current_ident]
                if other_active or limbo:
                    raise CaptureContractError(
                        "canonical OBSERVATION requires exclusive Python-thread execution; "
                        f"other_active_threads={len(other_active)} starting_threads={len(limbo)}"
                    )
            except BaseException:
                for owner, name, original in reversed(patches):
                    setattr(owner, name, original)
                raise
        self._exclusive_thread_boundary_state = {"lock": lock, "patches": patches}

    def _leave_exclusive_python_thread_boundary(self) -> None:
        state = getattr(self, "_exclusive_thread_boundary_state", None)
        if state is None:
            return
        self._exclusive_thread_boundary_state = None
        lock = state["lock"]
        patches = state["patches"]
        try:
            with lock:
                for owner, name, original in reversed(patches):
                    setattr(owner, name, original)
        except Exception as exc:
            raise CaptureContractError(
                "unable to restore ambient Python thread-start policy after canonical observation"
            ) from exc

    def _assert_live_state_authentication(self) -> None:
        # Cheap dispatch checks precede the inherited content-bound authentication.
        # Each layer adds its own checks; none rehashes tensors checked by its parent.
        self._assert_no_active_torch_override_modes()
        self._assert_no_registered_module_hooks()
        self._assert_cpu_dispatch_policy()
        super()._assert_live_state_authentication()

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
        self._assert_torch_runtime_identity()
        # The policy layer dispatches the full live-state chain exactly once.
        super().assert_execution_request(request)
        self._assert_model_runtime_attributes()

    def begin_observation(self) -> None:
        self._enter_exclusive_python_thread_boundary()
        try:
            # Recheck runtime ownership inside exclusion, before even the mode/policy
            # probes can be delegated to a substituted runtime object.
            self._assert_torch_runtime_identity()
            # Mode stacks are thread-local, so inspect them only after this thread
            # owns the exclusive observation boundary and before capture state mutates.
            self._assert_no_active_torch_override_modes()
            # Reauthenticate non-tensor execution attributes after concurrent Python
            # activity has been excluded, closing the final pre-first-forward window.
            self._assert_model_runtime_attributes()
            super().begin_observation()
        except BaseException:
            self._leave_exclusive_python_thread_boundary()
            raise

    def end_observation(self) -> None:
        try:
            super().end_observation()
        finally:
            self._leave_exclusive_python_thread_boundary()


__all__ = ["HuggingFacePyTorchBackend"]

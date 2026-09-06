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
            "_compiled_call_impl",
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
        # Keep the required cuBLAS environment check and the exact environment
        # receipt ahead of any CUDA runtime use. The receipt is repeated here,
        # even though the inherited policy layer also binds it, so the public
        # evidence boundary visibly authenticates initialization-time controls.
        self._validate_pre_cuda_environment(validated)
        self._validate_pre_mps_environment(validated)
        device = validated["backend"]["device"]
        self._canonical_cuda_environment = None
        if device.startswith("cuda:"):
            self._canonical_cuda_environment = self._cuda_environment_state()

        # PyTorch does not expose a reliable public MPS runtime-initialized predicate.
        # Fail closed instead: an MPS observation must own the process's first torch
        # import, before any process-state snapshot can touch MPS RNG/runtime state.
        self._assert_pristine_mps_import_state(device)
        try:
            import torch as process_torch
        except ImportError:
            # Preserve the canonical optional-dependency failure from the audited backend.
            super().__init__(validated)
            raise CaptureBackendUnavailable("canonical capture requires PyTorch")

        # This check must precede process-state snapshotting because querying CUDA RNG
        # state can itself initialize the runtime. ROCm is also rejected before that point.
        self._assert_pristine_cuda_runtime(process_torch, device)
        ambient = self._snapshot_torch_process_state(process_torch)
        try:
            super().__init__(validated)
        finally:
            # The parent already restores its own construction snapshot. Restore the
            # state observed at this final public boundary as well, and fail closed if
            # exact restoration is impossible.
            self._restore_torch_process_state(process_torch, ambient)

        # Freeze ordinary non-parameter/module attributes that participate in execution,
        # such as attention scaling factors. This closes the construction-to-first-run
        # mutation window without treating mutable post-forward caches as canonical state.
        self._canonical_model_runtime_attributes = self._model_runtime_attributes_seal()
        self._exclusive_thread_boundary_state: dict[str, Any] | None = None
        self._assert_no_registered_module_hooks()

    def _assert_no_global_module_hooks(self) -> None:
        """Reject PyTorch process-global module hooks that can rewrite any model execution."""
        torch = getattr(self, "_torch", None)
        if torch is None:
            return
        nn = getattr(torch, "nn", None)
        modules = getattr(nn, "modules", None) if nn is not None else None
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
        """Reject process-global and per-module execution hooks on the authenticated graph."""
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
                if attribute_name in self._MODULE_RUNTIME_ATTRIBUTE_EXCLUDES or callable(value):
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

    @staticmethod
    def _blocked_thread_start(*_args: Any, **_kwargs: Any) -> None:
        raise CaptureContractError(
            "canonical OBSERVATION requires exclusive Python-thread execution; "
            "starting another Python thread during capture is forbidden"
        )

    def _enter_exclusive_python_thread_boundary(self) -> None:
        """Fail closed unless the observation owns the process's Python execution thread."""
        if self._exclusive_thread_boundary_state is not None:
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
                (_thread, "start_joinable_thread"),
            ]
            try:
                for owner, name in targets:
                    original = getattr(owner, name, None)
                    if not callable(original):
                        continue
                    patches.append((owner, name, original))
                    setattr(owner, name, self._blocked_thread_start)
                current_ident = threading.get_ident()
                other_active = [ident for ident in active if ident != current_ident]
                if other_active or limbo:
                    raise CaptureContractError(
                        "canonical OBSERVATION requires exclusive Python-thread execution; "
                        f"other_active_threads={len(other_active)} starting_threads={len(limbo)}"
                    )
            except Exception:
                for owner, name, original in reversed(patches):
                    setattr(owner, name, original)
                raise
        self._exclusive_thread_boundary_state = {"lock": lock, "patches": patches}

    def _leave_exclusive_python_thread_boundary(self) -> None:
        state = self._exclusive_thread_boundary_state
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
        super()._assert_live_state_authentication()
        # Keep the content-bound model seal explicit at the final public evidence
        # boundary. The inherited guard already checks it, but this duplicate
        # assertion makes the byte-authentication invariant locally auditable.
        expected_content = getattr(self, "_canonical_model_content_state", None)
        if expected_content is not None and self._model_content_state_seal() != expected_content:
            raise CaptureContractError(
                "live model tensor contents changed after authenticated checkpoint loading"
            )
        # The inherited tensor/tokenizer/executable seals do not include PyTorch's
        # mutable hook registries. Reject them at every existing live-state guard,
        # including immediately before and after each hidden-state forward.
        self._assert_no_registered_module_hooks()

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
        super().assert_execution_request(request)
        self._assert_model_runtime_attributes()

    def begin_observation(self) -> None:
        self._enter_exclusive_python_thread_boundary()
        try:
            # Reauthenticate non-tensor execution attributes after concurrent Python
            # activity has been excluded, closing the final pre-first-forward window.
            self._assert_model_runtime_attributes()
            super().begin_observation()
        except Exception:
            self._leave_exclusive_python_thread_boundary()
            raise

    def end_observation(self) -> None:
        try:
            super().end_observation()
        finally:
            self._leave_exclusive_python_thread_boundary()


__all__ = ["HuggingFacePyTorchBackend"]

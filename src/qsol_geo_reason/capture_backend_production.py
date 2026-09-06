"""Final fail-closed production boundary for canonical GEO-CAP-001 observations.

This wrapper sits above the audited isolated backend and closes host/runtime mutation
windows that cannot be represented truthfully by ordinary manifest metadata.
"""
from __future__ import annotations

import sys
from typing import Any, Mapping

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


__all__ = ["HuggingFacePyTorchBackend"]

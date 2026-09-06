"""Final fail-closed production boundary for canonical GEO-CAP-001 observations.

This wrapper sits above the audited isolated backend and closes host/runtime mutation
windows that cannot be represented truthfully by ordinary manifest metadata.
"""
from __future__ import annotations

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

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        # Keep the required cuBLAS environment check ahead of any CUDA runtime use.
        self._validate_pre_cuda_environment(validated)
        device = validated["backend"]["device"]

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

    def _assert_no_registered_module_hooks(self) -> None:
        """Reject externally registered execution hooks on the authenticated model graph."""
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
        # The inherited tensor/tokenizer/executable seals do not include PyTorch's
        # mutable hook registries. Reject them at every existing live-state guard,
        # including immediately before and after each hidden-state forward.
        self._assert_no_registered_module_hooks()


__all__ = ["HuggingFacePyTorchBackend"]

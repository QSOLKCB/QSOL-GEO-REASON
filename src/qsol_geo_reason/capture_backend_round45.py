"""Round-45 hardening for fresh runtimes, CPU pooling identity, and UTF-8 provenance.

The canonical OBSERVATION lane performs CPU float64 pooling for every source device.
This layer therefore binds the host CPU and denormal policy on every lane, requires a
fresh PyTorch/Transformers import boundary before construction, and closes the final
snapshot-path encoding hole before bundle hashing.
"""
from __future__ import annotations

import sys
from typing import Any, Mapping

from .capture_backend_round44 import (
    HuggingFacePyTorchBackend as _Round44Backend,
    _assert_package_callable_matches_source,
)
from .capture_common import CaptureContractError
from .capture_hardware import _concrete_cpu_identity, _is_concrete_cpu_identity
from . import capture_backend_production as _production
from . import capture_provenance as _capture_provenance


_ORIGINAL_CANONICAL_SNAPSHOT_PATH = _capture_provenance._is_canonical_snapshot_path
_ORIGINAL_VALIDATE_BACKEND_METADATA = _capture_provenance._validate_backend_metadata


def _is_canonical_snapshot_path_round45(value: Any) -> bool:
    """Reject paths that cannot be represented by canonical UTF-8 JSON bytes."""
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return False
    return _ORIGINAL_CANONICAL_SNAPSHOT_PATH(value)


# _validate_snapshot_receipt() resolves this helper through module globals at call
# time, including when invoked by the public verifier imported later by capture.py.
_capture_provenance._is_canonical_snapshot_path = _is_canonical_snapshot_path_round45


def _validate_backend_metadata_round45(
    observed: Mapping[str, Any], request: Mapping[str, Any], evidence_class: str
) -> None:
    """Validate new pooling-instrument fields without weakening the established set."""
    if evidence_class != "OBSERVATION":
        _ORIGINAL_VALIDATE_BACKEND_METADATA(observed, request, evidence_class)
        return

    # Round 44's exact-key validator predates the explicit denormal receipt. Remove
    # only that newly authenticated leaf while delegating every established check,
    # then validate the leaf here. This preserves fail-closed unknown-field handling.
    legacy_observed = dict(observed)
    marker = object()
    flush_denormal = legacy_observed.pop("cpu_flush_denormal", marker)
    _ORIGINAL_VALIDATE_BACKEND_METADATA(legacy_observed, request, evidence_class)

    if not _is_concrete_cpu_identity(observed.get("cpu_processor")):
        raise CaptureContractError(
            "canonical OBSERVATION CPU pooling requires a concrete processor model identity"
        )
    if flush_denormal is marker or flush_denormal is not False:
        raise CaptureContractError(
            "canonical OBSERVATION requires cpu_flush_denormal=false"
        )


_capture_provenance._validate_backend_metadata = _validate_backend_metadata_round45


def _assert_transformers_instance_loader_source_bound(
    module: Any, instance: Any, label: str
) -> None:
    """Bind the concrete loader inherited by a returned model/tokenizer instance."""
    owner = type(instance)
    loader = getattr(owner, "from_pretrained", None)
    if loader is None:
        raise CaptureContractError(
            f"canonical Transformers {label} class does not expose from_pretrained"
        )
    _assert_package_callable_matches_source(
        module,
        loader,
        f"Transformers concrete {label} loader {owner.__qualname__}.from_pretrained",
    )


class HuggingFacePyTorchBackend(_Round44Backend):
    """Newest canonical boundary for Round-45 runtime/provenance hardening."""

    @staticmethod
    def _assert_pristine_mps_import_state(
        device: str, modules: Mapping[str, Any] | None = None
    ) -> None:
        """Require every OBSERVATION lane to own first Torch/Transformers import.

        A pre-imported runtime can already contain caller-installed concrete
        ``from_pretrained`` replacements or persistent ``torch.library`` dispatcher
        registrations. Canonical capture therefore rejects that process state instead
        of attempting to reconstruct an unknowable clean baseline afterward.
        """
        del device  # the fresh-runtime rule applies to CPU, CUDA, and MPS alike
        module_table = sys.modules if modules is None else modules
        preloaded: list[str] = []
        for root in ("torch", "transformers"):
            if root in module_table or any(
                name.startswith(root + ".") for name in module_table
            ):
                preloaded.append(root)
        if preloaded:
            raise CaptureContractError(
                "canonical OBSERVATION requires a fresh PyTorch/Transformers import "
                "boundary; preloaded=" + ",".join(preloaded)
            )

    def _force_round45_cpu_flush_denormal_policy(self) -> None:
        """Canonicalize CPU underflow semantics for model execution and pooling."""
        torch = getattr(self, "_torch", None)
        setter = getattr(torch, "set_flush_denormal", None) if torch is not None else None
        if not callable(setter):
            raise CaptureContractError(
                "canonical OBSERVATION requires torch.set_flush_denormal"
            )
        try:
            supported = setter(False)
        except Exception as exc:
            raise CaptureContractError(
                "unable to establish canonical CPU flush-denormal policy"
            ) from exc
        if supported is not True:
            raise CaptureContractError(
                "canonical OBSERVATION cannot establish gradual-underflow CPU semantics"
            )
        self._canonical_cpu_flush_denormal = False

    def _assert_round45_cpu_pooling_identity(self) -> str:
        processor = getattr(self, "_canonical_cpu_processor", None)
        if not _is_concrete_cpu_identity(processor):
            raise CaptureContractError(
                "canonical OBSERVATION CPU pooling processor identity is missing"
            )
        return processor.strip()

    def _assert_round45_transformers_delegates(self) -> None:
        module = getattr(self, "_transformers", None)
        if module is None:
            return
        tokenizer = getattr(self, "_tokenizer", None)
        model = getattr(self, "_model", None)
        if tokenizer is not None:
            _assert_transformers_instance_loader_source_bound(
                module, tokenizer, "tokenizer"
            )
        if model is not None:
            _assert_transformers_instance_loader_source_bound(
                module, model, "model"
            )

    def _assert_autocast_disabled(self) -> None:
        # Core calls this after binding self._torch but before any snapshot/model load,
        # and again before every hidden-state forward. Establish the CPU underflow
        # policy in both places so accelerator captures have identical CPU pooling
        # semantics even if ambient code changed the process policy between calls.
        self._force_round45_cpu_flush_denormal_policy()
        super()._assert_autocast_disabled()

        # CPU float64 pooling occurs for every source device. Resolve a concrete host
        # identity before checkpoint execution, not only for CPU model execution.
        processor = getattr(self, "_canonical_cpu_processor", None)
        if not _is_concrete_cpu_identity(processor):
            self._canonical_cpu_processor = _concrete_cpu_identity()

    def _assert_cpu_dispatch_policy(self) -> None:
        super()._assert_cpu_dispatch_policy()
        self._assert_round45_cpu_pooling_identity()
        # A fresh import boundary prevents pre-construction delegated-loader patches;
        # after loading, additionally bind the concrete classes that Auto* resolved.
        self._assert_round45_transformers_delegates()

    def _assert_live_state_authentication(self) -> None:
        self._force_round45_cpu_flush_denormal_policy()
        self._assert_round45_cpu_pooling_identity()
        super()._assert_live_state_authentication()

    def metadata(self) -> Mapping[str, Any]:
        self._force_round45_cpu_flush_denormal_policy()
        processor = self._assert_round45_cpu_pooling_identity()
        data = dict(super().metadata())
        data["cpu_processor"] = processor
        data["cpu_flush_denormal"] = False
        return data


# Preserve the exact-type OBSERVATION gate before capture_execute imports production.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

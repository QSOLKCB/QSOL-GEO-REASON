"""Final production hardening for GEO-CAP-001 canonical OBSERVATION.

This layer keeps construction-only trust anchors outside caller-writable instance
state. The pre-deserialization checks are attached to the real core loader path, so
software fixtures that deliberately replace the lower loader keep testing the
production boundary they were written for without weakening real OBSERVATION loads.
"""
from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_snapshot import _snapshot_file_hashes
from .capture_validation import validate_capture_request
from . import capture_backend_core as _core
from . import capture_backend_audit as _audit
from . import capture_backend_production as _production


_BaseAuditedBackend = _audit.HuggingFacePyTorchBackend
_TREE_RECEIPT_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")
_LEGACY_PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt")
_ORIGINAL_CORE_INIT = _core.HuggingFacePyTorchBackend.__init__


# The preregistered Hub-tree receipts, trusted pre-load snapshot bytes, and final
# runtime provenance are closure-owned. Instance attribute rewrites therefore cannot
# redefine what construction originally authenticated.
def _make_final_construction_vault():
    preflight: weakref.WeakKeyDictionary[
        Any,
        tuple[str, str, Path, Path, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
    ] = weakref.WeakKeyDictionary()
    baselines: weakref.WeakKeyDictionary[
        Any,
        tuple[str, str, str, str, str, str],
    ] = weakref.WeakKeyDictionary()

    def remember_preflight(
        instance: Any,
        value: tuple[
            str,
            str,
            Path,
            Path,
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str], ...],
        ],
    ) -> None:
        if instance in preflight:
            raise CaptureContractError("final construction preflight was already initialized")
        preflight[instance] = value

    def recall_preflight(
        instance: Any,
    ) -> tuple[
        str,
        str,
        Path,
        Path,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
    ] | None:
        return preflight.get(instance)

    def remember_baseline(
        instance: Any, baseline: tuple[str, str, str, str, str, str]
    ) -> None:
        if instance in baselines:
            raise CaptureContractError("final construction provenance baseline was already initialized")
        baselines[instance] = baseline

    def recall_baseline(instance: Any) -> tuple[str, str, str, str, str, str] | None:
        return baselines.get(instance)

    return remember_preflight, recall_preflight, remember_baseline, recall_baseline


(
    _remember_final_construction_preflight,
    _recall_final_construction_preflight,
    _remember_final_construction_baseline,
    _recall_final_construction_baseline,
) = _make_final_construction_vault()
del _make_final_construction_vault


def _require_frozen_tree_receipts(model: Mapping[str, Any]) -> tuple[str, str]:
    missing = [field for field in _TREE_RECEIPT_FIELDS if field not in model]
    if missing:
        raise CaptureContractError(
            "canonical OBSERVATION requires frozen Hub commit-tree SHA-256 receipts before "
            "model loading: " + ", ".join(missing)
        )
    return str(model["revision_tree_sha256"]), str(model["tokenizer_revision_tree_sha256"])


def _assert_safetensors_only_checkpoint(model_hashes: Mapping[str, str]) -> None:
    """Reject every pickle-capable checkpoint lane before Transformers is invoked."""
    safetensors = sorted(
        path
        for path in model_hashes
        if isinstance(path, str) and path.lower().endswith(".safetensors")
    )
    legacy = sorted(
        path
        for path in model_hashes
        if isinstance(path, str) and path.lower().endswith(_LEGACY_PICKLE_SUFFIXES)
    )
    if not safetensors:
        raise CaptureContractError(
            "canonical OBSERVATION requires Safetensors checkpoint weights before deserialization"
        )
    if legacy:
        preview = ", ".join(legacy[:3])
        raise CaptureContractError(
            "canonical OBSERVATION forbids pickle-capable checkpoint artifacts before loading: "
            + preview
        )


def _hardened_core_init(instance: Any, request: Mapping[str, Any]) -> None:
    """Authenticate the frozen snapshot before entering the actual core model loader.

    The production wrapper owns the exclusive Python-thread boundary before it calls
    this core constructor. Tests that replace the inherited loader never enter this
    function, while a real canonical load cannot reach Transformers deserialization
    until both Hub-tree receipts and the Safetensors-only checkpoint policy pass.
    """
    validated = validate_capture_request(request)
    model_cfg = validated["model"]
    model_tree_receipt, tokenizer_tree_receipt = _require_frozen_tree_receipts(model_cfg)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CaptureBackendUnavailable(
            "canonical capture requires optional capture dependencies; install qsol-geo-reason[capture]"
        ) from exc

    model_snapshot = Path(
        snapshot_download(
            repo_id=model_cfg["identifier"],
            revision=model_cfg["revision"],
            local_files_only=True,
        )
    )
    tokenizer_snapshot = Path(
        snapshot_download(
            repo_id=model_cfg["tokenizer_identifier"],
            revision=model_cfg["tokenizer_revision"],
            local_files_only=True,
        )
    )
    trusted_model_before = _snapshot_file_hashes(
        model_snapshot,
        model_cfg["revision"],
        "model",
        expected_tree_receipt_sha256=model_tree_receipt,
    )
    trusted_tokenizer_before = _snapshot_file_hashes(
        tokenizer_snapshot,
        model_cfg["tokenizer_revision"],
        "tokenizer",
        expected_tree_receipt_sha256=tokenizer_tree_receipt,
    )
    _assert_safetensors_only_checkpoint(trusted_model_before)
    _remember_final_construction_preflight(
        instance,
        (
            model_tree_receipt,
            tokenizer_tree_receipt,
            model_snapshot,
            tokenizer_snapshot,
            tuple(sorted(trusted_model_before.items())),
            tuple(sorted(trusted_tokenizer_before.items())),
        ),
    )

    _ORIGINAL_CORE_INIT(instance, validated)

    # The inherited core already brackets model/tokenizer loading with local SHA-256
    # receipts. Re-authenticate those bytes against the preregistered Hub-tree receipt
    # and require the inherited provenance maps to describe the same trusted files.
    trusted_model_after = _snapshot_file_hashes(
        model_snapshot,
        model_cfg["revision"],
        "model",
        expected_tree_receipt_sha256=model_tree_receipt,
    )
    trusted_tokenizer_after = _snapshot_file_hashes(
        tokenizer_snapshot,
        model_cfg["tokenizer_revision"],
        "tokenizer",
        expected_tree_receipt_sha256=tokenizer_tree_receipt,
    )
    if trusted_model_before != trusted_model_after:
        raise CaptureContractError("trusted model snapshot changed during canonical loading")
    if trusted_tokenizer_before != trusted_tokenizer_after:
        raise CaptureContractError("trusted tokenizer snapshot changed during canonical loading")
    if dict(getattr(instance, "_model_snapshot_hashes", {})) != trusted_model_after:
        raise CaptureContractError("loaded model receipt does not match the trusted Hub tree")
    if dict(getattr(instance, "_tokenizer_snapshot_hashes", {})) != trusted_tokenizer_after:
        raise CaptureContractError("loaded tokenizer receipt does not match the trusted Hub tree")

    torch_build = getattr(instance, "_torch_build_provenance", None)
    if not isinstance(torch_build, Mapping):
        raise CaptureContractError("canonical PyTorch build provenance baseline is missing")
    attention = getattr(instance, "_attention_implementation", None)
    model = getattr(instance, "_model", None)
    config = getattr(model, "config", None)
    live_attention = getattr(config, "_attn_implementation", None)
    if attention != live_attention or not isinstance(attention, str):
        raise CaptureContractError(
            "canonical attention implementation is not bound to the loaded model configuration"
        )

    _remember_final_construction_baseline(
        instance,
        (
            model_tree_receipt,
            tokenizer_tree_receipt,
            sha256_json(dict(torch_build)),
            attention,
            sha256_json(dict(sorted(trusted_model_after.items()))),
            sha256_json(dict(sorted(trusted_tokenizer_after.items()))),
        ),
    )


# Harden only the real core loader. Production constructor tests that deliberately
# replace the inherited loader keep their original boundary semantics, while every
# unmocked OBSERVATION path reaches this wrapper before model deserialization.
_core.HuggingFacePyTorchBackend.__init__ = _hardened_core_init


class HuggingFacePyTorchBackend(_BaseAuditedBackend):
    """Final canonical backend with preregistered Hub-tree and runtime trust anchors."""

    def _assert_cpu_dispatch_policy(self) -> None:
        # The inherited production constructor calls this after authenticated loading.
        # Preserve every earlier audit check first, then reassert the external seal.
        super()._assert_cpu_dispatch_policy()
        self._assert_final_construction_baseline()

    def _assert_final_construction_baseline(self) -> None:
        baseline = _recall_final_construction_baseline(self)
        if baseline is None:
            # Dependency-free fixtures may deliberately replace the lower loader.
            return
        (
            _model_tree_receipt,
            _tokenizer_tree_receipt,
            expected_torch_build,
            expected_attention,
            expected_model_snapshot,
            expected_tokenizer_snapshot,
        ) = baseline

        torch_build = getattr(self, "_torch_build_provenance", None)
        if not isinstance(torch_build, Mapping) or sha256_json(dict(torch_build)) != expected_torch_build:
            raise CaptureContractError(
                "PyTorch build provenance changed after authenticated backend construction"
            )

        model = getattr(self, "_model", None)
        config = getattr(model, "config", None)
        live_attention = getattr(config, "_attn_implementation", None)
        if (
            getattr(self, "_attention_implementation", None) != expected_attention
            or live_attention != expected_attention
        ):
            raise CaptureContractError(
                "attention implementation changed after authenticated model construction"
            )

        model_hashes = getattr(self, "_model_snapshot_hashes", None)
        tokenizer_hashes = getattr(self, "_tokenizer_snapshot_hashes", None)
        if not isinstance(model_hashes, Mapping) or not isinstance(tokenizer_hashes, Mapping):
            raise CaptureContractError("trusted snapshot provenance maps are missing")
        if sha256_json(dict(sorted(model_hashes.items()))) != expected_model_snapshot:
            raise CaptureContractError("trusted model snapshot provenance changed after construction")
        if sha256_json(dict(sorted(tokenizer_hashes.items()))) != expected_tokenizer_snapshot:
            raise CaptureContractError("trusted tokenizer snapshot provenance changed after construction")

    def _assert_live_state_authentication(self) -> None:
        # Keep this explicit as well as inherited: the final concrete boundary must
        # visibly reject late module-hook injection before any OBSERVATION reuse.
        self._assert_no_registered_module_hooks()
        super()._assert_live_state_authentication()
        self._assert_final_construction_baseline()

    def assert_execution_request(self, request: Mapping[str, Any]) -> None:
        super().assert_execution_request(request)
        baseline = _recall_final_construction_baseline(self)
        if baseline is None:
            return
        model = request["model"]
        requested = (
            model.get("revision_tree_sha256"),
            model.get("tokenizer_revision_tree_sha256"),
        )
        if requested != baseline[:2]:
            raise CaptureContractError(
                "backend Hub commit-tree receipts do not match the execution request"
            )
        self._assert_final_construction_baseline()

    def metadata(self) -> Mapping[str, Any]:
        # Authenticate the external construction baseline before the inherited core
        # expands caller-writable instance maps into persisted provenance.
        self._assert_final_construction_baseline()
        return super().metadata()


# capture.py imports this module before capture_execute imports the production module.
# Rebind the canonical exact-type boundary to this final audited subclass.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

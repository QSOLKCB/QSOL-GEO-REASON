"""Final production hardening for GEO-CAP-001 canonical OBSERVATION.

This layer keeps construction-only trust anchors outside caller-writable instance
state and performs the checks that must precede the inherited Transformers loader.
"""
from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_snapshot import _snapshot_file_hashes
from .capture_validation import validate_capture_request
from . import capture_backend_audit as _audit
from . import capture_backend_production as _production


_BaseAuditedBackend = _audit.HuggingFacePyTorchBackend
_TREE_RECEIPT_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")
_LEGACY_PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt")


def _make_final_construction_vault():
    baselines: weakref.WeakKeyDictionary[
        Any,
        tuple[str, str, str, str, str, str],
    ] = weakref.WeakKeyDictionary()

    def remember(instance: Any, baseline: tuple[str, str, str, str, str, str]) -> None:
        if instance in baselines:
            raise CaptureContractError("final construction provenance baseline was already initialized")
        baselines[instance] = baseline

    def recall(instance: Any) -> tuple[str, str, str, str, str, str] | None:
        return baselines.get(instance)

    return remember, recall


_remember_final_construction_baseline, _recall_final_construction_baseline = (
    _make_final_construction_vault()
)
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
        path for path in model_hashes
        if isinstance(path, str) and path.lower().endswith(".safetensors")
    )
    legacy = sorted(
        path for path in model_hashes
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


class HuggingFacePyTorchBackend(_BaseAuditedBackend):
    """Final canonical backend with preregistered Hub-tree and loader trust anchors."""

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        model_cfg = validated["model"]
        model_tree_receipt, tokenizer_tree_receipt = _require_frozen_tree_receipts(model_cfg)

        # Resolve and authenticate both cache snapshots before importing/loading the
        # PyTorch/Transformers model path in the inherited constructor. The tree JSON
        # is trusted only when its exact bytes match the preregistered request receipt.
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

        super().__init__(validated)

        # Re-authenticate against the frozen tree after loading and require the
        # inherited receipts to describe those exact trusted bytes. A cache/tree swap
        # during construction therefore cannot become accepted provenance.
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
        if dict(getattr(self, "_model_snapshot_hashes", {})) != trusted_model_after:
            raise CaptureContractError("loaded model receipt does not match the trusted Hub tree")
        if dict(getattr(self, "_tokenizer_snapshot_hashes", {})) != trusted_tokenizer_after:
            raise CaptureContractError("loaded tokenizer receipt does not match the trusted Hub tree")

        torch_build = getattr(self, "_torch_build_provenance", None)
        if not isinstance(torch_build, Mapping):
            raise CaptureContractError("canonical PyTorch build provenance baseline is missing")
        attention = getattr(self, "_attention_implementation", None)
        live_attention = getattr(getattr(self, "_model", None).config, "_attn_implementation", None)
        if attention != live_attention or not isinstance(attention, str):
            raise CaptureContractError(
                "canonical attention implementation is not bound to the loaded model configuration"
            )

        _remember_final_construction_baseline(
            self,
            (
                model_tree_receipt,
                tokenizer_tree_receipt,
                sha256_json(dict(torch_build)),
                attention,
                sha256_json(dict(sorted(trusted_model_after.items()))),
                sha256_json(dict(sorted(trusted_tokenizer_after.items()))),
            ),
        )

    def _assert_final_construction_baseline(self) -> None:
        baseline = _recall_final_construction_baseline(self)
        if baseline is None:
            # Dependency-free fixtures may bypass production construction.
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

        live_attention = getattr(getattr(self, "_model", None).config, "_attn_implementation", None)
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

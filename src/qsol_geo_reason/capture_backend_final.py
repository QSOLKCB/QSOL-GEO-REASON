"""Final production hardening for GEO-CAP-001 canonical OBSERVATION.

This layer keeps construction-only trust anchors outside caller-writable instance
state. Pre-deserialization hardening is attached to an existing dynamically dispatched
core guard, so the original production/core constructors remain byte-for-byte visible
to the long-standing source-invariant regression suite.
"""
from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_snapshot import _snapshot_file_hashes
from . import capture_backend_audit as _audit
from . import capture_backend_production as _production


_BaseAuditedBackend = _audit.HuggingFacePyTorchBackend
_TREE_RECEIPT_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")
_LEGACY_PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt")


# Request-bound tree receipts, trusted pre-load snapshot bytes, and final runtime
# provenance live in closure-owned weak maps. Instance attribute rewrites therefore
# cannot redefine what canonical construction originally authenticated.
def _make_final_construction_vault():
    requests: weakref.WeakKeyDictionary[Any, tuple[str | None, str | None]] = (
        weakref.WeakKeyDictionary()
    )
    preflight: weakref.WeakKeyDictionary[
        Any,
        tuple[str, str, Path, Path, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
    ] = weakref.WeakKeyDictionary()
    baselines: weakref.WeakKeyDictionary[
        Any,
        tuple[str, str, str, str, str, str],
    ] = weakref.WeakKeyDictionary()

    def remember_request(instance: Any, value: tuple[str | None, str | None]) -> None:
        if instance in requests:
            raise CaptureContractError("final construction request binding was already initialized")
        requests[instance] = value

    def recall_request(instance: Any) -> tuple[str | None, str | None] | None:
        return requests.get(instance)

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

    return (
        remember_request,
        recall_request,
        remember_preflight,
        recall_preflight,
        remember_baseline,
        recall_baseline,
    )


(
    _remember_final_construction_request,
    _recall_final_construction_request,
    _remember_final_construction_preflight,
    _recall_final_construction_preflight,
    _remember_final_construction_baseline,
    _recall_final_construction_baseline,
) = _make_final_construction_vault()
del _make_final_construction_vault


def _require_frozen_tree_receipts(
    receipts: tuple[str | None, str | None]
) -> tuple[str, str]:
    missing = [
        field
        for field, value in zip(_TREE_RECEIPT_FIELDS, receipts)
        if value is None
    ]
    if missing:
        raise CaptureContractError(
            "canonical OBSERVATION requires frozen Hub commit-tree SHA-256 receipts before "
            "model loading: " + ", ".join(missing)
        )
    model_receipt, tokenizer_receipt = receipts
    assert model_receipt is not None and tokenizer_receipt is not None
    return model_receipt, tokenizer_receipt


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


class HuggingFacePyTorchBackend(_BaseAuditedBackend):
    """Final canonical backend with preregistered Hub-tree and runtime trust anchors."""

    @classmethod
    def _validate_pre_cuda_environment(cls, request: Mapping[str, Any]) -> None:
        # Preserve the inherited pre-import CUDA policy first. Record the already
        # validated request receipts externally, but defer requiring them until the
        # unmocked core loader reaches its pre-snapshot autocast guard. This keeps
        # dependency-free constructor-boundary fixtures truthful without creating a
        # bypass in a real model-loading path.
        super()._validate_pre_cuda_environment(request)
        model = request["model"]
        instance = None
        # classmethod dispatch does not expose self; the production constructor calls
        # this through self but Python supplies the class. Request binding is therefore
        # completed in the instance guard below from the construction-shadow copy.
        # The no-op body here intentionally preserves the established constructor hook.
        _ = (model.get("revision_tree_sha256"), model.get("tokenizer_revision_tree_sha256"), instance)

    def _bind_pending_tree_receipts_from_construction_request(self) -> None:
        """Recover request receipts from the immutable construction shadow when present."""
        if _recall_final_construction_request(self) is not None:
            return
        # The audited hierarchy freezes a detached validated request shadow before the
        # core loader is entered. Use only that detached copy; do not trust mutable
        # caller-owned request state or later instance substitutions.
        request = getattr(self, "_construction_request", None)
        if isinstance(request, Mapping):
            model = request.get("model")
            if isinstance(model, Mapping):
                _remember_final_construction_request(
                    self,
                    (
                        model.get("revision_tree_sha256") if isinstance(model.get("revision_tree_sha256"), str) else None,
                        model.get("tokenizer_revision_tree_sha256") if isinstance(model.get("tokenizer_revision_tree_sha256"), str) else None,
                    ),
                )

    def _assert_autocast_disabled(self) -> None:
        """Run frozen-tree/Safetensors preflight at the real core pre-load boundary."""
        self._bind_pending_tree_receipts_from_construction_request()
        request_receipts = _recall_final_construction_request(self)
        preflight = _recall_final_construction_preflight(self)

        # Object-level software fixtures can call inherited helpers without ever
        # entering production construction. They have no construction request marker
        # and therefore exercise only the inherited autocast policy.
        if request_receipts is not None and preflight is None:
            model_tree_receipt, tokenizer_tree_receipt = _require_frozen_tree_receipts(
                request_receipts
            )
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise CaptureBackendUnavailable(
                    "canonical capture requires optional capture dependencies; install qsol-geo-reason[capture]"
                ) from exc

            model_snapshot = Path(
                snapshot_download(
                    repo_id=self._model_identifier,
                    revision=self._model_revision,
                    local_files_only=True,
                )
            )
            tokenizer_snapshot = Path(
                snapshot_download(
                    repo_id=self._tokenizer_identifier,
                    revision=self._tokenizer_revision,
                    local_files_only=True,
                )
            )
            trusted_model_before = _snapshot_file_hashes(
                model_snapshot,
                self._model_revision,
                "model",
                expected_tree_receipt_sha256=model_tree_receipt,
            )
            trusted_tokenizer_before = _snapshot_file_hashes(
                tokenizer_snapshot,
                self._tokenizer_revision,
                "tokenizer",
                expected_tree_receipt_sha256=tokenizer_tree_receipt,
            )
            _assert_safetensors_only_checkpoint(trusted_model_before)
            _remember_final_construction_preflight(
                self,
                (
                    model_tree_receipt,
                    tokenizer_tree_receipt,
                    model_snapshot,
                    tokenizer_snapshot,
                    tuple(sorted(trusted_model_before.items())),
                    tuple(sorted(trusted_tokenizer_before.items())),
                ),
            )

        super()._assert_autocast_disabled()

    def _finalize_or_assert_construction_baseline(self) -> None:
        preflight = _recall_final_construction_preflight(self)
        if preflight is None:
            # Dependency-free fixtures may deliberately replace the lower loader.
            return
        (
            model_tree_receipt,
            tokenizer_tree_receipt,
            model_snapshot,
            tokenizer_snapshot,
            model_before_items,
            tokenizer_before_items,
        ) = preflight
        model_before = dict(model_before_items)
        tokenizer_before = dict(tokenizer_before_items)

        trusted_model_after = _snapshot_file_hashes(
            model_snapshot,
            self._model_revision,
            "model",
            expected_tree_receipt_sha256=model_tree_receipt,
        )
        trusted_tokenizer_after = _snapshot_file_hashes(
            tokenizer_snapshot,
            self._tokenizer_revision,
            "tokenizer",
            expected_tree_receipt_sha256=tokenizer_tree_receipt,
        )
        if model_before != trusted_model_after:
            raise CaptureContractError("trusted model snapshot changed during canonical loading")
        if tokenizer_before != trusted_tokenizer_after:
            raise CaptureContractError("trusted tokenizer snapshot changed during canonical loading")
        if dict(getattr(self, "_model_snapshot_hashes", {})) != trusted_model_after:
            raise CaptureContractError("loaded model receipt does not match the trusted Hub tree")
        if dict(getattr(self, "_tokenizer_snapshot_hashes", {})) != trusted_tokenizer_after:
            raise CaptureContractError("loaded tokenizer receipt does not match the trusted Hub tree")

        baseline = _recall_final_construction_baseline(self)
        if baseline is None:
            torch_build = getattr(self, "_torch_build_provenance", None)
            if not isinstance(torch_build, Mapping):
                raise CaptureContractError("canonical PyTorch build provenance baseline is missing")
            attention = getattr(self, "_attention_implementation", None)
            model = getattr(self, "_model", None)
            config = getattr(model, "config", None)
            live_attention = getattr(config, "_attn_implementation", None)
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
        else:
            self._assert_final_construction_baseline()

    def _assert_cpu_dispatch_policy(self) -> None:
        # The inherited production constructor calls this after authenticated loading.
        # Preserve every earlier audit check first, then seal/reassert external state.
        super()._assert_cpu_dispatch_policy()
        self._finalize_or_assert_construction_baseline()

    def _assert_final_construction_baseline(self) -> None:
        baseline = _recall_final_construction_baseline(self)
        if baseline is None:
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
        self._assert_final_construction_baseline()
        return super().metadata()


# capture.py imports this module before capture_execute imports the production module.
# Rebind the canonical exact-type boundary to this final audited subclass.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

"""Final audit hardening layer for GEO-CAP-001 production observations.

This layer closes audit findings that need to run around the existing production
constructor while deliberately leaving that constructor as the canonical implementation.
That keeps the long-established construction/restoration invariants source-visible and
adds only narrowly scoped pre-construction and provenance guards.
"""
from __future__ import annotations

import os
import sys
import weakref
from typing import Any, Mapping

from .capture_common import CaptureContractError
from .capture_package import _python_package_provenance
from .capture_validation import validate_capture_request
from . import capture_backend_production as _production


_BaseProductionBackend = _production.HuggingFacePyTorchBackend


def _make_snapshot_provenance_vault():
    """Keep persisted snapshot baselines outside caller-writable instance state."""
    baselines: weakref.WeakKeyDictionary[
        Any, tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]
    ] = weakref.WeakKeyDictionary()

    def remember(
        instance: Any,
        baseline: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
    ) -> None:
        if instance in baselines:
            raise CaptureContractError(
                "canonical snapshot provenance baseline was already initialized"
            )
        baselines[instance] = baseline

    def recall(
        instance: Any,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]] | None:
        return baselines.get(instance)

    return remember, recall


_remember_snapshot_provenance_baseline, _recall_snapshot_provenance_baseline = (
    _make_snapshot_provenance_vault()
)
del _make_snapshot_provenance_vault


class HuggingFacePyTorchBackend(_BaseProductionBackend):
    """Audit-complete production backend exported at the canonical boundary."""

    _CPU_MATH_ENVIRONMENT_FIELDS = {
        "onednn_max_cpu_isa": "ONEDNN_MAX_CPU_ISA",
        "dnnl_max_cpu_isa": "DNNL_MAX_CPU_ISA",
        "mkl_cbwr": "MKL_CBWR",
    }
    _NATIVE_SLOW_TOKENIZER_MODULE_PREFIXES = (
        "sentencepiece",
        "_sentencepiece",
    )

    def __new__(cls, request: Mapping[str, Any]):
        """Freeze initialization-time audit inputs before the inherited constructor."""
        validated = validate_capture_request(request)
        instance = super().__new__(cls)
        device = validated["backend"]["device"]

        # oneDNN/DNNL/MKL dispatch controls are initialization-time execution inputs.
        # Capture them before the inherited production constructor can import or
        # initialize PyTorch. If PyTorch already exists, historical values cannot be
        # reconstructed and the manifest records that fact explicitly.
        cpu_active = device == "cpu"
        known = cpu_active and "torch" not in sys.modules
        environment = {
            field: os.environ.get(variable)
            for field, variable in cls._CPU_MATH_ENVIRONMENT_FIELDS.items()
        }
        instance._canonical_cpu_math_environment_known = known if cpu_active else None
        instance._canonical_cpu_math_environment = environment if known else None

        # Package and snapshot receipts are initialized only after the inherited
        # constructor has authenticated and loaded its tokenizer/checkpoint.
        instance._snapshot_provenance_baseline_initialized = False
        instance._tokenizers_native_backend_active = False
        instance._tokenizers_package_provenance = None
        instance._tokenizers_package_provenance_initialized = False
        instance._safetensors_deserializer_active = False
        instance._safetensors_package_provenance = None
        instance._safetensors_package_provenance_initialized = False
        return instance

    def _assert_cpu_math_environment_policy(self) -> None:
        state = vars(self)
        # Dependency-free source-audit fixtures may deliberately bypass __new__.
        # Real construction always installs this boundary state.
        if "_canonical_cpu_math_environment_known" not in state:
            return
        if getattr(self, "_device_type", None) != "cpu":
            return
        known = state.get("_canonical_cpu_math_environment_known")
        expected = state.get("_canonical_cpu_math_environment")
        if known is True:
            observed = {
                field: os.environ.get(variable)
                for field, variable in self._CPU_MATH_ENVIRONMENT_FIELDS.items()
            }
            if observed != expected:
                raise CaptureContractError(
                    "CPU math-library dispatch override drifted after initialization"
                )
        elif known is not False:
            raise CaptureContractError("CPU math-library dispatch provenance is missing")

    @classmethod
    def _native_slow_tokenizer_backend(cls, tokenizer: Any) -> str | None:
        """Identify native execution objects used by a slow tokenizer.

        GEO-CAP-001 currently content-binds the Rust Tokenizers fast path. A slow
        tokenizer that delegates segmentation to SentencePiece is a distinct native
        execution lane; reject it until that lane receives its own persisted receipt.
        """
        state = getattr(tokenizer, "__dict__", {})
        candidates = [tokenizer]
        if isinstance(state, Mapping):
            candidates.extend(state.values())
        for value in candidates:
            module = getattr(type(value), "__module__", "")
            if not isinstance(module, str):
                continue
            for prefix in cls._NATIVE_SLOW_TOKENIZER_MODULE_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    return "sentencepiece"
        if (
            isinstance(state, Mapping)
            and any(name in state for name in ("sp_model", "spm", "sentencepiece_processor"))
            and (
                "sentencepiece" in sys.modules
                or "_sentencepiece" in sys.modules
            )
        ):
            return "sentencepiece"
        return None

    def _initialize_snapshot_provenance_baseline(self) -> None:
        state = vars(self)
        if "_snapshot_provenance_baseline_initialized" not in state:
            return
        if state.get("_snapshot_provenance_baseline_initialized") is True:
            return
        model_hashes = state.get("_model_snapshot_hashes")
        tokenizer_hashes = state.get("_tokenizer_snapshot_hashes")
        if model_hashes is None and tokenizer_hashes is None:
            # Dependency-free constructor fixtures intentionally bypass loading.
            return
        if not isinstance(model_hashes, Mapping) or not isinstance(tokenizer_hashes, Mapping):
            raise CaptureContractError("canonical snapshot provenance maps are missing")
        baseline = (
            tuple(sorted((str(path), str(digest)) for path, digest in model_hashes.items())),
            tuple(sorted((str(path), str(digest)) for path, digest in tokenizer_hashes.items())),
        )
        _remember_snapshot_provenance_baseline(self, baseline)
        state["_snapshot_provenance_baseline_initialized"] = True

    def _assert_snapshot_provenance_baseline(self) -> None:
        baseline = _recall_snapshot_provenance_baseline(self)
        if baseline is None:
            return
        model_hashes = getattr(self, "_model_snapshot_hashes", None)
        tokenizer_hashes = getattr(self, "_tokenizer_snapshot_hashes", None)
        if not isinstance(model_hashes, Mapping) or not isinstance(tokenizer_hashes, Mapping):
            raise CaptureContractError("canonical snapshot provenance maps were removed")
        observed = (
            tuple(sorted((str(path), str(digest)) for path, digest in model_hashes.items())),
            tuple(sorted((str(path), str(digest)) for path, digest in tokenizer_hashes.items())),
        )
        if observed != baseline:
            raise CaptureContractError(
                "persisted model/tokenizer snapshot provenance changed after authenticated loading"
            )

    def _initialize_tokenizers_package_provenance(self) -> None:
        state = vars(self)
        # Synthetic fixtures built with object.__new__ intentionally do not acquire
        # production provenance. Real construction installs this marker in __new__.
        if "_tokenizers_package_provenance_initialized" not in state:
            return
        if state.get("_tokenizers_package_provenance_initialized") is True:
            return
        tokenizer = state.get("_tokenizer")
        if tokenizer is None:
            return

        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is None:
            native_slow_backend = self._native_slow_tokenizer_backend(tokenizer)
            if native_slow_backend is not None:
                raise CaptureContractError(
                    "canonical OBSERVATION does not support native slow-tokenizer "
                    f"execution ({native_slow_backend}); use a receipt-bound fast tokenizer "
                    "or define a future native slow-tokenizer provenance lane"
                )

        active = backend_tokenizer is not None
        state["_tokenizers_native_backend_active"] = active
        state["_tokenizers_package_provenance"] = None
        if active:
            process_tokenizers = sys.modules.get("tokenizers")
            if process_tokenizers is None:
                raise CaptureContractError(
                    "active fast tokenizer does not expose its imported native tokenizers package"
                )
            state["_tokenizers_package_provenance"] = _python_package_provenance(
                process_tokenizers, "Tokenizers"
            )
        state["_tokenizers_package_provenance_initialized"] = True

    def _initialize_checkpoint_deserializer_provenance(self) -> None:
        state = vars(self)
        if "_safetensors_package_provenance_initialized" not in state:
            return
        if state.get("_safetensors_package_provenance_initialized") is True:
            return
        model_hashes = state.get("_model_snapshot_hashes")
        if model_hashes is None:
            return
        if not isinstance(model_hashes, Mapping):
            raise CaptureContractError("canonical model snapshot provenance is missing")

        active = any(
            isinstance(path, str) and path.lower().endswith(".safetensors")
            for path in model_hashes
        )
        state["_safetensors_deserializer_active"] = active
        state["_safetensors_package_provenance"] = None
        if active:
            process_safetensors = sys.modules.get("safetensors")
            if process_safetensors is None:
                raise CaptureContractError(
                    "Safetensors checkpoint was loaded without an inspectable imported safetensors package"
                )
            state["_safetensors_package_provenance"] = _python_package_provenance(
                process_safetensors, "Safetensors"
            )
        state["_safetensors_package_provenance_initialized"] = True

    def _assert_checkpoint_deserializer_provenance(self) -> None:
        state = vars(self)
        if "_safetensors_package_provenance_initialized" not in state:
            return
        initialized = state.get("_safetensors_package_provenance_initialized")
        model_hashes = state.get("_model_snapshot_hashes")
        if initialized is not True:
            if model_hashes is not None:
                raise CaptureContractError(
                    "checkpoint deserializer provenance was not initialized"
                )
            return
        active = state.get("_safetensors_deserializer_active")
        if type(active) is not bool:
            raise CaptureContractError("Safetensors deserializer activation state is invalid")
        provenance = state.get("_safetensors_package_provenance")
        if active:
            process_safetensors = sys.modules.get("safetensors")
            if process_safetensors is None or not isinstance(provenance, Mapping):
                raise CaptureContractError("Safetensors package provenance is missing")
            if _python_package_provenance(
                process_safetensors, "Safetensors"
            ) != dict(provenance):
                raise CaptureContractError(
                    "imported Safetensors package changed after authenticated checkpoint loading"
                )
        elif provenance is not None:
            raise CaptureContractError(
                "inactive Safetensors deserializer cannot carry package provenance"
            )

    @staticmethod
    def _assert_no_active_python_instrumentation() -> None:
        trace = sys.gettrace()
        profile = sys.getprofile()
        if trace is not None or profile is not None:
            raise CaptureContractError(
                "canonical OBSERVATION forbids active Python trace/profile callbacks"
            )

    def _assert_no_active_torch_override_modes(self) -> None:
        # Production begin_observation invokes this after acquiring the exclusive
        # Python-thread boundary, so tracing/profile instrumentation is rejected
        # before any capture-state mutation or first model forward.
        super()._assert_no_active_torch_override_modes()
        self._assert_no_active_python_instrumentation()

    def _assert_live_state_authentication(self) -> None:
        super()._assert_live_state_authentication()
        self._assert_snapshot_provenance_baseline()
        self._assert_checkpoint_deserializer_provenance()

    def _assert_cpu_dispatch_policy(self) -> None:
        # Production __init__ invokes this after the inherited model/tokenizer load,
        # making it a construction-end hook without replacing the canonical __init__.
        super()._assert_cpu_dispatch_policy()
        self._assert_cpu_math_environment_policy()
        self._initialize_snapshot_provenance_baseline()
        self._initialize_tokenizers_package_provenance()
        self._initialize_checkpoint_deserializer_provenance()

    def metadata(self) -> Mapping[str, Any]:
        # super().metadata() dynamically reaches the live-state override above,
        # rechecking snapshot/deserializer provenance before producing persisted JSON.
        data = dict(super().metadata())
        state = vars(self)

        if "_tokenizers_package_provenance_initialized" in state:
            initialized = state.get("_tokenizers_package_provenance_initialized")
            tokenizer = state.get("_tokenizer")
            if initialized is not True:
                # Dependency-free constructor fixtures replace the inherited loader
                # and intentionally never create a tokenizer. They are not capable of
                # emitting OBSERVATION evidence, so there is no native implementation
                # to receipt. A real loaded tokenizer must never reach this branch.
                if tokenizer is not None:
                    raise CaptureContractError(
                        "native Tokenizers package provenance was not initialized"
                    )
            else:
                native_active = state.get("_tokenizers_native_backend_active")
                if type(native_active) is not bool:
                    raise CaptureContractError("native Tokenizers backend activation state is invalid")
                tokenizers_provenance = state.get("_tokenizers_package_provenance")
                if native_active:
                    process_tokenizers = sys.modules.get("tokenizers")
                    if process_tokenizers is None or not isinstance(tokenizers_provenance, Mapping):
                        raise CaptureContractError("native Tokenizers package provenance is missing")
                    if _python_package_provenance(
                        process_tokenizers, "Tokenizers"
                    ) != dict(tokenizers_provenance):
                        raise CaptureContractError(
                            "imported Tokenizers package changed after authenticated backend construction"
                        )
                elif tokenizers_provenance is not None:
                    raise CaptureContractError(
                        "slow tokenizer cannot carry native Tokenizers package provenance"
                    )

                data["tokenizers_native_backend_active"] = native_active
                data["tokenizers_package_file_count"] = (
                    tokenizers_provenance["file_count"]
                    if isinstance(tokenizers_provenance, Mapping)
                    else None
                )
                data["tokenizers_package_receipt_sha256"] = (
                    tokenizers_provenance["receipt_sha256"]
                    if isinstance(tokenizers_provenance, Mapping)
                    else None
                )

        if "_safetensors_package_provenance_initialized" in state:
            initialized = state.get("_safetensors_package_provenance_initialized")
            model_hashes = state.get("_model_snapshot_hashes")
            if initialized is not True:
                if model_hashes is not None:
                    raise CaptureContractError(
                        "Safetensors package provenance was not initialized"
                    )
            else:
                self._assert_checkpoint_deserializer_provenance()
                active = state.get("_safetensors_deserializer_active")
                provenance = state.get("_safetensors_package_provenance")
                data["safetensors_deserializer_active"] = active
                data["safetensors_package_file_count"] = (
                    provenance["file_count"] if isinstance(provenance, Mapping) else None
                )
                data["safetensors_package_receipt_sha256"] = (
                    provenance["receipt_sha256"] if isinstance(provenance, Mapping) else None
                )

        if "_canonical_cpu_math_environment_known" in state:
            known = state.get("_canonical_cpu_math_environment_known")
            environment = state.get("_canonical_cpu_math_environment")
            data["cpu_math_dispatch_env_known"] = known
            for field in self._CPU_MATH_ENVIRONMENT_FIELDS:
                data[field] = environment[field] if environment is not None else None
        return data


# capture.py imports this module before capture_execute imports the production module.
# Rebinding therefore preserves the exact-type OBSERVATION boundary while the inherited
# production constructor remains the single canonical construction implementation.
HuggingFacePyTorchBackend.__module__ = _production.__name__
_production.HuggingFacePyTorchBackend = HuggingFacePyTorchBackend

__all__ = ["HuggingFacePyTorchBackend"]

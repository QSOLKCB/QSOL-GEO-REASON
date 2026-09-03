"""Shared constants and low-level validation helpers for GEO-CAP-001."""
from __future__ import annotations
import hashlib
import math
import re
from typing import Any, Mapping, Protocol, Sequence
from .canonical import sha256_json

CAPTURE_PROTOCOL_ID = "GEO-CAP-001"
CAPTURE_SCHEMA_VERSION = "1.0.0"
_CAPTURE_PHASE = "replayed_prefix"
_PRODUCTION_BACKEND = "huggingface-pytorch"
_ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
_ALLOWED_CONTEXT_MODES = {"cumulative", "isolated"}
_ALLOWED_POOLING_MODES = {"last_token", "step_mean", "context_mean", "bounded_context_mean"}
_ALLOWED_DETERMINISM = {"required", "best_effort"}
_ALLOWED_EVIDENCE = {"SIMULATION", "OBSERVATION"}
_LOADING_INFO_KEYS = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
_HF_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BLOCK_CONTAINER_PATHS = ("layers", "h", "decoder.layers", "transformer.h", "gpt_neox.layers")
_LAYER_INDEX_SEMANTICS = (
    "indices address the canonical decoder hidden-state sequence: index 0 is "
    "the input to decoder block 0; indices 1..N-1 are inputs to subsequent "
    "decoder blocks; index N is the final base-model hidden state"
)
_STEP_SPAN_SEMANTICS = (
    "changed_token_span begins at the longest common token-ID prefix "
    "between the baseline context and the current rendered context"
)

_MANIFEST_KEYS = {
    "schema_version", "protocol_id", "run_id", "repository_commit", "request_sha256",
    "model", "backend_request", "backend_observed", "capture", "determinism",
    "generation_parameters", "run_manifest_id", "artifacts", "manifest_sha256",
}
_TRAJECTORY_KEYS = {
    "schema_version", "protocol_id", "evidence_class", "replication_status", "run_id",
    "run_manifest_id", "repository_commit", "representation_definition", "steps",
    "trajectory_sha256",
}
_REPRESENTATION_KEYS = {
    "context_mode", "phase", "layers", "layer_index_semantics", "pooling",
    "step_span_semantics", "prefix_input_ids", "prefix_input_ids_sha256",
}
_ARTIFACT_KEYS = {"capture_request_sha256", "captured_trajectory_sha256"}
_SIMULATION_BACKEND_KEYS = {
    "name", "observed_model_commit", "observed_tokenizer_commit", "device", "dtype",
    "quantization", "local_files_only", "trust_remote_code", "use_cache", "capture_phase",
    "kv_cache_reuse", "determinism_mode",
}
_PRODUCTION_BACKEND_KEYS = {
    "name", "python_version", "platform", "torch_version", "transformers_version",
    "tokenizers_version", "huggingface_hub_version", "model_class", "tokenizer_class",
    "observed_model_commit", "observed_tokenizer_commit", "checkpoint_loading_clean",
    "quantization_config_present", "model_reports_quantized", "attention_implementation",
    "device", "cpu_machine", "cpu_processor", "cpu_instruction_flags", "torch_num_threads",
    "torch_num_interop_threads", "omp_num_threads", "mkl_num_threads", "cuda_device_name",
    "cuda_device_capability", "cuda_build_version", "cudnn_version", "nvidia_driver_version",
    "float32_matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
    "nvidia_tf32_override", "torch_allow_tf32_cublas_override", "cublas_workspace_config",
    "mps_device_active", "mps_built", "mps_available", "mps_mac_model", "mps_cpu_brand",
    "mps_macos_version", "dtype", "observed_hidden_state_dtypes", "pool_accumulation_dtype",
    "pool_accumulation_device", "hidden_state_capture_strategy", "hidden_state_block_path",
    "hidden_state_count", "snapshot_authentication", "model_snapshot_file_count",
    "model_snapshot_file_sha256", "model_snapshot_receipt_sha256",
    "tokenizer_snapshot_file_count", "tokenizer_snapshot_file_sha256",
    "tokenizer_snapshot_receipt_sha256", "quantization", "offloading", "local_files_only",
    "trust_remote_code", "use_cache", "capture_phase", "kv_cache_reuse",
    "deterministic_algorithms_enabled", "determinism_mode",
}


class CaptureContractError(ValueError):
    """Raised when a request, backend, or artifact violates GEO-CAP-001."""


class CaptureBackendUnavailable(RuntimeError):
    """Raised when optional local capture dependencies are unavailable."""


class CaptureBackend(Protocol):
    def tokenize(self, text: str) -> list[int]: ...

    def hidden_states(
        self,
        input_ids: Sequence[int],
        layer_indices: Sequence[int],
        *,
        pool_span: tuple[int, int],
    ) -> Mapping[int, Mapping[str, Any]]: ...

    def metadata(self) -> Mapping[str, Any]: ...


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaptureContractError(f"{where} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str] | frozenset[str] = frozenset(), where: str
) -> None:
    missing = required - set(value)
    if missing:
        raise CaptureContractError(f"{where} missing required fields: {sorted(missing)}")
    unknown = set(value) - required - set(optional)
    if unknown:
        raise CaptureContractError(f"{where} contains unknown fields: {sorted(unknown)}")


def _require_nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureContractError(f"{where} must be a non-empty string")
    return value


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise CaptureContractError(f"{where} must be boolean")
    return value


def _require_git_sha(value: Any, where: str) -> str:
    text = _require_nonempty_string(value, where)
    if len(text) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise CaptureContractError(f"{where} must be a 40-hex Git commit SHA")
    return text.lower()


def _require_nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CaptureContractError(f"{where} must be a non-negative integer")
    return value


def _require_hf_repo_id(value: Any, where: str) -> str:
    text = _require_nonempty_string(value, where)
    if len(text) > 96 or "\\" in text or text.startswith(("~", "/", ".")):
        raise CaptureContractError(f"{where} must be a Hugging Face Hub repository identifier, not a local path")
    parts = text.split("/")
    if len(parts) != 2:
        raise CaptureContractError(f"{where} must have canonical 'namespace/repository' form")
    for part in parts:
        if not _HF_REPO_COMPONENT.fullmatch(part) or part in {".", ".."} or part.endswith((".", "-")):
            raise CaptureContractError(f"{where} must have canonical 'namespace/repository' form")
    return text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    i = 0
    while i < limit and left[i] == right[i]:
        i += 1
    return i


def _compose_text(prefix: str, segments: Sequence[str], joiner: str) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.extend(segments)
    return joiner.join(parts)


def _validate_token_ids(input_ids: Sequence[int], where: str, *, allow_empty: bool = False) -> list[int]:
    if not input_ids and not allow_empty:
        raise CaptureContractError(f"{where} tokenized to zero tokens")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in input_ids):
        raise CaptureContractError(f"{where} tokenizer returned an invalid token ID")
    return [int(v) for v in input_ids]


def _pool_span(*, token_count: int, mode: str, changed_span: tuple[int, int], window_tokens: int | None) -> tuple[int, int]:
    a, b = changed_span
    if token_count < 1:
        raise CaptureContractError("cannot pool an empty token sequence")
    if a < 0 or b > token_count or a >= b:
        raise CaptureContractError(f"invalid changed token span [{a}, {b}) for {token_count} tokens")
    if mode == "last_token":
        return token_count - 1, token_count
    if mode == "step_mean":
        return a, b
    if mode == "context_mean":
        return 0, token_count
    if mode == "bounded_context_mean":
        if window_tokens is None or window_tokens < 1:
            raise CaptureContractError("bounded_context_mean requires window_tokens >= 1")
        return max(0, token_count - window_tokens), token_count
    raise CaptureContractError(f"unsupported pooling mode {mode!r}")


def _validate_backend_layer(value: Any, *, layer_index: int, expected_dimension: int | None, where: str) -> tuple[list[float], int, str]:
    if not isinstance(value, Mapping):
        raise CaptureContractError(f"{where} must be a pooled layer record")
    _require_exact_keys(value, required={"vector", "vector_dimension", "observed_dtype"}, where=where)
    dimension = value["vector_dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise CaptureContractError(f"{where}.vector_dimension must be >= 1")
    if expected_dimension is not None and dimension != expected_dimension:
        raise CaptureContractError(f"layer {layer_index} vector dimension changed from {expected_dimension} to {dimension}")
    vector = value["vector"]
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != dimension:
        raise CaptureContractError(f"{where}.vector length must equal vector_dimension {dimension}")
    normalized: list[float] = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CaptureContractError(f"{where}.vector contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise CaptureContractError(f"{where}.vector contains a non-finite value")
        normalized.append(number)
    return normalized, dimension, _require_nonempty_string(value["observed_dtype"], f"{where}.observed_dtype")

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
_SIMULATION_BACKEND = "software-simulation"
_ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
_ALLOWED_OBSERVED_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_ALLOWED_CONTEXT_MODES = {"cumulative", "isolated"}
_ALLOWED_POOLING_MODES = {"last_token", "step_mean", "context_mean", "bounded_context_mean"}
_ALLOWED_DETERMINISM = {"required", "best_effort"}
_ALLOWED_EVIDENCE = {"SIMULATION", "OBSERVATION"}
_ALLOWED_ATTENTION_IMPLEMENTATIONS = {"eager", "sdpa"}
_TORCH_LONG_MAX = 2**63 - 1
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
    "torch_package_file_count", "torch_package_receipt_sha256",
    "transformers_package_file_count", "transformers_package_receipt_sha256",
    "tokenizers_native_backend_active", "tokenizers_package_file_count",
    "tokenizers_package_receipt_sha256",
    "safetensors_deserializer_active", "safetensors_package_file_count",
    "safetensors_package_receipt_sha256",
    "torch_build_config", "torch_build_config_sha256",
    "tokenizers_version", "huggingface_hub_version", "model_class", "tokenizer_class",
    "observed_model_commit", "observed_tokenizer_commit", "checkpoint_loading_clean",
    "quantization_config_present", "model_reports_quantized", "attention_implementation",
    "device", "cpu_machine", "cpu_processor", "cpu_instruction_flags", "torch_num_threads",
    "cpu_aten_capability", "aten_cpu_capability_env", "aten_cpu_capability_env_known",
    "cpu_math_dispatch_env_known", "onednn_max_cpu_isa", "dnnl_max_cpu_isa", "mkl_cbwr",
    "torch_num_interop_threads", "omp_num_threads", "mkl_num_threads",
    "cpu_mkldnn_enabled", "cpu_mkldnn_matmul_fp32_precision", "cpu_flush_denormal", "cuda_device_name",
    "cuda_device_capability", "cuda_resolved_device_index", "cuda_device_uuid",
    "cuda_visible_devices", "cuda_build_version", "cudnn_version", "nvidia_driver_version",
    "float32_matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
    "cudnn_benchmark", "cudnn_deterministic",
    "cuda_matmul_allow_fp16_reduced_precision_reduction",
    "cuda_matmul_allow_bf16_reduced_precision_reduction",
    "cuda_matmul_allow_fp16_accumulation",
    "sdpa_flash_enabled", "sdpa_mem_efficient_enabled", "sdpa_math_enabled", "sdpa_cudnn_enabled",
    "nvidia_tf32_override", "torch_allow_tf32_cublas_override", "cublas_workspace_config",
    "mps_device_active", "mps_built", "mps_available", "mps_mac_model", "mps_cpu_brand",
    "mps_macos_version", "mps_fallback_env", "mps_fast_math_env", "autocast_disabled",
    "dtype", "observed_hidden_state_dtypes", "pool_accumulation_dtype",
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


def _require_observed_dtype(value: Any, where: str) -> str:
    text = _require_nonempty_string(value, where)
    if text not in _ALLOWED_OBSERVED_DTYPES:
        raise CaptureContractError(f"{where} must be a supported floating dtype: {sorted(_ALLOWED_OBSERVED_DTYPES)}")
    return text


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
        if (
            not _HF_REPO_COMPONENT.fullmatch(part)
            or part in {".", ".."}
            or part.endswith((".", "-"))
            or "--" in part
            or ".." in part
        ):
            raise CaptureContractError(f"{where} must have canonical 'namespace/repository' form")
    return text


def _require_nonempty_sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise CaptureContractError(f"{where} must be a non-empty array")
    return value


def _require_finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureContractError(f"{where} must be numeric")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise CaptureContractError(f"{where} must be a representable finite number") from exc
    if not math.isfinite(numeric):
        raise CaptureContractError(f"{where} must be finite")
    return numeric


def _pool_span(mode: str, token_count: int, changed_start: int, window_tokens: int | None) -> tuple[int, int]:
    if token_count <= 0:
        raise CaptureContractError("cannot pool an empty token sequence")
    if mode == "last_token":
        return token_count - 1, token_count
    if mode == "context_mean":
        return 0, token_count
    if mode == "step_mean":
        return changed_start, token_count
    if mode == "bounded_context_mean":
        assert window_tokens is not None
        return max(0, token_count - window_tokens), token_count
    raise CaptureContractError(f"unsupported pooling mode: {mode}")


def _validate_token_ids(token_ids: Any, where: str) -> list[int]:
    if not isinstance(token_ids, (list, tuple)):
        raise CaptureContractError(f"{where} must be an integer sequence")
    result: list[int] = []
    for index, value in enumerate(token_ids):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CaptureContractError(f"{where}[{index}] must be an integer")
        if value < 0 or value > _TORCH_LONG_MAX:
            raise CaptureContractError(
                f"{where}[{index}] must fit canonical torch.long range [0, {_TORCH_LONG_MAX}]"
            )
        result.append(value)
    return result


def _finite_vector(values: Sequence[Any], where: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise CaptureContractError(f"{where} must be a non-empty numeric vector")
    return [_require_finite_number(v, f"{where}[{i}]") for i, v in enumerate(values)]


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _validate_step_span(start: int, stop: int, token_count: int, where: str) -> None:
    for name, value in (("start", start), ("stop", stop), ("token_count", token_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CaptureContractError(f"{where} {name} must be an integer")
    if not 0 <= start < stop <= token_count:
        raise CaptureContractError(f"{where} must satisfy 0 <= start < stop <= token_count")


def _canonical_step_text(prefix: str, joiner: str, steps: Sequence[Mapping[str, Any]], index: int, mode: str) -> str:
    if mode == "cumulative":
        return prefix + "".join(joiner + steps[i]["text"] for i in range(index + 1))
    if mode == "isolated":
        return prefix + joiner + steps[index]["text"]
    raise CaptureContractError(f"unsupported context mode: {mode}")

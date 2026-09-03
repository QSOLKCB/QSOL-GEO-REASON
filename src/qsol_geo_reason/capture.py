"""Canonical local-model hidden-state capture for QSOL-GEO-REASON Phase 2A.

The protocol core stops at representation capture. It does not interpret a
vector as a belief, proof state, truth state, gauge field, thermodynamic
variable, or mechanism of reasoning.

Generic adapters default to ``SIMULATION`` evidence. ``OBSERVATION`` is
accepted only for the concrete local-only Hugging Face / PyTorch adapter.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .canonical import canonical_json_bytes, sha256_json


CAPTURE_PROTOCOL_ID = "GEO-CAP-001"
CAPTURE_SCHEMA_VERSION = "1.0.0"
_CAPTURE_PHASE = "replayed_prefix"
_PRODUCTION_BACKEND = "huggingface-pytorch"
_ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
_ALLOWED_CONTEXT_MODES = {"cumulative", "isolated"}
_ALLOWED_POOLING_MODES = {
    "last_token",
    "step_mean",
    "context_mean",
    "bounded_context_mean",
}
_ALLOWED_DETERMINISM = {"required", "best_effort"}
_ALLOWED_EVIDENCE = {"SIMULATION", "OBSERVATION"}
_LOADING_INFO_KEYS = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
_HF_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LAYER_INDEX_SEMANTICS = (
    "indices address the canonical decoder hidden-state sequence: index 0 is "
    "the input to decoder block 0; indices 1..N-1 are inputs to subsequent "
    "decoder blocks; index N is the final base-model hidden state"
)
_STEP_SPAN_SEMANTICS = (
    "changed_token_span begins at the longest common token-ID prefix "
    "between the baseline context and the current rendered context"
)
_BLOCK_CONTAINER_PATHS = (
    "layers",
    "h",
    "decoder.layers",
    "transformer.h",
    "gpt_neox.layers",
)
_MANIFEST_KEYS = {
    "schema_version",
    "protocol_id",
    "run_id",
    "repository_commit",
    "request_sha256",
    "model",
    "backend_request",
    "backend_observed",
    "capture",
    "determinism",
    "generation_parameters",
    "run_manifest_id",
    "artifacts",
    "manifest_sha256",
}
_TRAJECTORY_KEYS = {
    "schema_version",
    "protocol_id",
    "evidence_class",
    "replication_status",
    "run_id",
    "run_manifest_id",
    "repository_commit",
    "representation_definition",
    "steps",
    "trajectory_sha256",
}
_REPRESENTATION_KEYS = {
    "context_mode",
    "phase",
    "layers",
    "layer_index_semantics",
    "pooling",
    "step_span_semantics",
}
_ARTIFACT_KEYS = {"capture_request_sha256", "captured_trajectory_sha256"}
_SIMULATION_BACKEND_KEYS = {
    "name",
    "observed_model_commit",
    "observed_tokenizer_commit",
    "device",
    "dtype",
    "quantization",
    "local_files_only",
    "trust_remote_code",
    "use_cache",
    "capture_phase",
    "kv_cache_reuse",
    "determinism_mode",
}
_PRODUCTION_BACKEND_KEYS = {
    "name",
    "python_version",
    "platform",
    "torch_version",
    "transformers_version",
    "tokenizers_version",
    "huggingface_hub_version",
    "model_class",
    "tokenizer_class",
    "observed_model_commit",
    "observed_tokenizer_commit",
    "checkpoint_loading_clean",
    "quantization_config_present",
    "model_reports_quantized",
    "attention_implementation",
    "device",
    "cpu_machine",
    "cpu_processor",
    "cpu_instruction_flags",
    "torch_num_threads",
    "torch_num_interop_threads",
    "omp_num_threads",
    "mkl_num_threads",
    "cuda_device_name",
    "cuda_device_capability",
    "cuda_build_version",
    "cudnn_version",
    "nvidia_driver_version",
    "dtype",
    "observed_hidden_state_dtypes",
    "pool_accumulation_dtype",
    "pool_accumulation_device",
    "hidden_state_capture_strategy",
    "hidden_state_block_path",
    "hidden_state_count",
    "quantization",
    "offloading",
    "local_files_only",
    "trust_remote_code",
    "use_cache",
    "capture_phase",
    "kv_cache_reuse",
    "deterministic_algorithms_enabled",
    "determinism_mode",
}


class CaptureContractError(ValueError):
    """Raised when a request, backend, or artifact violates GEO-CAP-001."""


class CaptureBackendUnavailable(RuntimeError):
    """Raised when optional local capture dependencies are unavailable."""


class CaptureBackend(Protocol):
    """Backend surface consumed by the backend-independent protocol core."""

    def tokenize(self, text: str) -> list[int]:
        ...

    def hidden_states(
        self,
        input_ids: Sequence[int],
        layer_indices: Sequence[int],
        *,
        pool_span: tuple[int, int],
    ) -> Mapping[int, Mapping[str, Any]]:
        """Return one pooled vector record for each requested layer."""

    def metadata(self) -> Mapping[str, Any]:
        ...


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaptureContractError(f"{where} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    where: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise CaptureContractError(f"{where} missing required fields: {sorted(missing)}")
    unknown = set(value) - required - optional
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
    """Require a canonical Hub ``namespace/repository`` identifier, never a path."""

    text = _require_nonempty_string(value, where)
    if len(text) > 96 or "\\" in text or text.startswith(("~", "/", ".")):
        raise CaptureContractError(
            f"{where} must be a Hugging Face Hub repository identifier, not a local path"
        )
    parts = text.split("/")
    if len(parts) != 2:
        raise CaptureContractError(
            f"{where} must have canonical 'namespace/repository' form"
        )
    for part in parts:
        if (
            not _HF_REPO_COMPONENT.fullmatch(part)
            or part in {".", ".."}
            or part.endswith((".", "-"))
        ):
            raise CaptureContractError(
                f"{where} must have canonical 'namespace/repository' form"
            )
    return text


def validate_capture_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied, strictly validated GEO-CAP-001 request."""

    if not isinstance(request, dict):
        raise CaptureContractError("capture request must be an object")
    root = copy.deepcopy(request)
    _require_exact_keys(
        root,
        required={
            "schema_version",
            "protocol_id",
            "run_id",
            "model",
            "backend",
            "capture",
            "determinism",
            "generation_parameters",
            "steps",
        },
        optional={"notes"},
        where="capture request",
    )
    if root["schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise CaptureContractError(f"schema_version must be {CAPTURE_SCHEMA_VERSION!r}")
    if root["protocol_id"] != CAPTURE_PROTOCOL_ID:
        raise CaptureContractError(f"protocol_id must be {CAPTURE_PROTOCOL_ID!r}")
    _require_nonempty_string(root["run_id"], "run_id")

    model = _require_object(root["model"], "model")
    _require_exact_keys(
        model,
        required={
            "identifier",
            "revision",
            "revision_kind",
            "tokenizer_identifier",
            "tokenizer_revision",
            "tokenizer_revision_kind",
        },
        where="model",
    )
    model["identifier"] = _require_hf_repo_id(model["identifier"], "model.identifier")
    model["tokenizer_identifier"] = _require_hf_repo_id(
        model["tokenizer_identifier"], "model.tokenizer_identifier"
    )
    if model["revision_kind"] != "hf_commit":
        raise CaptureContractError("model.revision_kind must be 'hf_commit'")
    if model["tokenizer_revision_kind"] != "hf_commit":
        raise CaptureContractError("model.tokenizer_revision_kind must be 'hf_commit'")
    model["revision"] = _require_git_sha(model["revision"], "model.revision")
    model["tokenizer_revision"] = _require_git_sha(
        model["tokenizer_revision"], "model.tokenizer_revision"
    )

    backend = _require_object(root["backend"], "backend")
    _require_exact_keys(
        backend,
        required={
            "name",
            "local_files_only",
            "trust_remote_code",
            "device",
            "dtype",
            "quantization",
        },
        where="backend",
    )
    if backend["name"] != _PRODUCTION_BACKEND:
        raise CaptureContractError(
            f"backend.name must be {_PRODUCTION_BACKEND!r} for GEO-CAP-001"
        )
    if _require_bool(backend["local_files_only"], "backend.local_files_only") is not True:
        raise CaptureContractError("backend.local_files_only must be true")
    if _require_bool(backend["trust_remote_code"], "backend.trust_remote_code") is not False:
        raise CaptureContractError("backend.trust_remote_code must be false")
    _require_nonempty_string(backend["device"], "backend.device")
    if backend["dtype"] not in _ALLOWED_DTYPES:
        raise CaptureContractError(f"backend.dtype must be one of {sorted(_ALLOWED_DTYPES)}")
    if backend["quantization"] != "none":
        raise CaptureContractError(
            "GEO-CAP-001 canonical capture requires backend.quantization='none'"
        )

    capture = _require_object(root["capture"], "capture")
    _require_exact_keys(
        capture,
        required={
            "context_mode",
            "phase",
            "layers",
            "pooling",
            "prefix_text",
            "step_joiner",
        },
        where="capture",
    )
    if capture["context_mode"] not in _ALLOWED_CONTEXT_MODES:
        raise CaptureContractError(
            f"capture.context_mode must be one of {sorted(_ALLOWED_CONTEXT_MODES)}"
        )
    if capture["phase"] != _CAPTURE_PHASE:
        raise CaptureContractError(
            f"GEO-CAP-001 currently supports capture.phase={_CAPTURE_PHASE!r} only"
        )
    if not isinstance(capture["prefix_text"], str):
        raise CaptureContractError("capture.prefix_text must be a string")
    if not isinstance(capture["step_joiner"], str):
        raise CaptureContractError("capture.step_joiner must be a string")

    layers = capture["layers"]
    if not isinstance(layers, list) or not layers:
        raise CaptureContractError("capture.layers must be a non-empty array")
    normalized_layers = [
        _require_nonnegative_int(layer, f"capture.layers[{idx}]")
        for idx, layer in enumerate(layers)
    ]
    if len(set(normalized_layers)) != len(normalized_layers):
        raise CaptureContractError("capture.layers must contain unique indices")
    capture["layers"] = normalized_layers

    pooling = _require_object(capture["pooling"], "capture.pooling")
    _require_exact_keys(
        pooling,
        required={"mode"},
        optional={"window_tokens"},
        where="capture.pooling",
    )
    mode = pooling["mode"]
    if mode not in _ALLOWED_POOLING_MODES:
        raise CaptureContractError(
            f"capture.pooling.mode must be one of {sorted(_ALLOWED_POOLING_MODES)}"
        )
    if mode == "bounded_context_mean":
        if "window_tokens" not in pooling:
            raise CaptureContractError(
                "capture.pooling.window_tokens is required for bounded_context_mean"
            )
        window = _require_nonnegative_int(
            pooling["window_tokens"], "capture.pooling.window_tokens"
        )
        if window == 0:
            raise CaptureContractError("capture.pooling.window_tokens must be >= 1")
        pooling["window_tokens"] = window
    elif "window_tokens" in pooling:
        raise CaptureContractError(
            "capture.pooling.window_tokens is only valid for bounded_context_mean"
        )

    determinism = _require_object(root["determinism"], "determinism")
    _require_exact_keys(determinism, required={"mode", "seed"}, where="determinism")
    if determinism["mode"] not in _ALLOWED_DETERMINISM:
        raise CaptureContractError(
            f"determinism.mode must be one of {sorted(_ALLOWED_DETERMINISM)}"
        )
    determinism["seed"] = _require_nonnegative_int(
        determinism["seed"], "determinism.seed"
    )

    generation = _require_object(root["generation_parameters"], "generation_parameters")
    if "generation_used" not in generation:
        raise CaptureContractError(
            "generation_parameters.generation_used is required for replay capture"
        )
    generation_used = _require_bool(
        generation["generation_used"], "generation_parameters.generation_used"
    )
    if generation_used:
        raise CaptureContractError(
            "GEO-CAP-001 replay capture requires generation_parameters.generation_used=false"
        )
    incompatible = {
        key: value
        for key, value in generation.items()
        if key != "generation_used" and value is not None
    }
    if incompatible:
        raise CaptureContractError(
            "generation settings are incompatible with generation_used=false: "
            + ", ".join(sorted(incompatible))
        )

    steps = root["steps"]
    if not isinstance(steps, list) or not steps:
        raise CaptureContractError("steps must be a non-empty array")
    seen_ids: set[str] = set()
    for idx, step_value in enumerate(steps):
        step = _require_object(step_value, f"steps[{idx}]")
        _require_exact_keys(step, required={"step_id", "text"}, where=f"steps[{idx}]")
        step_id = _require_nonempty_string(step["step_id"], f"steps[{idx}].step_id")
        if step_id in seen_ids:
            raise CaptureContractError(f"duplicate step_id {step_id!r}")
        seen_ids.add(step_id)
        _require_nonempty_string(step["text"], f"steps[{idx}].text")

    if "notes" in root and not isinstance(root["notes"], str):
        raise CaptureContractError("notes must be a string")
    return root


def _validate_loading_info(loading_info: Any) -> None:
    """Reject checkpoint loads that synthesize, discard, or mismatch parameters."""

    if not isinstance(loading_info, dict):
        raise CaptureContractError("Transformers did not return checkpoint loading information")
    problems: dict[str, list[Any]] = {}
    for key in _LOADING_INFO_KEYS:
        value = loading_info.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, (list, tuple)):
            raise CaptureContractError(f"checkpoint loading info {key!r} has invalid shape")
        if value:
            problems[key] = list(value)
    if problems:
        summary = "; ".join(f"{key}={value!r}" for key, value in problems.items())
        raise CaptureContractError(
            "checkpoint load was not exact; canonical observation rejected: " + summary
        )


def _quantization_reasons(model: Any) -> list[str]:
    """Return reasons a loaded model is not an unquantized canonical object."""

    reasons: list[str] = []
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "quantization_config", None) is not None:
        reasons.append("config.quantization_config")
    if bool(getattr(model, "is_quantized", False)):
        reasons.append("model.is_quantized")
    if getattr(model, "hf_quantizer", None) is not None:
        reasons.append("model.hf_quantizer")
    return reasons


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _compose_text(prefix: str, segments: Sequence[str], joiner: str) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.extend(segments)
    return joiner.join(parts)


def _validate_token_ids(input_ids: Sequence[int], where: str) -> list[int]:
    if not input_ids:
        raise CaptureContractError(f"{where} tokenized to zero tokens")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in input_ids
    ):
        raise CaptureContractError(f"{where} tokenizer returned an invalid token ID")
    return [int(token_id) for token_id in input_ids]


def _mean_rows(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise CaptureContractError("cannot pool an empty token span")
    dimension = len(rows[0])
    if dimension == 0:
        raise CaptureContractError("cannot pool zero-dimensional vectors")
    for row in rows:
        if len(row) != dimension:
            raise CaptureContractError("pooling rows have inconsistent dimensions")
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CaptureContractError("pooling rows contain invalid values")
    return [
        math.fsum(float(row[axis]) for row in rows) / len(rows)
        for axis in range(dimension)
    ]


def _pool_span(
    *,
    token_count: int,
    mode: str,
    changed_span: tuple[int, int],
    window_tokens: int | None,
) -> tuple[int, int]:
    changed_start, changed_end = changed_span
    if token_count < 1:
        raise CaptureContractError("cannot pool an empty token sequence")
    if changed_start < 0 or changed_end > token_count or changed_start >= changed_end:
        raise CaptureContractError(
            f"invalid changed token span [{changed_start}, {changed_end}) for {token_count} tokens"
        )
    if mode == "last_token":
        return token_count - 1, token_count
    if mode == "step_mean":
        return changed_start, changed_end
    if mode == "context_mean":
        return 0, token_count
    if mode == "bounded_context_mean":
        if window_tokens is None or window_tokens < 1:
            raise CaptureContractError("bounded_context_mean requires window_tokens >= 1")
        return max(0, token_count - window_tokens), token_count
    raise CaptureContractError(f"unsupported pooling mode {mode!r}")


def _pool_hidden_state(
    matrix: Sequence[Sequence[float]],
    *,
    mode: str,
    changed_span: tuple[int, int],
    window_tokens: int | None,
) -> tuple[list[float], tuple[int, int]]:
    """Reference software pooling used by tests and non-tensor adapters."""

    span = _pool_span(
        token_count=len(matrix),
        mode=mode,
        changed_span=changed_span,
        window_tokens=window_tokens,
    )
    vector = _mean_rows(matrix[span[0] : span[1]])
    if not vector or any(not math.isfinite(value) for value in vector):
        raise CaptureContractError("pooled representation contains non-finite values")
    return vector, span


def _validate_backend_layer(
    value: Any,
    *,
    layer_index: int,
    expected_dimension: int | None,
    where: str,
) -> tuple[list[float], int, str]:
    if not isinstance(value, Mapping):
        raise CaptureContractError(f"{where} must be a pooled layer record")
    _require_exact_keys(
        value,
        required={"vector", "vector_dimension", "observed_dtype"},
        where=where,
    )
    dimension = value["vector_dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise CaptureContractError(f"{where}.vector_dimension must be >= 1")
    if expected_dimension is not None and dimension != expected_dimension:
        raise CaptureContractError(
            f"layer {layer_index} vector dimension changed from {expected_dimension} to {dimension}"
        )
    vector = value["vector"]
    if (
        not isinstance(vector, Sequence)
        or isinstance(vector, (str, bytes))
        or len(vector) != dimension
    ):
        raise CaptureContractError(
            f"{where}.vector length must equal vector_dimension {dimension}"
        )
    normalized: list[float] = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CaptureContractError(f"{where}.vector contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise CaptureContractError(f"{where}.vector contains a non-finite value")
        normalized.append(number)
    observed_dtype = _require_nonempty_string(
        value["observed_dtype"], f"{where}.observed_dtype"
    )
    return normalized, dimension, observed_dtype


def _capture_steps(
    request: Mapping[str, Any], backend: CaptureBackend
) -> list[dict[str, Any]]:
    capture_cfg = request["capture"]
    pooling_cfg = capture_cfg["pooling"]
    pooling_mode = pooling_cfg["mode"]
    window_tokens = pooling_cfg.get("window_tokens")
    prefix = capture_cfg["prefix_text"]
    joiner = capture_cfg["step_joiner"]
    context_mode = capture_cfg["context_mode"]
    layers: list[int] = capture_cfg["layers"]

    raw_prefix_ids = backend.tokenize(prefix)
    if prefix:
        prefix_ids = _validate_token_ids(raw_prefix_ids, "prefix_text")
    else:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_prefix_ids
        ):
            raise CaptureContractError("tokenizer returned an invalid prefix token ID")
        prefix_ids = [int(value) for value in raw_prefix_ids]

    previous_ids = prefix_ids
    cumulative_segments: list[str] = []
    output: list[dict[str, Any]] = []
    vector_dimensions: dict[int, int] = {}

    for step_index, step in enumerate(request["steps"]):
        if context_mode == "cumulative":
            cumulative_segments.append(step["text"])
            rendered = _compose_text(prefix, cumulative_segments, joiner)
            baseline_ids = previous_ids
        else:
            rendered = _compose_text(prefix, [step["text"]], joiner)
            baseline_ids = prefix_ids

        input_ids = _validate_token_ids(
            backend.tokenize(rendered), f"step {step['step_id']!r}"
        )
        changed_start = _common_prefix_length(baseline_ids, input_ids)
        changed_span = (changed_start, len(input_ids))
        if changed_start == len(input_ids):
            raise CaptureContractError(
                f"step {step['step_id']!r} adds no changed token span under the frozen tokenizer"
            )
        pool_span = _pool_span(
            token_count=len(input_ids),
            mode=pooling_mode,
            changed_span=changed_span,
            window_tokens=window_tokens,
        )

        selected = backend.hidden_states(input_ids, layers, pool_span=pool_span)
        if not isinstance(selected, Mapping):
            raise CaptureContractError(
                "backend.hidden_states must return a layer-indexed mapping"
            )
        if set(selected) != set(layers):
            raise CaptureContractError(
                "backend hidden-state selection does not exactly match requested layers"
            )

        layer_records: list[dict[str, Any]] = []
        for layer_index in layers:
            vector, dimension, observed_dtype = _validate_backend_layer(
                selected[layer_index],
                layer_index=layer_index,
                expected_dimension=vector_dimensions.get(layer_index),
                where=f"step {step['step_id']!r} layer {layer_index}",
            )
            vector_dimensions.setdefault(layer_index, dimension)
            layer_records.append(
                {
                    "layer_index": layer_index,
                    "vector_dimension": dimension,
                    "observed_dtype": observed_dtype,
                    "pool_span": [pool_span[0], pool_span[1]],
                    "vector": vector,
                    "vector_sha256": sha256_json(vector),
                }
            )

        output.append(
            {
                "step_index": step_index,
                "step_id": step["step_id"],
                "rendered_text_sha256": _sha256_text(rendered),
                "input_ids": input_ids,
                "input_ids_sha256": sha256_json(input_ids),
                "token_count": len(input_ids),
                "changed_token_span": [changed_span[0], changed_span[1]],
                "phase": _CAPTURE_PHASE,
                "layers": layer_records,
            }
        )
        if context_mode == "cumulative":
            previous_ids = input_ids
    return output


def _validate_backend_identity(
    observed_backend: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    if observed_backend.get("name") != _PRODUCTION_BACKEND:
        raise CaptureContractError(f"backend metadata name must be {_PRODUCTION_BACKEND!r}")
    if observed_backend.get("observed_model_commit") != request["model"]["revision"]:
        raise CaptureContractError(
            "observed model commit does not match the frozen request revision"
        )
    if (
        observed_backend.get("observed_tokenizer_commit")
        != request["model"]["tokenizer_revision"]
    ):
        raise CaptureContractError(
            "observed tokenizer commit does not match the frozen request revision"
        )


def _validate_backend_metadata_shape(
    observed_backend: Mapping[str, Any], evidence_class: str
) -> None:
    required = (
        _PRODUCTION_BACKEND_KEYS if evidence_class == "OBSERVATION" else _SIMULATION_BACKEND_KEYS
    )
    _require_exact_keys(
        observed_backend,
        required=set(required),
        where=f"{evidence_class.lower()} backend_observed",
    )


def _cpu_hardware_metadata(torch_module: Any) -> dict[str, Any]:
    cpu_model: str | None = platform.processor() or None
    cpu_flags: str | None = None
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        first = text.split("\n\n", 1)[0]
        fields: dict[str, str] = {}
        for line in first.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        cpu_model = (
            fields.get("model name")
            or fields.get("Processor")
            or fields.get("Hardware")
            or cpu_model
        )
        cpu_flags = fields.get("flags") or fields.get("Features")
    except OSError:
        pass

    def _thread_value(name: str) -> int | None:
        function = getattr(torch_module, name, None)
        if function is None:
            return None
        try:
            value = function()
        except Exception:
            return None
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    return {
        "cpu_machine": platform.machine() or None,
        "cpu_processor": cpu_model,
        "cpu_instruction_flags": cpu_flags,
        "torch_num_threads": _thread_value("get_num_threads"),
        "torch_num_interop_threads": _thread_value("get_num_interop_threads"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _get_dotted_attr(root: Any, path: str) -> Any:
    value = root
    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def _config_hidden_layer_count(config: Any) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise CaptureContractError(
        "canonical selective hidden-state capture requires an explicit positive layer count in model config"
    )


def _resolve_hidden_state_layout(model: Any) -> tuple[Any, str, Any]:
    """Resolve one unambiguous decoder-block sequence for selective capture."""

    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise CaptureContractError("model does not expose a canonical base_model")
    expected = _config_hidden_layer_count(getattr(model, "config", None))
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for path in _BLOCK_CONTAINER_PATHS:
        blocks = _get_dotted_attr(base_model, path)
        if blocks is None:
            continue
        try:
            length = len(blocks)
            blocks[0]
        except (TypeError, IndexError, KeyError):
            continue
        if length != expected or id(blocks) in seen:
            continue
        seen.add(id(blocks))
        candidates.append((path, blocks))
    if len(candidates) != 1:
        found = [path for path, _ in candidates]
        raise CaptureContractError(
            "canonical selective hidden-state capture requires exactly one recognized "
            f"decoder block sequence of length {expected}; candidates={found!r}"
        )
    path, blocks = candidates[0]
    return base_model, path, blocks


def _extract_hidden_tensor(value: Any) -> Any:
    if value is None:
        return None
    hidden = getattr(value, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(value, Mapping) and value.get("last_hidden_state") is not None:
        return value["last_hidden_state"]
    if isinstance(value, (tuple, list)) and value:
        return _extract_hidden_tensor(value[0])
    if hasattr(value, "ndim") and hasattr(value, "shape") and hasattr(value, "detach"):
        return value
    return None


class HuggingFacePyTorchBackend:
    """Direct local-only Hugging Face / PyTorch replay backend."""

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise CaptureBackendUnavailable(
                "canonical capture requires optional capture dependencies; "
                "install qsol-geo-reason[capture]"
            ) from exc

        self._torch = torch
        self._transformers = transformers
        self._observed_hidden_state_dtypes: dict[int, set[str]] = {}
        backend = validated["backend"]
        model_cfg = validated["model"]
        determinism = validated["determinism"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._device = backend["device"]
        self._dtype_name = backend["dtype"]
        self._determinism_mode = determinism["mode"]

        torch.manual_seed(determinism["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(determinism["seed"])
        if determinism["mode"] == "required":
            torch.use_deterministic_algorithms(True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["tokenizer_identifier"],
            revision=model_cfg["tokenizer_revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            model_cfg["identifier"],
            revision=model_cfg["revision"],
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype_map[backend["dtype"]],
            output_loading_info=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise CaptureContractError(
                "Transformers did not return (model, loading_info) for canonical load"
            )
        self._model, loading_info = loaded
        _validate_loading_info(loading_info)
        quantization = _quantization_reasons(self._model)
        if quantization:
            raise CaptureContractError(
                "canonical capture forbids checkpoint/config quantization; detected: "
                + ", ".join(quantization)
            )
        self._base_model, self._block_path, self._blocks = _resolve_hidden_state_layout(
            self._model
        )
        self._hidden_state_count = len(self._blocks) + 1
        self._checkpoint_loading_clean = True
        self._model.to(self._device)
        self._model.eval()

    def tokenize(self, text: str) -> list[int]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        return [int(token_id) for token_id in encoded["input_ids"]]

    def _pool_tensor_record(
        self,
        tensor: Any,
        *,
        layer_index: int,
        token_count: int,
        pool_span: tuple[int, int],
    ) -> Mapping[str, Any]:
        """Pool one selected state on CPU with bounded float64 temporary memory."""

        torch = self._torch
        if tensor is None:
            raise CaptureContractError(f"selective hook for layer {layer_index} produced no tensor")
        if tensor.ndim == 3:
            if int(tensor.shape[0]) != 1:
                raise CaptureContractError(
                    f"layer {layer_index} hidden-state batch dimension must be 1"
                )
            matrix = tensor[0]
        elif tensor.ndim == 2:
            matrix = tensor
        else:
            raise CaptureContractError(
                f"layer {layer_index} hidden-state rank {tensor.ndim} is unsupported"
            )
        if int(matrix.shape[0]) != token_count or int(matrix.shape[1]) < 1:
            raise CaptureContractError(
                f"layer {layer_index} hidden-state shape is incompatible with token capture"
            )
        start, end = pool_span
        if start < 0 or end > token_count or start >= end:
            raise CaptureContractError("backend received an invalid pool span")

        observed_dtype = str(matrix.dtype).removeprefix("torch.")
        self._observed_hidden_state_dtypes.setdefault(layer_index, set()).add(
            observed_dtype
        )
        dimension = int(matrix.shape[1])

        if end - start == 1:
            # Move to CPU before the float64 conversion. This is required for
            # MPS, which does not support float64 tensors on-device.
            pooled = matrix[start].detach().to(device="cpu").to(dtype=torch.float64)
        else:
            accumulator = torch.zeros(dimension, dtype=torch.float64, device="cpu")
            chunk_tokens = 256
            for chunk_start in range(start, end, chunk_tokens):
                chunk_end = min(end, chunk_start + chunk_tokens)
                chunk = (
                    matrix[chunk_start:chunk_end]
                    .detach()
                    .to(device="cpu")
                    .to(dtype=torch.float64)
                )
                accumulator.add_(chunk.sum(dim=0, dtype=torch.float64))
            pooled = accumulator / (end - start)

        if not bool(torch.isfinite(pooled).all().item()):
            raise CaptureContractError(
                f"layer {layer_index} pooled representation contains non-finite values"
            )
        return {
            "vector": pooled.tolist(),
            "vector_dimension": dimension,
            "observed_dtype": observed_dtype,
        }

    def hidden_states(
        self,
        input_ids: Sequence[int],
        layer_indices: Sequence[int],
        *,
        pool_span: tuple[int, int],
    ) -> Mapping[int, Mapping[str, Any]]:
        """Capture only requested states with hooks; never retain the full tuple."""

        torch = self._torch
        requested = tuple(layer_indices)
        if any(index < 0 or index >= self._hidden_state_count for index in requested):
            bad = next(
                index
                for index in requested
                if index < 0 or index >= self._hidden_state_count
            )
            raise CaptureContractError(
                f"requested layer {bad} outside backend hidden-state range "
                f"[0, {self._hidden_state_count - 1}]"
            )

        selected: dict[int, Mapping[str, Any]] = {}
        handles: list[Any] = []
        token_count = len(input_ids)

        def capture(layer_index: int, value: Any) -> None:
            if layer_index in selected:
                raise CaptureContractError(
                    f"selective hidden-state hook for layer {layer_index} fired more than once"
                )
            tensor = _extract_hidden_tensor(value)
            selected[layer_index] = self._pool_tensor_record(
                tensor,
                layer_index=layer_index,
                token_count=token_count,
                pool_span=pool_span,
            )

        def make_pre_hook(layer_index: int):
            def hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
                value = kwargs.get("hidden_states")
                if value is None and args:
                    value = args[0]
                capture(layer_index, value)

            return hook

        for layer_index in requested:
            if layer_index < len(self._blocks):
                handle = self._blocks[layer_index].register_forward_pre_hook(
                    make_pre_hook(layer_index), with_kwargs=True
                )
                handles.append(handle)

        if len(self._blocks) in requested:
            final_index = len(self._blocks)

            def final_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                capture(final_index, output)

            handles.append(self._base_model.register_forward_hook(final_hook))

        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self._device)
        mask = torch.ones_like(ids)
        try:
            with torch.inference_mode():
                self._model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        if set(selected) != set(requested):
            missing = sorted(set(requested) - set(selected))
            raise CaptureContractError(
                f"selective hidden-state hooks did not capture requested layers: {missing}"
            )
        return selected

    @staticmethod
    def _installed_version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _nvidia_driver_version() -> str | None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        versions = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
        return ",".join(versions) if versions else None

    def metadata(self) -> Mapping[str, Any]:
        torch = self._torch
        model_config = self._model.config
        observed_model_commit = getattr(model_config, "_commit_hash", None)
        observed_tokenizer_commit = (
            getattr(self._tokenizer, "_commit_hash", None)
            or self._tokenizer.init_kwargs.get("_commit_hash")
        )
        attention_implementation = getattr(model_config, "_attn_implementation", None)
        cuda_active = str(self._device).startswith("cuda") and torch.cuda.is_available()
        cuda_device = None
        cuda_capability = None
        if cuda_active:
            try:
                cuda_device = torch.cuda.get_device_name(self._device)
                capability = torch.cuda.get_device_capability(self._device)
                cuda_capability = f"{capability[0]}.{capability[1]}"
            except Exception:
                cuda_device = None
                cuda_capability = None
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:
            cudnn_version = None
        return {
            "name": _PRODUCTION_BACKEND,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": self._transformers.__version__,
            "tokenizers_version": self._installed_version("tokenizers"),
            "huggingface_hub_version": self._installed_version("huggingface-hub"),
            "model_class": type(self._model).__name__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "observed_model_commit": observed_model_commit,
            "observed_tokenizer_commit": observed_tokenizer_commit,
            "checkpoint_loading_clean": self._checkpoint_loading_clean,
            "quantization_config_present": bool(
                getattr(model_config, "quantization_config", None) is not None
            ),
            "model_reports_quantized": bool(getattr(self._model, "is_quantized", False)),
            "attention_implementation": attention_implementation,
            "device": str(self._device),
            **_cpu_hardware_metadata(torch),
            "cuda_device_name": cuda_device,
            "cuda_device_capability": cuda_capability,
            "cuda_build_version": getattr(torch.version, "cuda", None),
            "cudnn_version": cudnn_version,
            "nvidia_driver_version": self._nvidia_driver_version() if cuda_active else None,
            "dtype": self._dtype_name,
            "observed_hidden_state_dtypes": {
                str(layer): sorted(values)
                for layer, values in sorted(self._observed_hidden_state_dtypes.items())
            },
            "pool_accumulation_dtype": "float64",
            "pool_accumulation_device": "cpu",
            "hidden_state_capture_strategy": "selective_forward_hooks",
            "hidden_state_block_path": self._block_path,
            "hidden_state_count": self._hidden_state_count,
            "quantization": "none",
            "offloading": "none",
            "local_files_only": True,
            "trust_remote_code": False,
            "use_cache": False,
            "capture_phase": _CAPTURE_PHASE,
            "kv_cache_reuse": False,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "determinism_mode": self._determinism_mode,
        }


def execute_capture(
    request: Mapping[str, Any],
    *,
    implementation_revision: str,
    backend: CaptureBackend,
    evidence_class: str = "SIMULATION",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute GEO-CAP-001 and return ``(run_manifest, trajectory)``."""

    validated = validate_capture_request(request)
    implementation_revision = _require_git_sha(
        implementation_revision, "implementation_revision"
    )
    if evidence_class not in _ALLOWED_EVIDENCE:
        raise CaptureContractError(
            f"evidence_class must be one of {sorted(_ALLOWED_EVIDENCE)}"
        )
    if evidence_class == "OBSERVATION" and type(backend) is not HuggingFacePyTorchBackend:
        raise CaptureContractError(
            "OBSERVATION capture requires the concrete HuggingFacePyTorchBackend"
        )

    steps = _capture_steps(validated, backend)
    observed_backend = dict(backend.metadata())
    _validate_backend_identity(observed_backend, validated)
    _validate_backend_metadata_shape(observed_backend, evidence_class)

    request_sha256 = sha256_json(validated)
    manifest_identity = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "protocol_id": CAPTURE_PROTOCOL_ID,
        "run_id": validated["run_id"],
        "repository_commit": implementation_revision,
        "request_sha256": request_sha256,
        "model": validated["model"],
        "backend_request": validated["backend"],
        "backend_observed": observed_backend,
        "capture": validated["capture"],
        "determinism": validated["determinism"],
        "generation_parameters": validated["generation_parameters"],
    }
    run_manifest_id = sha256_json(manifest_identity)
    trajectory_payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "protocol_id": CAPTURE_PROTOCOL_ID,
        "evidence_class": evidence_class,
        "replication_status": "not_attempted",
        "run_id": validated["run_id"],
        "run_manifest_id": run_manifest_id,
        "repository_commit": implementation_revision,
        "representation_definition": {
            "context_mode": validated["capture"]["context_mode"],
            "phase": validated["capture"]["phase"],
            "layers": validated["capture"]["layers"],
            "layer_index_semantics": _LAYER_INDEX_SEMANTICS,
            "pooling": validated["capture"]["pooling"],
            "step_span_semantics": _STEP_SPAN_SEMANTICS,
        },
        "steps": steps,
    }
    trajectory_sha256 = sha256_json(trajectory_payload)
    trajectory = {**trajectory_payload, "trajectory_sha256": trajectory_sha256}
    manifest_payload = {
        **manifest_identity,
        "run_manifest_id": run_manifest_id,
        "artifacts": {
            "capture_request_sha256": request_sha256,
            "captured_trajectory_sha256": trajectory_sha256,
        },
    }
    manifest = {**manifest_payload, "manifest_sha256": sha256_json(manifest_payload)}
    return manifest, trajectory


def _without(mapping: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key != field}


def _require_span(value: Any, *, token_count: int, where: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise CaptureContractError(f"{where} must be a two-integer span")
    start, end = value
    if start < 0 or end > token_count or start >= end:
        raise CaptureContractError(f"{where} is outside the token sequence")
    return start, end


def verify_capture_bundle(
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check all canonical bundle identities before persistence."""

    validated = validate_capture_request(request)
    if canonical_json_bytes(validated) != canonical_json_bytes(request):
        raise CaptureContractError(
            "capture request must be normalized before writing the canonical bundle"
        )
    if not isinstance(manifest, dict) or not isinstance(trajectory, dict):
        raise CaptureContractError("manifest and trajectory must be objects")

    # Validate declared object shapes before authenticating hashes. Otherwise a
    # caller could make an unknown field self-consistent merely by rehashing it.
    _require_exact_keys(manifest, required=_MANIFEST_KEYS, where="run manifest")
    _require_exact_keys(trajectory, required=_TRAJECTORY_KEYS, where="captured trajectory")

    if trajectory["evidence_class"] not in _ALLOWED_EVIDENCE:
        raise CaptureContractError("trajectory evidence_class is invalid")
    if trajectory["replication_status"] != "not_attempted":
        raise CaptureContractError("trajectory replication_status must be not_attempted")

    artifacts = _require_object(manifest["artifacts"], "manifest artifacts")
    _require_exact_keys(artifacts, required=_ARTIFACT_KEYS, where="manifest artifacts")
    representation = _require_object(
        trajectory["representation_definition"], "trajectory representation_definition"
    )
    _require_exact_keys(
        representation,
        required=_REPRESENTATION_KEYS,
        where="trajectory representation_definition",
    )

    request_sha = sha256_json(validated)
    if manifest["request_sha256"] != request_sha:
        raise CaptureContractError("manifest request_sha256 does not match capture request")
    if artifacts["capture_request_sha256"] != request_sha:
        raise CaptureContractError("manifest artifact request hash does not match capture request")

    expected_representation = {
        "context_mode": validated["capture"]["context_mode"],
        "phase": validated["capture"]["phase"],
        "layers": validated["capture"]["layers"],
        "layer_index_semantics": _LAYER_INDEX_SEMANTICS,
        "pooling": validated["capture"]["pooling"],
        "step_span_semantics": _STEP_SPAN_SEMANTICS,
    }
    for field, expected in expected_representation.items():
        if representation[field] != expected:
            raise CaptureContractError(
                f"trajectory representation_definition.{field} does not match request"
            )

    trajectory_steps = trajectory["steps"]
    if not isinstance(trajectory_steps, list) or len(trajectory_steps) != len(validated["steps"]):
        raise CaptureContractError("trajectory step count does not match request")

    prefix = validated["capture"]["prefix_text"]
    joiner = validated["capture"]["step_joiner"]
    context_mode = validated["capture"]["context_mode"]
    pooling = validated["capture"]["pooling"]
    cumulative_segments: list[str] = []
    previous_ids: list[int] | None = None

    for index, (request_step, trajectory_step) in enumerate(
        zip(validated["steps"], trajectory_steps, strict=True)
    ):
        if not isinstance(trajectory_step, dict):
            raise CaptureContractError(f"trajectory step {index} is not an object")
        _require_exact_keys(
            trajectory_step,
            required={
                "step_index",
                "step_id",
                "rendered_text_sha256",
                "input_ids",
                "input_ids_sha256",
                "token_count",
                "changed_token_span",
                "phase",
                "layers",
            },
            where=f"trajectory step {index}",
        )
        if trajectory_step["step_index"] != index:
            raise CaptureContractError(f"trajectory step_index mismatch at {index}")
        if trajectory_step["step_id"] != request_step["step_id"]:
            raise CaptureContractError(f"trajectory step_id mismatch at {index}")
        if trajectory_step["phase"] != _CAPTURE_PHASE:
            raise CaptureContractError(f"trajectory phase mismatch at {index}")

        if context_mode == "cumulative":
            cumulative_segments.append(request_step["text"])
            rendered = _compose_text(prefix, cumulative_segments, joiner)
        else:
            rendered = _compose_text(prefix, [request_step["text"]], joiner)
        if trajectory_step["rendered_text_sha256"] != _sha256_text(rendered):
            raise CaptureContractError(f"rendered text hash mismatch at step {index}")

        input_ids = _validate_token_ids(
            trajectory_step["input_ids"], f"trajectory step {index}"
        )
        if trajectory_step["token_count"] != len(input_ids):
            raise CaptureContractError(f"token_count mismatch at step {index}")
        if trajectory_step["input_ids_sha256"] != sha256_json(input_ids):
            raise CaptureContractError(f"input_ids_sha256 mismatch at step {index}")
        changed_span = _require_span(
            trajectory_step["changed_token_span"],
            token_count=len(input_ids),
            where=f"trajectory step {index}.changed_token_span",
        )
        if context_mode == "cumulative" and previous_ids is not None:
            if changed_span[0] != _common_prefix_length(previous_ids, input_ids):
                raise CaptureContractError(f"changed token span mismatch at step {index}")
        expected_pool = _pool_span(
            token_count=len(input_ids),
            mode=pooling["mode"],
            changed_span=changed_span,
            window_tokens=pooling.get("window_tokens"),
        )

        layer_records = trajectory_step["layers"]
        requested_layers = validated["capture"]["layers"]
        if not isinstance(layer_records, list) or len(layer_records) != len(requested_layers):
            raise CaptureContractError(f"layer count mismatch at step {index}")
        for position, (requested_layer, layer_record) in enumerate(
            zip(requested_layers, layer_records, strict=True)
        ):
            if not isinstance(layer_record, dict):
                raise CaptureContractError(
                    f"trajectory step {index} layer {position} is not an object"
                )
            _require_exact_keys(
                layer_record,
                required={
                    "layer_index",
                    "vector_dimension",
                    "observed_dtype",
                    "pool_span",
                    "vector",
                    "vector_sha256",
                },
                where=f"trajectory step {index} layer {position}",
            )
            if layer_record["layer_index"] != requested_layer:
                raise CaptureContractError(
                    f"layer identity mismatch at step {index} position {position}"
                )
            dimension = layer_record["vector_dimension"]
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise CaptureContractError(
                    f"invalid vector_dimension at step {index} layer {requested_layer}"
                )
            _require_nonempty_string(
                layer_record["observed_dtype"],
                f"trajectory step {index} layer {requested_layer}.observed_dtype",
            )
            pool_span = _require_span(
                layer_record["pool_span"],
                token_count=len(input_ids),
                where=f"trajectory step {index} layer {requested_layer}.pool_span",
            )
            if pool_span != expected_pool:
                raise CaptureContractError(
                    f"pool span mismatch at step {index} layer {requested_layer}"
                )
            vector = layer_record["vector"]
            if (
                not isinstance(vector, list)
                or len(vector) != dimension
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in vector
                )
            ):
                raise CaptureContractError(
                    f"vector content/dimension mismatch at step {index} layer {requested_layer}"
                )
            if layer_record["vector_sha256"] != sha256_json(vector):
                raise CaptureContractError(
                    f"vector_sha256 mismatch at step {index} layer {requested_layer}"
                )
        if context_mode == "cumulative":
            previous_ids = input_ids

    stored_trajectory_sha = trajectory["trajectory_sha256"]
    actual_trajectory_sha = sha256_json(_without(trajectory, "trajectory_sha256"))
    if stored_trajectory_sha != actual_trajectory_sha:
        raise CaptureContractError("captured trajectory SHA-256 is invalid")
    if artifacts["captured_trajectory_sha256"] != actual_trajectory_sha:
        raise CaptureContractError("manifest trajectory hash does not match trajectory")

    actual_manifest_sha = sha256_json(_without(manifest, "manifest_sha256"))
    if manifest["manifest_sha256"] != actual_manifest_sha:
        raise CaptureContractError("run manifest SHA-256 is invalid")

    manifest_identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "protocol_id",
            "run_id",
            "repository_commit",
            "request_sha256",
            "model",
            "backend_request",
            "backend_observed",
            "capture",
            "determinism",
            "generation_parameters",
        )
    }
    actual_run_manifest_id = sha256_json(manifest_identity)
    if manifest["run_manifest_id"] != actual_run_manifest_id:
        raise CaptureContractError("run_manifest_id is invalid")

    for field in ("schema_version", "protocol_id", "run_id", "repository_commit"):
        if trajectory[field] != manifest[field]:
            raise CaptureContractError(f"trajectory {field} does not match manifest")
    if trajectory["run_manifest_id"] != actual_run_manifest_id:
        raise CaptureContractError("trajectory run_manifest_id does not match manifest")

    for field in ("model", "backend", "capture", "determinism", "generation_parameters"):
        manifest_field = "backend_request" if field == "backend" else field
        if manifest[manifest_field] != validated[field]:
            raise CaptureContractError(f"manifest {manifest_field} does not match request")

    observed_backend = _require_object(manifest["backend_observed"], "manifest backend_observed")
    _validate_backend_identity(observed_backend, validated)
    _validate_backend_metadata_shape(observed_backend, trajectory["evidence_class"])
    if trajectory["evidence_class"] == "OBSERVATION":
        required_observed = {
            "checkpoint_loading_clean": True,
            "quantization_config_present": False,
            "model_reports_quantized": False,
            "hidden_state_capture_strategy": "selective_forward_hooks",
            "pool_accumulation_dtype": "float64",
            "pool_accumulation_device": "cpu",
            "quantization": "none",
            "local_files_only": True,
            "trust_remote_code": False,
            "use_cache": False,
            "capture_phase": _CAPTURE_PHASE,
            "kv_cache_reuse": False,
        }
        for field, expected in required_observed.items():
            if observed_backend[field] != expected:
                raise CaptureContractError(
                    f"observation backend provenance field {field} is invalid"
                )
    return validated


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_capture_bundle(
    output_dir: Path,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> None:
    """Verify and atomically publish a new canonical capture-bundle directory."""

    validated = verify_capture_bundle(request, manifest, trajectory)
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise CaptureContractError(
            "output_dir already exists; canonical capture bundles are immutable publications"
        )

    payloads = {
        "capture-request.json": validated,
        "run-manifest.json": manifest,
        "captured-trajectory.json": trajectory,
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(parent))
    )
    published = False
    try:
        for name, payload in payloads.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(canonical_json_bytes(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        if output_dir.exists() or output_dir.is_symlink():
            raise CaptureContractError(
                "output_dir appeared during publication; refusing to replace it"
            )
        os.replace(staging, output_dir)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

"""Request/checkpoint validation for GEO-CAP-001."""
from __future__ import annotations
import copy
import re
from typing import Any, Mapping
from .capture_common import (CAPTURE_SCHEMA_VERSION, CAPTURE_PROTOCOL_ID, _ALLOWED_CONTEXT_MODES, _ALLOWED_DETERMINISM, _ALLOWED_DTYPES, _ALLOWED_POOLING_MODES, _CAPTURE_PHASE, _PRODUCTION_BACKEND, _LOADING_INFO_KEYS, CaptureContractError, _require_bool, _require_exact_keys, _require_git_sha, _require_hf_repo_id, _require_nonempty_string, _require_nonnegative_int, _require_object)

_MAX_TORCH_SEED = (1 << 64) - 1
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")


def validate_capture_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise CaptureContractError("capture request must be an object")
    root = copy.deepcopy(request)
    _require_exact_keys(
        root,
        required={"schema_version", "protocol_id", "run_id", "model", "backend", "capture", "determinism", "generation_parameters", "steps"},
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
        required={"identifier", "revision", "revision_kind", "tokenizer_identifier", "tokenizer_revision", "tokenizer_revision_kind"},
        where="model",
    )
    model["identifier"] = _require_hf_repo_id(model["identifier"], "model.identifier")
    model["tokenizer_identifier"] = _require_hf_repo_id(model["tokenizer_identifier"], "model.tokenizer_identifier")
    if model["revision_kind"] != "hf_commit":
        raise CaptureContractError("model.revision_kind must be 'hf_commit'")
    if model["tokenizer_revision_kind"] != "hf_commit":
        raise CaptureContractError("model.tokenizer_revision_kind must be 'hf_commit'")
    model["revision"] = _require_git_sha(model["revision"], "model.revision")
    model["tokenizer_revision"] = _require_git_sha(model["tokenizer_revision"], "model.tokenizer_revision")

    backend = _require_object(root["backend"], "backend")
    _require_exact_keys(
        backend,
        required={"name", "local_files_only", "trust_remote_code", "device", "dtype", "quantization"},
        where="backend",
    )
    if backend["name"] != _PRODUCTION_BACKEND:
        raise CaptureContractError(f"backend.name must be {_PRODUCTION_BACKEND!r} for GEO-CAP-001")
    if _require_bool(backend["local_files_only"], "backend.local_files_only") is not True:
        raise CaptureContractError("backend.local_files_only must be true")
    if _require_bool(backend["trust_remote_code"], "backend.trust_remote_code") is not False:
        raise CaptureContractError("backend.trust_remote_code must be false")
    device = _require_nonempty_string(backend["device"], "backend.device")
    if device not in {"cpu", "mps"} and not _CUDA_DEVICE.fullmatch(device):
        raise CaptureContractError("backend.device must be one of 'cpu', 'mps', or 'cuda:N' with an explicit non-negative CUDA index")
    if backend["dtype"] not in _ALLOWED_DTYPES:
        raise CaptureContractError(f"backend.dtype must be one of {sorted(_ALLOWED_DTYPES)}")
    if backend["quantization"] != "none":
        raise CaptureContractError("GEO-CAP-001 canonical capture requires backend.quantization='none'")

    capture = _require_object(root["capture"], "capture")
    _require_exact_keys(
        capture,
        required={"context_mode", "phase", "layers", "pooling", "prefix_text", "step_joiner"},
        where="capture",
    )
    if capture["context_mode"] not in _ALLOWED_CONTEXT_MODES:
        raise CaptureContractError(f"capture.context_mode must be one of {sorted(_ALLOWED_CONTEXT_MODES)}")
    if capture["phase"] != _CAPTURE_PHASE:
        raise CaptureContractError(f"GEO-CAP-001 currently supports capture.phase={_CAPTURE_PHASE!r} only")
    if not isinstance(capture["prefix_text"], str):
        raise CaptureContractError("capture.prefix_text must be a string")
    if not isinstance(capture["step_joiner"], str):
        raise CaptureContractError("capture.step_joiner must be a string")
    layers = capture["layers"]
    if not isinstance(layers, list) or not layers:
        raise CaptureContractError("capture.layers must be a non-empty array")
    normalized_layers = [_require_nonnegative_int(v, f"capture.layers[{i}]") for i, v in enumerate(layers)]
    if len(set(normalized_layers)) != len(normalized_layers):
        raise CaptureContractError("capture.layers must contain unique indices")
    capture["layers"] = normalized_layers
    pooling = _require_object(capture["pooling"], "capture.pooling")
    _require_exact_keys(pooling, required={"mode"}, optional={"window_tokens"}, where="capture.pooling")
    if pooling["mode"] not in _ALLOWED_POOLING_MODES:
        raise CaptureContractError(f"capture.pooling.mode must be one of {sorted(_ALLOWED_POOLING_MODES)}")
    if pooling["mode"] == "bounded_context_mean":
        if "window_tokens" not in pooling:
            raise CaptureContractError("capture.pooling.window_tokens is required for bounded_context_mean")
        window = _require_nonnegative_int(pooling["window_tokens"], "capture.pooling.window_tokens")
        if window == 0:
            raise CaptureContractError("capture.pooling.window_tokens must be >= 1")
        pooling["window_tokens"] = window
    elif "window_tokens" in pooling:
        raise CaptureContractError("capture.pooling.window_tokens is only valid for bounded_context_mean")

    determinism = _require_object(root["determinism"], "determinism")
    _require_exact_keys(determinism, required={"mode", "seed"}, where="determinism")
    if determinism["mode"] not in _ALLOWED_DETERMINISM:
        raise CaptureContractError(f"determinism.mode must be one of {sorted(_ALLOWED_DETERMINISM)}")
    seed = _require_nonnegative_int(determinism["seed"], "determinism.seed")
    if seed > _MAX_TORCH_SEED:
        raise CaptureContractError(f"determinism.seed must be <= {_MAX_TORCH_SEED} for torch.manual_seed")
    determinism["seed"] = seed

    generation = _require_object(root["generation_parameters"], "generation_parameters")
    if "generation_used" not in generation:
        raise CaptureContractError("generation_parameters.generation_used is required for replay capture")
    if _require_bool(generation["generation_used"], "generation_parameters.generation_used"):
        raise CaptureContractError("GEO-CAP-001 replay capture requires generation_parameters.generation_used=false")
    incompatible = {k: v for k, v in generation.items() if k != "generation_used" and v is not None}
    if incompatible:
        raise CaptureContractError("generation settings are incompatible with generation_used=false: " + ", ".join(sorted(incompatible)))

    steps = root["steps"]
    if not isinstance(steps, list) or not steps:
        raise CaptureContractError("steps must be a non-empty array")
    seen: set[str] = set()
    for i, value in enumerate(steps):
        step = _require_object(value, f"steps[{i}]")
        _require_exact_keys(step, required={"step_id", "text"}, where=f"steps[{i}]")
        step_id = _require_nonempty_string(step["step_id"], f"steps[{i}].step_id")
        if step_id in seen:
            raise CaptureContractError(f"duplicate step_id {step_id!r}")
        seen.add(step_id)
        _require_nonempty_string(step["text"], f"steps[{i}].text")
    if "notes" in root and not isinstance(root["notes"], str):
        raise CaptureContractError("notes must be a string")
    return root


def _validate_loading_info(loading_info: Any) -> None:
    if not isinstance(loading_info, dict):
        raise CaptureContractError("Transformers did not return checkpoint loading information")
    problems: dict[str, list[Any]] = {}
    for key in _LOADING_INFO_KEYS:
        value = loading_info.get(key, []) or []
        if not isinstance(value, (list, tuple)):
            raise CaptureContractError(f"checkpoint loading info {key!r} has invalid shape")
        if value:
            problems[key] = list(value)
    if problems:
        summary = "; ".join(f"{key}={value!r}" for key, value in problems.items())
        raise CaptureContractError("checkpoint load was not exact; canonical observation rejected: " + summary)


def _quantization_reasons(model: Any) -> list[str]:
    reasons: list[str] = []
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "quantization_config", None) is not None:
        reasons.append("config.quantization_config")
    if bool(getattr(model, "is_quantized", False)):
        reasons.append("model.is_quantized")
    if getattr(model, "hf_quantizer", None) is not None:
        reasons.append("model.hf_quantizer")
    return reasons

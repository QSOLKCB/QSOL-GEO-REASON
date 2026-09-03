"""Canonical local-model hidden-state capture for Phase 2A.

This module deliberately separates *capture* from later geometric analysis.  A
captured vector is an extracted representation state under GEO-CAP-001, not a
belief, proof state, semantic truth state, or mechanism of reasoning.

The default production backend is a direct Hugging Face / PyTorch forward pass.
Heavy dependencies are imported lazily so the Phase 1 package and its tests do
not require torch or transformers.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
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
_ALLOWED_POOLING_MODES = {"last_token", "step_mean", "context_mean", "bounded_context_mean"}
_ALLOWED_DETERMINISM = {"required", "best_effort"}


class CaptureContractError(ValueError):
    """Raised when a capture request or backend output violates GEO-CAP-001."""


class CaptureBackendUnavailable(RuntimeError):
    """Raised when the optional local capture dependencies are unavailable."""


class CaptureBackend(Protocol):
    """Minimal backend surface consumed by the protocol engine.

    `tokenize` and `hidden_states` intentionally expose the exact discrete input
    IDs and per-token hidden states used by the core.  This keeps step
    segmentation and pooling in one backend-independent, testable place.
    """

    def tokenize(self, text: str) -> list[int]:
        ...

    def hidden_states(self, input_ids: Sequence[int]) -> Sequence[Sequence[Sequence[float]]]:
        ...

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


def validate_capture_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a shallow-normalized GEO-CAP-001 request.

    The contract is intentionally strict.  Unknown fields are rejected so that a
    run cannot silently depend on an extraction choice that is absent from the
    published schema.
    """

    root = _require_object(dict(request), "capture request")
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
        raise CaptureContractError(
            f"schema_version must be {CAPTURE_SCHEMA_VERSION!r}"
        )
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
    for field in ("identifier", "tokenizer_identifier"):
        _require_nonempty_string(model[field], f"model.{field}")
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
        raise CaptureContractError(
            f"backend.dtype must be one of {sorted(_ALLOWED_DTYPES)}"
        )
    if backend["quantization"] != "none":
        raise CaptureContractError(
            "GEO-CAP-001 canonical capture requires backend.quantization='none'"
        )

    capture = _require_object(root["capture"], "capture")
    _require_exact_keys(
        capture,
        required={"context_mode", "phase", "layers", "pooling", "prefix_text", "step_joiner"},
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
    normalized_layers: list[int] = []
    for idx, layer in enumerate(layers):
        normalized_layers.append(_require_nonnegative_int(layer, f"capture.layers[{idx}]"))
    if len(set(normalized_layers)) != len(normalized_layers):
        raise CaptureContractError("capture.layers must contain unique indices")

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
    elif "window_tokens" in pooling:
        raise CaptureContractError(
            "capture.pooling.window_tokens is only valid for bounded_context_mean"
        )

    determinism = _require_object(root["determinism"], "determinism")
    _require_exact_keys(
        determinism,
        required={"mode", "seed"},
        where="determinism",
    )
    if determinism["mode"] not in _ALLOWED_DETERMINISM:
        raise CaptureContractError(
            f"determinism.mode must be one of {sorted(_ALLOWED_DETERMINISM)}"
        )
    _require_nonnegative_int(determinism["seed"], "determinism.seed")

    if not isinstance(root["generation_parameters"], dict):
        raise CaptureContractError("generation_parameters must be an object")

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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _compose_text(prefix: str, segments: Sequence[str], joiner: str) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.extend(segments)
    return joiner.join(parts)


def _validate_token_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    expected_tokens: int,
    where: str,
) -> tuple[int, int]:
    if len(matrix) != expected_tokens:
        raise CaptureContractError(
            f"{where} token count {len(matrix)} != input token count {expected_tokens}"
        )
    if expected_tokens == 0:
        raise CaptureContractError(f"{where} cannot be empty")

    dimension: int | None = None
    for token_idx, row in enumerate(matrix):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise CaptureContractError(f"{where}[{token_idx}] must be a vector")
        if dimension is None:
            dimension = len(row)
            if dimension == 0:
                raise CaptureContractError(f"{where} vectors must be non-empty")
        elif len(row) != dimension:
            raise CaptureContractError(f"{where} contains inconsistent vector dimensions")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CaptureContractError(f"{where} contains a non-numeric value")
            if not math.isfinite(float(value)):
                raise CaptureContractError(f"{where} contains a non-finite value")
    assert dimension is not None
    return expected_tokens, dimension


def _mean_rows(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise CaptureContractError("cannot pool an empty token span")
    dimension = len(rows[0])
    result: list[float] = []
    for axis in range(dimension):
        result.append(math.fsum(float(row[axis]) for row in rows) / len(rows))
    return result


def _pool_hidden_state(
    matrix: Sequence[Sequence[float]],
    *,
    mode: str,
    changed_span: tuple[int, int],
    window_tokens: int | None,
) -> tuple[list[float], tuple[int, int]]:
    token_count = len(matrix)
    changed_start, changed_end = changed_span
    if changed_start < 0 or changed_end > token_count or changed_start >= changed_end:
        raise CaptureContractError(
            f"invalid changed token span [{changed_start}, {changed_end}) for {token_count} tokens"
        )

    if mode == "last_token":
        start, end = token_count - 1, token_count
        vector = [float(value) for value in matrix[-1]]
    elif mode == "step_mean":
        start, end = changed_start, changed_end
        vector = _mean_rows(matrix[start:end])
    elif mode == "context_mean":
        start, end = 0, token_count
        vector = _mean_rows(matrix)
    elif mode == "bounded_context_mean":
        if window_tokens is None or window_tokens < 1:
            raise CaptureContractError("bounded_context_mean requires window_tokens >= 1")
        start, end = max(0, token_count - window_tokens), token_count
        vector = _mean_rows(matrix[start:end])
    else:
        raise CaptureContractError(f"unsupported pooling mode {mode!r}")

    if not vector or any(not math.isfinite(value) for value in vector):
        raise CaptureContractError("pooled representation contains non-finite values")
    return vector, (start, end)


def _capture_steps(request: Mapping[str, Any], backend: CaptureBackend) -> list[dict[str, Any]]:
    capture_cfg = request["capture"]
    pooling_cfg = capture_cfg["pooling"]
    pooling_mode = pooling_cfg["mode"]
    window_tokens = pooling_cfg.get("window_tokens")
    prefix = capture_cfg["prefix_text"]
    joiner = capture_cfg["step_joiner"]
    context_mode = capture_cfg["context_mode"]
    layers: list[int] = capture_cfg["layers"]

    prefix_ids = backend.tokenize(prefix)
    if prefix and not prefix_ids:
        raise CaptureContractError("tokenizer produced no tokens for non-empty prefix_text")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in prefix_ids
    ):
        raise CaptureContractError("tokenizer returned an invalid prefix token ID")

    previous_ids = prefix_ids
    cumulative_segments: list[str] = []
    output: list[dict[str, Any]] = []
    hidden_state_count: int | None = None
    vector_dimensions: dict[int, int] = {}

    for step_index, step in enumerate(request["steps"]):
        if context_mode == "cumulative":
            cumulative_segments.append(step["text"])
            rendered = _compose_text(prefix, cumulative_segments, joiner)
            baseline_ids = previous_ids
        else:
            rendered = _compose_text(prefix, [step["text"]], joiner)
            baseline_ids = prefix_ids

        input_ids = backend.tokenize(rendered)
        if not input_ids:
            raise CaptureContractError(f"step {step['step_id']!r} tokenized to zero tokens")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in input_ids
        ):
            raise CaptureContractError(
                f"step {step['step_id']!r} tokenizer returned an invalid token ID"
            )

        changed_start = _common_prefix_length(baseline_ids, input_ids)
        changed_span = (changed_start, len(input_ids))
        if changed_start == len(input_ids):
            raise CaptureContractError(
                f"step {step['step_id']!r} adds no changed token span under the frozen tokenizer"
            )

        states = backend.hidden_states(input_ids)
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise CaptureContractError("backend.hidden_states must return a sequence of layers")
        if hidden_state_count is None:
            hidden_state_count = len(states)
            if hidden_state_count == 0:
                raise CaptureContractError("backend returned no hidden-state layers")
        elif len(states) != hidden_state_count:
            raise CaptureContractError("backend hidden-state layer count changed across steps")

        layer_records: list[dict[str, Any]] = []
        for layer_index in layers:
            if layer_index >= len(states):
                raise CaptureContractError(
                    f"requested layer {layer_index} outside backend hidden_states range [0, {len(states) - 1}]"
                )
            matrix = states[layer_index]
            _, dimension = _validate_token_matrix(
                matrix,
                expected_tokens=len(input_ids),
                where=f"step {step['step_id']!r} layer {layer_index}",
            )
            previous_dimension = vector_dimensions.setdefault(layer_index, dimension)
            if previous_dimension != dimension:
                raise CaptureContractError(
                    f"layer {layer_index} vector dimension changed from {previous_dimension} to {dimension}"
                )
            vector, pool_span = _pool_hidden_state(
                matrix,
                mode=pooling_mode,
                changed_span=changed_span,
                window_tokens=window_tokens,
            )
            layer_records.append(
                {
                    "layer_index": layer_index,
                    "vector_dimension": dimension,
                    "pool_span": [pool_span[0], pool_span[1]],
                    "vector": vector,
                    "vector_sha256": sha256_json(vector),
                }
            )

        step_record = {
            "step_index": step_index,
            "step_id": step["step_id"],
            "rendered_text_sha256": _sha256_text(rendered),
            "input_ids": [int(token_id) for token_id in input_ids],
            "input_ids_sha256": sha256_json([int(token_id) for token_id in input_ids]),
            "token_count": len(input_ids),
            "changed_token_span": [changed_span[0], changed_span[1]],
            "phase": _CAPTURE_PHASE,
            "layers": layer_records,
        }
        output.append(step_record)
        if context_mode == "cumulative":
            previous_ids = input_ids

    return output


def execute_capture(
    request: Mapping[str, Any],
    *,
    implementation_revision: str,
    backend: CaptureBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one canonical capture and return `(run_manifest, trajectory)`.

    The caller owns persistence.  No result interpretation, geometry statistic,
    or hypothesis label is produced here.
    """

    validated = validate_capture_request(request)
    implementation_revision = _require_git_sha(
        implementation_revision, "implementation_revision"
    )

    observed_backend = dict(backend.metadata())
    if observed_backend.get("name") != _PRODUCTION_BACKEND:
        raise CaptureContractError(
            f"backend metadata name must be {_PRODUCTION_BACKEND!r}"
        )
    observed_model_commit = observed_backend.get("observed_model_commit")
    if observed_model_commit != validated["model"]["revision"]:
        raise CaptureContractError(
            "observed model commit does not match the frozen request revision"
        )
    observed_tokenizer_commit = observed_backend.get("observed_tokenizer_commit")
    if observed_tokenizer_commit != validated["model"]["tokenizer_revision"]:
        raise CaptureContractError(
            "observed tokenizer commit does not match the frozen request revision"
        )

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

    steps = _capture_steps(validated, backend)
    trajectory_payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "protocol_id": CAPTURE_PROTOCOL_ID,
        "evidence_class": "OBSERVATION",
        "replication_status": "not_attempted",
        "run_id": validated["run_id"],
        "run_manifest_id": run_manifest_id,
        "repository_commit": implementation_revision,
        "representation_definition": {
            "context_mode": validated["capture"]["context_mode"],
            "phase": validated["capture"]["phase"],
            "layers": validated["capture"]["layers"],
            "layer_index_semantics": (
                "indices address the backend outputs.hidden_states tuple; "
                "index 0 is the pre-block embedding state when supplied by the model"
            ),
            "pooling": validated["capture"]["pooling"],
            "step_span_semantics": (
                "changed_token_span begins at the longest common token-ID prefix "
                "between the baseline context and the current rendered context"
            ),
        },
        "steps": steps,
    }
    trajectory_sha256 = sha256_json(trajectory_payload)
    trajectory = {
        **trajectory_payload,
        "trajectory_sha256": trajectory_sha256,
    }

    manifest_payload = {
        **manifest_identity,
        "run_manifest_id": run_manifest_id,
        "artifacts": {
            "capture_request_sha256": request_sha256,
            "captured_trajectory_sha256": trajectory_sha256,
        },
    }
    manifest_sha256 = sha256_json(manifest_payload)
    manifest = {
        **manifest_payload,
        "manifest_sha256": manifest_sha256,
    }
    return manifest, trajectory


def write_capture_bundle(
    output_dir: Path,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> None:
    """Write canonical request, manifest, and trajectory JSON files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "capture-request.json": request,
        "run-manifest.json": manifest,
        "captured-trajectory.json": trajectory,
    }
    for name, payload in files.items():
        path = output_dir / name
        path.write_bytes(canonical_json_bytes(payload) + b"\n")


class HuggingFacePyTorchBackend:
    """Direct, local-only Hugging Face / PyTorch replay backend."""

    def __init__(self, request: Mapping[str, Any]):
        validated = validate_capture_request(request)
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise CaptureBackendUnavailable(
                "canonical capture requires the optional 'capture' dependencies: "
                "install qsol-geo-reason[capture]"
            ) from exc

        self._torch = torch
        self._transformers = transformers
        backend = validated["backend"]
        model_cfg = validated["model"]
        determinism = validated["determinism"]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[backend["dtype"]]
        device = backend["device"]

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
        self._model = AutoModelForCausalLM.from_pretrained(
            model_cfg["identifier"],
            revision=model_cfg["revision"],
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        )
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._dtype_name = backend["dtype"]
        self._determinism_mode = determinism["mode"]

    def tokenize(self, text: str) -> list[int]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        return [int(token_id) for token_id in input_ids]

    def hidden_states(self, input_ids: Sequence[int]) -> Sequence[Sequence[Sequence[float]]]:
        torch = self._torch
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self._device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            output = self._model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        states = output.hidden_states
        if states is None:
            raise CaptureContractError("model did not return hidden_states")
        return [
            state[0].detach().to(device="cpu", dtype=torch.float64).tolist()
            for state in states
        ]

    def metadata(self) -> Mapping[str, Any]:
        torch = self._torch
        model_config = self._model.config
        observed_commit = getattr(model_config, "_commit_hash", None)
        observed_tokenizer_commit = self._tokenizer.init_kwargs.get("_commit_hash")
        attn_impl = getattr(model_config, "_attn_implementation", None)
        try:
            tokenizers_version = metadata.version("tokenizers")
        except metadata.PackageNotFoundError:
            tokenizers_version = None
        try:
            hub_version = metadata.version("huggingface-hub")
        except metadata.PackageNotFoundError:
            hub_version = None

        cuda_device = None
        if str(self._device).startswith("cuda") and torch.cuda.is_available():
            try:
                cuda_device = torch.cuda.get_device_name(self._device)
            except Exception:
                cuda_device = None

        return {
            "name": _PRODUCTION_BACKEND,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": self._transformers.__version__,
            "tokenizers_version": tokenizers_version,
            "huggingface_hub_version": hub_version,
            "model_class": type(self._model).__name__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "observed_model_commit": observed_commit,
            "observed_tokenizer_commit": observed_tokenizer_commit,
            "attention_implementation": attn_impl,
            "device": str(self._device),
            "cuda_device_name": cuda_device,
            "dtype": self._dtype_name,
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

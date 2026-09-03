"""Canonical local-model hidden-state capture for QSOL-GEO-REASON Phase 2A.

The core in this module deliberately stops at representation capture.  It does
not interpret a vector as a belief, proof state, truth state, gauge field,
thermodynamic variable, or reasoning mechanism.

The generic protocol engine defaults to ``SIMULATION`` evidence so software
fixtures and third-party adapters cannot accidentally manufacture empirical
``OBSERVATION`` records.  ``OBSERVATION`` is accepted only when the concrete
backend is the local-only Hugging Face / PyTorch adapter defined here.
"""

from __future__ import annotations

import copy
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
_ALLOWED_POOLING_MODES = {
    "last_token",
    "step_mean",
    "context_mean",
    "bounded_context_mean",
}
_ALLOWED_DETERMINISM = {"required", "best_effort"}
_ALLOWED_EVIDENCE = {"SIMULATION", "OBSERVATION"}


class CaptureContractError(ValueError):
    """Raised when a capture request or backend output violates GEO-CAP-001."""


class CaptureBackendUnavailable(RuntimeError):
    """Raised when optional local capture dependencies are unavailable."""


class CaptureBackend(Protocol):
    """Minimal backend surface consumed by the backend-independent protocol core."""

    def tokenize(self, text: str) -> list[int]:
        ...

    def hidden_states(
        self, input_ids: Sequence[int]
    ) -> Sequence[Sequence[Sequence[float]]]:
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
    """Return a deep-copied, strictly validated GEO-CAP-001 request.

    Unknown fields are rejected so no material extraction choice can silently
    travel outside the published request contract.  The caller's object is not
    mutated while revisions are normalized to lowercase.
    """

    root = _require_object(copy.deepcopy(dict(request)), "capture request")
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
    _require_nonempty_string(model["identifier"], "model.identifier")
    _require_nonempty_string(
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
    _require_exact_keys(
        determinism,
        required={"mode", "seed"},
        where="determinism",
    )
    if determinism["mode"] not in _ALLOWED_DETERMINISM:
        raise CaptureContractError(
            f"determinism.mode must be one of {sorted(_ALLOWED_DETERMINISM)}"
        )
    determinism["seed"] = _require_nonnegative_int(
        determinism["seed"], "determinism.seed"
    )

    if not isinstance(root["generation_parameters"], dict):
        raise CaptureContractError("generation_parameters must be an object")

    steps = root["steps"]
    if not isinstance(steps, list) or not steps:
        raise CaptureContractError("steps must be a non-empty array")
    seen_ids: set[str] = set()
    for idx, step_value in enumerate(steps):
        step = _require_object(step_value, f"steps[{idx}]")
        _require_exact_keys(
            step,
            required={"step_id", "text"},
            where=f"steps[{idx}]",
        )
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


def _validate_token_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    expected_tokens: int,
    where: str,
) -> int:
    if len(matrix) != expected_tokens:
        raise CaptureContractError(
            f"{where} token count {len(matrix)} != input token count {expected_tokens}"
        )
    if expected_tokens == 0:
        raise CaptureContractError(f"{where} cannot be empty")

    dimension: int | None = None
    for token_index, row in enumerate(matrix):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise CaptureContractError(f"{where}[{token_index}] must be a vector")
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
    return dimension


def _mean_rows(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise CaptureContractError("cannot pool an empty token span")
    dimension = len(rows[0])
    return [
        math.fsum(float(row[axis]) for row in rows) / len(rows)
        for axis in range(dimension)
    ]


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
        prefix_ids = [int(value) for value in raw_prefix_ids]
        if any(value < 0 for value in prefix_ids):
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

        input_ids = _validate_token_ids(
            backend.tokenize(rendered), f"step {step['step_id']!r}"
        )
        changed_start = _common_prefix_length(baseline_ids, input_ids)
        changed_span = (changed_start, len(input_ids))
        if changed_start == len(input_ids):
            raise CaptureContractError(
                f"step {step['step_id']!r} adds no changed token span under the frozen tokenizer"
            )

        states = backend.hidden_states(input_ids)
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise CaptureContractError(
                "backend.hidden_states must return a sequence of layers"
            )
        if hidden_state_count is None:
            hidden_state_count = len(states)
            if hidden_state_count == 0:
                raise CaptureContractError("backend returned no hidden-state layers")
        elif len(states) != hidden_state_count:
            raise CaptureContractError(
                "backend hidden-state layer count changed across steps"
            )

        layer_records: list[dict[str, Any]] = []
        for layer_index in layers:
            if layer_index >= len(states):
                raise CaptureContractError(
                    f"requested layer {layer_index} outside backend hidden_states range "
                    f"[0, {len(states) - 1}]"
                )
            matrix = states[layer_index]
            dimension = _validate_token_matrix(
                matrix,
                expected_tokens=len(input_ids),
                where=f"step {step['step_id']!r} layer {layer_index}",
            )
            previous_dimension = vector_dimensions.setdefault(layer_index, dimension)
            if previous_dimension != dimension:
                raise CaptureContractError(
                    f"layer {layer_index} vector dimension changed from "
                    f"{previous_dimension} to {dimension}"
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
        self._model = AutoModelForCausalLM.from_pretrained(
            model_cfg["identifier"],
            revision=model_cfg["revision"],
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype_map[backend["dtype"]],
        )
        self._model.to(self._device)
        self._model.eval()

    def tokenize(self, text: str) -> list[int]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        return [int(token_id) for token_id in encoded["input_ids"]]

    def hidden_states(
        self, input_ids: Sequence[int]
    ) -> Sequence[Sequence[Sequence[float]]]:
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

    @staticmethod
    def _installed_version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    def metadata(self) -> Mapping[str, Any]:
        torch = self._torch
        model_config = self._model.config
        observed_model_commit = getattr(model_config, "_commit_hash", None)
        observed_tokenizer_commit = (
            getattr(self._tokenizer, "_commit_hash", None)
            or self._tokenizer.init_kwargs.get("_commit_hash")
        )
        attention_implementation = getattr(
            model_config, "_attn_implementation", None
        )
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
            "tokenizers_version": self._installed_version("tokenizers"),
            "huggingface_hub_version": self._installed_version("huggingface-hub"),
            "model_class": type(self._model).__name__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "observed_model_commit": observed_model_commit,
            "observed_tokenizer_commit": observed_tokenizer_commit,
            "attention_implementation": attention_implementation,
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


def execute_capture(
    request: Mapping[str, Any],
    *,
    implementation_revision: str,
    backend: CaptureBackend,
    evidence_class: str = "SIMULATION",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute GEO-CAP-001 and return ``(run_manifest, trajectory)``.

    Generic adapters and test doubles produce ``SIMULATION`` by default.
    ``OBSERVATION`` requires the concrete local-only
    :class:`HuggingFacePyTorchBackend`; metadata labels alone are insufficient.
    """

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

    observed_backend = dict(backend.metadata())
    if observed_backend.get("name") != _PRODUCTION_BACKEND:
        raise CaptureContractError(
            f"backend metadata name must be {_PRODUCTION_BACKEND!r}"
        )
    if observed_backend.get("observed_model_commit") != validated["model"]["revision"]:
        raise CaptureContractError(
            "observed model commit does not match the frozen request revision"
        )
    if (
        observed_backend.get("observed_tokenizer_commit")
        != validated["model"]["tokenizer_revision"]
    ):
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
        "steps": _capture_steps(validated, backend),
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
    manifest = {
        **manifest_payload,
        "manifest_sha256": sha256_json(manifest_payload),
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
    for name, payload in {
        "capture-request.json": request,
        "run-manifest.json": manifest,
        "captured-trajectory.json": trajectory,
    }.items():
        (output_dir / name).write_bytes(canonical_json_bytes(payload) + b"\n")

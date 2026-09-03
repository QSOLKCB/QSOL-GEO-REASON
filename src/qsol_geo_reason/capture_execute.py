"""Execution and capture-step construction for GEO-CAP-001."""
from __future__ import annotations
from typing import Any, Mapping
from .canonical import sha256_json
from .capture_backend import HuggingFacePyTorchBackend
from .capture_common import (CAPTURE_PROTOCOL_ID, CAPTURE_SCHEMA_VERSION, _ALLOWED_EVIDENCE, _CAPTURE_PHASE, _LAYER_INDEX_SEMANTICS, _STEP_SPAN_SEMANTICS, CaptureBackend, CaptureContractError, _common_prefix_length, _compose_text, _pool_span, _require_git_sha, _sha256_text, _validate_backend_layer, _validate_token_ids)
from .capture_validation import validate_capture_request
from .capture_provenance import _validate_backend_metadata

def _capture_steps(request: Mapping[str, Any], backend: CaptureBackend) -> tuple[list[dict[str, Any]], list[int]]:
    cfg = request["capture"]
    pooling = cfg["pooling"]
    prefix_ids = _validate_token_ids(backend.tokenize(cfg["prefix_text"]), "prefix_text", allow_empty=not bool(cfg["prefix_text"]))
    previous_ids = prefix_ids
    cumulative_segments: list[str] = []
    output: list[dict[str, Any]] = []
    dimensions: dict[int, int] = {}

    for step_index, step in enumerate(request["steps"]):
        if cfg["context_mode"] == "cumulative":
            cumulative_segments.append(step["text"])
            rendered = _compose_text(cfg["prefix_text"], cumulative_segments, cfg["step_joiner"])
            baseline_ids = previous_ids
        else:
            rendered = _compose_text(cfg["prefix_text"], [step["text"]], cfg["step_joiner"])
            baseline_ids = prefix_ids
        input_ids = _validate_token_ids(backend.tokenize(rendered), f"step {step['step_id']!r}")
        changed_start = _common_prefix_length(baseline_ids, input_ids)
        if changed_start == len(input_ids):
            raise CaptureContractError(f"step {step['step_id']!r} adds no changed token span under the frozen tokenizer")
        changed_span = (changed_start, len(input_ids))
        pool_span = _pool_span(
            token_count=len(input_ids), mode=pooling["mode"], changed_span=changed_span,
            window_tokens=pooling.get("window_tokens"),
        )
        selected = backend.hidden_states(input_ids, cfg["layers"], pool_span=pool_span)
        if not isinstance(selected, Mapping) or set(selected) != set(cfg["layers"]):
            raise CaptureContractError("backend hidden-state selection does not exactly match requested layers")
        layer_records: list[dict[str, Any]] = []
        for layer_index in cfg["layers"]:
            vector, dimension, observed_dtype = _validate_backend_layer(
                selected[layer_index], layer_index=layer_index,
                expected_dimension=dimensions.get(layer_index),
                where=f"step {step['step_id']!r} layer {layer_index}",
            )
            dimensions.setdefault(layer_index, dimension)
            layer_records.append({
                "layer_index": layer_index, "vector_dimension": dimension,
                "observed_dtype": observed_dtype, "pool_span": list(pool_span),
                "vector": vector, "vector_sha256": sha256_json(vector),
            })
        output.append({
            "step_index": step_index, "step_id": step["step_id"],
            "rendered_text_sha256": _sha256_text(rendered), "input_ids": input_ids,
            "input_ids_sha256": sha256_json(input_ids), "token_count": len(input_ids),
            "changed_token_span": list(changed_span), "phase": _CAPTURE_PHASE,
            "layers": layer_records,
        })
        if cfg["context_mode"] == "cumulative":
            previous_ids = input_ids
    return output, prefix_ids


def execute_capture(
    request: Mapping[str, Any], *, implementation_revision: str, backend: CaptureBackend,
    evidence_class: str = "SIMULATION",
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_capture_request(request)
    implementation_revision = _require_git_sha(implementation_revision, "implementation_revision")
    if evidence_class not in _ALLOWED_EVIDENCE:
        raise CaptureContractError(f"evidence_class must be one of {sorted(_ALLOWED_EVIDENCE)}")
    if evidence_class == "OBSERVATION" and type(backend) is not HuggingFacePyTorchBackend:
        raise CaptureContractError("OBSERVATION capture requires the concrete HuggingFacePyTorchBackend")
    steps, prefix_ids = _capture_steps(validated, backend)
    observed = dict(backend.metadata())
    _validate_backend_metadata(observed, validated, evidence_class)
    request_sha = sha256_json(validated)
    identity = {
        "schema_version": CAPTURE_SCHEMA_VERSION, "protocol_id": CAPTURE_PROTOCOL_ID,
        "run_id": validated["run_id"], "repository_commit": implementation_revision,
        "request_sha256": request_sha, "model": validated["model"], "backend_request": validated["backend"],
        "backend_observed": observed, "capture": validated["capture"], "determinism": validated["determinism"],
        "generation_parameters": validated["generation_parameters"],
    }
    run_manifest_id = sha256_json(identity)
    trajectory_payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION, "protocol_id": CAPTURE_PROTOCOL_ID,
        "evidence_class": evidence_class, "replication_status": "not_attempted", "run_id": validated["run_id"],
        "run_manifest_id": run_manifest_id, "repository_commit": implementation_revision,
        "representation_definition": {
            "context_mode": validated["capture"]["context_mode"], "phase": validated["capture"]["phase"],
            "layers": validated["capture"]["layers"], "layer_index_semantics": _LAYER_INDEX_SEMANTICS,
            "pooling": validated["capture"]["pooling"], "step_span_semantics": _STEP_SPAN_SEMANTICS,
            "prefix_input_ids": prefix_ids, "prefix_input_ids_sha256": sha256_json(prefix_ids),
        },
        "steps": steps,
    }
    trajectory_sha = sha256_json(trajectory_payload)
    trajectory = {**trajectory_payload, "trajectory_sha256": trajectory_sha}
    manifest_payload = {**identity, "run_manifest_id": run_manifest_id, "artifacts": {"capture_request_sha256": request_sha, "captured_trajectory_sha256": trajectory_sha}}
    manifest = {**manifest_payload, "manifest_sha256": sha256_json(manifest_payload)}
    return manifest, trajectory

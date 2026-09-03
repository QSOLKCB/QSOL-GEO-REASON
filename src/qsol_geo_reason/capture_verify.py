"""Cross-artifact verification for GEO-CAP-001."""
from __future__ import annotations
import math
from typing import Any, Mapping
from .canonical import canonical_json_bytes, sha256_json
from .capture_common import (
    CAPTURE_PROTOCOL_ID,
    CAPTURE_SCHEMA_VERSION,
    _ALLOWED_EVIDENCE,
    _ARTIFACT_KEYS,
    _CAPTURE_PHASE,
    _LAYER_INDEX_SEMANTICS,
    _MANIFEST_KEYS,
    _REPRESENTATION_KEYS,
    _STEP_SPAN_SEMANTICS,
    _TRAJECTORY_KEYS,
    CaptureContractError,
    _common_prefix_length,
    _compose_text,
    _pool_span,
    _require_exact_keys,
    _require_git_sha,
    _require_nonempty_string,
    _require_object,
    _sha256_text,
    _validate_token_ids,
)
from .capture_validation import validate_capture_request
from .capture_provenance import _validate_backend_metadata


def _without(mapping: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if k != field}


def _require_span(value: Any, *, token_count: int, where: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise CaptureContractError(f"{where} must be a two-integer span")
    start, end = value
    if start < 0 or end > token_count or start >= end:
        raise CaptureContractError(f"{where} is outside the token sequence")
    return start, end


def _validate_observed_dtype_map(
    observed_backend: Mapping[str, Any], recorded_dtypes: Mapping[int, set[str]]
) -> None:
    expected = {
        str(layer_index): sorted(values)
        for layer_index, values in sorted(recorded_dtypes.items())
    }
    if observed_backend.get("observed_hidden_state_dtypes") != expected:
        raise CaptureContractError(
            "backend observed_hidden_state_dtypes does not match trajectory layer records"
        )


def verify_capture_bundle(request: Mapping[str, Any], manifest: Mapping[str, Any], trajectory: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_capture_request(request)
    if canonical_json_bytes(validated) != canonical_json_bytes(request):
        raise CaptureContractError("capture request must be normalized before writing the canonical bundle")
    if not isinstance(manifest, dict) or not isinstance(trajectory, dict):
        raise CaptureContractError("manifest and trajectory must be objects")
    _require_exact_keys(manifest, required=_MANIFEST_KEYS, where="run manifest")
    _require_exact_keys(trajectory, required=_TRAJECTORY_KEYS, where="captured trajectory")

    if manifest["schema_version"] != CAPTURE_SCHEMA_VERSION or trajectory["schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise CaptureContractError(f"bundle schema_version must be {CAPTURE_SCHEMA_VERSION!r}")
    if manifest["protocol_id"] != CAPTURE_PROTOCOL_ID or trajectory["protocol_id"] != CAPTURE_PROTOCOL_ID:
        raise CaptureContractError(f"bundle protocol_id must be {CAPTURE_PROTOCOL_ID!r}")
    manifest_repository_commit = _require_git_sha(manifest["repository_commit"], "manifest.repository_commit")
    trajectory_repository_commit = _require_git_sha(trajectory["repository_commit"], "trajectory.repository_commit")
    if manifest_repository_commit != manifest["repository_commit"]:
        raise CaptureContractError("manifest.repository_commit must use canonical lowercase 40-hex form")
    if trajectory_repository_commit != trajectory["repository_commit"]:
        raise CaptureContractError("trajectory.repository_commit must use canonical lowercase 40-hex form")
    if trajectory_repository_commit != manifest_repository_commit:
        raise CaptureContractError("trajectory repository_commit does not match manifest")

    if trajectory["evidence_class"] not in _ALLOWED_EVIDENCE or trajectory["replication_status"] != "not_attempted":
        raise CaptureContractError("trajectory evidence/replication status is invalid")

    artifacts = _require_object(manifest["artifacts"], "manifest artifacts")
    _require_exact_keys(artifacts, required=_ARTIFACT_KEYS, where="manifest artifacts")
    representation = _require_object(trajectory["representation_definition"], "trajectory representation_definition")
    _require_exact_keys(representation, required=_REPRESENTATION_KEYS, where="trajectory representation_definition")
    request_sha = sha256_json(validated)
    if manifest["request_sha256"] != request_sha or artifacts["capture_request_sha256"] != request_sha:
        raise CaptureContractError("manifest request SHA-256 does not match capture request")

    prefix_ids = _validate_token_ids(representation["prefix_input_ids"], "trajectory prefix_input_ids", allow_empty=True)
    if representation["prefix_input_ids_sha256"] != sha256_json(prefix_ids):
        raise CaptureContractError("trajectory prefix_input_ids_sha256 is invalid")
    expected_representation = {
        "context_mode": validated["capture"]["context_mode"], "phase": validated["capture"]["phase"],
        "layers": validated["capture"]["layers"], "layer_index_semantics": _LAYER_INDEX_SEMANTICS,
        "pooling": validated["capture"]["pooling"], "step_span_semantics": _STEP_SPAN_SEMANTICS,
    }
    for key, expected in expected_representation.items():
        if representation[key] != expected:
            raise CaptureContractError(f"trajectory representation_definition.{key} does not match request")

    steps = trajectory["steps"]
    if not isinstance(steps, list) or len(steps) != len(validated["steps"]):
        raise CaptureContractError("trajectory step count does not match request")
    cfg, pooling = validated["capture"], validated["capture"]["pooling"]
    cumulative_segments: list[str] = []
    previous_ids = prefix_ids
    stable_dimensions: dict[int, int] = {}
    recorded_dtypes: dict[int, set[str]] = {}
    for index, (request_step, step) in enumerate(zip(validated["steps"], steps, strict=True)):
        if not isinstance(step, dict):
            raise CaptureContractError(f"trajectory step {index} is not an object")
        _require_exact_keys(step, required={"step_index", "step_id", "rendered_text_sha256", "input_ids", "input_ids_sha256", "token_count", "changed_token_span", "phase", "layers"}, where=f"trajectory step {index}")
        if step["step_index"] != index:
            raise CaptureContractError(f"trajectory step_index mismatch at {index}")
        if step["step_id"] != request_step["step_id"]:
            raise CaptureContractError(f"trajectory step_id mismatch at {index}")
        if step["phase"] != _CAPTURE_PHASE:
            raise CaptureContractError(f"trajectory phase mismatch at {index}")
        if cfg["context_mode"] == "cumulative":
            cumulative_segments.append(request_step["text"])
            rendered = _compose_text(cfg["prefix_text"], cumulative_segments, cfg["step_joiner"])
            baseline_ids = previous_ids
        else:
            rendered = _compose_text(cfg["prefix_text"], [request_step["text"]], cfg["step_joiner"])
            baseline_ids = prefix_ids
        if step["rendered_text_sha256"] != _sha256_text(rendered):
            raise CaptureContractError(f"rendered text hash mismatch at step {index}")
        input_ids = _validate_token_ids(step["input_ids"], f"trajectory step {index}")
        if step["token_count"] != len(input_ids):
            raise CaptureContractError(f"token_count mismatch at step {index}")
        if step["input_ids_sha256"] != sha256_json(input_ids):
            raise CaptureContractError(f"input_ids SHA-256 mismatch at step {index}")
        changed_span = _require_span(step["changed_token_span"], token_count=len(input_ids), where=f"trajectory step {index}.changed_token_span")
        expected_changed = (_common_prefix_length(baseline_ids, input_ids), len(input_ids))
        if changed_span != expected_changed:
            raise CaptureContractError(f"changed token span mismatch at step {index}")
        expected_pool = _pool_span(token_count=len(input_ids), mode=pooling["mode"], changed_span=changed_span, window_tokens=pooling.get("window_tokens"))
        records, requested_layers = step["layers"], cfg["layers"]
        if not isinstance(records, list) or len(records) != len(requested_layers):
            raise CaptureContractError(f"layer count mismatch at step {index}")
        for position, (requested_layer, record) in enumerate(zip(requested_layers, records, strict=True)):
            if not isinstance(record, dict):
                raise CaptureContractError(f"trajectory step {index} layer {position} is not an object")
            _require_exact_keys(record, required={"layer_index", "vector_dimension", "observed_dtype", "pool_span", "vector", "vector_sha256"}, where=f"trajectory step {index} layer {position}")
            if record["layer_index"] != requested_layer:
                raise CaptureContractError(f"layer identity mismatch at step {index} position {position}")
            dim = record["vector_dimension"]
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
                raise CaptureContractError(f"invalid vector_dimension at step {index} layer {requested_layer}")
            prior_dim = stable_dimensions.setdefault(requested_layer, dim)
            if dim != prior_dim:
                raise CaptureContractError(f"layer {requested_layer} vector dimension changed from {prior_dim} to {dim}")
            observed_dtype = _require_nonempty_string(
                record["observed_dtype"],
                f"trajectory step {index} layer {requested_layer}.observed_dtype",
            )
            recorded_dtypes.setdefault(requested_layer, set()).add(observed_dtype)
            if _require_span(record["pool_span"], token_count=len(input_ids), where=f"trajectory step {index} layer {requested_layer}.pool_span") != expected_pool:
                raise CaptureContractError(f"pool span mismatch at step {index} layer {requested_layer}")
            vector = record["vector"]
            if not isinstance(vector, list) or len(vector) != dim or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in vector):
                raise CaptureContractError(f"vector content/dimension mismatch at step {index} layer {requested_layer}")
            if record["vector_sha256"] != sha256_json(vector):
                raise CaptureContractError(f"vector SHA-256 mismatch at step {index} layer {requested_layer}")
        if cfg["context_mode"] == "cumulative":
            previous_ids = input_ids

    actual_traj_sha = sha256_json(_without(trajectory, "trajectory_sha256"))
    if trajectory["trajectory_sha256"] != actual_traj_sha or artifacts["captured_trajectory_sha256"] != actual_traj_sha:
        raise CaptureContractError("captured trajectory SHA-256 is invalid")
    actual_manifest_sha = sha256_json(_without(manifest, "manifest_sha256"))
    if manifest["manifest_sha256"] != actual_manifest_sha:
        raise CaptureContractError("run manifest SHA-256 is invalid")
    if manifest["run_id"] != validated["run_id"] or trajectory["run_id"] != validated["run_id"]:
        raise CaptureContractError("manifest/trajectory run_id does not match capture request")
    identity = {key: manifest[key] for key in ("schema_version", "protocol_id", "run_id", "repository_commit", "request_sha256", "model", "backend_request", "backend_observed", "capture", "determinism", "generation_parameters")}
    actual_run_id = sha256_json(identity)
    if manifest["run_manifest_id"] != actual_run_id or trajectory["run_manifest_id"] != actual_run_id:
        raise CaptureContractError("run_manifest_id is invalid")
    for field in ("schema_version", "protocol_id", "repository_commit"):
        if trajectory[field] != manifest[field]:
            raise CaptureContractError(f"trajectory {field} does not match manifest")
    for field in ("model", "capture", "determinism", "generation_parameters"):
        if manifest[field] != validated[field]:
            raise CaptureContractError(f"manifest {field} does not match request")
    if manifest["backend_request"] != validated["backend"]:
        raise CaptureContractError("manifest backend_request does not match request")
    observed = _require_object(manifest["backend_observed"], "manifest backend_observed")
    _validate_backend_metadata(observed, validated, trajectory["evidence_class"])
    if trajectory["evidence_class"] == "OBSERVATION":
        _validate_observed_dtype_map(observed, recorded_dtypes)
    return validated

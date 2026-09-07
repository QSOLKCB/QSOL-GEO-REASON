"""Execution and capture-step construction for GEO-CAP-001."""
from __future__ import annotations
from typing import Any, Mapping
from .canonical import sha256_json
from .capture_backend import _ExplicitNoAttentionBaseModel
from .capture_backend_production import HuggingFacePyTorchBackend
from .capture_common import (CAPTURE_PROTOCOL_ID, CAPTURE_SCHEMA_VERSION, _ALLOWED_EVIDENCE, _CAPTURE_PHASE, _LAYER_INDEX_SEMANTICS, _SIMULATION_BACKEND, _STEP_SPAN_SEMANTICS, CaptureBackend, CaptureContractError, _common_prefix_length, _compose_text, _pool_span, _require_git_sha, _sha256_text, _validate_backend_layer, _validate_token_ids)
from .capture_dispatch import _MathSDPABaseModel
from .capture_execution_state import _callable_execution_identity
from .capture_validation import validate_capture_request
from .capture_provenance import _resolve_hidden_state_layout, _validate_backend_metadata
from .provenance import SourceIdentityError, resolve_implementation_revision


_OBSERVATION_BACKEND_EXECUTION_METHODS = (
    "assert_execution_request",
    "begin_observation",
    "end_observation",
    "tokenize",
    "hidden_states",
    "metadata",
)
_VOLATILE_CALLABLE_RECEIPT_FIELDS = frozenset(
    {"identity", "bound_self", "class_identity", "call_identity"}
)


def _unwrap_backend_descriptor(value: Any) -> Any:
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    return value


def _trusted_callable_receipt(value: Any) -> dict[str, Any]:
    """Return the stable executable part of a callable identity receipt.

    Python object addresses are process-local bookkeeping, not executable semantics.
    The canonical import-time baseline therefore binds the callable's module and
    qualified name, marshalled implementation code, callable implementation code,
    defaults and keyword defaults. Replacing a backend method with different code
    still fails closed, while an equivalent function object restored by a test or
    instrumentation transaction does not permanently poison the process.
    """
    receipt = dict(_callable_execution_identity(value))
    for field in _VOLATILE_CALLABLE_RECEIPT_FIELDS:
        receipt.pop(field, None)
    nested = receipt.get("partial_function")
    if isinstance(nested, Mapping):
        nested_copy = dict(nested)
        for field in _VOLATILE_CALLABLE_RECEIPT_FIELDS:
            nested_copy.pop(field, None)
        receipt["partial_function"] = nested_copy
    return receipt


def _freeze_observation_backend_callables() -> tuple[
    frozenset[str],
    tuple[tuple[type, str, Mapping[str, Any]], ...],
    tuple[tuple[str, Mapping[str, Any]], ...],
]:
    """Capture the trusted QSOL adapter callable surface at module import time."""
    names: set[str] = set()
    class_bindings: list[tuple[type, str, Mapping[str, Any]]] = []
    for cls in HuggingFacePyTorchBackend.__mro__:
        if not getattr(cls, "__module__", "").startswith("qsol_geo_reason"):
            continue
        for name, descriptor in vars(cls).items():
            target = _unwrap_backend_descriptor(descriptor)
            if not callable(target):
                continue
            names.add(name)
            class_bindings.append(
                (cls, name, _trusted_callable_receipt(target))
            )

    execution_bindings: list[tuple[str, Mapping[str, Any]]] = []
    for name in _OBSERVATION_BACKEND_EXECUTION_METHODS:
        target = _unwrap_backend_descriptor(getattr(HuggingFacePyTorchBackend, name, None))
        if not callable(target):
            raise RuntimeError(
                f"trusted canonical OBSERVATION backend is missing execution method {name!r}"
            )
        execution_bindings.append(
            (name, _trusted_callable_receipt(target))
        )
    return frozenset(names), tuple(class_bindings), tuple(execution_bindings)


(
    _OBSERVATION_BACKEND_CALLABLE_NAMES,
    _OBSERVATION_BACKEND_CLASS_CALLABLES,
    _OBSERVATION_BACKEND_EXECUTION_BASELINE,
) = _freeze_observation_backend_callables()
_TRUSTED_RESOLVE_HIDDEN_STATE_LAYOUT = _resolve_hidden_state_layout


def _assert_observation_backend_execution_methods(backend: HuggingFacePyTorchBackend) -> None:
    """Reject instance or class substitutions on the canonical adapter dispatch surface."""
    # The expected class implementation is an import-time executable receipt, not
    # a fresh lookup from the mutable class. This detects class monkey-patching and
    # in-place Python function code changes without depending on object addresses.
    for cls, name, expected_receipt in _OBSERVATION_BACKEND_CLASS_CALLABLES:
        current = _unwrap_backend_descriptor(vars(cls).get(name))
        if (
            not callable(current)
            or _trusted_callable_receipt(current) != dict(expected_receipt)
        ):
            raise CaptureContractError(
                "canonical OBSERVATION backend class execution method changed after trusted import: "
                f"{cls.__module__}.{cls.__qualname__}.{name}"
            )

    instance_state = vars(backend)
    shadowed = sorted(
        name for name in instance_state if name in _OBSERVATION_BACKEND_CALLABLE_NAMES
    )
    if shadowed:
        raise CaptureContractError(
            "canonical OBSERVATION backend execution methods cannot be overridden on the "
            "instance: " + ", ".join(shadowed)
        )

    # Keep explicit evidence-boundary checks for the public adapter methods, but
    # compare against the frozen import-time executable receipt rather than the
    # mutable class or a process-local function address.
    for name, expected_receipt in _OBSERVATION_BACKEND_EXECUTION_BASELINE:
        resolved = getattr(backend, name, None)
        resolved_target = getattr(resolved, "__func__", resolved)
        if (
            not callable(resolved)
            or _trusted_callable_receipt(resolved_target) != dict(expected_receipt)
        ):
            raise CaptureContractError(
                f"canonical OBSERVATION backend execution method {name!r} is not the trusted import-time implementation"
            )


def _assert_observation_backend_routing_state(backend: HuggingFacePyTorchBackend) -> None:
    """Bind non-callable adapter routing to the live authenticated model graph."""
    try:
        expected_base_model, expected_path, expected_blocks = (
            _TRUSTED_RESOLVE_HIDDEN_STATE_LAYOUT(backend._model)
        )
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError(
            "unable to authenticate canonical OBSERVATION adapter routing state"
        ) from exc

    if getattr(backend, "_block_path", None) != expected_path:
        raise CaptureContractError("canonical OBSERVATION decoder block path changed after construction")
    if getattr(backend, "_blocks", None) is not expected_blocks:
        raise CaptureContractError(
            "canonical OBSERVATION decoder block routing object changed after construction"
        )
    if getattr(backend, "_hidden_state_count", None) != len(expected_blocks) + 1:
        raise CaptureContractError("canonical OBSERVATION hidden-state routing count is invalid")

    # The policy facade always installs the no-attention wrapper. CPU/MPS SDPA
    # adds the math-only wrapper outside it; all other lanes use the former
    # directly. Bind exact wrapper classes, delegate identities, and backend owner.
    routed = getattr(backend, "_base_model", None)
    if (
        getattr(backend, "_device_type", None) in {"cpu", "mps"}
        and getattr(backend, "_attention_implementation", None) == "sdpa"
    ):
        if type(routed) is not _MathSDPABaseModel:
            raise CaptureContractError(
                "canonical OBSERVATION base-model SDPA routing wrapper changed after construction"
            )
        if getattr(routed, "_backend", None) is not backend:
            raise CaptureContractError(
                "canonical OBSERVATION SDPA routing wrapper is bound to the wrong backend"
            )
        routed = getattr(routed, "_module", None)

    if type(routed) is not _ExplicitNoAttentionBaseModel:
        raise CaptureContractError(
            "canonical OBSERVATION no-attention routing wrapper changed after construction"
        )
    routed = getattr(routed, "_module", None)
    if routed is not expected_base_model:
        raise CaptureContractError(
            "canonical OBSERVATION base-model routing identity changed after construction"
        )


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

    observation_started = False
    if evidence_class == "OBSERVATION":
        if type(backend) is not HuggingFacePyTorchBackend:
            raise CaptureContractError("OBSERVATION capture requires the concrete HuggingFacePyTorchBackend")
        _assert_observation_backend_execution_methods(backend)
        _assert_observation_backend_routing_state(backend)
        try:
            implementation_revision = resolve_implementation_revision(
                implementation_revision, require_checkout=True
            )
        except SourceIdentityError as exc:
            raise CaptureContractError(
                f"OBSERVATION implementation revision is not bound to the executing checkout: {exc}"
            ) from exc
        backend.assert_execution_request(validated)
        backend._observed_hidden_state_dtypes.clear()
        backend.begin_observation()
        observation_started = True

    try:
        steps, prefix_ids = _capture_steps(validated, backend)
        observed = dict(backend.metadata())
        if evidence_class == "SIMULATION":
            observed["name"] = _SIMULATION_BACKEND
        _validate_backend_metadata(observed, validated, evidence_class)
    finally:
        if observation_started:
            backend.end_observation()

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

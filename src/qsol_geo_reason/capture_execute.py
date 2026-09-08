"""Execution and capture-step construction for GEO-CAP-001."""
from __future__ import annotations
import functools
import inspect
import types
import weakref
from collections import deque
from typing import Any, Mapping
from .canonical import sha256_json
from .capture_backend import _ExplicitNoAttentionBaseModel
from .capture_backend_production import HuggingFacePyTorchBackend
from .capture_common import (CAPTURE_PROTOCOL_ID, CAPTURE_SCHEMA_VERSION, _ALLOWED_EVIDENCE, _CAPTURE_PHASE, _LAYER_INDEX_SEMANTICS, _SIMULATION_BACKEND, _STEP_SPAN_SEMANTICS, _TORCH_LONG_MAX, CaptureBackend, CaptureContractError, _common_prefix_length, _compose_text, _pool_span, _require_git_sha, _sha256_text, _validate_backend_layer, _validate_token_ids)
from .capture_dispatch import _MathSDPABaseModel, _global_paths
from .capture_validation import validate_capture_request
from .capture_provenance import _resolve_hidden_state_layout, _validate_backend_metadata
from .capture_runtime import _assert_torch_execution_surface
from .provenance import SourceIdentityError, resolve_implementation_revision


_OBSERVATION_BACKEND_EXECUTION_METHODS = (
    "assert_execution_request",
    "begin_observation",
    "end_observation",
    "tokenize",
    "hidden_states",
    "metadata",
)


def _unwrap_backend_descriptor(value: Any) -> Any:
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    return value


def _stable_code_constant(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return {"code": _stable_code_structure(value)}
    if isinstance(value, tuple):
        return {"tuple": [_stable_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_stable_code_constant(item) for item in value]
        return {"frozenset": sorted(items, key=repr)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _stable_code_structure(code: types.CodeType) -> dict[str, Any]:
    """Describe executable Python code without CPython adaptive/runtime state."""
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "consts": [_stable_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "name": code.co_name,
        "qualname": code.co_qualname,
        "firstlineno": code.co_firstlineno,
        "linetable": code.co_linetable.hex(),
        "exceptiontable": code.co_exceptiontable.hex(),
    }


def _trusted_callable_receipt(value: Any) -> dict[str, Any]:
    """Return a stable import-time executable receipt for a backend callable."""
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    call_impl = getattr(type(function), "__call__", None)
    call_code = getattr(call_impl, "__code__", None)
    return {
        "kind": "callable",
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
        "code_sha256": sha256_json(_stable_code_structure(code)) if isinstance(code, types.CodeType) else None,
        "call_module": getattr(call_impl, "__module__", None),
        "call_qualname": getattr(call_impl, "__qualname__", None),
        "call_code_sha256": sha256_json(_stable_code_structure(call_code)) if isinstance(call_code, types.CodeType) else None,
        "defaults": repr(getattr(function, "__defaults__", None)),
        "kwdefaults": repr(getattr(function, "__kwdefaults__", None)),
    }


def _trusted_execution_dependencies_sha256(roots: list[Any]) -> str:
    """Bind QSOL transitive Python globals/closures with stable executable receipts.

    QSOL-owned Python functions are followed transitively. External/stdlib callables
    remain content-bound leaf receipts rather than having their mutable module caches
    traversed. This authenticates the adapter code we ship without mistaking ordinary
    lazy state such as platform/pathlib caches for executable substitution.
    """
    pending: deque[types.FunctionType] = deque()
    queued: set[int] = set()
    nodes: dict[str, Any] = {}
    references = 0
    max_nodes, max_references = 4096, 32768

    def reference(value: Any, *, containers: frozenset[int] = frozenset()) -> Any:
        nonlocal references
        references += 1
        if references > max_references:
            raise CaptureContractError("trusted backend execution dependency reference limit exceeded")
        target = getattr(value, "__func__", value)
        if isinstance(target, types.FunctionType):
            module_name = getattr(target, "__module__", "")
            if module_name.startswith("qsol_geo_reason") and id(target) not in queued:
                if len(queued) >= max_nodes:
                    raise CaptureContractError("trusted backend execution dependency callable limit exceeded")
                queued.add(id(target))
                pending.append(target)
            return {"callable": _trusted_callable_receipt(target)}
        if isinstance(value, functools.partial):
            return {
                "partial": _trusted_callable_receipt(value),
                "function": reference(value.func),
            }
        if callable(value):
            return {"callable": _trusted_callable_receipt(value)}
        if isinstance(value, weakref.WeakKeyDictionary):
            # Closure-owned live trust vaults intentionally change as production
            # backends are constructed and collected. Their accessor functions are
            # authenticated above, but the mutable vault entries themselves are not
            # executable dependencies and must not poison the import-time receipt.
            return {
                "opaque_trust_anchor": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                )
            }
        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in containers:
                return {"cycle": True}
            if len(value) > max_references:
                raise CaptureContractError("trusted backend execution dependency registry limit exceeded")
            seen = containers | {id(value)}
            items = value.items() if isinstance(value, Mapping) else enumerate(value)
            return {
                "container_type": type(value).__qualname__,
                "entries": {
                    str(key): reference(item, containers=seen)
                    for key, item in items
                    if callable(item) or isinstance(item, (Mapping, list, tuple))
                },
            }
        if value is None or type(value) in (bool, int, float, str):
            return {"value": value}
        if isinstance(value, types.ModuleType):
            return {"module": value.__name__}
        return {
            "class_module": type(value).__module__,
            "class_qualname": type(value).__qualname__,
        }

    root_receipts = [reference(root) for root in roots]
    while pending:
        function = pending.popleft()
        globals_map = function.__globals__
        builtins_map = function.__builtins__
        if not isinstance(builtins_map, Mapping):
            builtins_map = vars(builtins_map)
        edges: dict[str, Any] = {}
        for path in sorted(_global_paths(function.__code__)):
            name = path[0]
            if name in globals_map:
                value = globals_map[name]
            elif name in builtins_map:
                value = builtins_map[name]
            else:
                edges[".".join(path)] = {"missing": True}
                continue
            try:
                for component in path[1:]:
                    value = inspect.getattr_static(value, component)
            except AttributeError:
                edges[".".join(path)] = {"unresolved_static_attribute": True}
            else:
                edges[".".join(path)] = reference(value)
        closure: dict[str, Any] = {}
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                closure[name] = {"empty_cell": True}
            else:
                closure[name] = reference(value)
        receipt = _trusted_callable_receipt(function)
        node_key = (
            f"{receipt['module']}:{receipt['qualname']}:"
            f"{receipt['code_sha256']}"
        )
        nodes[node_key] = {
            "function": receipt,
            "globals": edges,
            "closure": closure,
        }
    return sha256_json({"roots": root_receipts, "dependencies": nodes})


def _freeze_observation_backend_callables() -> tuple[
    frozenset[str],
    tuple[tuple[type, str, Mapping[str, Any]], ...],
    tuple[tuple[str, Mapping[str, Any]], ...],
    str,
]:
    """Capture the trusted QSOL adapter callable surface at module import time."""
    names: set[str] = set()
    class_bindings: list[tuple[type, str, Mapping[str, Any]]] = []
    dependency_roots: list[Any] = []
    for cls in HuggingFacePyTorchBackend.__mro__:
        if not getattr(cls, "__module__", "").startswith("qsol_geo_reason"):
            continue
        for name, descriptor in vars(cls).items():
            target = _unwrap_backend_descriptor(descriptor)
            if not callable(target):
                continue
            names.add(name)
            class_bindings.append((cls, name, _trusted_callable_receipt(target)))
            dependency_roots.append(target)

    execution_bindings: list[tuple[str, Mapping[str, Any]]] = []
    for name in _OBSERVATION_BACKEND_EXECUTION_METHODS:
        target = _unwrap_backend_descriptor(getattr(HuggingFacePyTorchBackend, name, None))
        if not callable(target):
            raise RuntimeError(
                f"trusted canonical OBSERVATION backend is missing execution method {name!r}"
            )
        execution_bindings.append((name, _trusted_callable_receipt(target)))
    return (
        frozenset(names),
        tuple(class_bindings),
        tuple(execution_bindings),
        _trusted_execution_dependencies_sha256(dependency_roots),
    )


(
    _OBSERVATION_BACKEND_CALLABLE_NAMES,
    _OBSERVATION_BACKEND_CLASS_CALLABLES,
    _OBSERVATION_BACKEND_EXECUTION_BASELINE,
    _OBSERVATION_BACKEND_DEPENDENCY_BASELINE,
) = _freeze_observation_backend_callables()
_TRUSTED_RESOLVE_HIDDEN_STATE_LAYOUT = _resolve_hidden_state_layout


def _assert_observation_backend_execution_methods(backend: HuggingFacePyTorchBackend) -> None:
    """Reject instance, class, or global substitutions on the canonical adapter surface."""
    current_roots: list[Any] = []
    for cls, name, expected_receipt in _OBSERVATION_BACKEND_CLASS_CALLABLES:
        current = _unwrap_backend_descriptor(vars(cls).get(name))
        if not callable(current) or _trusted_callable_receipt(current) != dict(expected_receipt):
            raise CaptureContractError(
                "canonical OBSERVATION backend class execution method changed after trusted import: "
                f"{cls.__module__}.{cls.__qualname__}.{name}"
            )
        current_roots.append(current)

    if _trusted_execution_dependencies_sha256(current_roots) != _OBSERVATION_BACKEND_DEPENDENCY_BASELINE:
        raise CaptureContractError(
            "canonical OBSERVATION backend execution dependencies changed after trusted import"
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

    for name, expected_receipt in _OBSERVATION_BACKEND_EXECUTION_BASELINE:
        resolved = getattr(backend, name, None)
        resolved_target = getattr(resolved, "__func__", resolved)
        if not callable(resolved) or _trusted_callable_receipt(resolved_target) != dict(expected_receipt):
            raise CaptureContractError(
                f"canonical OBSERVATION backend execution method {name!r} is not the trusted import-time implementation"
            )


def _assert_observation_backend_routing_state(backend: HuggingFacePyTorchBackend) -> None:
    """Bind non-callable adapter routing to the live authenticated model graph."""
    _assert_torch_execution_surface(getattr(backend, "_torch", None))
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


def _capture_steps(
    request: Mapping[str, Any], backend: CaptureBackend, *, max_token_id: int | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    cfg = request["capture"]
    pooling = cfg["pooling"]
    prefix_ids = _validate_token_ids(
        backend.tokenize(cfg["prefix_text"]), "prefix_text",
        allow_empty=not bool(cfg["prefix_text"]), max_value=max_token_id,
    )
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
        input_ids = _validate_token_ids(
            backend.tokenize(rendered), f"step {step['step_id']!r}", max_value=max_token_id
        )
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

    production_backend_instance = isinstance(backend, HuggingFacePyTorchBackend)
    production_backend = type(backend) is HuggingFacePyTorchBackend
    if production_backend_instance and evidence_class != "OBSERVATION":
        raise CaptureContractError(
            "HuggingFacePyTorchBackend instances, including subclasses, may execute "
            "only as OBSERVATION; use a software simulation backend for SIMULATION"
        )

    if evidence_class == "OBSERVATION":
        if not production_backend:
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

    try:
        if evidence_class == "OBSERVATION":
            backend.begin_observation()
            # begin_observation() owns the exclusive Python-thread boundary. Repeat
            # adapter and routing authentication only after that exclusion is held,
            # so a short-lived pre-boundary mutator cannot race the first tokenization.
            _assert_observation_backend_execution_methods(backend)
            _assert_observation_backend_routing_state(backend)
        steps, prefix_ids = _capture_steps(
            validated, backend,
            max_token_id=_TORCH_LONG_MAX if evidence_class == "OBSERVATION" else None,
        )
        observed = dict(backend.metadata())
        if evidence_class == "SIMULATION":
            observed["name"] = _SIMULATION_BACKEND
        _validate_backend_metadata(observed, validated, evidence_class)
    finally:
        if production_backend and (
            getattr(backend, "_observation_active", False)
            or getattr(backend, "_exclusive_thread_boundary_state", None) is not None
        ):
            # Also cover a partially entered production boundary whose thread-start
            # patches are active even though inherited observation activation has not
            # completed yet. end_observation() is idempotent for the inactive parent
            # session and always releases the production thread boundary in finally.
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
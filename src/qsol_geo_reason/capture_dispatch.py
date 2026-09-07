"""Dispatch-policy receipts and bounded Python execution-dependency authentication.

These receipts describe the live instrument, not a sandbox against arbitrary native
code. No model tensor storage is read by execution-dependency authentication.
"""
from __future__ import annotations

import dis
import functools
import inspect
import types
from collections import deque
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .capture_execution_state import _callable_execution_identity

_CPU_CAPABILITIES = frozenset({"DEFAULT", "VSX", "Z VECTOR", "NO AVX", "AVX2", "AVX512", "SVE256"})


def _fp16_accumulation_state(torch: Any) -> bool | None:
    matmul = getattr(getattr(getattr(torch, "backends", None), "cuda", None), "matmul", None)
    try:
        value = getattr(matmul, "allow_fp16_accumulation")
    except AttributeError:
        return None
    except Exception as exc:
        raise CaptureContractError("unable to inspect CUDA FP16 accumulation policy") from exc
    if type(value) is not bool:
        raise CaptureContractError("CUDA allow_fp16_accumulation must be boolean when exposed")
    return value


def _effective_cpu_capability(torch: Any) -> str:
    cpu = getattr(getattr(torch, "backends", None), "cpu", None)
    getter = getattr(cpu, "get_cpu_capability", None)
    if not callable(getter):
        raise CaptureContractError("canonical CPU capture requires effective ATen CPU capability")
    try:
        capability = getter()
    except Exception as exc:
        raise CaptureContractError("unable to establish effective ATen CPU capability") from exc
    if type(capability) is not str or capability not in _CPU_CAPABILITIES:
        raise CaptureContractError("effective ATen CPU capability is not a supported dispatch identity")
    return capability


def _validate_dispatch_metadata(observed: Mapping[str, Any], device: str) -> None:
    accumulation = observed.get("cuda_matmul_allow_fp16_accumulation")
    if accumulation is not None and accumulation is not False:
        raise CaptureContractError("canonical CUDA FP16 accumulation must be false or unavailable/null")
    if not device.startswith("cuda:") and accumulation is not None:
        raise CaptureContractError("CUDA FP16 accumulation must be null outside CUDA")

    # Every canonical device performs final pooling on CPU in float64, so the
    # effective ATen CPU dispatch lane is material execution provenance for CPU,
    # CUDA, and MPS observations alike.
    capability = observed.get("cpu_aten_capability")
    environment = observed.get("aten_cpu_capability_env")
    known = observed.get("aten_cpu_capability_env_known")
    if type(capability) is not str or capability not in _CPU_CAPABILITIES:
        raise CaptureContractError(
            "canonical production provenance requires effective ATen CPU capability for CPU pooling"
        )
    if type(known) is not bool:
        raise CaptureContractError("aten_cpu_capability_env_known must be boolean for CPU pooling provenance")
    if environment is not None and type(environment) is not str:
        raise CaptureContractError("aten_cpu_capability_env must be string or null")
    if not known and environment is not None:
        raise CaptureContractError("unknown initialization-time ATen override must be null")

    if device == "mps":
        identity = (observed.get("mps_mac_model"), observed.get("mps_cpu_brand"))
        if not any(type(value) is str and value.strip() for value in identity):
            raise CaptureContractError(
                "canonical MPS provenance requires a concrete non-whitespace Mac model or Apple chip/CPU identity"
            )


class _MathSDPABaseModel:
    """Force and verify math SDPA at the actual CPU/MPS base-model call boundary."""

    def __init__(self, module: Any, backend: Any):
        self._module = module
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        backend = self._backend
        # The namespace is named cuda, but these are the global SDPA selector
        # controls also used by CPU/MPS. Existing ambient-state cleanup owns them.
        backend._force_sdpa_math_policy()
        result = self._module(*args, **kwargs)
        backend._last_sdpa_policy = backend._assert_sdpa_math_policy()
        return result


def _global_paths(code: types.CodeType) -> set[tuple[str, ...]]:
    """Find globals and statically named attribute chains, including nested code."""
    result: set[tuple[str, ...]] = set()
    codes = [code]
    while codes:
        current = codes.pop()
        instructions = list(dis.get_instructions(current))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
                continue
            path = [str(instruction.argval)]
            result.add(tuple(path))
            for following in instructions[index + 1:]:
                if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                path.append(str(following.argval))
                result.add(tuple(path))
        codes.extend(item for item in current.co_consts if isinstance(item, types.CodeType))
    return result


def _execution_dependencies_sha256(roots: Sequence[Any]) -> str:
    """Bind referenced global/qualified/closure callables, with cycle/work limits.

    Python functions are traversed transitively; native callables are identity-bound
    leaves. Module dictionaries are never swept wholesale, and descriptors are not
    invoked to discover dependencies. Callable registries are explicitly bound.
    """
    pending: deque[Any] = deque()
    queued: set[int] = set()
    nodes: dict[str, Any] = {}
    references = 0
    max_nodes, max_references = 4096, 32768

    def reference(value: Any, *, containers: frozenset[int] = frozenset()) -> Any:
        nonlocal references
        references += 1
        if references > max_references:
            raise CaptureContractError("execution dependency reference limit exceeded")
        target = getattr(value, "__func__", value)
        if isinstance(target, types.FunctionType):
            if id(target) not in queued:
                if len(queued) >= max_nodes:
                    raise CaptureContractError("execution dependency callable limit exceeded")
                queued.add(id(target))
                pending.append(target)
        if isinstance(value, functools.partial):
            return {
                "callable": _callable_execution_identity(value),
                "function": reference(value.func),
            }
        if callable(value):
            return _callable_execution_identity(value)
        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in containers:
                return {"cycle": id(value)}
            if len(value) > max_references:
                raise CaptureContractError("execution dependency registry limit exceeded")
            seen = containers | {id(value)}
            items = value.items() if isinstance(value, Mapping) else enumerate(value)
            return {
                "container": id(value),
                "callables": {
                    str(key): reference(item, containers=seen)
                    for key, item in items
                    if callable(item) or isinstance(item, (Mapping, list, tuple))
                },
            }
        # Scalar identity is stable by value. Non-callable objects/modules are
        # identity-bound without traversing model tensors or calling custom repr.
        if value is None or type(value) in (bool, int, float, str):
            return {"value": value}
        return {"identity": id(value), "class_identity": id(type(value))}

    root_receipts = [reference(root) for root in roots]
    while pending:
        function = pending.popleft()
        globals_map = function.__globals__
        builtins_map = function.__builtins__
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
                # A runtime-only/lazy attribute is not silently guessed. Its
                # owner identity is still bound by the shorter path receipt.
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
        nodes[str(id(function))] = {
            "function": _callable_execution_identity(function),
            "globals": edges,
            "closure": closure,
        }
    return sha256_json({"roots": root_receipts, "dependencies": nodes})


def _model_execution_dependency_roots(model: Any) -> list[Any]:
    """Bind model forwards, call dispatch, and self-resolved Python helpers."""
    roots: list[Any] = []
    for _name, module in model.named_modules():
        # nn.Module.__call__ dynamically enters _wrapped_call_impl/_call_impl before
        # forward. Bind the effective inherited dispatch functions explicitly so a
        # class- or process-level call-path substitution changes the executable seal.
        for dispatch_name in ("__call__", "_wrapped_call_impl", "_call_impl"):
            try:
                dispatch = inspect.getattr_static(module, dispatch_name)
            except AttributeError:
                continue
            if isinstance(dispatch, (staticmethod, classmethod)):
                dispatch = dispatch.__func__
            if callable(dispatch):
                roots.append(dispatch)
                if len(roots) > 4096:
                    raise CaptureContractError("model execution dependency root limit exceeded")

        pending = [getattr(module, "forward", None)]
        seen: set[int] = set()
        while pending:
            value = pending.pop()
            target = getattr(value, "__func__", value)
            if not callable(target) or id(target) in seen:
                continue
            seen.add(id(target))
            roots.append(value)
            if len(roots) > 4096:
                raise CaptureContractError("model execution dependency root limit exceeded")
            code = getattr(target, "__code__", None)
            if not isinstance(code, types.CodeType):
                continue
            for name in code.co_names:
                try:
                    helper = inspect.getattr_static(module, name)
                except AttributeError:
                    continue
                if isinstance(helper, (staticmethod, classmethod)):
                    helper = helper.__func__
                if isinstance(helper, (types.FunctionType, types.MethodType, functools.partial)):
                    pending.append(helper)
        # Instance-level overrides can contain their own global dependencies.
        roots.extend(value for value in vars(module).values()
                     if isinstance(value, (types.FunctionType, types.MethodType, functools.partial)))
        if len(roots) > 4096:
            raise CaptureContractError("model execution dependency root limit exceeded")
    return roots

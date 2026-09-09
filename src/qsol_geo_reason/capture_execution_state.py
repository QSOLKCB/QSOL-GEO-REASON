"""Small, inspectable execution-state receipts for canonical capture."""
from __future__ import annotations

import _thread
import functools
import hashlib
import marshal
import sys
import threading
from typing import Any, Mapping

from .capture_common import CaptureContractError


def _assert_exclusive_interpreter_thread() -> None:
    """Include raw _thread workers, not only threading's opt-in registry.

    This is a CPython/GIL in-process precondition, not an OS/native-code sandbox.
    Call while new thread starts are blocked by the production boundary.
    """
    frames = getattr(sys, "_current_frames", None)
    count_threads = getattr(_thread, "_count", None)
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if (
        sys.implementation.name != "cpython"
        or not callable(frames)
        or not callable(count_threads)
        or (callable(gil_enabled) and gil_enabled() is not True)
    ):
        raise CaptureContractError(
            "canonical OBSERVATION requires GIL-enabled CPython thread-state inspection"
        )
    current = _thread.get_ident()
    if current != threading.main_thread().ident:
        raise CaptureContractError(
            "canonical OBSERVATION requires exclusive Python-thread execution on the main thread"
        )
    try:
        frame_ids = set(frames())
        workers = count_threads()
    except Exception as exc:
        raise CaptureContractError("unable to inspect active CPython thread states") from exc
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise CaptureContractError("invalid CPython worker-thread count")
    if frame_ids != {current} or workers != 0:
        raise CaptureContractError(
            "canonical OBSERVATION requires exclusive Python-thread execution; "
            f"other_interpreter_threads={len(frame_ids - {current})} raw_or_managed_workers={workers}"
        )


def _callable_execution_identity(value: Any) -> Mapping[str, Any]:
    """Bind callable helpers without invoking them or relying on their repr alone."""
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    call_impl = getattr(type(function), "__call__", None)
    call_code = getattr(call_impl, "__code__", None)
    identity: dict[str, Any] = {
        "kind": "callable",
        "identity": id(function),
        "bound_self": id(getattr(value, "__self__", None)),
        "class_identity": id(type(function)),
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
        "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else None,
        "call_identity": id(call_impl),
        "call_code_sha256": hashlib.sha256(marshal.dumps(call_code)).hexdigest() if call_code is not None else None,
        "defaults": repr(getattr(function, "__defaults__", None)),
        "kwdefaults": repr(getattr(function, "__kwdefaults__", None)),
    }
    if isinstance(value, functools.partial):
        identity["partial_function"] = _callable_execution_identity(value.func)
        identity["partial_args"] = repr(value.args)
        identity["partial_keywords"] = repr(value.keywords)
    return identity


def _cudnn_algorithm_policy_state(torch: Any) -> dict[str, bool]:
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    result: dict[str, bool] = {}
    for name in ("benchmark", "deterministic"):
        value = getattr(cudnn, name, None)
        if not isinstance(value, bool):
            raise CaptureContractError(f"canonical CUDA capture requires boolean torch.backends.cudnn.{name}")
        result[f"cudnn_{name}"] = value
    return result

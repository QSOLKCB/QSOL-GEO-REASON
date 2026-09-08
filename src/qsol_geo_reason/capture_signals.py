"""Fail-closed asynchronous-signal policy for canonical in-process observations."""
from __future__ import annotations

import signal
from typing import Any

from .capture_common import CaptureContractError


def _allowed_handler(signum: Any, handler: Any) -> bool:
    if handler in (signal.SIG_DFL, signal.SIG_IGN):
        return True
    # CPython installs this ordinary default KeyboardInterrupt handler for SIGINT.
    return (
        getattr(signal, "SIGINT", None) == signum
        and handler is getattr(signal, "default_int_handler", None)
    )


def _assert_no_async_signal_instrumentation() -> None:
    """Reject custom handlers and armed interval timers during OBSERVATION.

    Python signal callbacks execute on the main interpreter thread and are therefore
    outside the thread-exclusion boundary.  A canonical in-process observation only
    proceeds when no custom asynchronous Python handler or interval timer can run in
    the tokenization/forward/pooling window.
    """
    valid_signals = getattr(signal, "valid_signals", None)
    if not callable(valid_signals):
        raise CaptureContractError(
            "canonical OBSERVATION cannot inspect the process signal-handler surface"
        )
    try:
        candidates = sorted(valid_signals(), key=int)
    except Exception as exc:
        raise CaptureContractError(
            "canonical OBSERVATION cannot enumerate process signal handlers"
        ) from exc

    uncatchable = {
        value
        for value in (getattr(signal, "SIGKILL", None), getattr(signal, "SIGSTOP", None))
        if value is not None
    }
    for signum in candidates:
        if signum in uncatchable:
            continue
        try:
            handler = signal.getsignal(signum)
        except (OSError, ValueError):
            # Some platforms expose numbers in valid_signals that Python cannot query.
            continue
        except Exception as exc:
            raise CaptureContractError(
                f"unable to inspect signal handler for {signum!r}"
            ) from exc
        if not _allowed_handler(signum, handler):
            raise CaptureContractError(
                f"canonical OBSERVATION forbids custom asynchronous signal handler for {signum!r}"
            )

    getitimer = getattr(signal, "getitimer", None)
    if callable(getitimer):
        for name in ("ITIMER_REAL", "ITIMER_VIRTUAL", "ITIMER_PROF"):
            timer = getattr(signal, name, None)
            if timer is None:
                continue
            try:
                remaining, interval = getitimer(timer)
            except (OSError, ValueError):
                continue
            except Exception as exc:
                raise CaptureContractError(
                    f"unable to inspect asynchronous interval timer {name}"
                ) from exc
            if remaining > 0.0 or interval > 0.0:
                raise CaptureContractError(
                    f"canonical OBSERVATION forbids armed asynchronous interval timer {name}"
                )


__all__ = ["_assert_no_async_signal_instrumentation"]

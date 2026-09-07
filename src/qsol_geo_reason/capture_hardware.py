"""Concrete hardware identity helpers for canonical capture provenance."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .capture_common import CaptureContractError

_GENERIC_CPU_IDENTITIES = frozenset(
    {
        "",
        "unknown",
        "generic",
        "cpu",
        "processor",
        "x86_64",
        "amd64",
        "x64",
        "i386",
        "i686",
        "arm",
        "arm64",
        "aarch64",
        "ppc64",
        "ppc64le",
        "s390x",
    }
)


def _is_concrete_cpu_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() not in _GENERIC_CPU_IDENTITIES
    )


def _read_linux_cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for block in text.split("\n\n"):
        fields = {
            line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
            for line in block.splitlines()
            if ":" in line
        }
        for key in ("model name", "Processor", "Hardware", "cpu model"):
            value = fields.get(key)
            if _is_concrete_cpu_identity(value):
                return value.strip()
    return None


def _sysctl_cpu_model() -> str | None:
    if platform.system() != "Darwin":
        return None
    for name in ("machdep.cpu.brand_string", "hw.model"):
        try:
            completed = subprocess.run(
                ["sysctl", "-n", name],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        value = completed.stdout.strip()
        if completed.returncode == 0 and _is_concrete_cpu_identity(value):
            return value
    return None


def _concrete_cpu_identity() -> str:
    """Return a non-generic CPU model identity or fail closed."""
    candidates = (
        _read_linux_cpu_model(),
        _sysctl_cpu_model(),
        platform.processor() or None,
        platform.uname().processor or None,
        os.environ.get("PROCESSOR_IDENTIFIER"),
    )
    for value in candidates:
        if _is_concrete_cpu_identity(value):
            return value.strip()
    raise CaptureContractError(
        "canonical CPU observation requires a concrete processor model identity"
    )


__all__ = ["_concrete_cpu_identity", "_is_concrete_cpu_identity"]

"""Verify the complete Phase 2A Python 3.11 CPU reference runtime lock.

Reject missing, mismatched, and unexpected runtime distributions as well as any
interpreter/platform outside the selected CPython 3.11 Linux x86_64 lane.
"""
from __future__ import annotations

import json
import platform
import sys

from qsol_geo_reason import capture_reference_environment as _reference
from qsol_geo_reason.capture_common import CaptureContractError


LOCK = _reference.LOCK


def _locked_versions():
    return _reference._locked_versions()


def _installed_runtime_versions():
    return _reference._installed_runtime_versions()


def _current_python_version() -> tuple[int, int, int]:
    value = sys.version_info
    return value.major, value.minor, value.micro


def _current_python_implementation() -> str:
    return platform.python_implementation()


def _current_platform_system() -> str:
    return platform.system()


def _current_platform_machine() -> str:
    return platform.machine()


def verify_reference_environment() -> dict[str, object]:
    try:
        return _reference._build_reference_environment_receipt_from_state(
            locked=_locked_versions(),
            installed=_installed_runtime_versions(),
            python_version=_current_python_version(),
            python_implementation=_current_python_implementation(),
            platform_system=_current_platform_system(),
            platform_machine=_current_platform_machine(),
            lock_sha256=_reference._lock_sha256(),
        )
    except CaptureContractError as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> int:
    print(json.dumps(verify_reference_environment(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

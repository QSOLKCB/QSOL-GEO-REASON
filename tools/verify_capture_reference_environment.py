"""Verify the complete Phase 2A Python 3.11 CPU reference runtime lock.

Canonical lock: constraints/capture-reference-py311.txt.
Reject missing, mismatched, and unexpected runtime distributions as well as any
interpreter/platform outside the selected final CPython 3.11 Linux x86_64 lane. The
receipt content-binds the exact importable Hugging Face Hub package tree, its complete
locked execution dependency closure, the direct Torch/Transformers/Tokenizers/
Safetensors capture packages, and every remaining capture-time transitive executable
package in the frozen reference closure.

This verifier is intentionally bytecode-write-free. Canonical production rejects any
``src/qsol_geo_reason`` bytecode before import, so the preflight must not dirty a clean
checkout merely by importing the package it verifies. The documented invocation also
uses ``python -B`` as defense in depth.
"""
from __future__ import annotations

import json
import platform
import sys

# This must precede every qsol_geo_reason import. ``-B`` is also documented for the
# operator/CI path, but keeping the script self-protecting prevents a plain invocation
# from creating source-tree __pycache__ files that the production launcher must reject.
sys.dont_write_bytecode = True

from qsol_geo_reason import capture_reference_environment as _reference
from qsol_geo_reason.capture_common import CaptureContractError


LOCK = _reference.LOCK


def _locked_versions():
    return _reference._locked_versions()


def _installed_runtime_versions():
    return _reference._installed_runtime_versions()


def _current_hub_package_provenance():
    return _reference._huggingface_hub_package_provenance()


def _current_hub_transport_package_provenance():
    return _reference._hub_transport_package_provenance()


def _current_capture_direct_package_provenance():
    return _reference._capture_direct_package_provenance()


def _current_capture_transitive_package_provenance():
    return _reference._capture_transitive_package_provenance()


def _current_python_version() -> tuple[int, int, int]:
    value = sys.version_info
    return value.major, value.minor, value.micro


def _current_python_releaselevel() -> str:
    return sys.version_info.releaselevel


def _current_python_serial() -> int:
    return sys.version_info.serial


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
            hub_package_provenance=_current_hub_package_provenance(),
            hub_transport_package_provenance=_current_hub_transport_package_provenance(),
            capture_direct_package_provenance=_current_capture_direct_package_provenance(),
            capture_transitive_package_provenance=_current_capture_transitive_package_provenance(),
            python_releaselevel=_current_python_releaselevel(),
            python_serial=_current_python_serial(),
            lock_sha256=_reference._lock_sha256(),
        )
    except CaptureContractError as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> int:
    print(json.dumps(verify_reference_environment(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
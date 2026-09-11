"""Deterministic archived Python 3.11 reference-environment receipt for unit fixtures."""
from __future__ import annotations

from qsol_geo_reason import capture_reference_environment as reference


def reference_environment_receipt() -> dict:
    locked = reference._locked_versions()
    return reference._build_reference_environment_receipt_from_state(
        locked=locked,
        installed=dict(locked),
        python_version=(3, 11, 16),
        python_implementation="CPython",
        platform_system="Linux",
        platform_machine="x86_64",
        lock_sha256=reference._lock_sha256(),
    )

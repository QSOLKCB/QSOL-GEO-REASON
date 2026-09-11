"""Deterministic archived Python 3.11 reference-environment receipt for unit fixtures."""
from __future__ import annotations

from qsol_geo_reason import capture_reference_environment as reference


def hub_package_provenance() -> dict[str, object]:
    return {
        "file_count": 137,
        "receipt_sha256": "7" * 64,
    }


def reference_environment_receipt() -> dict:
    locked = reference._locked_versions()
    return reference._build_reference_environment_receipt_from_state(
        locked=locked,
        installed=dict(locked),
        python_version=(3, 11, 16),
        python_implementation="CPython",
        platform_system="Linux",
        platform_machine="x86_64",
        hub_package_provenance=hub_package_provenance(),
        lock_sha256=reference._lock_sha256(),
    )

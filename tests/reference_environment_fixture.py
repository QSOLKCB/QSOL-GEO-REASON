"""Deterministic archived Python 3.11 reference-environment receipt for unit fixtures."""
from __future__ import annotations

from qsol_geo_reason import capture_reference_environment as reference


def hub_package_provenance() -> dict[str, object]:
    return {
        "file_count": 137,
        "receipt_sha256": "7" * 64,
    }


def _package_map(names: list[str], *, offset: int) -> dict[str, dict[str, object]]:
    return {
        canonical: {
            "file_count": offset + index + 1,
            "receipt_sha256": f"{offset + index + 1:064x}",
        }
        for index, canonical in enumerate(names)
    }


def hub_transport_package_provenance() -> dict[str, dict[str, object]]:
    return _package_map(sorted(reference.HUB_TRANSPORT_PACKAGE_IMPORTS), offset=100)


def capture_direct_package_provenance() -> dict[str, dict[str, object]]:
    return _package_map(sorted(reference.CAPTURE_DIRECT_PACKAGE_IMPORTS), offset=200)


def capture_transitive_package_provenance() -> dict[str, dict[str, object]]:
    return _package_map(sorted(reference.CAPTURE_TRANSITIVE_PACKAGE_IMPORTS), offset=300)


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
        hub_transport_package_provenance=hub_transport_package_provenance(),
        capture_direct_package_provenance=capture_direct_package_provenance(),
        capture_transitive_package_provenance=capture_transitive_package_provenance(),
        lock_sha256=reference._lock_sha256(),
    )

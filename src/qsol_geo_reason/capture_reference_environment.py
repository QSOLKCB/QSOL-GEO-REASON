"""Machine-verifiable Phase 2A reference-environment receipts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
import re
import sys
import types
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .capture_package import _python_package_provenance


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "constraints" / "capture-reference-py311.txt"
REFERENCE_ENVIRONMENT_SCHEMA_VERSION = "1.1.0"
REFERENCE_LANE = "capture-reference-py311-linux-x86_64-cpu"
_BOOTSTRAP_OR_PROJECT = frozenset({"pip", "setuptools", "wheel", "qsol-geo-reason"})
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "lane",
        "status",
        "python_implementation",
        "python_version",
        "platform_system",
        "platform_machine",
        "lock_sha256",
        "distribution_count",
        "distributions",
        "huggingface_hub_package_file_count",
        "huggingface_hub_package_receipt_sha256",
        "environment_receipt_sha256",
    }
)


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_versions() -> dict[str, tuple[str, str]]:
    locked: dict[str, tuple[str, str]] = {}
    try:
        lines = LOCK.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CaptureContractError(f"unable to read reference runtime lock: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise CaptureContractError(
                f"reference lock line {line_number} must be one exact name==version pin"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise CaptureContractError(f"reference lock line {line_number} is malformed")
        canonical = _canonical_name(name)
        if canonical in locked:
            raise CaptureContractError(
                f"reference lock contains duplicate distribution {name}"
            )
        locked[canonical] = (name, version)
    if not locked:
        raise CaptureContractError("reference lock contains no runtime distributions")
    return locked


def _installed_runtime_versions() -> dict[str, tuple[str, str]]:
    installed: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        canonical = _canonical_name(name)
        if canonical in _BOOTSTRAP_OR_PROJECT:
            continue
        version = distribution.version
        previous = installed.get(canonical)
        if previous is not None and previous != (name, version):
            raise CaptureContractError(
                "multiple installed distributions normalize to the same name: "
                f"{previous[0]} and {name}"
            )
        installed[canonical] = (name, version)
    return installed


def _lock_sha256() -> str:
    try:
        return hashlib.sha256(LOCK.read_bytes()).hexdigest()
    except OSError as exc:
        raise CaptureContractError(f"unable to hash reference runtime lock: {exc}") from exc


def _canonical_machine(value: str) -> str:
    lowered = value.strip().lower()
    return "x86_64" if lowered in {"x86_64", "amd64"} else lowered


def _huggingface_hub_package_provenance() -> dict[str, Any]:
    """Content-bind the importable Hub package tree without importing package code."""
    try:
        spec = importlib.util.find_spec("huggingface_hub")
    except (ImportError, AttributeError, ValueError) as exc:
        raise CaptureContractError(
            "reference lane cannot locate the locked huggingface-hub package"
        ) from exc
    if spec is None or not isinstance(spec.origin, str) or not spec.origin.strip():
        raise CaptureContractError(
            "reference lane cannot locate the locked huggingface-hub package"
        )
    probe = types.SimpleNamespace(__file__=spec.origin)
    return _python_package_provenance(probe, "Hugging Face Hub reference environment")


def _validate_hub_package_provenance(
    provenance: Mapping[str, Any],
) -> tuple[int, str]:
    if not isinstance(provenance, Mapping):
        raise CaptureContractError("Hugging Face Hub package provenance must be an object")
    count = provenance.get("file_count")
    receipt = provenance.get("receipt_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError(
            "Hugging Face Hub package provenance file_count must be a positive integer"
        )
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(ch not in "0123456789abcdef" for ch in receipt)
    ):
        raise CaptureContractError(
            "Hugging Face Hub package provenance receipt_sha256 must be lowercase 64-hex"
        )
    return count, receipt


def _require_reference_platform(
    *,
    python_version: tuple[int, int, int],
    python_implementation: str,
    platform_system: str,
    platform_machine: str,
) -> tuple[str, str]:
    if python_implementation != "CPython":
        raise CaptureContractError(
            f"reference lane requires CPython, observed {python_implementation or '<empty>'}"
        )
    if python_version[:2] != (3, 11):
        observed = ".".join(str(part) for part in python_version)
        raise CaptureContractError(
            f"reference lane requires Python 3.11, observed {observed}"
        )
    if platform_system != "Linux":
        raise CaptureContractError(
            f"reference lane requires Linux, observed {platform_system or '<empty>'}"
        )
    machine = _canonical_machine(platform_machine)
    if machine != "x86_64":
        raise CaptureContractError(
            f"reference lane requires x86_64, observed {platform_machine or '<empty>'}"
        )
    return ".".join(str(part) for part in python_version), machine


def _build_reference_environment_receipt_from_state(
    *,
    locked: Mapping[str, tuple[str, str]],
    installed: Mapping[str, tuple[str, str]],
    python_version: tuple[int, int, int],
    python_implementation: str,
    platform_system: str,
    platform_machine: str,
    hub_package_provenance: Mapping[str, Any],
    lock_sha256: str | None = None,
) -> dict[str, Any]:
    version_text, machine = _require_reference_platform(
        python_version=python_version,
        python_implementation=python_implementation,
        platform_system=platform_system,
        platform_machine=platform_machine,
    )
    hub_file_count, hub_receipt_sha256 = _validate_hub_package_provenance(
        hub_package_provenance
    )

    missing = sorted(set(locked) - set(installed))
    unexpected = sorted(set(installed) - set(locked))
    mismatched = sorted(
        canonical
        for canonical in set(locked) & set(installed)
        if installed[canonical][1] != locked[canonical][1]
    )
    if missing or unexpected or mismatched:
        details: list[str] = []
        if missing:
            details.append(
                "missing="
                + ",".join(
                    f"{locked[name][0]}=={locked[name][1]}" for name in missing
                )
            )
        if unexpected:
            details.append(
                "unexpected="
                + ",".join(
                    f"{installed[name][0]}=={installed[name][1]}" for name in unexpected
                )
            )
        if mismatched:
            details.append(
                "mismatched="
                + ",".join(
                    f"{locked[name][0]} expected {locked[name][1]} observed {installed[name][1]}"
                    for name in mismatched
                )
            )
        raise CaptureContractError(
            "reference runtime does not match complete lock: " + "; ".join(details)
        )

    distributions = {
        canonical: locked[canonical][1]
        for canonical in sorted(locked)
    }
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_ENVIRONMENT_SCHEMA_VERSION,
        "lane": REFERENCE_LANE,
        "status": "locked",
        "python_implementation": python_implementation,
        "python_version": version_text,
        "platform_system": platform_system,
        "platform_machine": machine,
        "lock_sha256": lock_sha256 if lock_sha256 is not None else _lock_sha256(),
        "distribution_count": len(distributions),
        "distributions": distributions,
        "huggingface_hub_package_file_count": hub_file_count,
        "huggingface_hub_package_receipt_sha256": hub_receipt_sha256,
    }
    payload["environment_receipt_sha256"] = sha256_json(payload)
    return payload


def build_reference_environment_receipt() -> dict[str, Any]:
    """Measure the exact interpreter, package tree, and locked runtime used here."""
    version = sys.version_info
    return _build_reference_environment_receipt_from_state(
        locked=_locked_versions(),
        installed=_installed_runtime_versions(),
        python_version=(version.major, version.minor, version.micro),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        hub_package_provenance=_huggingface_hub_package_provenance(),
        lock_sha256=_lock_sha256(),
    )


def verify_reference_environment_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one archived reference-environment receipt without trusting the current env."""
    if not isinstance(receipt, dict):
        raise CaptureContractError("reference environment receipt must be an object")
    actual_keys = set(receipt)
    missing = sorted(_RECEIPT_KEYS - actual_keys)
    extra = sorted(actual_keys - _RECEIPT_KEYS)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if extra:
            parts.append("extra=" + ",".join(extra))
        raise CaptureContractError(
            "reference environment receipt keys are not canonical: " + "; ".join(parts)
        )
    if receipt["schema_version"] != REFERENCE_ENVIRONMENT_SCHEMA_VERSION:
        raise CaptureContractError("reference environment schema_version is invalid")
    if receipt["lane"] != REFERENCE_LANE or receipt["status"] != "locked":
        raise CaptureContractError("reference environment lane/status is invalid")
    if receipt["python_implementation"] != "CPython":
        raise CaptureContractError("reference environment must record CPython")
    python_version = receipt["python_version"]
    if not isinstance(python_version, str) or re.fullmatch(r"3\.11\.[0-9]+", python_version) is None:
        raise CaptureContractError("reference environment must record an exact Python 3.11 patch version")
    if receipt["platform_system"] != "Linux" or receipt["platform_machine"] != "x86_64":
        raise CaptureContractError("reference environment platform must be Linux x86_64")

    locked = _locked_versions()
    expected_distributions = {
        canonical: locked[canonical][1]
        for canonical in sorted(locked)
    }
    distributions = receipt["distributions"]
    if not isinstance(distributions, dict) or distributions != expected_distributions:
        raise CaptureContractError(
            "reference environment distributions do not equal the complete checked-in lock"
        )
    if receipt["distribution_count"] != len(expected_distributions):
        raise CaptureContractError("reference environment distribution_count is invalid")
    lock_sha = receipt["lock_sha256"]
    if lock_sha != _lock_sha256():
        raise CaptureContractError("reference environment lock_sha256 does not match the checked-in lock")
    _validate_hub_package_provenance(
        {
            "file_count": receipt["huggingface_hub_package_file_count"],
            "receipt_sha256": receipt["huggingface_hub_package_receipt_sha256"],
        }
    )
    observed_hash = receipt["environment_receipt_sha256"]
    if (
        not isinstance(observed_hash, str)
        or len(observed_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in observed_hash)
    ):
        raise CaptureContractError("environment_receipt_sha256 must be lowercase 64-hex")
    expected_hash = sha256_json(
        {key: value for key, value in receipt.items() if key != "environment_receipt_sha256"}
    )
    if observed_hash != expected_hash:
        raise CaptureContractError("reference environment receipt SHA-256 is invalid")
    return dict(receipt)


def verify_current_reference_environment(
    expected_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the current process to be the selected reference lane and optionally match preparation."""
    current = build_reference_environment_receipt()
    if expected_receipt is not None:
        expected = verify_reference_environment_receipt(expected_receipt)
        if current != expected:
            raise CaptureContractError(
                "current reference environment does not match the preparation reference environment receipt"
            )
    return current


__all__ = [
    "LOCK",
    "REFERENCE_ENVIRONMENT_SCHEMA_VERSION",
    "REFERENCE_LANE",
    "build_reference_environment_receipt",
    "verify_current_reference_environment",
    "verify_reference_environment_receipt",
]

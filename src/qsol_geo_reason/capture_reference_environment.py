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
from .capture_package import (
    _authenticate_package_bytecode,
    _distribution_package_provenance,
    _python_package_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "constraints" / "capture-reference-py311.txt"
REFERENCE_ENVIRONMENT_SCHEMA_VERSION = "1.4.0"
REFERENCE_LANE = "capture-reference-py311-linux-x86_64-cpu"
_BOOTSTRAP_OR_PROJECT = frozenset({"pip", "setuptools", "wheel", "qsol-geo-reason"})
HUB_TRANSPORT_PACKAGE_IMPORTS = {
    "requests": "requests",
    "urllib3": "urllib3",
    "certifi": "certifi",
    "charset-normalizer": "charset_normalizer",
    "idna": "idna",
}
# Direct executable packages used by canonical capture. These were previously
# represented only by per-run manifests, which did not bind observation-time bytes
# back to the environment frozen during preparation.
CAPTURE_DIRECT_PACKAGE_IMPORTS = {
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
}
# Complete locked dependency closure that huggingface_hub 0.23.5 may execute while
# resolving/downloading snapshots, including Requests' TLS/HTTP dependencies.
HUB_EXECUTION_PACKAGE_IMPORTS = {
    "certifi": "certifi",
    "charset-normalizer": "charset_normalizer",
    "filelock": "filelock",
    "fsspec": "fsspec",
    "idna": "idna",
    "packaging": "packaging",
    "pyyaml": "yaml",
    "requests": "requests",
    "tqdm": "tqdm",
    "typing-extensions": "typing_extensions",
    "urllib3": "urllib3",
}
# Locked executable transitive distributions not represented by the direct capture
# map or dedicated Hub package. Some packages intentionally overlap the explicit Hub
# execution map: the latter records the online-preparation trust boundary as a closed
# set, while this map records the capture-time transitive closure.
CAPTURE_TRANSITIVE_PACKAGE_IMPORTS = {
    "attrs": "attrs",
    "filelock": "filelock",
    "fsspec": "fsspec",
    "jinja2": "jinja2",
    "jsonschema": "jsonschema",
    "jsonschema-specifications": "jsonschema_specifications",
    "markupsafe": "markupsafe",
    "mpmath": "mpmath",
    "networkx": "networkx",
    "numpy": "numpy",
    "packaging": "packaging",
    "pyyaml": "yaml",
    "referencing": "referencing",
    "regex": "regex",
    "rpds-py": "rpds",
    "sympy": "sympy",
    "tqdm": "tqdm",
    "typing-extensions": "typing_extensions",
}
_PACKAGE_DISTRIBUTION_BY_IMPORT = {
    "huggingface_hub": "huggingface-hub",
    **{import_name: canonical for canonical, import_name in HUB_TRANSPORT_PACKAGE_IMPORTS.items()},
    **{import_name: canonical for canonical, import_name in HUB_EXECUTION_PACKAGE_IMPORTS.items()},
    **{import_name: canonical for canonical, import_name in CAPTURE_DIRECT_PACKAGE_IMPORTS.items()},
    **{import_name: canonical for canonical, import_name in CAPTURE_TRANSITIVE_PACKAGE_IMPORTS.items()},
}
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
        "hub_transport_package_provenance",
        "hub_execution_package_provenance",
        "capture_direct_package_provenance",
        "capture_transitive_package_provenance",
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
        if previous is not None:
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


def _single_module_provenance_without_import(origin: Path, where: str) -> dict[str, Any]:
    """Content-bind a single-file module plus any executable caches without importing it."""
    try:
        source = origin.resolve(strict=True)
        if not source.is_file() or source.suffix != ".py":
            raise CaptureContractError(
                f"{where} single-module origin is not a regular Python source file: {origin}"
            )
        root = source.parent
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise CaptureContractError(f"unable to read {where} single-module source") from exc

    hashes = {source.name: hashlib.sha256(source_bytes).hexdigest()}
    caches: set[Path] = set()
    pycache = root / "__pycache__"
    try:
        if pycache.is_dir():
            caches.update(pycache.glob(f"{source.stem}.*.pyc"))
            caches.update(pycache.glob(f"{source.stem}.*.pyo"))
        for suffix in (".pyc", ".pyo"):
            candidate = source.with_suffix(suffix)
            if candidate.is_file():
                caches.add(candidate)
    except OSError as exc:
        raise CaptureContractError(
            f"unable to inspect executable caches for {where}"
        ) from exc
    for cache in sorted(caches, key=lambda value: value.as_posix()):
        _authenticate_package_bytecode(root, cache, hashes, where)
    return {
        "file_count": 1,
        "receipt_sha256": sha256_json(hashes),
    }


def _package_provenance_without_import(import_name: str, where: str) -> dict[str, Any]:
    """Content-bind the import surface and all distribution-owned runtime files."""
    distribution_name = _PACKAGE_DISTRIBUTION_BY_IMPORT.get(import_name)
    if distribution_name is None:
        raise CaptureContractError(
            f"{where} has no canonical distribution identity for import {import_name}"
        )
    return _distribution_package_provenance(
        distribution_name,
        import_name,
        where,
    )


def _huggingface_hub_package_provenance() -> dict[str, Any]:
    return _package_provenance_without_import(
        "huggingface_hub",
        "Hugging Face Hub reference environment",
    )


def _package_provenance_map(
    imports: Mapping[str, str],
    where: str,
) -> dict[str, dict[str, Any]]:
    return {
        canonical: _package_provenance_without_import(
            import_name,
            f"{where} dependency {canonical}",
        )
        for canonical, import_name in sorted(imports.items())
    }


def _hub_transport_package_provenance() -> dict[str, dict[str, Any]]:
    """Content-bind the locked Requests/TLS distribution payload used for Hub transport."""
    return _package_provenance_map(
        HUB_TRANSPORT_PACKAGE_IMPORTS,
        "Hugging Face Hub transport",
    )


def _hub_execution_package_provenance() -> dict[str, dict[str, Any]]:
    """Content-bind every locked dependency executable by online Hub preparation."""
    return _package_provenance_map(
        HUB_EXECUTION_PACKAGE_IMPORTS,
        "Hugging Face Hub execution",
    )


def _capture_direct_package_provenance() -> dict[str, dict[str, Any]]:
    """Content-bind direct canonical capture packages at preparation time."""
    return _package_provenance_map(
        CAPTURE_DIRECT_PACKAGE_IMPORTS,
        "capture direct",
    )


def _capture_transitive_package_provenance() -> dict[str, dict[str, Any]]:
    """Content-bind the locked distribution-owned transitive closure used by capture-time imports."""
    return _package_provenance_map(
        CAPTURE_TRANSITIVE_PACKAGE_IMPORTS,
        "capture transitive",
    )


def _validate_package_provenance(
    provenance: Mapping[str, Any],
    where: str,
) -> tuple[int, str]:
    if not isinstance(provenance, Mapping):
        raise CaptureContractError(f"{where} provenance must be an object")
    count = provenance.get("file_count")
    receipt = provenance.get("receipt_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CaptureContractError(
            f"{where} provenance file_count must be a positive integer"
        )
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(ch not in "0123456789abcdef" for ch in receipt)
    ):
        raise CaptureContractError(
            f"{where} provenance receipt_sha256 must be lowercase 64-hex"
        )
    return count, receipt


def _validate_hub_package_provenance(
    provenance: Mapping[str, Any],
) -> tuple[int, str]:
    return _validate_package_provenance(provenance, "Hugging Face Hub package")


def _validate_package_provenance_map(
    provenance: Mapping[str, Any],
    expected_imports: Mapping[str, str],
    where: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(provenance, Mapping):
        raise CaptureContractError(f"{where} package provenance must be an object")
    expected = set(expected_imports)
    actual = set(provenance)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CaptureContractError(
            f"{where} package provenance keys are not canonical: "
            f"missing={missing}; extra={extra}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for canonical in sorted(expected):
        count, receipt = _validate_package_provenance(
            provenance[canonical],
            f"{where} dependency {canonical}",
        )
        normalized[canonical] = {
            "file_count": count,
            "receipt_sha256": receipt,
        }
    return normalized


def _validate_hub_transport_package_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return _validate_package_provenance_map(
        provenance,
        HUB_TRANSPORT_PACKAGE_IMPORTS,
        "Hugging Face Hub transport",
    )


def _validate_hub_execution_package_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return _validate_package_provenance_map(
        provenance,
        HUB_EXECUTION_PACKAGE_IMPORTS,
        "Hugging Face Hub execution",
    )


def _validate_capture_direct_package_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return _validate_package_provenance_map(
        provenance,
        CAPTURE_DIRECT_PACKAGE_IMPORTS,
        "capture direct",
    )


def _validate_capture_transitive_package_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return _validate_package_provenance_map(
        provenance,
        CAPTURE_TRANSITIVE_PACKAGE_IMPORTS,
        "capture transitive",
    )


def _require_reference_platform(
    *,
    python_version: tuple[int, int, int],
    python_implementation: str,
    platform_system: str,
    platform_machine: str,
    python_releaselevel: str = "final",
    python_serial: int = 0,
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
    if python_releaselevel != "final" or python_serial != 0:
        observed = ".".join(str(part) for part in python_version)
        raise CaptureContractError(
            "reference lane requires a final CPython release, observed "
            f"{observed} {python_releaselevel}{python_serial}"
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
    hub_transport_package_provenance: Mapping[str, Any],
    hub_execution_package_provenance: Mapping[str, Any],
    capture_direct_package_provenance: Mapping[str, Any],
    capture_transitive_package_provenance: Mapping[str, Any],
    python_releaselevel: str = "final",
    python_serial: int = 0,
    lock_sha256: str | None = None,
) -> dict[str, Any]:
    version_text, machine = _require_reference_platform(
        python_version=python_version,
        python_implementation=python_implementation,
        platform_system=platform_system,
        platform_machine=platform_machine,
        python_releaselevel=python_releaselevel,
        python_serial=python_serial,
    )
    hub_file_count, hub_receipt_sha256 = _validate_hub_package_provenance(
        hub_package_provenance
    )
    transport_provenance = _validate_hub_transport_package_provenance(
        hub_transport_package_provenance
    )
    hub_execution_provenance = _validate_hub_execution_package_provenance(
        hub_execution_package_provenance
    )
    direct_provenance = _validate_capture_direct_package_provenance(
        capture_direct_package_provenance
    )
    transitive_provenance = _validate_capture_transitive_package_provenance(
        capture_transitive_package_provenance
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
        "hub_transport_package_provenance": transport_provenance,
        "hub_execution_package_provenance": hub_execution_provenance,
        "capture_direct_package_provenance": direct_provenance,
        "capture_transitive_package_provenance": transitive_provenance,
    }
    payload["environment_receipt_sha256"] = sha256_json(payload)
    return payload


def build_reference_environment_receipt() -> dict[str, Any]:
    """Measure the exact final interpreter and distribution-owned runtime used here."""
    version = sys.version_info
    return _build_reference_environment_receipt_from_state(
        locked=_locked_versions(),
        installed=_installed_runtime_versions(),
        python_version=(version.major, version.minor, version.micro),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        hub_package_provenance=_huggingface_hub_package_provenance(),
        hub_transport_package_provenance=_hub_transport_package_provenance(),
        hub_execution_package_provenance=_hub_execution_package_provenance(),
        capture_direct_package_provenance=_capture_direct_package_provenance(),
        capture_transitive_package_provenance=_capture_transitive_package_provenance(),
        python_releaselevel=version.releaselevel,
        python_serial=version.serial,
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
        raise CaptureContractError(
            "reference environment must record an exact Python 3.11 patch version"
        )
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
        raise CaptureContractError(
            "reference environment lock_sha256 does not match the checked-in lock"
        )
    _validate_hub_package_provenance(
        {
            "file_count": receipt["huggingface_hub_package_file_count"],
            "receipt_sha256": receipt["huggingface_hub_package_receipt_sha256"],
        }
    )
    normalized_transport = _validate_hub_transport_package_provenance(
        receipt["hub_transport_package_provenance"]
    )
    if normalized_transport != receipt["hub_transport_package_provenance"]:
        raise CaptureContractError(
            "reference environment Hub transport package provenance is not canonical"
        )
    normalized_hub_execution = _validate_hub_execution_package_provenance(
        receipt["hub_execution_package_provenance"]
    )
    if normalized_hub_execution != receipt["hub_execution_package_provenance"]:
        raise CaptureContractError(
            "reference environment Hub execution package provenance is not canonical"
        )
    normalized_direct = _validate_capture_direct_package_provenance(
        receipt["capture_direct_package_provenance"]
    )
    if normalized_direct != receipt["capture_direct_package_provenance"]:
        raise CaptureContractError(
            "reference environment capture direct package provenance is not canonical"
        )
    normalized_transitive = _validate_capture_transitive_package_provenance(
        receipt["capture_transitive_package_provenance"]
    )
    if normalized_transitive != receipt["capture_transitive_package_provenance"]:
        raise CaptureContractError(
            "reference environment capture transitive package provenance is not canonical"
        )
    observed_hash = receipt["environment_receipt_sha256"]
    if (
        not isinstance(observed_hash, str)
        or len(observed_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in observed_hash)
    ):
        raise CaptureContractError("environment_receipt_sha256 must be lowercase 64-hex")
    expected_hash = sha256_json(
        {
            key: value
            for key, value in receipt.items()
            if key != "environment_receipt_sha256"
        }
    )
    if observed_hash != expected_hash:
        raise CaptureContractError("reference environment receipt SHA-256 is invalid")
    return dict(receipt)


def verify_current_reference_environment(
    expected_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require current executable bytes to equal the reference lane and optional preparation receipt."""
    current = build_reference_environment_receipt()
    if expected_receipt is not None:
        expected = verify_reference_environment_receipt(expected_receipt)
        if current != expected:
            raise CaptureContractError(
                "current reference environment does not match the preparation reference environment receipt"
            )
    return current


__all__ = [
    "CAPTURE_DIRECT_PACKAGE_IMPORTS",
    "CAPTURE_TRANSITIVE_PACKAGE_IMPORTS",
    "HUB_EXECUTION_PACKAGE_IMPORTS",
    "HUB_TRANSPORT_PACKAGE_IMPORTS",
    "LOCK",
    "REFERENCE_ENVIRONMENT_SCHEMA_VERSION",
    "REFERENCE_LANE",
    "build_reference_environment_receipt",
    "verify_current_reference_environment",
    "verify_reference_environment_receipt",
]
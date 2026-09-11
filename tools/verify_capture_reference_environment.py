"""Verify the complete Phase 2A Python 3.11 CPU reference runtime lock."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "constraints" / "capture-reference-py311.txt"
_BOOTSTRAP_OR_PROJECT = frozenset({"pip", "setuptools", "wheel", "qsol-geo-reason"})


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_versions() -> dict[str, tuple[str, str]]:
    locked: dict[str, tuple[str, str]] = {}
    for line_number, raw in enumerate(LOCK.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(
                f"reference lock line {line_number} must be one exact name==version pin"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise RuntimeError(f"reference lock line {line_number} is malformed")
        canonical = _canonical_name(name)
        if canonical in locked:
            raise RuntimeError(f"reference lock contains duplicate distribution {name}")
        locked[canonical] = (name, version)
    if not locked:
        raise RuntimeError("reference lock contains no runtime distributions")
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
            raise RuntimeError(
                "multiple installed distributions normalize to the same name: "
                f"{previous[0]} and {name}"
            )
        installed[canonical] = (name, version)
    return installed


def verify_reference_environment() -> dict[str, object]:
    locked = _locked_versions()
    installed = _installed_runtime_versions()

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
                + ",".join(f"{locked[name][0]}=={locked[name][1]}" for name in missing)
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
        raise RuntimeError("reference runtime does not match complete lock: " + "; ".join(details))

    return {
        "status": "locked",
        "distribution_count": len(locked),
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
    }


def main() -> int:
    print(json.dumps(verify_reference_environment(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

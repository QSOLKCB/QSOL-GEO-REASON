#!/usr/bin/env python3
"""Regenerate and byte-compare the frozen Phase 1 reference result."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsol_geo_reason.canonical import canonical_json_bytes, sha256_json
from qsol_geo_reason.simulation import (
    EVIDENCE_CLASS,
    PROTOCOL_ID,
    REPLICATION_STATUS,
    SCHEMA_VERSION,
    run_recipe,
)

_METADATA_KEYS = {
    "schema_version",
    "protocol_id",
    "recipe_id",
    "implementation_revision",
    "recipe_sha256",
    "artifact_sha256",
    "evidence_class",
    "replication_status",
}


def _identity_errors(recipe: dict, metadata: dict, frozen: dict, regenerated: dict) -> list[str]:
    errors: list[str] = []
    if set(metadata) != _METADATA_KEYS:
        missing = sorted(_METADATA_KEYS - set(metadata))
        unknown = sorted(set(metadata) - _METADATA_KEYS)
        if missing:
            errors.append(f"metadata missing keys: {', '.join(missing)}")
        if unknown:
            errors.append(f"metadata unknown keys: {', '.join(unknown)}")

    expected_recipe_hash = sha256_json(recipe)
    identities = {
        "schema_version": (metadata.get("schema_version"), recipe.get("schema_version"), frozen.get("schema_version"), regenerated.get("schema_version"), SCHEMA_VERSION),
        "protocol_id": (metadata.get("protocol_id"), frozen.get("protocol_id"), regenerated.get("protocol_id"), PROTOCOL_ID),
        "recipe_id": (metadata.get("recipe_id"), recipe.get("recipe_id"), frozen.get("recipe_id"), regenerated.get("recipe_id")),
        "implementation_revision": (metadata.get("implementation_revision"), frozen.get("implementation_revision"), regenerated.get("implementation_revision")),
        "recipe_sha256": (metadata.get("recipe_sha256"), frozen.get("recipe_sha256"), regenerated.get("recipe_sha256"), expected_recipe_hash),
        "artifact_sha256": (metadata.get("artifact_sha256"), frozen.get("artifact_sha256"), regenerated.get("artifact_sha256")),
        "evidence_class": (metadata.get("evidence_class"), frozen.get("evidence_class"), regenerated.get("evidence_class"), EVIDENCE_CLASS),
        "replication_status": (metadata.get("replication_status"), frozen.get("replication_status"), regenerated.get("replication_status"), REPLICATION_STATUS),
    }
    for name, values in identities.items():
        if any(value != values[0] for value in values[1:]):
            errors.append(f"{name} identity mismatch: {values!r}")
    return errors


def main() -> int:
    recipe = json.loads((ROOT / "recipes/reference-suite.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "fixtures/reference-metadata.json").read_text(encoding="utf-8"))
    expected_bytes = (ROOT / "fixtures/reference-result.json").read_bytes()
    expected = json.loads(expected_bytes)

    implementation_revision = metadata.get("implementation_revision")
    if not isinstance(implementation_revision, str) or not implementation_revision:
        print("invalid or missing metadata implementation_revision", file=sys.stderr)
        return 1

    actual = run_recipe(recipe, implementation_revision=implementation_revision)
    identity_errors = _identity_errors(recipe, metadata, expected, actual)
    if identity_errors:
        print("reference metadata identity mismatch", file=sys.stderr)
        for error in identity_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    regenerated = canonical_json_bytes(actual) + b"\n"
    if regenerated != expected_bytes:
        print("reference fixture mismatch", file=sys.stderr)
        print(f"expected artifact_sha256: {expected.get('artifact_sha256')}", file=sys.stderr)
        print(f"actual artifact_sha256:   {actual.get('artifact_sha256')}", file=sys.stderr)
        for key in sorted(set(expected) | set(actual)):
            if expected.get(key) != actual.get(key):
                print(f"top-level difference: {key}", file=sys.stderr)
        return 1

    print("reference metadata identities verified")
    print("reference fixture verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

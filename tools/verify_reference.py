#!/usr/bin/env python3
"""Regenerate and byte-compare the frozen Phase 1 reference result."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsol_geo_reason.canonical import canonical_json_bytes
from qsol_geo_reason.simulation import run_recipe


def main() -> int:
    recipe = json.loads((ROOT / "recipes/reference-suite.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "fixtures/reference-metadata.json").read_text(encoding="utf-8"))
    expected_bytes = (ROOT / "fixtures/reference-result.json").read_bytes()
    expected = json.loads(expected_bytes)
    actual = run_recipe(recipe, implementation_revision=metadata["implementation_revision"])
    regenerated = canonical_json_bytes(actual) + b"\n"
    if regenerated != expected_bytes:
        print("reference fixture mismatch", file=sys.stderr)
        print(f"expected artifact_sha256: {expected.get('artifact_sha256')}", file=sys.stderr)
        print(f"actual artifact_sha256:   {actual.get('artifact_sha256')}", file=sys.stderr)
        for key in sorted(set(expected) | set(actual)):
            if expected.get(key) != actual.get(key):
                print(f"top-level difference: {key}", file=sys.stderr)
        return 1
    print("reference fixture verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

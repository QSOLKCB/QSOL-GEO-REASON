#!/usr/bin/env python3
"""Regenerate and byte-compare the frozen Phase 1 reference result."""

import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qsol_geo_reason.canonical import canonical_json_bytes
from qsol_geo_reason.simulation import run_recipe


def main() -> int:
    recipe = json.loads((ROOT / "recipes/reference-suite.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "fixtures/reference-metadata.json").read_text(encoding="utf-8"))
    expected = (ROOT / "fixtures/reference-result.json").read_bytes()
    regenerated = canonical_json_bytes(run_recipe(recipe, implementation_revision=metadata["implementation_revision"])) + b"\n"
    if regenerated != expected:
        print("reference fixture mismatch", file=sys.stderr); return 1
    print("reference fixture verified"); return 0

if __name__ == "__main__": raise SystemExit(main())

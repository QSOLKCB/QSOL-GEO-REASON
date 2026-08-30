#!/usr/bin/env python3
"""Generate a Phase 1 reference result for an explicit implementation revision."""

import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qsol_geo_reason.canonical import canonical_json_bytes
from qsol_geo_reason.simulation import run_recipe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures/reference-result.json")
    args = parser.parse_args()
    recipe = json.loads((ROOT / "recipes/reference-suite.json").read_text(encoding="utf-8"))
    result = run_recipe(recipe, implementation_revision=args.implementation_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result) + b"\n")
    print(result["artifact_sha256"])
    return 0

if __name__ == "__main__": raise SystemExit(main())

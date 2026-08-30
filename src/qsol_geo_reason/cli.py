"""Command-line entry point for deterministic reference simulations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .canonical import canonical_json_bytes
from .provenance import SourceIdentityError, resolve_implementation_revision
from .simulation import run_recipe


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a QSOL-GEO-REASON synthetic geometry recipe")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation-revision",
        default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION"),
        help="Immutable implementation revision. If omitted, a clean source Git checkout is required and HEAD is used.",
    )
    args = parser.parse_args()
    try:
        implementation_revision = resolve_implementation_revision(args.implementation_revision)
    except SourceIdentityError as exc:
        parser.error(str(exc))

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    result = run_recipe(recipe, implementation_revision=implementation_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result) + b"\n")
    print(result["artifact_sha256"])
    return 0

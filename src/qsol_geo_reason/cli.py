"""Command-line entry point for deterministic reference simulations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .canonical import canonical_json_bytes
from .simulation import run_recipe


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a QSOL-GEO-REASON synthetic geometry recipe")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION") or _git_revision(), help="Immutable implementation revision; defaults to env var or current git HEAD.")
    args = parser.parse_args()
    if not args.implementation_revision:
        parser.error("--implementation-revision is required outside a git checkout")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    result = run_recipe(recipe, implementation_revision=args.implementation_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result) + b"\n")
    print(result["artifact_sha256"])
    return 0

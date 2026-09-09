"""Fresh-process worker for canonical GEO-CAP-001 OBSERVATION capture.

This module is intentionally internal.  ``qsol-geo-capture`` launches it with
CPython isolated mode (``-I``) and bytecode writes disabled (``-B``), after removing
Python/loader injection environment variables.  The worker then imports the canonical
backend stack and performs one offline observation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_EXTERNAL_CAPTURE_PREFIXES = (
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
)


def _assert_fresh_worker_boundary() -> None:
    if os.environ.get("QSOL_GEO_CAPTURE_FRESH_WORKER") != "1":
        raise RuntimeError("canonical capture worker must be launched by qsol-geo-capture")
    if not sys.flags.isolated or not sys.flags.no_user_site or not sys.flags.ignore_environment:
        raise RuntimeError("canonical capture worker requires CPython isolated mode (-I)")
    contaminated = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _EXTERNAL_CAPTURE_PREFIXES)
    )
    if contaminated:
        raise RuntimeError(
            "canonical capture worker inherited preloaded production dependencies: "
            + ", ".join(contaminated[:5])
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-revision")
    args = parser.parse_args(argv)

    try:
        _assert_fresh_worker_boundary()

        # Import only after the worker boundary is established.  In particular no
        # Torch/Transformers/Hub code has executed in this interpreter before here.
        from .capture import (
            CaptureBackendUnavailable,
            CaptureContractError,
            HuggingFacePyTorchBackend,
            execute_capture,
            validate_capture_request,
            write_capture_bundle,
        )
        from . import capture_execute
        from .provenance import SourceIdentityError

        request = json.loads(args.request.read_text(encoding="utf-8"))
        validated = validate_capture_request(request)
        implementation_revision = capture_execute.resolve_implementation_revision(
            args.implementation_revision,
            require_checkout=True,
        )
        backend = HuggingFacePyTorchBackend(validated)
        manifest, trajectory = execute_capture(
            validated,
            implementation_revision=implementation_revision,
            backend=backend,
            evidence_class="OBSERVATION",
        )
        write_capture_bundle(args.output_dir, validated, manifest, trajectory)
        print(manifest["manifest_sha256"])
        return 0
    except (
        SourceIdentityError,
        CaptureContractError,
        CaptureBackendUnavailable,
        json.JSONDecodeError,
        UnicodeError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"qsol-geo-capture worker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

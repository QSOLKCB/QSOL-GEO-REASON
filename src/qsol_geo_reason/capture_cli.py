"""Command-line entry point for GEO-CAP-001 canonical local capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .capture import (
    CaptureBackendUnavailable,
    CaptureContractError,
    HuggingFacePyTorchBackend,
    execute_capture,
    validate_capture_request,
    write_capture_bundle,
)
from .provenance import SourceIdentityError, resolve_implementation_revision


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture local-model hidden states under the GEO-CAP-001 "
            "canonical Hugging Face/PyTorch replay protocol"
        )
    )
    parser.add_argument("request", type=Path, help="GEO-CAP-001 request JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--implementation-revision",
        default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION"),
        help=(
            "Immutable repository revision. If omitted, a clean source Git "
            "checkout is required and HEAD is used."
        ),
    )
    args = parser.parse_args()

    try:
        implementation_revision = resolve_implementation_revision(
            args.implementation_revision
        )
        request = json.loads(args.request.read_text(encoding="utf-8"))
        validated = validate_capture_request(request)
        backend = HuggingFacePyTorchBackend(validated)
        manifest, trajectory = execute_capture(
            validated,
            implementation_revision=implementation_revision,
            backend=backend,
            evidence_class="OBSERVATION",
        )
    except (
        SourceIdentityError,
        CaptureContractError,
        CaptureBackendUnavailable,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        parser.error(str(exc))

    write_capture_bundle(args.output_dir, validated, manifest, trajectory)
    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

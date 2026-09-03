"""Command-line entry point for GEO-CAP-001 canonical local capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .canonical import sha256_json
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
            "Validate or capture local-model hidden states under the GEO-CAP-001 "
            "canonical Hugging Face/PyTorch replay protocol"
        )
    )
    parser.add_argument("request", type=Path, help="GEO-CAP-001 request JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run the complete canonical request validator, including semantic "
            "constraints such as unique step IDs, without loading a model."
        ),
    )
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
        request = json.loads(args.request.read_text(encoding="utf-8"))
        validated = validate_capture_request(request)
        if args.validate_only:
            print(sha256_json(validated))
            return 0
        if args.output_dir is None:
            raise CaptureContractError("--output-dir is required unless --validate-only is used")
        implementation_revision = resolve_implementation_revision(
            args.implementation_revision
        )
        backend = HuggingFacePyTorchBackend(validated)
        manifest, trajectory = execute_capture(
            validated,
            implementation_revision=implementation_revision,
            backend=backend,
            evidence_class="OBSERVATION",
        )
        write_capture_bundle(args.output_dir, validated, manifest, trajectory)
    except (
        SourceIdentityError,
        CaptureContractError,
        CaptureBackendUnavailable,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        parser.error(str(exc))

    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

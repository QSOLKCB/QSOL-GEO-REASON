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
from .capture_hub_tree import prepare_tree_receipts
from .provenance import SourceIdentityError, resolve_implementation_revision


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, prepare, or capture local-model hidden states under the GEO-CAP-001 "
            "canonical Hugging Face/PyTorch replay protocol"
        )
    )
    parser.add_argument("request", type=Path, help="GEO-CAP-001 request JSON")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run the complete canonical request validator, including semantic "
            "constraints such as unique step IDs, without loading a model."
        ),
    )
    mode.add_argument(
        "--prepare-tree-receipts",
        action="store_true",
        help=(
            "ONLINE WARM-UP: query the Hugging Face Hub for each exact immutable "
            "model/tokenizer commit, warm its snapshot, create QSOL's authenticated "
            "trees/<commit>.json cache artifact, and print the two SHA-256 receipt "
            "fields that must be frozen into the request before offline capture."
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
        if args.prepare_tree_receipts:
            receipts = prepare_tree_receipts(validated)
            print(json.dumps(receipts, sort_keys=True, separators=(",", ":")))
            return 0
        if args.validate_only:
            print(sha256_json(validated))
            return 0
        if args.output_dir is None:
            raise CaptureContractError("--output-dir is required unless a validation/warm-up mode is used")
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

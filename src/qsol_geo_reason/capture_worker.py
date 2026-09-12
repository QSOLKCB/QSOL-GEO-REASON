"""Fresh-process worker for canonical GEO-CAP-001 OBSERVATION capture.

This module is intentionally internal. ``qsol-geo-capture`` launches it with
CPython isolated/no-site mode (``-I -S``) and bytecode writes disabled (``-B``), after
removing Python/loader injection environment variables. The worker then imports the
canonical backend stack and performs one offline observation.
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
        raise RuntimeError(
            "canonical capture worker must be launched by qsol-geo-capture"
        )
    if (
        not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.no_user_site
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError(
            "canonical capture worker requires CPython isolated no-site mode (-I -S)"
        )
    contaminated = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _EXTERNAL_CAPTURE_PREFIXES
        )
    )
    if contaminated:
        raise RuntimeError(
            "canonical capture worker inherited preloaded production dependencies: "
            + ", ".join(contaminated[:5])
        )


def _assert_execution_receipt_outside_bundle(
    output_dir: Path,
    execution_receipt: Path | None,
) -> None:
    """Reject occurrence receipts that would mutate an immutable published bundle."""
    if execution_receipt is None:
        return
    try:
        bundle = Path(output_dir).resolve(strict=False)
        receipt = Path(execution_receipt).resolve(strict=False)
        receipt.relative_to(bundle)
    except ValueError:
        return
    except OSError as exc:
        raise RuntimeError("unable to resolve execution receipt publication path") from exc
    raise RuntimeError(
        "--execution-receipt must be outside --output-dir; canonical bundle directories are immutable"
    )


def _assert_execution_identity(
    execution_id: str | None,
    execution_receipt: Path | None,
) -> None:
    """Validate occurrence identity before reservation, imports, or model work."""
    if (execution_id is None) != (execution_receipt is None):
        raise RuntimeError(
            "canonical execution identity requires both --execution-id and --execution-receipt"
        )
    if execution_id is not None and not execution_id.strip():
        raise RuntimeError("--execution-id must be a non-empty string")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-revision")
    parser.add_argument("--execution-id")
    parser.add_argument("--execution-receipt", type=Path)
    args = parser.parse_args(argv)

    receipt_reservation = None

    # Establish all pre-import trust/publication boundaries before any production
    # dependency or model backend is imported into this interpreter.
    try:
        _assert_fresh_worker_boundary()
        _assert_execution_identity(args.execution_id, args.execution_receipt)
        _assert_execution_receipt_outside_bundle(
            args.output_dir,
            args.execution_receipt,
        )
        if args.execution_receipt is not None:
            from .execution_receipt_reservation import (
                reserve_execution_receipt_destination,
            )

            receipt_reservation = reserve_execution_receipt_destination(
                args.execution_receipt
            )
    except RuntimeError as exc:
        print(f"qsol-geo-capture worker: {exc}", file=sys.stderr)
        return 2

    from .capture import (
        CaptureBackendUnavailable,
        CaptureContractError,
        HuggingFacePyTorchBackend,
        execute_capture,
        validate_capture_request,
        write_capture_bundle,
    )
    from . import capture_execute
    from .capture_execution import (
        build_execution_receipt,
        verify_execution_receipt,
    )
    from .execution_receipt_reservation import (
        commit_execution_receipt_reservation,
        release_execution_receipt_reservation,
    )
    from .provenance import SourceIdentityError

    bundle_published = False
    try:
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

        # Publish the immutable scientific bundle first. The occurrence receipt name
        # has already been durably reserved outside that directory, so a pre-existing
        # or unwritable destination cannot strand a provenance-less successful bundle.
        write_capture_bundle(args.output_dir, validated, manifest, trajectory)
        bundle_published = True
        if args.execution_id is not None:
            execution_receipt = build_execution_receipt(
                execution_id=args.execution_id,
                bundle_dir=args.output_dir,
            )
            verify_execution_receipt(
                execution_receipt,
                bundle_dir=args.output_dir,
            )
            if receipt_reservation is None:
                raise RuntimeError("execution receipt reservation is missing")
            commit_execution_receipt_reservation(
                receipt_reservation,
                execution_receipt,
            )

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
        if receipt_reservation is not None and not bundle_published:
            release_execution_receipt_reservation(receipt_reservation)
        print(f"qsol-geo-capture worker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

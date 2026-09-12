"""Command-line entry point for GEO-CAP-001 canonical local capture.

Validation and online Hub-tree preparation run in the command process. A production
OBSERVATION never does: it is delegated to a fresh CPython isolated worker so a caller's
already-imported Python modules are outside the canonical execution boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .canonical import sha256_json
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_hub_tree import prepare_tree_receipts
from .capture_validation import validate_capture_request
from .no_site_subprocess import isolated_package_command


_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_LOADER_ENV_NAMES = frozenset({"GLIBC_TUNABLES", "LIBPATH", "SHLIB_PATH"})


def _fresh_worker_environment() -> dict[str, str]:
    """Remove Python/native-loader injection controls before the worker starts."""
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("PYTHON"):
            continue
        if upper.startswith(_LOADER_ENV_PREFIXES) or upper in _LOADER_ENV_NAMES:
            continue
        environment[key] = value
    environment.update(
        {
            "QSOL_GEO_CAPTURE_FRESH_WORKER": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _run_fresh_worker(
    request_path: Path,
    output_dir: Path,
    implementation_revision: str | None,
    *,
    execution_id: str | None = None,
    execution_receipt: Path | None = None,
) -> str:
    if (execution_id is None) != (execution_receipt is None):
        raise CaptureContractError(
            "execution occurrence identity requires both execution_id and execution_receipt"
        )
    worker_args = [
        str(request_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if implementation_revision:
        worker_args.extend(["--implementation-revision", implementation_revision])
    if execution_id is not None:
        worker_args.extend(
            [
                "--execution-id",
                execution_id,
                "--execution-receipt",
                str(execution_receipt.resolve()),
            ]
        )
    command = isolated_package_command(
        "qsol_geo_reason.capture_worker",
        worker_args,
    )
    try:
        completed = subprocess.run(
            command,
            env=_fresh_worker_environment(),
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise CaptureContractError(
            "unable to launch the fresh canonical capture worker"
        ) from exc
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "capture worker failed"
        )
        raise CaptureContractError(detail)
    lines = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    if (
        len(lines) != 1
        or len(lines[0]) != 64
        or any(ch not in "0123456789abcdef" for ch in lines[0])
    ):
        raise CaptureContractError(
            "fresh capture worker returned a malformed manifest receipt"
        )
    return lines[0]


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
            "ONLINE WARM-UP: query the canonical Hugging Face Hub for each exact immutable "
            "model/tokenizer commit, warm its snapshot, create QSOL's authenticated "
            "trees/<commit>.json cache artifact, and print the two SHA-256 receipt "
            "fields that must be frozen into the request before offline capture."
        ),
    )
    parser.add_argument(
        "--implementation-revision",
        default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION"),
        help=(
            "Immutable repository revision. If omitted, the fresh worker requires a "
            "clean source Git checkout and binds its HEAD."
        ),
    )
    parser.add_argument(
        "--execution-id",
        help=(
            "Optional per-execution occurrence identity. Must be paired with "
            "--execution-receipt; it does not modify the frozen capture request."
        ),
    )
    parser.add_argument(
        "--execution-receipt",
        type=Path,
        help=(
            "Optional no-replace path for the worker-generated execution receipt bound "
            "to the published canonical bundle."
        ),
    )
    args = parser.parse_args()

    try:
        if (args.execution_id is None) != (args.execution_receipt is None):
            raise CaptureContractError(
                "--execution-id and --execution-receipt must be supplied together"
            )
        request = json.loads(args.request.read_text(encoding="utf-8"))
        validated = validate_capture_request(request)
        if args.prepare_tree_receipts:
            if args.execution_id is not None:
                raise CaptureContractError(
                    "execution occurrence identity is not valid during Hub preparation"
                )
            receipts = prepare_tree_receipts(validated)
            print(
                json.dumps(
                    receipts,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.validate_only:
            if args.execution_id is not None:
                raise CaptureContractError(
                    "execution occurrence identity is not valid during validation-only mode"
                )
            print(sha256_json(validated))
            return 0
        if args.output_dir is None:
            raise CaptureContractError(
                "--output-dir is required unless a validation/warm-up mode is used"
            )
        manifest_sha256 = _run_fresh_worker(
            args.request,
            args.output_dir,
            args.implementation_revision,
            execution_id=args.execution_id,
            execution_receipt=args.execution_receipt,
        )
    except (
        CaptureContractError,
        CaptureBackendUnavailable,
        json.JSONDecodeError,
        UnicodeError,
        OSError,
    ) as exc:
        parser.error(str(exc))

    print(manifest_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

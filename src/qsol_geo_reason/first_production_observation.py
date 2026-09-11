"""Authenticated orchestration for GEO-CAP-001-EXP-001.

All evidence-producing preparation, staging, worker orchestration, failure preservation,
and replay-verdict publication live under the directly authenticated package tree. The
``tools/run_first_production_observation.py`` file is only an isolated-process launcher.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_bytes, sha256_json
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_hub_tree import prepare_tree_receipts
from .capture_preparation import (
    build_preparation_receipt,
    verify_preparation_receipt,
)
from .capture_publish import _ensure_parent_directory_durable, _fsync_directory
from .capture_replay import REPLAY_BUNDLE_FILES, build_replay_verdict, verify_replay_verdict
from .capture_validation import validate_capture_request
from .provenance import SourceIdentityError, resolve_implementation_revision
from .tracked_artifact import authenticate_tracked_file_against_revision


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "GEO-CAP-001-EXP-001"
TEMPLATE = ROOT / "experiments" / "GEO-CAP-001-EXP-001.request.template.json"
BUNDLE_FILES = REPLAY_BUNDLE_FILES
TREE_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(f"unable to read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureContractError(f"{path} must contain a JSON object")
    return value


def _load_template(repository_commit: str | None = None) -> dict[str, Any]:
    if repository_commit is not None:
        try:
            authenticate_tracked_file_against_revision(TEMPLATE, repository_commit)
        except SourceIdentityError as exc:
            raise CaptureContractError(
                f"frozen experiment template is not bound to repository revision {repository_commit}: {exc}"
            ) from exc
    template = validate_capture_request(_read_json(TEMPLATE))
    model = template["model"]
    if any(field in model for field in TREE_FIELDS):
        raise CaptureContractError(
            "the committed experiment template must not contain machine-materialized Hub tree receipts"
        )
    return template


def _without_tree_receipts(request: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(request))
    model = value["model"]
    for field in TREE_FIELDS:
        model.pop(field, None)
    return value


def _assert_exact_experiment_request(
    request: Mapping[str, Any],
    *,
    require_receipts: bool,
    repository_commit: str | None = None,
) -> dict[str, Any]:
    validated = validate_capture_request(request)
    template = _load_template(repository_commit)
    if _without_tree_receipts(validated) != template:
        raise CaptureContractError(
            f"request does not match the frozen {EXPERIMENT_ID} experiment declaration"
        )
    model = validated["model"]
    present = [field in model for field in TREE_FIELDS]
    if require_receipts and not all(present):
        raise CaptureContractError(
            "final production request is missing one or both authenticated Hub tree receipts; run prepare first"
        )
    if any(present) and not all(present):
        raise CaptureContractError("Hub tree receipts must be present as a complete pair")
    return validated


def _default_preparation_receipt_path(request_path: Path) -> Path:
    """Return the canonical sidecar path emitted by ``prepare`` for one request path."""
    request_path = Path(request_path)
    if request_path.suffix == ".json":
        return request_path.with_name(
            f"{request_path.stem}.preparation-receipt.json"
        )
    return request_path.with_name(f"{request_path.name}.preparation-receipt.json")


def _create_output_root_durable(output_root: Path) -> None:
    """Create one immutable observation root and durably publish its directory entry."""
    _ensure_parent_directory_durable(output_root.parent)
    try:
        output_root.mkdir()
    except FileExistsError as exc:
        raise CaptureContractError(
            f"refusing to reuse output root {output_root}; observation evidence directories are immutable"
        ) from exc
    except OSError as exc:
        raise CaptureContractError(
            f"unable to create observation output root {output_root}: {exc}"
        ) from exc
    try:
        _fsync_directory(output_root.parent)
    except OSError as exc:
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise CaptureContractError(
            f"unable to durably publish observation output root {output_root}: {exc}"
        ) from exc


def _unique_sibling(path: Path, marker: str) -> Path:
    while True:
        candidate = path.with_name(
            f".{path.name}.{marker}.{os.getpid()}.{secrets.token_hex(8)}"
        )
        if not candidate.exists():
            return candidate


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Stage, fsync, and publish immutable canonical JSON with no-replace semantics."""
    _ensure_parent_directory_durable(path.parent)
    if path.exists():
        raise CaptureContractError(f"refusing to overwrite existing artifact {path}")
    try:
        payload = canonical_json_bytes(value) + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CaptureContractError(f"unable to canonicalize artifact {path}") from exc
    temporary = _unique_sibling(path, "tmp")
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CaptureContractError(
                f"refusing to overwrite existing artifact {path}"
            ) from exc
        published = True
        _fsync_directory(path.parent)
    except CaptureContractError:
        raise
    except OSError as exc:
        if published:
            try:
                path.unlink()
            except OSError:
                pass
        raise CaptureContractError(f"unable to persist {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def prepare(
    output: Path,
    implementation_revision: str | None = None,
) -> str:
    """Online warm-up with a clean-revision provenance receipt for the final request."""
    preparation_receipt_path = _default_preparation_receipt_path(output)
    if output.exists():
        raise CaptureContractError(f"refusing to overwrite existing artifact {output}")
    if preparation_receipt_path.exists():
        raise CaptureContractError(
            f"refusing to overwrite existing artifact {preparation_receipt_path}"
        )

    try:
        preparation_repository_commit = resolve_implementation_revision(
            implementation_revision,
            require_checkout=True,
        )
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"unable to bind online preparation to a clean repository revision: {exc}"
        ) from exc

    # The preregistered experiment declaration is outside src/, so authenticate its
    # literal working-tree bytes against the same bound revision before trusting it.
    template = _load_template(preparation_repository_commit)
    receipts = prepare_tree_receipts(template)
    final_request = json.loads(json.dumps(template))
    final_request["model"].update(receipts)
    final_request = _assert_exact_experiment_request(
        final_request,
        require_receipts=True,
        repository_commit=preparation_repository_commit,
    )

    try:
        preparation_repository_commit = resolve_implementation_revision(
            preparation_repository_commit,
            require_checkout=True,
        )
        authenticate_tracked_file_against_revision(
            TEMPLATE,
            preparation_repository_commit,
        )
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"online preparation inputs changed before publication: {exc}"
        ) from exc

    preparation_receipt = build_preparation_receipt(
        request=final_request,
        repository_commit=preparation_repository_commit,
        experiment_id=EXPERIMENT_ID,
    )
    verify_preparation_receipt(
        preparation_receipt,
        request=final_request,
        experiment_id=EXPERIMENT_ID,
    )

    _exclusive_write_json(preparation_receipt_path, preparation_receipt)
    try:
        _exclusive_write_json(output, final_request)
    except BaseException:
        try:
            preparation_receipt_path.unlink(missing_ok=True)
            _fsync_directory(preparation_receipt_path.parent)
        except OSError:
            pass
        raise
    return sha256_json(final_request)


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _run_capture(
    request_path: Path,
    output_dir: Path,
    implementation_revision: str,
    execution_id: str,
    execution_receipt_path: Path,
) -> str:
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "qsol_geo_reason.capture_cli",
        str(request_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--implementation-revision",
        implementation_revision,
        "--execution-id",
        execution_id,
        "--execution-receipt",
        str(execution_receipt_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_offline_environment(),
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "capture failed"
        raise CaptureContractError(detail)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if (
        len(lines) != 1
        or len(lines[0]) != 64
        or any(ch not in "0123456789abcdef" for ch in lines[0])
    ):
        raise CaptureContractError(
            "capture CLI returned a malformed manifest SHA-256 receipt"
        )
    if not execution_receipt_path.is_file():
        raise CaptureContractError(
            f"capture worker did not publish execution receipt {execution_receipt_path}"
        )
    return lines[0]


def _completed_run_directories(output_root: Path) -> list[str]:
    completed: list[str] = []
    for name in ("run-a", "run-b"):
        directory = output_root / name
        if directory.is_dir() and all(
            (directory / item).is_file() for item in BUNDLE_FILES
        ):
            completed.append(name)
    return completed


def _preserve_failed_attempt(
    output_root: Path,
    exc: BaseException,
    repository_commit: str,
    execution_ids: Mapping[str, str],
    preparation_receipt: Mapping[str, Any],
) -> Path:
    marker = {
        "schema_version": "1.0.0",
        "protocol_id": "GEO-CAP-001",
        "experiment_id": EXPERIMENT_ID,
        "repository_commit": repository_commit,
        "preparation_repository_commit": preparation_receipt[
            "preparation_repository_commit"
        ],
        "preparation_receipt_sha256": preparation_receipt[
            "preparation_receipt_sha256"
        ],
        "planned_execution_ids": dict(execution_ids),
        "attempt_status": "failed_before_replay_verdict",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "completed_run_directories": _completed_run_directories(output_root),
        "interpretation": (
            "This directory records an incomplete production attempt. It is not a "
            "successful replay result and must not be substituted for the final experiment."
        ),
    }
    try:
        _exclusive_write_json(output_root / "execution-failure.json", marker)
    except CaptureContractError:
        pass
    failed_path = output_root.with_name(
        f"{output_root.name}.failed-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        os.rename(output_root, failed_path)
    except OSError:
        return output_root
    try:
        _fsync_directory(failed_path.parent)
    except OSError:
        return failed_path
    return failed_path


def observe(
    request_path: Path,
    output_root: Path,
    implementation_revision: str | None,
    preparation_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run two offline canonical observations from one prepared immutable request."""
    if output_root.exists():
        raise CaptureContractError(
            f"refusing to reuse output root {output_root}; observation evidence directories are immutable"
        )

    # Authenticate the executing package revision before accepting the external
    # preregistration template as the definition of this experiment.
    try:
        repository_commit = resolve_implementation_revision(
            implementation_revision,
            require_checkout=True,
        )
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"unable to bind production observation to a clean repository revision: {exc}"
        ) from exc

    request = _assert_exact_experiment_request(
        _read_json(request_path),
        require_receipts=True,
        repository_commit=repository_commit,
    )
    source_preparation_receipt_path = (
        Path(preparation_receipt_path)
        if preparation_receipt_path is not None
        else _default_preparation_receipt_path(request_path)
    )
    preparation_receipt = verify_preparation_receipt(
        _read_json(source_preparation_receipt_path),
        request=request,
        experiment_id=EXPERIMENT_ID,
    )

    attempt_nonce = secrets.token_hex(16)
    execution_ids = {
        "run-a": f"{EXPERIMENT_ID}:{attempt_nonce}:run-a",
        "run-b": f"{EXPERIMENT_ID}:{attempt_nonce}:run-b",
    }
    _create_output_root_durable(output_root)

    try:
        validated_request_path = output_root / "validated-request.json"
        archived_preparation_receipt_path = output_root / "preparation-receipt.json"
        _exclusive_write_json(validated_request_path, request)
        _exclusive_write_json(
            archived_preparation_receipt_path,
            preparation_receipt,
        )
        if _read_json(validated_request_path) != request:
            raise CaptureContractError(
                "validated request snapshot changed immediately after publication"
            )
        archived_preparation_receipt = verify_preparation_receipt(
            _read_json(archived_preparation_receipt_path),
            request=request,
            experiment_id=EXPERIMENT_ID,
        )
        if archived_preparation_receipt != preparation_receipt:
            raise CaptureContractError(
                "preparation receipt snapshot changed immediately after publication"
            )

        run_a = output_root / "run-a"
        run_b = output_root / "run-b"
        run_a_execution_receipt = output_root / "run-a-execution-receipt.json"
        run_b_execution_receipt = output_root / "run-b-execution-receipt.json"
        manifest_a = _run_capture(
            validated_request_path,
            run_a,
            repository_commit,
            execution_ids["run-a"],
            run_a_execution_receipt,
        )
        manifest_b = _run_capture(
            validated_request_path,
            run_b,
            repository_commit,
            execution_ids["run-b"],
            run_b_execution_receipt,
        )

        verdict = build_replay_verdict(
            request=request,
            validated_request_path=validated_request_path,
            preparation_receipt_path=archived_preparation_receipt_path,
            run_a_dir=run_a,
            run_b_dir=run_b,
            run_a_execution_receipt_path=run_a_execution_receipt,
            run_b_execution_receipt_path=run_b_execution_receipt,
            run_a_manifest_receipt=manifest_a,
            run_b_manifest_receipt=manifest_b,
            experiment_id=EXPERIMENT_ID,
        )
        verdict_path = output_root / "replay-verdict.json"
        _exclusive_write_json(verdict_path, verdict)
        persisted = _read_json(verdict_path)
        verify_replay_verdict(
            persisted,
            request=request,
            validated_request_path=validated_request_path,
            preparation_receipt_path=archived_preparation_receipt_path,
            run_a_dir=run_a,
            run_b_dir=run_b,
            run_a_execution_receipt_path=run_a_execution_receipt,
            run_b_execution_receipt_path=run_b_execution_receipt,
            run_a_manifest_receipt=manifest_a,
            run_b_manifest_receipt=manifest_b,
            experiment_id=EXPERIMENT_ID,
        )
        return persisted, 0 if persisted["replay_outcome"] == "byte_identical" else 2
    except BaseException as exc:
        failed_path = _preserve_failed_attempt(
            output_root,
            exc,
            repository_commit,
            execution_ids,
            preparation_receipt,
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise CaptureContractError(
            f"production observation attempt failed: {exc}; incomplete evidence preserved at {failed_path}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and execute the frozen GEO-CAP-001-EXP-001 first production observation"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help=(
            "ONLINE: warm exact canonical-Hub commits and write the final request plus preparation provenance"
        ),
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--implementation-revision",
        default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION"),
        help=(
            "Optional immutable repository commit to bind preparation explicitly. "
            "If omitted, preparation requires a clean Git checkout and binds HEAD."
        ),
    )
    observe_parser = subparsers.add_parser(
        "observe",
        help="OFFLINE: execute the prepared request twice and record the replay verdict",
    )
    observe_parser.add_argument("--request", type=Path, required=True)
    observe_parser.add_argument("--output-root", type=Path, required=True)
    observe_parser.add_argument(
        "--preparation-receipt",
        type=Path,
        help=(
            "Preparation provenance receipt emitted by prepare. If omitted, derive the "
            "canonical sidecar path from --request."
        ),
    )
    observe_parser.add_argument(
        "--implementation-revision",
        default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION"),
        help=(
            "Optional immutable repository commit to bind explicitly. If omitted, the "
            "authenticated package orchestrator requires a clean Git checkout and binds HEAD."
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            request_sha256 = prepare(
                args.output,
                args.implementation_revision,
            )
            print(request_sha256)
            return 0
        verdict, status = observe(
            args.request,
            args.output_root,
            args.implementation_revision,
            args.preparation_receipt,
        )
        print(json.dumps(verdict, sort_keys=True, separators=(",", ":")))
        return status
    except (
        CaptureContractError,
        CaptureBackendUnavailable,
        SourceIdentityError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

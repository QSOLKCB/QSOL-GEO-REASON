"""Freeze and execute GEO-CAP-001-EXP-001 without weakening the evidence boundary.

The experiment has two deliberately separate stages:

1. ``prepare`` is the only online stage. It warms the exact immutable Hugging Face
   snapshot(s), creates authenticated QSOL Hub-tree receipts, injects those receipts
   into the preregistered request template, and writes one final request.
2. ``observe`` is offline. It snapshots the validated final request, executes that
   exact snapshot twice through the canonical fresh-worker production backend,
   verifies both bundles, and records a machine-verifiable replay verdict.

A replay divergence is retained as a result. A failed attempt is moved to a uniquely
named failure directory instead of occupying the requested final output path. This tool
never upgrades a capture beyond OBSERVATION and never interprets a hidden-state
trajectory as mechanism evidence.
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

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture_common import (
    CaptureBackendUnavailable,
    CaptureContractError,
)
from qsol_geo_reason.capture_hub_tree import prepare_tree_receipts
from qsol_geo_reason.capture_replay import (
    REPLAY_BUNDLE_FILES,
    build_replay_verdict,
    verify_replay_verdict,
)
from qsol_geo_reason.capture_validation import validate_capture_request
from qsol_geo_reason.provenance import (
    SourceIdentityError,
    resolve_implementation_revision,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "GEO-CAP-001-EXP-001"
TEMPLATE = ROOT / "experiments" / "GEO-CAP-001-EXP-001.request.template.json"
BUNDLE_FILES = REPLAY_BUNDLE_FILES
TREE_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(f"unable to read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureContractError(f"{path} must contain a JSON object")
    return value


def _load_template() -> dict[str, Any]:
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
    request: Mapping[str, Any], *, require_receipts: bool
) -> dict[str, Any]:
    validated = validate_capture_request(request)
    template = _load_template()
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


def _fsync_directory(path: Path) -> None:
    """Durably publish directory metadata on POSIX filesystems."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_parent_directory_durable(path: Path) -> None:
    """Create missing directories one at a time and fsync each new parent entry."""
    path = Path(path)
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise CaptureContractError(
                "unable to locate an existing ancestor for experiment artifact publication"
            )
        current = parent
    if not current.is_dir():
        raise CaptureContractError(
            "experiment artifact parent ancestry must contain only directories"
        )

    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise CaptureContractError(
                    "experiment artifact parent path was replaced by a non-directory"
                )
        else:
            # The new child name lives in directory.parent. Sync that containing
            # directory immediately so every newly created ancestor is reachable
            # after a crash before deeper descendants are published.
            _fsync_directory(directory.parent)


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
        # No experiment artifact exists yet. Best-effort removal keeps a failed
        # durability probe from poisoning the explicit retry path.
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise CaptureContractError(
            f"unable to durably publish observation output root {output_root}: {exc}"
        ) from exc


def _unique_sibling(path: Path, marker: str) -> Path:
    """Return an unpredictable sibling path without creating it."""
    while True:
        candidate = path.with_name(
            f".{path.name}.{marker}.{os.getpid()}.{secrets.token_hex(8)}"
        )
        if not candidate.exists():
            return candidate


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Stage, fsync, and publish immutable JSON with no-replace semantics."""
    _ensure_parent_directory_durable(path.parent)
    if path.exists():
        raise CaptureContractError(f"refusing to overwrite existing artifact {path}")

    payload = _canonical_bytes(value)
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
        if published:
            try:
                _fsync_directory(path.parent)
            except OSError:
                pass


def prepare(output: Path) -> str:
    """Online warm-up that materializes the final immutable production request."""
    template = _load_template()
    receipts = prepare_tree_receipts(template)
    final_request = json.loads(json.dumps(template))
    final_request["model"].update(receipts)
    final_request = _assert_exact_experiment_request(
        final_request, require_receipts=True
    )
    _exclusive_write_json(output, final_request)
    return sha256_json(final_request)


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
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
    implementation_revision: str | None,
) -> str:
    # Security boundary: this is a fixed argv vector with shell=False. User-supplied
    # paths are individual argv elements and are never interpolated into a shell string.
    command = [
        sys.executable,
        "-m",
        "qsol_geo_reason.capture_cli",
        str(request_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if implementation_revision:
        command.extend(["--implementation-revision", implementation_revision])
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
) -> Path:
    """Move an incomplete attempt aside so the requested output path can be retried."""
    marker = {
        "schema_version": "1.0.0",
        "protocol_id": "GEO-CAP-001",
        "experiment_id": EXPERIMENT_ID,
        "repository_commit": repository_commit,
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
        # The partial capture directories remain useful forensic evidence even when
        # the filesystem cannot accept the marker itself.
        pass

    failed_path = output_root.with_name(
        f"{output_root.name}.failed-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        os.rename(output_root, failed_path)
        _fsync_directory(failed_path.parent)
        return failed_path
    except OSError:
        # Preserve rather than delete evidence if the failure directory cannot move.
        return output_root


def observe(
    request_path: Path,
    output_root: Path,
    implementation_revision: str | None,
) -> tuple[dict[str, Any], int]:
    """Run two offline canonical observations from one immutable request snapshot."""
    request = _assert_exact_experiment_request(
        _read_json(request_path), require_receipts=True
    )
    if output_root.exists():
        raise CaptureContractError(
            f"refusing to reuse output root {output_root}; observation evidence directories are immutable"
        )

    # Bind the executing checkout before any evidence directory is created and before
    # either worker starts. Both workers receive this exact revision, and a failed
    # attempt records the same identity even when no run bundle was published.
    try:
        repository_commit = resolve_implementation_revision(
            implementation_revision,
            require_checkout=True,
        )
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"unable to bind production observation to a clean repository revision: {exc}"
        ) from exc

    _create_output_root_durable(output_root)

    try:
        # Snapshot the already validated request before either worker starts. Both
        # subprocesses consume this staged artifact rather than re-opening the caller's
        # mutable source path. The semantic replay verifier later requires each bundled
        # capture-request.json to equal this exact request including both Hub receipts.
        validated_request_path = output_root / "validated-request.json"
        _exclusive_write_json(validated_request_path, request)
        if _read_json(validated_request_path) != request:
            raise CaptureContractError(
                "validated request snapshot changed immediately after publication"
            )

        run_a = output_root / "run-a"
        run_b = output_root / "run-b"
        manifest_a = _run_capture(
            validated_request_path, run_a, repository_commit
        )
        manifest_b = _run_capture(
            validated_request_path, run_b, repository_commit
        )

        verdict = build_replay_verdict(
            request=request,
            validated_request_path=validated_request_path,
            run_a_dir=run_a,
            run_b_dir=run_b,
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
            run_a_dir=run_a,
            run_b_dir=run_b,
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
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise CaptureContractError(
            f"production observation attempt failed: {exc}; incomplete evidence preserved at {failed_path}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and execute the frozen GEO-CAP-001-EXP-001 first production observation"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="ONLINE: warm exact Hub commits and write the final request with tree receipts",
    )
    prepare_parser.add_argument("--output", type=Path, required=True)

    observe_parser = subparsers.add_parser(
        "observe",
        help="OFFLINE: execute the final request twice and record the replay verdict",
    )
    observe_parser.add_argument("--request", type=Path, required=True)
    observe_parser.add_argument("--output-root", type=Path, required=True)
    observe_parser.add_argument(
        "--implementation-revision",
        default=os.environ.get("QSOL_GEO_REASON_IMPLEMENTATION_REVISION"),
        help=(
            "Optional immutable repository commit to bind explicitly. If omitted, the canonical worker requires a clean Git checkout and binds HEAD."
        ),
    )

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            request_sha256 = prepare(args.output)
            print(request_sha256)
            return 0
        verdict, status = observe(
            args.request,
            args.output_root,
            args.implementation_revision,
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

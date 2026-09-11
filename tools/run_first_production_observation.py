"""Freeze and execute GEO-CAP-001-EXP-001 without weakening the evidence boundary.

The experiment has two deliberately separate stages:

1. ``prepare`` is the only online stage. It warms the exact immutable Hugging Face
   snapshot(s), creates authenticated QSOL Hub-tree receipts, injects those receipts
   into the preregistered request template, and writes one final request.
2. ``observe`` is offline. It executes that exact final request twice through the
   canonical fresh-worker production backend, verifies both bundles, and records a
   replay verdict without rewriting either observation bundle.

A replay divergence is retained as a result. A failed attempt is moved to a uniquely
named failure directory instead of occupying the requested final output path. This tool
never upgrades a capture beyond OBSERVATION and never interprets a hidden-state
trajectory as mechanism evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import verify_capture_bundle
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_hub_tree import prepare_tree_receipts
from qsol_geo_reason.capture_validation import validate_capture_request


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "GEO-CAP-001-EXP-001"
TEMPLATE = ROOT / "experiments" / "GEO-CAP-001-EXP-001.request.template.json"
BUNDLE_FILES = (
    "capture-request.json",
    "run-manifest.json",
    "captured-trajectory.json",
)
TREE_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _unique_sibling(path: Path, marker: str) -> Path:
    """Return an unpredictable sibling path without creating it."""
    while True:
        candidate = path.with_name(
            f".{path.name}.{marker}.{os.getpid()}.{secrets.token_hex(8)}"
        )
        if not candidate.exists():
            return candidate


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write durable JSON and publish it only after the complete payload is synced.

    A sibling temporary file is fully written and fsynced first. A hard-link publish
    provides no-replace semantics: an existing destination can never be overwritten.
    Any ordinary pre-publication failure removes the temporary file, so a later retry
    is not blocked by a partial destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_bundle(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return tuple(  # type: ignore[return-value]
        _read_json(directory / name) for name in BUNDLE_FILES
    )


def _verify_observation_bundle(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request, manifest, trajectory = _read_bundle(directory)
    _assert_exact_experiment_request(request, require_receipts=True)
    verify_capture_bundle(request, manifest, trajectory)
    if trajectory.get("evidence_class") != "OBSERVATION":
        raise CaptureContractError(f"{directory} did not emit OBSERVATION evidence")
    return request, manifest, trajectory


def _file_receipts(directory: Path) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for name in BUNDLE_FILES:
        try:
            payload = (directory / name).read_bytes()
        except OSError as exc:
            raise CaptureContractError(
                f"unable to hash {directory / name}: {exc}"
            ) from exc
        receipts[name] = _sha256_bytes(payload)
    return receipts


def _completed_run_directories(output_root: Path) -> list[str]:
    completed: list[str] = []
    for name in ("run-a", "run-b"):
        directory = output_root / name
        if directory.is_dir() and all((directory / item).is_file() for item in BUNDLE_FILES):
            completed.append(name)
    return completed


def _preserve_failed_attempt(output_root: Path, exc: BaseException) -> Path:
    """Move an incomplete attempt aside so the requested output path can be retried."""
    marker = {
        "schema_version": "1.0.0",
        "protocol_id": "GEO-CAP-001",
        "experiment_id": EXPERIMENT_ID,
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
        # The partial capture directories themselves remain useful forensic evidence
        # even if a full filesystem cannot accept the marker.
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
    """Run two offline canonical observations and preserve the replay outcome."""
    request = _assert_exact_experiment_request(
        _read_json(request_path), require_receipts=True
    )
    if output_root.exists():
        raise CaptureContractError(
            f"refusing to reuse output root {output_root}; observation evidence directories are immutable"
        )
    output_root.mkdir(parents=True, exist_ok=False)

    try:
        run_a = output_root / "run-a"
        run_b = output_root / "run-b"
        manifest_a = _run_capture(request_path, run_a, implementation_revision)
        manifest_b = _run_capture(request_path, run_b, implementation_revision)

        _, first_manifest, first_trajectory = _verify_observation_bundle(run_a)
        _, second_manifest, second_trajectory = _verify_observation_bundle(run_b)
        receipts_a = _file_receipts(run_a)
        receipts_b = _file_receipts(run_b)
        equality = {
            name: receipts_a[name] == receipts_b[name] for name in BUNDLE_FILES
        }
        byte_identical = all(equality.values()) and manifest_a == manifest_b

        verdict: dict[str, Any] = {
            "schema_version": "1.0.0",
            "protocol_id": "GEO-CAP-001",
            "experiment_id": EXPERIMENT_ID,
            "evidence_class": "OBSERVATION",
            "request_sha256": sha256_json(request),
            "run_a_manifest_receipt_sha256": manifest_a,
            "run_b_manifest_receipt_sha256": manifest_b,
            "run_a_bundle_file_sha256": receipts_a,
            "run_b_bundle_file_sha256": receipts_b,
            "bundle_file_byte_equality": equality,
            "manifest_receipts_equal": manifest_a == manifest_b,
            "trajectory_sha256_equal": (
                first_trajectory.get("trajectory_sha256")
                == second_trajectory.get("trajectory_sha256")
            ),
            "repository_commit_equal": (
                first_manifest.get("repository_commit")
                == second_manifest.get("repository_commit")
            ),
            "replay_outcome": "byte_identical" if byte_identical else "diverged",
            "interpretation": (
                "Deterministic replay established for this exact request/backend/runtime pair."
                if byte_identical
                else "Deterministic replay was not established. Preserve both observations and investigate the recorded divergence; do not tune or discard it."
            ),
        }
        _exclusive_write_json(output_root / "replay-verdict.json", verdict)
        return verdict, 0 if byte_identical else 2
    except BaseException as exc:
        failed_path = _preserve_failed_attempt(output_root, exc)
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
    except (CaptureContractError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

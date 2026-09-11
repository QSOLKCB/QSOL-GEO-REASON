"""Per-execution provenance receipts for canonical GEO-CAP-001 capture bundles.

A capture request may be frozen and replayed byte-for-byte.  ``run_id`` and
``run_manifest_id`` therefore identify scientific/content state, not the physical
occurrence of an execution.  This module adds a separate occurrence identity without
mutating the preregistered request or the three canonical bundle files.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_bytes, sha256_json
from .capture_common import (
    CAPTURE_PROTOCOL_ID,
    CAPTURE_SCHEMA_VERSION,
    CaptureContractError,
    _require_git_sha,
    _require_nonempty_string,
)
from .capture_publish import _ensure_parent_directory_durable, _fsync_directory
from .capture_verify import verify_capture_bundle

EXECUTION_RECEIPT_SCHEMA_VERSION = "1.0.0"
EXECUTION_RECEIPT_FILENAME = "execution-receipt.json"
_EXECUTION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "execution_id",
        "request_run_id",
        "repository_commit",
        "request_sha256",
        "run_manifest_id",
        "manifest_sha256",
        "trajectory_sha256",
        "capture_request_file_sha256",
        "run_manifest_file_sha256",
        "captured_trajectory_file_sha256",
        "execution_receipt_sha256",
    }
)


def _require_sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptureContractError(f"{where} must be a lowercase 64-hex SHA-256")
    return value


def _canonical_file_sha256(value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _receipt_payload(
    *,
    execution_id: str,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    validated = verify_capture_bundle(request, manifest, trajectory)
    execution_id = _require_nonempty_string(execution_id, "execution_id")
    repository_commit = _require_git_sha(
        manifest.get("repository_commit"), "manifest.repository_commit"
    )
    return {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "protocol_id": CAPTURE_PROTOCOL_ID,
        "execution_id": execution_id,
        "request_run_id": validated["run_id"],
        "repository_commit": repository_commit,
        "request_sha256": sha256_json(validated),
        "run_manifest_id": _require_sha256(
            manifest.get("run_manifest_id"), "manifest.run_manifest_id"
        ),
        "manifest_sha256": _require_sha256(
            manifest.get("manifest_sha256"), "manifest.manifest_sha256"
        ),
        "trajectory_sha256": _require_sha256(
            trajectory.get("trajectory_sha256"), "trajectory.trajectory_sha256"
        ),
        "capture_request_file_sha256": _canonical_file_sha256(validated),
        "run_manifest_file_sha256": _canonical_file_sha256(manifest),
        "captured_trajectory_file_sha256": _canonical_file_sha256(trajectory),
    }


def build_execution_receipt(
    *,
    execution_id: str,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-hashed occurrence receipt bound to one verified bundle."""
    payload = _receipt_payload(
        execution_id=execution_id,
        request=request,
        manifest=manifest,
        trajectory=trajectory,
    )
    return {**payload, "execution_receipt_sha256": sha256_json(payload)}


def verify_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on edited or cross-bundle execution occurrence metadata."""
    if not isinstance(receipt, dict):
        raise CaptureContractError("execution receipt must be an object")
    actual_keys = set(receipt)
    if actual_keys != _EXECUTION_RECEIPT_KEYS:
        missing = sorted(_EXECUTION_RECEIPT_KEYS - actual_keys)
        extra = sorted(actual_keys - _EXECUTION_RECEIPT_KEYS)
        raise CaptureContractError(
            f"execution receipt keys are not canonical: missing={missing}; extra={extra}"
        )
    if receipt["schema_version"] != EXECUTION_RECEIPT_SCHEMA_VERSION:
        raise CaptureContractError("execution receipt schema_version is invalid")
    if receipt["protocol_id"] != CAPTURE_PROTOCOL_ID:
        raise CaptureContractError("execution receipt protocol_id is invalid")

    expected_payload = _receipt_payload(
        execution_id=_require_nonempty_string(
            receipt["execution_id"], "execution receipt execution_id"
        ),
        request=request,
        manifest=manifest,
        trajectory=trajectory,
    )
    observed_payload = {
        key: receipt[key] for key in _EXECUTION_RECEIPT_KEYS if key != "execution_receipt_sha256"
    }
    if canonical_json_bytes(observed_payload) != canonical_json_bytes(expected_payload):
        raise CaptureContractError(
            "execution receipt does not match the verified capture bundle"
        )
    receipt_sha = _require_sha256(
        receipt["execution_receipt_sha256"], "execution_receipt_sha256"
    )
    if receipt_sha != sha256_json(expected_payload):
        raise CaptureContractError("execution receipt SHA-256 is invalid")
    return dict(receipt)


def write_execution_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Durably publish one execution receipt with no-replace semantics."""
    path = Path(path)
    _ensure_parent_directory_durable(path.parent)
    if path.exists():
        raise CaptureContractError(f"refusing to overwrite execution receipt {path}")
    try:
        payload = canonical_json_bytes(receipt) + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CaptureContractError("execution receipt is not canonical JSON") from exc

    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
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
                f"refusing to overwrite execution receipt {path}"
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
        raise CaptureContractError(f"unable to persist execution receipt {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "EXECUTION_RECEIPT_FILENAME",
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "build_execution_receipt",
    "verify_execution_receipt",
    "write_execution_receipt",
]

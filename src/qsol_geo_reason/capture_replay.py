"""Machine-verifiable replay evidence for canonical GEO-CAP-001 observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import CaptureContractError
from .capture_verify import verify_capture_bundle

REPLAY_VERDICT_SCHEMA_VERSION = "1.0.0"
REPLAY_VERDICT_PROTOCOL_ID = "GEO-CAP-001"
REPLAY_REPLICATION_STATUS = "not_attempted"
REPLAY_BUNDLE_FILES = (
    "capture-request.json",
    "run-manifest.json",
    "captured-trajectory.json",
)
REPLAY_INTERPRETATIONS = {
    "byte_identical": (
        "Deterministic replay established for this exact request/backend/runtime pair."
    ),
    "diverged": (
        "Deterministic replay was not established. Preserve both observations and "
        "investigate the recorded divergence; do not tune or discard it."
    ),
}
_VERDICT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "experiment_id",
        "evidence_class",
        "replication_status",
        "request_sha256",
        "validated_request_artifact_sha256",
        "run_a_manifest_receipt_sha256",
        "run_b_manifest_receipt_sha256",
        "run_a_bundle_file_sha256",
        "run_b_bundle_file_sha256",
        "bundle_file_byte_equality",
        "manifest_receipts_equal",
        "trajectory_sha256_equal",
        "repository_commit_equal",
        "replay_outcome",
        "interpretation",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(
            f"unable to read replay evidence JSON from {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CaptureContractError(
            f"replay evidence file {path} must contain a JSON object"
        )
    return value


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CaptureContractError(
            f"unable to hash replay evidence file {path}: {exc}"
        ) from exc


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], where: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if extra:
            parts.append("extra=" + ",".join(extra))
        raise CaptureContractError(
            f"{where} keys are not canonical: {'; '.join(parts)}"
        )


def _require_nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureContractError(f"{where} must be a non-empty string")
    return value


def _require_sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptureContractError(
            f"{where} must be a lowercase 64-hex SHA-256"
        )
    return value


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise CaptureContractError(f"{where} must be boolean")
    return value


def _require_receipt_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CaptureContractError(f"{where} must be an object")
    expected = frozenset(REPLAY_BUNDLE_FILES)
    _require_exact_keys(value, expected, where)
    return {
        name: _require_sha256(value[name], f"{where}.{name}")
        for name in REPLAY_BUNDLE_FILES
    }


def _require_equality_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise CaptureContractError("bundle_file_byte_equality must be an object")
    expected = frozenset(REPLAY_BUNDLE_FILES)
    _require_exact_keys(value, expected, "bundle_file_byte_equality")
    return {
        name: _require_bool(
            value[name], f"bundle_file_byte_equality.{name}"
        )
        for name in REPLAY_BUNDLE_FILES
    }


def _bundle_file_receipts(directory: Path) -> dict[str, str]:
    return {
        name: _sha256_file(directory / name)
        for name in REPLAY_BUNDLE_FILES
    }


def _load_verified_observation_bundle(
    directory: Path,
    expected_request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = _read_json(directory / "capture-request.json")
    manifest = _read_json(directory / "run-manifest.json")
    trajectory = _read_json(directory / "captured-trajectory.json")
    if request != expected_request:
        raise CaptureContractError(
            f"{directory} bundled request does not equal the immutable validated request snapshot"
        )
    verify_capture_bundle(request, manifest, trajectory)
    if trajectory.get("evidence_class") != "OBSERVATION":
        raise CaptureContractError(
            f"{directory} is not an OBSERVATION capture bundle"
        )
    return request, manifest, trajectory


def _actual_replay_state(
    *,
    request: Mapping[str, Any],
    validated_request_path: Path,
    run_a_dir: Path,
    run_b_dir: Path,
    run_a_manifest_receipt: str,
    run_b_manifest_receipt: str,
) -> dict[str, Any]:
    snapshot = _read_json(validated_request_path)
    if snapshot != request:
        raise CaptureContractError(
            "validated request artifact does not equal the in-memory frozen request"
        )
    request_artifact_sha256 = _sha256_file(validated_request_path)

    _, manifest_a, trajectory_a = _load_verified_observation_bundle(
        run_a_dir, request
    )
    _, manifest_b, trajectory_b = _load_verified_observation_bundle(
        run_b_dir, request
    )

    manifest_a_sha = _require_sha256(
        manifest_a.get("manifest_sha256"), "run-a manifest.manifest_sha256"
    )
    manifest_b_sha = _require_sha256(
        manifest_b.get("manifest_sha256"), "run-b manifest.manifest_sha256"
    )
    if (
        _require_sha256(
            run_a_manifest_receipt, "run-a CLI manifest receipt"
        )
        != manifest_a_sha
    ):
        raise CaptureContractError(
            "run-a CLI manifest receipt does not match run-manifest.json"
        )
    if (
        _require_sha256(
            run_b_manifest_receipt, "run-b CLI manifest receipt"
        )
        != manifest_b_sha
    ):
        raise CaptureContractError(
            "run-b CLI manifest receipt does not match run-manifest.json"
        )

    receipts_a = _bundle_file_receipts(run_a_dir)
    receipts_b = _bundle_file_receipts(run_b_dir)
    equality = {
        name: receipts_a[name] == receipts_b[name]
        for name in REPLAY_BUNDLE_FILES
    }
    manifest_equal = manifest_a_sha == manifest_b_sha
    trajectory_equal = (
        trajectory_a.get("trajectory_sha256")
        == trajectory_b.get("trajectory_sha256")
    )
    repository_equal = (
        manifest_a.get("repository_commit")
        == manifest_b.get("repository_commit")
    )
    byte_identical = all(equality.values()) and manifest_equal

    return {
        "request_artifact_sha256": request_artifact_sha256,
        "manifest_a_sha256": manifest_a_sha,
        "manifest_b_sha256": manifest_b_sha,
        "receipts_a": receipts_a,
        "receipts_b": receipts_b,
        "equality": equality,
        "manifest_equal": manifest_equal,
        "trajectory_equal": trajectory_equal,
        "repository_equal": repository_equal,
        "outcome": "byte_identical" if byte_identical else "diverged",
    }


def build_replay_verdict(
    *,
    request: Mapping[str, Any],
    validated_request_path: Path,
    run_a_dir: Path,
    run_b_dir: Path,
    run_a_manifest_receipt: str,
    run_b_manifest_receipt: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Construct a verdict solely from verified immutable request/bundle artifacts."""
    experiment_id = _require_nonempty_string(experiment_id, "experiment_id")
    state = _actual_replay_state(
        request=request,
        validated_request_path=validated_request_path,
        run_a_dir=run_a_dir,
        run_b_dir=run_b_dir,
        run_a_manifest_receipt=run_a_manifest_receipt,
        run_b_manifest_receipt=run_b_manifest_receipt,
    )
    outcome = state["outcome"]
    verdict: dict[str, Any] = {
        "schema_version": REPLAY_VERDICT_SCHEMA_VERSION,
        "protocol_id": REPLAY_VERDICT_PROTOCOL_ID,
        "experiment_id": experiment_id,
        "evidence_class": "OBSERVATION",
        "replication_status": REPLAY_REPLICATION_STATUS,
        "request_sha256": sha256_json(request),
        "validated_request_artifact_sha256": state[
            "request_artifact_sha256"
        ],
        "run_a_manifest_receipt_sha256": state["manifest_a_sha256"],
        "run_b_manifest_receipt_sha256": state["manifest_b_sha256"],
        "run_a_bundle_file_sha256": state["receipts_a"],
        "run_b_bundle_file_sha256": state["receipts_b"],
        "bundle_file_byte_equality": state["equality"],
        "manifest_receipts_equal": state["manifest_equal"],
        "trajectory_sha256_equal": state["trajectory_equal"],
        "repository_commit_equal": state["repository_equal"],
        "replay_outcome": outcome,
        "interpretation": REPLAY_INTERPRETATIONS[outcome],
    }
    verify_replay_verdict(
        verdict,
        request=request,
        validated_request_path=validated_request_path,
        run_a_dir=run_a_dir,
        run_b_dir=run_b_dir,
        run_a_manifest_receipt=run_a_manifest_receipt,
        run_b_manifest_receipt=run_b_manifest_receipt,
        experiment_id=experiment_id,
    )
    return verdict


def verify_replay_verdict(
    verdict: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    validated_request_path: Path,
    run_a_dir: Path,
    run_b_dir: Path,
    run_a_manifest_receipt: str,
    run_b_manifest_receipt: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Fail closed on malformed, edited, or cross-bundle-inconsistent replay evidence."""
    if not isinstance(verdict, dict):
        raise CaptureContractError("replay verdict must be an object")
    _require_exact_keys(verdict, _VERDICT_KEYS, "replay verdict")
    if verdict["schema_version"] != REPLAY_VERDICT_SCHEMA_VERSION:
        raise CaptureContractError(
            "replay verdict schema_version is invalid"
        )
    if verdict["protocol_id"] != REPLAY_VERDICT_PROTOCOL_ID:
        raise CaptureContractError("replay verdict protocol_id is invalid")
    expected_experiment_id = _require_nonempty_string(
        experiment_id, "expected experiment_id"
    )
    observed_experiment_id = _require_nonempty_string(
        verdict["experiment_id"], "replay verdict experiment_id"
    )
    if observed_experiment_id != expected_experiment_id:
        raise CaptureContractError("replay verdict experiment_id is invalid")
    if verdict["evidence_class"] != "OBSERVATION":
        raise CaptureContractError(
            "replay verdict evidence_class must be OBSERVATION"
        )
    if verdict["replication_status"] != REPLAY_REPLICATION_STATUS:
        raise CaptureContractError(
            "replay verdict replication_status must be not_attempted"
        )

    request_sha = _require_sha256(
        verdict["request_sha256"], "request_sha256"
    )
    request_artifact_sha = _require_sha256(
        verdict["validated_request_artifact_sha256"],
        "validated_request_artifact_sha256",
    )
    manifest_a_sha = _require_sha256(
        verdict["run_a_manifest_receipt_sha256"],
        "run_a_manifest_receipt_sha256",
    )
    manifest_b_sha = _require_sha256(
        verdict["run_b_manifest_receipt_sha256"],
        "run_b_manifest_receipt_sha256",
    )
    receipts_a = _require_receipt_map(
        verdict["run_a_bundle_file_sha256"], "run_a_bundle_file_sha256"
    )
    receipts_b = _require_receipt_map(
        verdict["run_b_bundle_file_sha256"], "run_b_bundle_file_sha256"
    )
    equality = _require_equality_map(verdict["bundle_file_byte_equality"])
    manifest_equal = _require_bool(
        verdict["manifest_receipts_equal"], "manifest_receipts_equal"
    )
    trajectory_equal = _require_bool(
        verdict["trajectory_sha256_equal"], "trajectory_sha256_equal"
    )
    repository_equal = _require_bool(
        verdict["repository_commit_equal"], "repository_commit_equal"
    )
    outcome = verdict["replay_outcome"]
    if not isinstance(outcome, str) or outcome not in REPLAY_INTERPRETATIONS:
        raise CaptureContractError(
            "replay_outcome must be byte_identical or diverged"
        )
    if verdict["interpretation"] != REPLAY_INTERPRETATIONS[outcome]:
        raise CaptureContractError(
            "replay verdict interpretation does not match replay_outcome"
        )

    state = _actual_replay_state(
        request=request,
        validated_request_path=validated_request_path,
        run_a_dir=run_a_dir,
        run_b_dir=run_b_dir,
        run_a_manifest_receipt=run_a_manifest_receipt,
        run_b_manifest_receipt=run_b_manifest_receipt,
    )
    expected = {
        "request_sha256": sha256_json(request),
        "validated_request_artifact_sha256": state[
            "request_artifact_sha256"
        ],
        "run_a_manifest_receipt_sha256": state["manifest_a_sha256"],
        "run_b_manifest_receipt_sha256": state["manifest_b_sha256"],
        "run_a_bundle_file_sha256": state["receipts_a"],
        "run_b_bundle_file_sha256": state["receipts_b"],
        "bundle_file_byte_equality": state["equality"],
        "manifest_receipts_equal": state["manifest_equal"],
        "trajectory_sha256_equal": state["trajectory_equal"],
        "repository_commit_equal": state["repository_equal"],
        "replay_outcome": state["outcome"],
        "interpretation": REPLAY_INTERPRETATIONS[state["outcome"]],
    }
    observed = {
        "request_sha256": request_sha,
        "validated_request_artifact_sha256": request_artifact_sha,
        "run_a_manifest_receipt_sha256": manifest_a_sha,
        "run_b_manifest_receipt_sha256": manifest_b_sha,
        "run_a_bundle_file_sha256": receipts_a,
        "run_b_bundle_file_sha256": receipts_b,
        "bundle_file_byte_equality": equality,
        "manifest_receipts_equal": manifest_equal,
        "trajectory_sha256_equal": trajectory_equal,
        "repository_commit_equal": repository_equal,
        "replay_outcome": outcome,
        "interpretation": verdict["interpretation"],
    }
    if observed != expected:
        raise CaptureContractError(
            "replay verdict does not match the verified request and observation bundles"
        )
    return dict(verdict)


__all__ = [
    "REPLAY_BUNDLE_FILES",
    "REPLAY_REPLICATION_STATUS",
    "REPLAY_VERDICT_PROTOCOL_ID",
    "REPLAY_VERDICT_SCHEMA_VERSION",
    "build_replay_verdict",
    "verify_replay_verdict",
]

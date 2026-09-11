"""Machine-verifiable provenance for GEO-CAP-001 online preparation."""
from __future__ import annotations

from typing import Any, Mapping

from .canonical import sha256_json
from .capture_common import (
    CAPTURE_PROTOCOL_ID,
    CaptureContractError,
    _require_git_sha,
    _require_nonempty_string,
)
from .capture_hub_tree import CANONICAL_HF_ENDPOINT
from .capture_validation import validate_capture_request

PREPARATION_RECEIPT_SCHEMA_VERSION = "1.0.0"
_TREE_FIELDS = ("revision_tree_sha256", "tokenizer_revision_tree_sha256")
_PREPARATION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "experiment_id",
        "preparation_repository_commit",
        "hub_endpoint",
        "request_sha256",
        "model_repository",
        "model_revision",
        "tokenizer_repository",
        "tokenizer_revision",
        "revision_tree_sha256",
        "tokenizer_revision_tree_sha256",
        "preparation_receipt_sha256",
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


def _without(mapping: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key != field}


def build_preparation_receipt(
    *,
    request: Mapping[str, Any],
    repository_commit: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Bind one finalized prepared request to the clean code revision that prepared it."""
    validated = validate_capture_request(request)
    repository_commit = _require_git_sha(
        repository_commit, "preparation repository_commit"
    )
    experiment_id = _require_nonempty_string(experiment_id, "experiment_id")
    model = validated["model"]
    for field in _TREE_FIELDS:
        _require_sha256(model.get(field), f"request.model.{field}")

    payload: dict[str, Any] = {
        "schema_version": PREPARATION_RECEIPT_SCHEMA_VERSION,
        "protocol_id": CAPTURE_PROTOCOL_ID,
        "experiment_id": experiment_id,
        "preparation_repository_commit": repository_commit,
        "hub_endpoint": CANONICAL_HF_ENDPOINT,
        "request_sha256": sha256_json(validated),
        "model_repository": model["identifier"],
        "model_revision": model["revision"],
        "tokenizer_repository": model["tokenizer_identifier"],
        "tokenizer_revision": model["tokenizer_revision"],
        "revision_tree_sha256": model["revision_tree_sha256"],
        "tokenizer_revision_tree_sha256": model[
            "tokenizer_revision_tree_sha256"
        ],
    }
    payload["preparation_receipt_sha256"] = sha256_json(payload)
    return payload


def verify_preparation_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    """Fail closed on edited or request-inconsistent online-preparation provenance."""
    if not isinstance(receipt, dict):
        raise CaptureContractError("preparation receipt must be an object")
    actual_keys = set(receipt)
    missing = sorted(_PREPARATION_RECEIPT_KEYS - actual_keys)
    extra = sorted(actual_keys - _PREPARATION_RECEIPT_KEYS)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if extra:
            parts.append("extra=" + ",".join(extra))
        raise CaptureContractError(
            "preparation receipt keys are not canonical: " + "; ".join(parts)
        )

    validated = validate_capture_request(request)
    expected_experiment_id = _require_nonempty_string(
        experiment_id, "expected experiment_id"
    )
    observed_experiment_id = _require_nonempty_string(
        receipt["experiment_id"], "preparation receipt experiment_id"
    )
    if observed_experiment_id != expected_experiment_id:
        raise CaptureContractError("preparation receipt experiment_id is invalid")
    if receipt["schema_version"] != PREPARATION_RECEIPT_SCHEMA_VERSION:
        raise CaptureContractError("preparation receipt schema_version is invalid")
    if receipt["protocol_id"] != CAPTURE_PROTOCOL_ID:
        raise CaptureContractError("preparation receipt protocol_id is invalid")
    _require_git_sha(
        receipt["preparation_repository_commit"],
        "preparation receipt repository_commit",
    )
    if receipt["hub_endpoint"] != CANONICAL_HF_ENDPOINT:
        raise CaptureContractError(
            "preparation receipt hub_endpoint is not the canonical Hugging Face endpoint"
        )

    model = validated["model"]
    expected = {
        "request_sha256": sha256_json(validated),
        "model_repository": model["identifier"],
        "model_revision": model["revision"],
        "tokenizer_repository": model["tokenizer_identifier"],
        "tokenizer_revision": model["tokenizer_revision"],
        "revision_tree_sha256": _require_sha256(
            model.get("revision_tree_sha256"), "request.model.revision_tree_sha256"
        ),
        "tokenizer_revision_tree_sha256": _require_sha256(
            model.get("tokenizer_revision_tree_sha256"),
            "request.model.tokenizer_revision_tree_sha256",
        ),
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise CaptureContractError(
                f"preparation receipt {field} does not match the finalized request"
            )

    observed_hash = _require_sha256(
        receipt["preparation_receipt_sha256"],
        "preparation_receipt_sha256",
    )
    expected_hash = sha256_json(
        _without(receipt, "preparation_receipt_sha256")
    )
    if observed_hash != expected_hash:
        raise CaptureContractError("preparation receipt SHA-256 is invalid")
    return dict(receipt)


__all__ = [
    "PREPARATION_RECEIPT_SCHEMA_VERSION",
    "build_preparation_receipt",
    "verify_preparation_receipt",
]

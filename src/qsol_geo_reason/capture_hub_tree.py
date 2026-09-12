"""Trusted online warm-up/export for GEO-CAP-001 Hub tree receipts.

The ordinary Hugging Face cache does not contain QSOL's ``trees/<commit>.json``
artifact. This module creates that artifact explicitly from the canonical Hugging Face
Hub while the machine is online, warms the exact immutable snapshot, and returns the
SHA-256 that must then be frozen into the offline capture request.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_provenance import _is_canonical_snapshot_path
from .capture_snapshot import _TREE_CACHE_FORMAT_VERSION, _cached_hub_commit_tree
from .capture_validation import validate_capture_request

CANONICAL_HF_ENDPOINT = "https://huggingface.co"


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _repo_file_record(item: Any, where: str) -> tuple[str, dict[str, Any]] | None:
    """Normalize one ``huggingface_hub`` RepoFile across supported client versions."""
    path = _field(item, "path")
    size = _field(item, "size")
    blob_id = _field(item, "blob_id")
    if blob_id is None and size is None:
        return None
    if not _is_canonical_snapshot_path(path):
        raise CaptureContractError(f"Hub returned a noncanonical {where} tree path")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise CaptureContractError(f"Hub returned an invalid {where} tree size for {path!r}")
    if (
        not isinstance(blob_id, str)
        or len(blob_id) != 40
        or any(ch not in "0123456789abcdef" for ch in blob_id)
    ):
        raise CaptureContractError(
            f"Hub returned an invalid Git blob identity for {where} {path!r}"
        )

    record: dict[str, Any] = {"size": size, "blob_id": blob_id}
    lfs = _field(item, "lfs")
    if lfs is not None:
        lfs_sha256 = _field(lfs, "sha256")
        lfs_size = _field(lfs, "size")
        if (
            not isinstance(lfs_sha256, str)
            or len(lfs_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in lfs_sha256)
        ):
            raise CaptureContractError(
                f"Hub returned invalid LFS SHA-256 for {where} {path!r}"
            )
        if isinstance(lfs_size, bool) or not isinstance(lfs_size, int) or lfs_size < 0:
            raise CaptureContractError(
                f"Hub returned invalid LFS size for {where} {path!r}"
            )
        if lfs_size != size:
            raise CaptureContractError(
                f"Hub returned contradictory LFS size for {where} {path!r}"
            )
        record["lfs_sha256"] = lfs_sha256
        record["lfs_size"] = lfs_size
    return str(path), record


def _write_tree_artifact(
    snapshot: Path, commit: str, files: Mapping[str, Any], where: str
) -> str:
    if snapshot.parent.name != "snapshots" or snapshot.name.lower() != commit.lower():
        raise CaptureContractError(
            f"{where} warm-up did not resolve the requested immutable commit into a canonical snapshot"
        )
    payload = {
        "format_version": _TREE_CACHE_FORMAT_VERSION,
        "files": dict(sorted(files.items())),
    }
    tree_bytes = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    receipt = hashlib.sha256(tree_bytes).hexdigest()
    tree_dir = snapshot.parent.parent / "trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    destination = tree_dir / f"{commit.lower()}.json"
    temporary = tree_dir / f".{commit.lower()}.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(tree_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise CaptureContractError(
            f"unable to persist trusted {where} Hub tree artifact"
        ) from exc

    _cached_hub_commit_tree(
        snapshot.resolve(),
        commit,
        where,
        expected_tree_receipt_sha256=receipt,
    )
    return receipt


def _prepare_one(
    api: Any,
    snapshot_download: Any,
    repo_id: str,
    commit: str,
    where: str,
) -> str:
    try:
        info = api.model_info(repo_id=repo_id, revision=commit)
        resolved = getattr(info, "sha", None)
        if not isinstance(resolved, str) or resolved.lower() != commit.lower():
            raise CaptureContractError(
                f"trusted Hub warm-up for {where} did not resolve exactly to {commit}"
            )
        files: dict[str, dict[str, Any]] = {}
        for item in api.list_repo_tree(
            repo_id=repo_id, revision=commit, recursive=True
        ):
            normalized = _repo_file_record(item, where)
            if normalized is None:
                continue
            path, record = normalized
            if path in files:
                raise CaptureContractError(
                    f"Hub returned duplicate {where} tree path {path!r}"
                )
            files[path] = record
        if not files:
            raise CaptureContractError(
                f"trusted Hub warm-up returned no files for {where}"
            )
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=commit,
                local_files_only=False,
                endpoint=CANONICAL_HF_ENDPOINT,
            )
        )
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError(
            f"trusted online Hub warm-up failed for {where} {repo_id}@{commit}: {exc}"
        ) from exc
    return _write_tree_artifact(snapshot, commit, files, where)


def prepare_tree_receipts(request: Mapping[str, Any]) -> dict[str, str]:
    """Warm exact canonical-Hub revisions and export request-ready tree receipts."""
    validated = validate_capture_request(request)
    model = validated["model"]
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise CaptureBackendUnavailable(
            "Hub tree warm-up requires optional capture dependencies; install qsol-geo-reason[capture]"
        ) from exc

    # Never inherit HF_ENDPOINT from the ambient environment.  The preregistered
    # repository identity refers to the canonical public Hugging Face service, so both
    # metadata and snapshot retrieval are explicitly bound to that endpoint.
    api = HfApi(endpoint=CANONICAL_HF_ENDPOINT)
    cache: dict[tuple[str, str], str] = {}

    def prepare(repo_id: str, commit: str, where: str) -> str:
        key = (repo_id, commit)
        if key not in cache:
            cache[key] = _prepare_one(
                api, snapshot_download, repo_id, commit, where
            )
        return cache[key]

    return {
        "revision_tree_sha256": prepare(
            model["identifier"], model["revision"], "model"
        ),
        "tokenizer_revision_tree_sha256": prepare(
            model["tokenizer_identifier"],
            model["tokenizer_revision"],
            "tokenizer",
        ),
    }


__all__ = ["CANONICAL_HF_ENDPOINT", "prepare_tree_receipts"]

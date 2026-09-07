"""Snapshot byte authentication for GEO-CAP-001 production capture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .capture_common import CaptureContractError
from .capture_provenance import _is_canonical_snapshot_path


_TREE_CACHE_FORMAT_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1()
    digest.update(b"blob ")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _cached_hub_commit_tree(snapshot: Path, expected_commit: str, where: str) -> dict[str, Mapping[str, Any]]:
    """Load the Hub-derived immutable commit tree cached by ``snapshot_download``.

    Modern ``huggingface_hub`` stores ``trees/<commit>.json`` beside ``snapshots``.
    The tree is populated from the Hub repository tree endpoint and records each
    path's Git blob identity plus LFS content identity where applicable. Canonical
    OBSERVATION refuses basename-only snapshot attribution: the cached commit tree
    must exist and be structurally valid before local bytes can be associated with
    the requested Hub commit.
    """
    if snapshot.parent.name != "snapshots":
        raise CaptureContractError(
            f"{where} snapshot is not inside a canonical Hugging Face snapshots directory"
        )
    tree_path = snapshot.parent.parent / "trees" / f"{expected_commit.lower()}.json"
    try:
        payload = json.loads(tree_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaptureContractError(
            f"{where} snapshot lacks cached Hub commit-tree metadata for {expected_commit}; "
            "refresh the immutable revision with a current huggingface_hub snapshot_download before offline capture"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(
            f"unable to read cached Hub commit-tree metadata for {where} snapshot"
        ) from exc

    if not isinstance(payload, Mapping) or payload.get("format_version") != _TREE_CACHE_FORMAT_VERSION:
        raise CaptureContractError(f"{where} cached Hub commit-tree metadata has an unsupported format")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise CaptureContractError(f"{where} cached Hub commit tree contains no files")

    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_path, raw_info in files.items():
        if not _is_canonical_snapshot_path(raw_path) or not isinstance(raw_info, Mapping):
            raise CaptureContractError(f"{where} cached Hub commit tree is malformed")
        size = raw_info.get("size")
        blob_id = raw_info.get("blob_id")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CaptureContractError(f"{where} cached Hub tree has invalid size for {raw_path!r}")
        if not _lower_hex(blob_id, 40):
            raise CaptureContractError(f"{where} cached Hub tree has invalid Git blob identity for {raw_path!r}")

        lfs_sha256 = raw_info.get("lfs_sha256")
        lfs_size = raw_info.get("lfs_size")
        if lfs_sha256 is None:
            if lfs_size is not None:
                raise CaptureContractError(f"{where} cached Hub tree has incomplete LFS metadata for {raw_path!r}")
        else:
            if not _lower_hex(lfs_sha256, 64):
                raise CaptureContractError(f"{where} cached Hub tree has invalid LFS SHA-256 for {raw_path!r}")
            if isinstance(lfs_size, bool) or not isinstance(lfs_size, int) or lfs_size < 0:
                raise CaptureContractError(f"{where} cached Hub tree has invalid LFS size for {raw_path!r}")
            if lfs_size != size:
                raise CaptureContractError(f"{where} cached Hub tree has contradictory LFS size for {raw_path!r}")

        normalized[str(raw_path)] = raw_info
    return normalized


def _snapshot_file_hashes(snapshot: Path, expected_commit: str, where: str) -> dict[str, str]:
    """Authenticate local snapshot bytes against cached Hub commit-tree metadata.

    Pre/post SHA-256 receipts still bind the exact bytes consumed by Transformers,
    but a snapshot directory name alone is never accepted as proof of the declared
    Hub revision. Every local file must also match the Hub-derived cached commit tree:
    regular Git files are checked against their Git blob SHA-1 and LFS/Xet-backed
    files against their LFS SHA-256. Standard cache symlinks are additionally required
    to resolve into the repository's content-addressed ``blobs`` directory.
    """
    snapshot = snapshot.resolve()
    if snapshot.name.lower() != expected_commit.lower():
        raise CaptureContractError(
            f"{where} snapshot path is not bound to requested commit {expected_commit}"
        )

    expected = _cached_hub_commit_tree(snapshot, expected_commit, where)
    storage = snapshot.parent.parent.resolve()
    blob_root = (storage / "blobs").resolve()
    observed_paths: set[str] = set()
    hashes: dict[str, str] = {}

    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_dir() and not path.is_symlink():
            continue
        try:
            rel = path.relative_to(snapshot).as_posix()
            if not _is_canonical_snapshot_path(rel):
                raise CaptureContractError(
                    f"{where} snapshot contains a noncanonical artifact path: {rel!r}"
                )
            info = expected.get(rel)
            if info is None:
                raise CaptureContractError(
                    f"{where} snapshot contains a file absent from the cached Hub commit tree: {rel}"
                )
            try:
                target = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise CaptureContractError(
                    f"{where} snapshot contains an unreadable or broken artifact: {path}"
                ) from exc
            if not target.is_file():
                raise CaptureContractError(
                    f"{where} snapshot contains a non-regular artifact: {path}"
                )
            size = path.stat().st_size
            if size != info["size"]:
                raise CaptureContractError(
                    f"{where} snapshot size does not match cached Hub commit metadata: {rel}"
                )

            sha256 = _sha256_file(path)
            lfs_sha256 = info.get("lfs_sha256")
            expected_blob_name = lfs_sha256 if lfs_sha256 is not None else info["blob_id"]
            if lfs_sha256 is not None:
                if sha256 != lfs_sha256:
                    raise CaptureContractError(
                        f"{where} snapshot bytes do not match cached Hub LFS identity: {rel}"
                    )
            elif _git_blob_sha1(path, size) != info["blob_id"]:
                raise CaptureContractError(
                    f"{where} snapshot bytes do not match cached Hub Git blob identity: {rel}"
                )

            if path.is_symlink():
                if target.parent != blob_root or target.name != expected_blob_name:
                    raise CaptureContractError(
                        f"{where} snapshot symlink does not resolve to the commit-tree content-addressed blob: {rel}"
                    )

            observed_paths.add(rel)
            hashes[rel] = sha256
        except CaptureContractError:
            raise
        except OSError as exc:
            raise CaptureContractError(
                f"unable to authenticate {where} snapshot artifact {path}: {exc}"
            ) from exc

    missing = sorted(set(expected) - observed_paths)
    if missing:
        preview = ", ".join(missing[:3])
        raise CaptureContractError(
            f"{where} snapshot is incomplete relative to cached Hub commit tree: {preview}"
        )
    if not hashes:
        raise CaptureContractError(f"{where} snapshot contains no files")
    return hashes

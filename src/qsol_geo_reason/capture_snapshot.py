"""Snapshot byte authentication for GEO-CAP-001 production capture."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .capture_common import CaptureContractError
from .capture_provenance import _is_canonical_snapshot_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_file_hashes(snapshot: Path, expected_commit: str, where: str) -> dict[str, str]:
    """Hash snapshot-relative files, safely following normal HF blob symlinks.

    Hugging Face cache snapshots normally contain relative symlinks into a sibling
    ``blobs`` directory. The canonical receipt therefore authenticates the
    snapshot-relative key plus the bytes reached through that entry. Broken
    links, links to directories, FIFOs/devices, and noncanonical receipt keys
    fail closed.
    """
    snapshot = snapshot.resolve()
    if snapshot.name.lower() != expected_commit.lower():
        raise CaptureContractError(
            f"{where} snapshot path is not bound to requested commit {expected_commit}"
        )
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
            hashes[rel] = _sha256_file(path)
        except CaptureContractError:
            raise
        except OSError as exc:
            raise CaptureContractError(
                f"unable to hash {where} snapshot artifact {path}: {exc}"
            ) from exc
    if not hashes:
        raise CaptureContractError(f"{where} snapshot contains no files")
    return hashes

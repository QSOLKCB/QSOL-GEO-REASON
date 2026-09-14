"""Authentication of executable repository tools against an already bound revision."""
from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from .provenance import SourceIdentityError, _git_run, source_repo_root


def authenticate_tracked_tool_against_revision(path: Path, revision: str) -> str:
    """Bind one working-tree tool file directly to the named committed blob.

    This deliberately does not trust ``git status`` or index stat hints such as
    assume-unchanged/skip-worktree.  The committed blob is read from the named tree and
    the working-tree bytes are hashed directly with ``git hash-object --no-filters``.
    """
    root = source_repo_root().resolve()
    try:
        resolved = Path(path).resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise SourceIdentityError(
            "canonical experiment runner must be a regular file inside the executing checkout"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise SourceIdentityError("canonical experiment runner must be a regular tracked file")
    parts = PurePosixPath(relative).parts
    if not parts or parts[0] != "tools":
        raise SourceIdentityError("canonical experiment runner must live under tools/")

    try:
        listing = _git_run(
            root,
            "ls-tree",
            "-z",
            revision,
            "--",
            relative,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceIdentityError(
            "unable to read canonical experiment runner identity from Git"
        ) from exc
    records = tuple(record for record in listing.stdout.split("\0") if record)
    if len(records) != 1:
        raise SourceIdentityError("canonical experiment runner is not uniquely tracked at the bound revision")
    try:
        metadata, listed_path = records[0].split("\t", 1)
        mode, object_type, committed_oid = metadata.split(" ", 2)
    except ValueError as exc:
        raise SourceIdentityError("malformed canonical experiment runner Git tree entry") from exc
    if listed_path.replace("\\", "/") != relative:
        raise SourceIdentityError("canonical experiment runner Git tree path changed")
    if object_type != "blob" or mode not in {"100644", "100755"}:
        raise SourceIdentityError("canonical experiment runner must be a regular tracked blob")

    try:
        observed_oid = _git_run(
            root,
            "hash-object",
            "--no-filters",
            "--",
            relative,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceIdentityError(
            "unable to hash canonical experiment runner working-tree bytes"
        ) from exc
    if not observed_oid or observed_oid != committed_oid:
        raise SourceIdentityError(
            "canonical experiment runner bytes do not match the bound repository revision"
        )
    return committed_oid


__all__ = ["authenticate_tracked_tool_against_revision"]

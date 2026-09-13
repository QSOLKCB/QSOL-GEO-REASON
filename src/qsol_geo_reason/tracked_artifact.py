"""Direct authentication for tracked non-package research artifacts."""
from __future__ import annotations

import stat
import subprocess
from pathlib import Path

from .provenance import (
    SourceIdentityError,
    _trusted_git_environment,
    _trusted_git_executable,
    source_repo_root,
)


def _git(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    git = _trusted_git_executable()
    try:
        completed = subprocess.run(
            [str(git), "-C", str(root), *args],
            env=_trusted_git_environment(),
            check=True,
            capture_output=True,
            text=text,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceIdentityError(
            f"unable to authenticate tracked artifact with Git command: {' '.join(args)}"
        ) from exc
    _trusted_git_executable()
    return completed


def authenticate_tracked_file_against_revision(
    path: Path,
    revision: str,
    *,
    repo_root: Path | None = None,
) -> str:
    """Require raw working-tree bytes to equal one regular tracked blob at ``revision``.

    This deliberately bypasses the Git index so ``assume-unchanged`` and
    ``skip-worktree`` cannot hide a modified preregistration artifact. The requested
    pathname itself is authoritative: symlink replacement is rejected before any Git
    identity lookup, and the commit-tree path is derived from the unreplaced lexical
    pathname rather than from its resolved target. Replacement objects, alternate Git
    directories/object stores, config-injection variables, native-loader injection,
    and caller-controlled Git executable lookup are excluded from the identity-sensitive
    Git child.
    """
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in "0123456789abcdef" for ch in revision)
    ):
        raise SourceIdentityError("tracked artifact revision must be canonical lowercase 40-hex")

    root = Path(repo_root) if repo_root is not None else source_repo_root()
    try:
        root = root.resolve(strict=True)
        requested = Path(path)
        if not requested.is_absolute():
            requested = requested.absolute()
        relative = requested.relative_to(root).as_posix()
        requested_stat = requested.lstat()
    except (OSError, ValueError) as exc:
        raise SourceIdentityError(
            "tracked artifact must be an existing file inside the executing repository"
        ) from exc
    if stat.S_ISLNK(requested_stat.st_mode):
        raise SourceIdentityError(
            f"tracked artifact pathname must not be a symlink: {relative}"
        )
    if not stat.S_ISREG(requested_stat.st_mode):
        raise SourceIdentityError("tracked artifact must be a regular working-tree file")

    resolved_commit = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}").stdout.strip()
    if resolved_commit != revision:
        raise SourceIdentityError("tracked artifact revision did not resolve to the requested commit")

    listing = _git(root, "ls-tree", "-z", revision, "--", relative).stdout
    records = tuple(record for record in listing.split("\0") if record)
    if len(records) != 1:
        raise SourceIdentityError(
            f"tracked artifact is not represented by exactly one commit-tree entry: {relative}"
        )
    try:
        metadata, committed_path = records[0].split("\t", 1)
        mode, object_type, object_id = metadata.split(" ", 2)
    except ValueError as exc:
        raise SourceIdentityError("malformed tracked artifact tree entry") from exc
    if committed_path != relative or object_type != "blob" or mode not in {"100644", "100755"}:
        raise SourceIdentityError(
            f"tracked artifact must be a regular committed blob at the bound revision: {relative}"
        )

    committed_bytes = _git(root, "cat-file", "blob", object_id, text=False).stdout
    try:
        # Re-check immediately before reading so a symlink substitution between the
        # initial pathname check and Git lookup cannot silently redirect authentication.
        read_stat = requested.lstat()
        if stat.S_ISLNK(read_stat.st_mode) or not stat.S_ISREG(read_stat.st_mode):
            raise SourceIdentityError(
                f"tracked artifact pathname changed type during authentication: {relative}"
            )
        observed_bytes = requested.read_bytes()
        final_stat = requested.lstat()
    except SourceIdentityError:
        raise
    except OSError as exc:
        raise SourceIdentityError(f"unable to read tracked artifact bytes: {relative}") from exc
    if (
        stat.S_ISLNK(final_stat.st_mode)
        or not stat.S_ISREG(final_stat.st_mode)
        or (read_stat.st_dev, read_stat.st_ino) != (final_stat.st_dev, final_stat.st_ino)
    ):
        raise SourceIdentityError(
            f"tracked artifact pathname changed during authentication: {relative}"
        )
    if observed_bytes != committed_bytes:
        raise SourceIdentityError(
            "tracked artifact does not match the bound Git revision independently of index flags: "
            f"{relative}"
        )
    return relative


__all__ = ["authenticate_tracked_file_against_revision"]

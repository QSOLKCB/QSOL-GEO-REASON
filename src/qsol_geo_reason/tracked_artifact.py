"""Direct authentication for tracked non-package research artifacts."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .provenance import SourceIdentityError, source_repo_root


_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_LOADER_ENV_NAMES = frozenset({"GLIBC_TUNABLES", "LIBPATH", "SHLIB_PATH"})


def _trusted_git_environment() -> dict[str, str]:
    """Return a Git environment without repository or native-loader redirection knobs."""
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("GIT_"):
            continue
        if upper.startswith(_LOADER_ENV_PREFIXES) or upper in _LOADER_ENV_NAMES:
            continue
        environment[key] = value
    environment.update({"GIT_NO_REPLACE_OBJECTS": "1", "LC_ALL": "C"})
    return environment


def _git(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
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


def authenticate_tracked_file_against_revision(
    path: Path,
    revision: str,
    *,
    repo_root: Path | None = None,
) -> str:
    """Require raw working-tree bytes to equal one regular tracked blob at ``revision``.

    This deliberately bypasses the Git index so ``assume-unchanged`` and
    ``skip-worktree`` cannot hide a modified preregistration artifact. Replacement
    objects, alternate Git directories/object stores, config-injection variables, and
    native-loader injection are also excluded from the identity-sensitive Git child.
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
        artifact = Path(path).resolve(strict=True)
        relative = artifact.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise SourceIdentityError(
            "tracked artifact must be an existing file inside the executing repository"
        ) from exc
    if not artifact.is_file():
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
        observed_bytes = artifact.read_bytes()
    except OSError as exc:
        raise SourceIdentityError(f"unable to read tracked artifact bytes: {relative}") from exc
    if observed_bytes != committed_bytes:
        raise SourceIdentityError(
            "tracked artifact does not match the bound Git revision independently of index flags: "
            f"{relative}"
        )
    return relative


__all__ = ["authenticate_tracked_file_against_revision"]

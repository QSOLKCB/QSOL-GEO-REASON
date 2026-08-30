"""Source-identity helpers for provenance-bound simulation runs."""

from __future__ import annotations

import subprocess
from pathlib import Path


class SourceIdentityError(RuntimeError):
    pass


def source_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_source_revision(*, require_clean: bool = True) -> str | None:
    """Return HEAD for the source checkout, or None when not running from Git.

    When require_clean is true, any tracked or untracked working-tree change
    rejects revision binding because HEAD would not identify the executing
    source bytes.
    """
    root = source_repo_root()
    try:
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    git_root = Path(probe.stdout.strip()).resolve()
    if git_root != root.resolve():
        return None

    if require_clean:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise SourceIdentityError(
                "source checkout is dirty; commit or stash changes before binding an implementation revision"
            )

    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head:
        raise SourceIdentityError("git HEAD is empty")
    return head


def resolve_implementation_revision(explicit: str | None = None) -> str:
    """Resolve and, when possible, verify the implementation revision."""
    observed = git_source_revision(require_clean=True)
    if explicit:
        if observed is not None and observed != explicit:
            raise SourceIdentityError(
                f"implementation revision {explicit!r} does not match clean source HEAD {observed!r}"
            )
        return explicit
    if observed is None:
        raise SourceIdentityError(
            "implementation revision is required when the installed source is not in a Git checkout"
        )
    return observed

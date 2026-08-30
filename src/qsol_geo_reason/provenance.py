"""Source-identity helpers for provenance-bound simulation runs."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


class SourceIdentityError(RuntimeError):
    pass


_GENERATED_TOP_LEVEL = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}


def source_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_generated_untracked(path: str) -> bool:
    """Return True for untracked interpreter/build artifacts, never source."""
    normalized = path.strip().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        return False
    if "__pycache__" in parts:
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if parts[0] in _GENERATED_TOP_LEVEL:
        return True
    if normalized == ".coverage" or normalized.endswith((".pyc", ".pyo", ".pyd")):
        return True
    return False


def _status_has_source_changes(status_text: str) -> bool:
    for line in status_text.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if code == "??" and _is_generated_untracked(path):
            continue
        return True
    return False


def git_source_revision(*, require_clean: bool = True) -> str | None:
    """Return HEAD for the source checkout, or None when not running from Git.

    When require_clean is true, tracked changes and source-relevant untracked
    files reject revision binding because HEAD would not identify the executing
    source bytes. Untracked interpreter/build artifacts are ignored.
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
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
        if _status_has_source_changes(status.stdout):
            raise SourceIdentityError(
                "source checkout is dirty; commit or stash source-relevant changes before binding an implementation revision"
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

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
_IMPORTABLE_PACKAGE_ROOT = ("src", "qsol_geo_reason")
_BYTECODE_SUFFIXES = (".pyc", ".pyo")


def source_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_importable_package_bytecode(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    return (
        len(parts) >= len(_IMPORTABLE_PACKAGE_ROOT) + 1
        and tuple(parts[: len(_IMPORTABLE_PACKAGE_ROOT)]) == _IMPORTABLE_PACKAGE_ROOT
        and normalized.endswith(_BYTECODE_SUFFIXES)
    )


def _is_generated_untracked(path: str) -> bool:
    """Return True for disposable untracked artifacts, never importable source code."""
    normalized = path.strip().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        return False
    # Bytecode underneath the importable package can replace tracked Python source
    # at execution time when its cache header is valid. Never classify it as a
    # disposable cleanliness exception.
    if _is_importable_package_bytecode(normalized):
        return False
    if "__pycache__" in parts:
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if parts[0] in _GENERATED_TOP_LEVEL:
        return True
    # A .pyd in the importable source tree is executable source, not bytecode.
    # Build/cache directories are handled above; never exempt it by suffix.
    if normalized == ".coverage" or normalized.endswith(_BYTECODE_SUFFIXES):
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


def _ignored_importable_bytecode(root: Path) -> tuple[str, ...]:
    """Return ignored bytecode that can participate in package imports."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceIdentityError(
            "unable to inspect ignored importable bytecode for checkout-bound execution"
        ) from exc
    paths = tuple(
        sorted(
            path.replace("\\", "/")
            for path in result.stdout.split("\0")
            if path and _is_importable_package_bytecode(path)
        )
    )
    return paths


def git_source_revision(
    *, require_clean: bool = True, reject_importable_bytecode: bool = False
) -> str | None:
    """Return HEAD for the source checkout, or None when not running from Git.

    When require_clean is true, tracked changes and source-relevant untracked
    files reject revision binding because HEAD would not identify the executing
    source bytes. Ordinary build artifacts are ignored. Canonical OBSERVATION
    callers additionally reject ignored bytecode inside ``src/qsol_geo_reason``
    because those files are executable import inputs even when Git ignores them.
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
        if reject_importable_bytecode:
            bytecode = _ignored_importable_bytecode(root)
            if bytecode:
                shown = ", ".join(bytecode[:5])
                suffix = "" if len(bytecode) <= 5 else f" (+{len(bytecode) - 5} more)"
                raise SourceIdentityError(
                    "canonical observation forbids importable bytecode caches under "
                    f"src/qsol_geo_reason; remove them and run with bytecode writing disabled: {shown}{suffix}"
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


def resolve_implementation_revision(
    explicit: str | None = None, *, require_checkout: bool = False
) -> str:
    """Resolve and, when requested, require a clean checkout-bound revision."""
    observed = git_source_revision(
        require_clean=True,
        reject_importable_bytecode=require_checkout,
    )
    if require_checkout and observed is None:
        raise SourceIdentityError(
            "canonical observation requires execution from the clean QSOL-GEO-REASON Git checkout"
        )
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

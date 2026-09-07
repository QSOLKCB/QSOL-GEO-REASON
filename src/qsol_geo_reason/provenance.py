"""Source-identity helpers for provenance-bound simulation runs."""

from __future__ import annotations

import importlib.util
import marshal
import subprocess
import types
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
    """Return True for ordinary untracked interpreter/build artifacts, never source."""
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
    return tuple(
        sorted(
            path.replace("\\", "/")
            for path in result.stdout.split("\0")
            if path and _is_importable_package_bytecode(path)
        )
    )


def _bytecode_optimization_level(cache_path: Path) -> int:
    name = cache_path.name
    if ".opt-2." in name:
        return 2
    if ".opt-1." in name:
        return 1
    return 0


def _source_path_for_bytecode(cache_path: Path) -> Path:
    """Resolve a cache path to the source file whose code it may execute."""
    try:
        return Path(importlib.util.source_from_cache(str(cache_path)))
    except (ValueError, NotImplementedError):
        # Legacy sourceless-style caches can sit beside a module rather than in
        # __pycache__. They are canonical only when the corresponding tracked
        # source exists and recompiles to the exact same code object.
        if cache_path.parent.name == "__pycache__":
            raise SourceIdentityError(
                f"importable bytecode cache has no canonical source mapping: {cache_path}"
            )
        return cache_path.with_suffix(".py")


def _authenticate_importable_bytecode(root: Path, paths: tuple[str, ...]) -> None:
    """Accept only caches whose executable code exactly matches tracked source.

    The normal editable-install CLI imports this package before the final source
    provenance check, so CPython may create ignored ``__pycache__`` entries during
    that same trusted invocation. Blanket rejection would make the canonical CLI
    reject itself. Instead, every ignored importable cache is mapped to a tracked
    package source file and its marshalled code object is compared with a fresh
    compilation of the clean checkout source under the cache's optimization lane.
    Tampered, stale, foreign-interpreter, malformed, or sourceless caches fail closed.
    """
    root_resolved = root.resolve()
    for relative in paths:
        cache_path = root / PurePosixPath(relative)
        try:
            source_path = _source_path_for_bytecode(cache_path).resolve()
            source_relative = source_path.relative_to(root_resolved).as_posix()
        except (OSError, ValueError) as exc:
            raise SourceIdentityError(
                f"importable bytecode cache escapes the executing checkout: {relative}"
            ) from exc

        source_parts = PurePosixPath(source_relative).parts
        if (
            len(source_parts) < len(_IMPORTABLE_PACKAGE_ROOT) + 1
            or tuple(source_parts[: len(_IMPORTABLE_PACKAGE_ROOT)])
            != _IMPORTABLE_PACKAGE_ROOT
            or source_path.suffix != ".py"
            or not source_path.is_file()
        ):
            raise SourceIdentityError(
                f"importable bytecode cache is not backed by canonical package source: {relative}"
            )

        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--error-unmatch",
                    "--",
                    source_relative,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SourceIdentityError(
                f"importable bytecode cache is not backed by tracked source: {relative}"
            ) from exc

        try:
            raw = cache_path.read_bytes()
            if len(raw) < 16 or raw[:4] != importlib.util.MAGIC_NUMBER:
                raise ValueError("invalid or foreign bytecode header")
            observed = marshal.loads(raw[16:])
            if not isinstance(observed, types.CodeType):
                raise TypeError("bytecode payload is not a module code object")
            expected = compile(
                source_path.read_bytes(),
                str(source_path),
                "exec",
                dont_inherit=True,
                optimize=_bytecode_optimization_level(cache_path),
            )
        except (OSError, EOFError, ValueError, TypeError) as exc:
            raise SourceIdentityError(
                f"unable to authenticate importable bytecode cache against tracked source: {relative}"
            ) from exc

        if marshal.dumps(observed) != marshal.dumps(expected):
            raise SourceIdentityError(
                "importable bytecode cache does not match the clean tracked source: "
                f"{relative}"
            )


def git_source_revision(
    *, require_clean: bool = True, reject_importable_bytecode: bool = False
) -> str | None:
    """Return HEAD for the source checkout, or None when not running from Git.

    Ordinary provenance tolerates disposable interpreter caches. Canonical
    OBSERVATION callers additionally authenticate Git-ignored package bytecode
    against the clean tracked source because a timestamp/hash-valid cache can be
    an executable import input even though it is absent from normal ``git status``.
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
            _authenticate_importable_bytecode(root, _ignored_importable_bytecode(root))

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

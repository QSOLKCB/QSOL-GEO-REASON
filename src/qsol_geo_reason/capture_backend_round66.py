"""Round-66 checkout guard for Git-ignored Python source.

Round 60 already authenticates tracked package bytes, source-equivalent bytecode caches,
ignored native extensions, and the Git executable itself. A standard Git ignore rule
could still hide an untracked ``.py`` file beneath ``src/qsol_geo_reason`` from
``git status``. This final resolver wrapper enumerates that ignored source with its own
closure-sealed trusted Git runner before a checkout-bound OBSERVATION is accepted.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from . import capture_backend_round60 as _round60
from . import capture_execute as _capture_execute
from . import provenance as _provenance
from .capture_backend_round60 import HuggingFacePyTorchBackend


_IMPORTABLE_PACKAGE_ROOT = ("src", "qsol_geo_reason")


def _scan_ignored_python_sources_round66(
    root: Path,
    git_run: Any,
    _package_root: tuple[str, str] = _IMPORTABLE_PACKAGE_ROOT,
    _path_type: type[PurePosixPath] = PurePosixPath,
    _source_identity_error: type[BaseException] = _provenance.SourceIdentityError,
) -> tuple[str, ...]:
    """Return ignored Python source files beneath the importable package root."""
    try:
        result = git_run(
            root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            "/".join(_package_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        raise _source_identity_error(
            "unable to inspect ignored Python source for checkout-bound execution"
        ) from exc

    paths: list[str] = []
    for raw in result.stdout.split("\0"):
        if not raw:
            continue
        normalized = raw.replace("\\", "/")
        parts = _path_type(normalized).parts
        if (
            len(parts) >= len(_package_root) + 1
            and tuple(parts[: len(_package_root)]) == _package_root
            and normalized.lower().endswith(".py")
        ):
            paths.append(normalized)
    return tuple(sorted(paths))


def _make_ignored_python_source_guard_round66(
    git_run: Any,
    root: Path,
    scanner: Any,
    source_identity_error: type[BaseException],
):
    def guard() -> None:
        ignored = scanner(root, git_run)
        if ignored:
            preview = ", ".join(ignored[:3])
            raise source_identity_error(
                "ignored Python source exists inside the importable qsol_geo_reason package: "
                + preview
            )

    return guard


# Build a fresh runner while package import is still atomic. The Round-60 factory
# closure-binds system Git discovery, executable hashing, loader-environment stripping,
# subprocess execution, and post-command executable reauthentication.
_git_run_round66 = _round60._make_sealed_git_runner_round60()
_ignored_python_source_guard_round66 = _make_ignored_python_source_guard_round66(
    _git_run_round66,
    Path(__file__).resolve().parents[2],
    _scan_ignored_python_sources_round66,
    _provenance.SourceIdentityError,
)
del _make_ignored_python_source_guard_round66


def _make_round66_resolver(original_resolver: Any, ignored_guard: Any):
    def resolve_implementation_revision(
        explicit: str | None = None, *, require_checkout: bool = False
    ) -> str:
        revision = original_resolver(explicit, require_checkout=require_checkout)
        if require_checkout:
            ignored_guard()
        return revision

    return resolve_implementation_revision


_resolve_implementation_revision_round66 = _make_round66_resolver(
    _capture_execute.resolve_implementation_revision,
    _ignored_python_source_guard_round66,
)
del _make_round66_resolver
_capture_execute.resolve_implementation_revision = _resolve_implementation_revision_round66


__all__ = ["HuggingFacePyTorchBackend"]

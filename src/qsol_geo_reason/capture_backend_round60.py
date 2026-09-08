"""Round-60 correction for the sealed provenance wrapper call graph."""
from __future__ import annotations

import subprocess
import types
from pathlib import Path, PurePosixPath
from typing import Any

from .capture_backend_round59 import HuggingFacePyTorchBackend
from . import capture_backend_round44 as _round44
from . import capture_backend_round58 as _round58
from . import capture_execute as _capture_execute
from . import provenance as _provenance


_NATIVE_IMPORT_SUFFIXES = tuple(_round44._NATIVE_IMPORT_SUFFIXES)
_IMPORTABLE_PACKAGE_ROOT = ("src", "qsol_geo_reason")


def _clone_provenance_function_round60(
    value: types.FunctionType, private_globals: dict[str, Any]
) -> types.FunctionType:
    clone = types.FunctionType(
        value.__code__,
        private_globals,
        name=value.__name__,
        argdefs=value.__defaults__,
        closure=value.__closure__,
    )
    clone.__kwdefaults__ = value.__kwdefaults__
    clone.__annotations__ = dict(value.__annotations__)
    clone.__qualname__ = value.__qualname__
    clone.__doc__ = value.__doc__
    clone.__dict__.update(value.__dict__)
    return clone


def _make_ignored_native_guard_round60(sealed_git_run: Any):
    native_suffixes = _NATIVE_IMPORT_SUFFIXES
    importable_root = _IMPORTABLE_PACKAGE_ROOT
    path_type = PurePosixPath
    source_identity_error = _provenance.SourceIdentityError

    def is_importable_native_artifact(path: str) -> bool:
        normalized = path.strip().replace("\\", "/")
        parts = path_type(normalized).parts
        return (
            len(parts) >= 3
            and tuple(parts[:2]) == importable_root
            and normalized.lower().endswith(native_suffixes)
        )

    def assert_no_ignored_importable_native_artifacts(root: Path) -> None:
        try:
            result = sealed_git_run(
                root,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            raise source_identity_error(
                "unable to inspect ignored native artifacts for checkout-bound execution"
            ) from exc
        paths = tuple(
            sorted(
                path.replace("\\", "/")
                for path in result.stdout.split("\0")
                if path and is_importable_native_artifact(path)
            )
        )
        if paths:
            preview = ", ".join(paths[:3])
            raise source_identity_error(
                "ignored native artifact exists inside the importable qsol_geo_reason package: "
                + preview
            )

    return assert_no_ignored_importable_native_artifacts


def _make_sealed_provenance_entrypoints_round60():
    """Rebuild the Round-44 wrapper around a private provenance-native graph.

    Round 44 replaces ``provenance.git_source_revision`` with a function defined in
    ``capture_backend_round44``.  Such a wrapper must not be cloned against
    ``provenance`` globals: doing so loses its defining globals, including the saved
    original revision resolver.  Instead, clone only functions actually defined by
    ``provenance.py``, bind their Git entry point to Round 58's closure-sealed runner,
    then explicitly reconstruct Round 44's ignored-native-artifact guard around the
    sealed original ``git_source_revision``.
    """
    original_git_source_revision = _round44._ORIGINAL_GIT_SOURCE_REVISION
    if not (
        isinstance(original_git_source_revision, types.FunctionType)
        and original_git_source_revision.__globals__ is _provenance.__dict__
        and original_git_source_revision.__name__ == "git_source_revision"
    ):
        raise RuntimeError("unable to identify the original provenance revision function")

    private_globals = dict(vars(_provenance))
    clones: dict[str, types.FunctionType] = {}
    for name, value in tuple(private_globals.items()):
        if (
            isinstance(value, types.FunctionType)
            and value.__globals__ is _provenance.__dict__
        ):
            clones[name] = _clone_provenance_function_round60(value, private_globals)

    private_globals.update(clones)
    private_globals["_git_run"] = _round58._git_run_round58

    sealed_original = _clone_provenance_function_round60(
        original_git_source_revision, private_globals
    )
    sealed_source_repo_root = private_globals.get("source_repo_root")
    sealed_resolver = private_globals.get("resolve_implementation_revision")
    if not isinstance(sealed_source_repo_root, types.FunctionType) or not isinstance(
        sealed_resolver, types.FunctionType
    ):
        raise RuntimeError("unable to seal canonical provenance entrypoints")

    native_guard = _make_ignored_native_guard_round60(_round58._git_run_round58)

    def git_source_revision(
        *, require_clean: bool = True, reject_importable_bytecode: bool = False
    ) -> str | None:
        observed = sealed_original(
            require_clean=require_clean,
            reject_importable_bytecode=reject_importable_bytecode,
        )
        if observed is not None and require_clean and reject_importable_bytecode:
            native_guard(sealed_source_repo_root())
        return observed

    private_globals["git_source_revision"] = git_source_revision
    return git_source_revision, sealed_resolver


(
    _git_source_revision_round60,
    _resolve_implementation_revision_round60,
) = _make_sealed_provenance_entrypoints_round60()
del _make_sealed_provenance_entrypoints_round60


# Round 59 deliberately leaves provenance._git_run patchable for ordinary public
# provenance tests.  Canonical capture instead resolves source identity through this
# private graph, preserving the Round-44 native-artifact check without consulting
# that mutable compatibility slot.
_capture_execute.resolve_implementation_revision = _resolve_implementation_revision_round60


__all__ = ["HuggingFacePyTorchBackend"]

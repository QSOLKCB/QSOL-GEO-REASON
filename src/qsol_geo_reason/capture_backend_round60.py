"""Round-60 correction for the sealed provenance wrapper call graph."""
from __future__ import annotations

import os
import subprocess
import types
from pathlib import Path, PurePosixPath
from typing import Any

from .capture_backend_round59 import HuggingFacePyTorchBackend
from . import capture_backend_round44 as _round44
from . import capture_backend_round56 as _round56
from . import capture_execute as _capture_execute
from . import provenance as _provenance


_NATIVE_IMPORT_SUFFIXES = tuple(_round44._NATIVE_IMPORT_SUFFIXES)
_IMPORTABLE_PACKAGE_ROOT = ("src", "qsol_geo_reason")
_DYNAMIC_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_DYNAMIC_LOADER_ENV_NAMES = frozenset({"GLIBC_TUNABLES", "LIBPATH", "SHLIB_PATH"})


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


def _make_sealed_git_runner_round60():
    """Bind Git execution and environment sanitization without module-global lookups."""
    trusted_git = _round56._trusted_git_executable
    trusted_run = subprocess.run
    source_environment = os.environ
    devnull = os.devnull
    loader_prefixes = _DYNAMIC_LOADER_ENV_PREFIXES
    loader_names = _DYNAMIC_LOADER_ENV_NAMES

    def child_environment() -> dict[str, str]:
        environment: dict[str, str] = {}
        for key, value in source_environment.items():
            upper = key.upper()
            if upper.startswith("GIT_"):
                continue
            if upper.startswith(loader_prefixes):
                continue
            if upper in loader_names:
                continue
            environment[key] = value
        environment.update(
            {
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        return environment

    sealed_child_environment = child_environment

    def run(
        root: Path, *args: str, **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        git = trusted_git()
        completed = trusted_run(
            [str(git), "-C", str(root), *args],
            env=sealed_child_environment(),
            **kwargs,
        )
        trusted_git()
        return completed

    return run


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
    """Build authenticated canonical provenance entrypoints from a private graph.

    The resolver returned to ``capture_execute`` is a closure-bound wrapper, not a
    cloned provenance function whose writable ``__globals__`` exposes the private
    graph.  Every invocation also authenticates the private graph before and after
    source resolution.  Git execution uses a Round-60 runner whose sanitizer and
    subprocess dependencies are closure-bound, so rebinding Round-58 module globals
    cannot reintroduce Git or dynamic-loader environment controls.
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
    sealed_git_run = _make_sealed_git_runner_round60()
    private_globals["_git_run"] = sealed_git_run

    sealed_original = _clone_provenance_function_round60(
        original_git_source_revision, private_globals
    )
    sealed_source_repo_root = private_globals.get("source_repo_root")
    if not isinstance(sealed_source_repo_root, types.FunctionType):
        raise RuntimeError("unable to seal canonical provenance source-root entrypoint")

    expected_bindings = tuple(private_globals.items())
    expected_function_code = tuple(
        (value, value.__code__)
        for value in private_globals.values()
        if isinstance(value, types.FunctionType)
    )
    missing = object()
    source_identity_error = _provenance.SourceIdentityError
    native_guard = _make_ignored_native_guard_round60(sealed_git_run)

    def assert_private_graph_intact() -> None:
        if len(private_globals) != len(expected_bindings):
            raise source_identity_error("sealed canonical provenance globals were modified")
        for name, expected in expected_bindings:
            if private_globals.get(name, missing) is not expected:
                raise source_identity_error(
                    f"sealed canonical provenance binding was modified: {name}"
                )
        for function, expected_code in expected_function_code:
            if function.__code__ is not expected_code:
                raise source_identity_error(
                    "sealed canonical provenance callable code was modified"
                )

    def git_source_revision(
        *, require_clean: bool = True, reject_importable_bytecode: bool = False
    ) -> str | None:
        assert_private_graph_intact()
        observed = sealed_original(
            require_clean=require_clean,
            reject_importable_bytecode=reject_importable_bytecode,
        )
        assert_private_graph_intact()
        if observed is not None and require_clean and reject_importable_bytecode:
            native_guard(sealed_source_repo_root())
            assert_private_graph_intact()
        return observed

    def resolve_implementation_revision(
        explicit: str | None = None, *, require_checkout: bool = False
    ) -> str:
        observed = git_source_revision(
            require_clean=True,
            reject_importable_bytecode=require_checkout,
        )
        if require_checkout and observed is None:
            raise source_identity_error(
                "canonical observation requires execution from the clean "
                "QSOL-GEO-REASON Git checkout"
            )
        if explicit:
            if observed is not None and observed != explicit:
                raise source_identity_error(
                    f"implementation revision {explicit!r} does not match clean "
                    f"source HEAD {observed!r}"
                )
            return explicit
        if observed is None:
            raise source_identity_error(
                "implementation revision is required when the installed source is "
                "not in a Git checkout"
            )
        return observed

    return git_source_revision, resolve_implementation_revision


(
    _git_source_revision_round60,
    _resolve_implementation_revision_round60,
) = _make_sealed_provenance_entrypoints_round60()
del _make_sealed_provenance_entrypoints_round60


# Ordinary provenance remains patchable for its public/test compatibility surface.
# Canonical capture resolves source identity only through the authenticated private
# graph above and therefore does not consult Round-58's mutable compatibility slots.
_capture_execute.resolve_implementation_revision = _resolve_implementation_revision_round60


__all__ = ["HuggingFacePyTorchBackend"]

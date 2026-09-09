"""Round-60 correction for the sealed provenance wrapper call graph."""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import types
from pathlib import Path, PurePosixPath
from typing import Any

from .capture_backend_round59 import HuggingFacePyTorchBackend
from . import capture_backend_round44 as _round44
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


def _private_provenance_value_round60(value: Any) -> Any:
    """Detach mutable module values before cloning the provenance call graph."""
    if isinstance(value, set):
        return frozenset(value)
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _private_value_fingerprint_round60(value: Any) -> Any:
    """Return an immutable content fingerprint for mutable private graph values."""
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    ((repr(key), id(item)) for key, item in value.items()),
                    key=lambda item: item[0],
                )
            ),
        )
    if isinstance(value, (tuple, frozenset)):
        items = tuple(
            sorted(
                (
                    _private_value_fingerprint_round60(item)
                    if isinstance(item, (dict, tuple, frozenset))
                    else (type(item).__qualname__, repr(item))
                    for item in value
                ),
                key=repr,
            )
        )
        return (type(value).__qualname__, items)
    return (type(value).__qualname__, id(value))


def _make_sealed_git_runner_round60():
    """Bind the complete trusted-Git resolver and execution graph in closures.

    Round 56's resolver was itself a closure but still looked up candidate enumeration
    and executable hashing through writable Round-56 globals. Canonical provenance
    must not inherit those lookups. This factory therefore owns candidate selection,
    executable hashing, baseline authentication, child-environment sanitization and
    subprocess execution without consulting any Round module after construction.
    """
    path_type = Path
    source_identity_error = _provenance.SourceIdentityError
    sha256 = hashlib.sha256
    is_regular = stat.S_ISREG
    access = os.access
    x_ok = os.X_OK
    process_name = os.name
    trusted_run = subprocess.run
    source_environment = os.environ
    devnull = os.devnull
    loader_prefixes = _DYNAMIC_LOADER_ENV_PREFIXES
    loader_names = _DYNAMIC_LOADER_ENV_NAMES

    if process_name == "nt":
        candidates = (
            path_type(r"C:\Program Files\Git\cmd\git.exe"),
            path_type(r"C:\Program Files\Git\bin\git.exe"),
        )
    else:
        candidates = (
            path_type("/usr/bin/git"),
            path_type("/bin/git"),
            path_type("/run/current-system/sw/bin/git"),
        )

    def hash_executable(path: Path) -> tuple[Path, str]:
        try:
            resolved = path.resolve(strict=True)
            info = resolved.stat()
        except OSError as exc:
            raise source_identity_error(
                f"unable to resolve trusted Git executable {path}"
            ) from exc
        if not is_regular(info.st_mode) or not access(resolved, x_ok):
            raise source_identity_error(
                f"trusted Git executable is not an executable regular file: {resolved}"
            )
        digest = sha256()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise source_identity_error(
                f"unable to hash trusted Git executable {resolved}"
            ) from exc
        return resolved, digest.hexdigest()

    baseline: tuple[Path, str] | None = None

    def trusted_git() -> Path:
        nonlocal baseline
        if baseline is None:
            last_error: BaseException | None = None
            for candidate in candidates:
                try:
                    baseline = hash_executable(candidate)
                    break
                except source_identity_error as exc:
                    last_error = exc
            if baseline is None:
                raise source_identity_error(
                    "canonical source identity requires a trusted system Git executable"
                ) from last_error
        path, expected_digest = baseline
        observed_path, observed_digest = hash_executable(path)
        if observed_path != path or observed_digest != expected_digest:
            raise source_identity_error(
                "trusted Git executable changed after source-identity initialization"
            )
        return path

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
    """Build authenticated canonical provenance entrypoints from a private graph."""
    original_git_source_revision = _round44._ORIGINAL_GIT_SOURCE_REVISION
    if not (
        isinstance(original_git_source_revision, types.FunctionType)
        and original_git_source_revision.__globals__ is _provenance.__dict__
        and original_git_source_revision.__name__ == "git_source_revision"
    ):
        raise RuntimeError("unable to identify the original provenance revision function")

    private_globals = {
        name: _private_provenance_value_round60(value)
        for name, value in vars(_provenance).items()
    }
    clones: dict[str, types.FunctionType] = {}
    for name, value in tuple(private_globals.items()):
        original_value = vars(_provenance).get(name)
        if (
            isinstance(original_value, types.FunctionType)
            and original_value.__globals__ is _provenance.__dict__
        ):
            clones[name] = _clone_provenance_function_round60(
                original_value, private_globals
            )

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
    expected_value_fingerprints = tuple(
        (name, _private_value_fingerprint_round60(value))
        for name, value in expected_bindings
        if isinstance(value, (dict, tuple, frozenset))
    )
    expected_function_code = tuple(
        (value, value.__code__)
        for value in private_globals.values()
        if isinstance(value, types.FunctionType)
    )
    missing = object()
    source_identity_error = _provenance.SourceIdentityError
    native_guard = _make_ignored_native_guard_round60(sealed_git_run)
    fingerprint = _private_value_fingerprint_round60

    def assert_private_graph_intact() -> None:
        if len(private_globals) != len(expected_bindings):
            raise source_identity_error("sealed canonical provenance globals were modified")
        for name, expected in expected_bindings:
            if private_globals.get(name, missing) is not expected:
                raise source_identity_error(
                    f"sealed canonical provenance binding was modified: {name}"
                )
        for name, expected in expected_value_fingerprints:
            if fingerprint(private_globals[name]) != expected:
                raise source_identity_error(
                    f"sealed canonical provenance value was modified: {name}"
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


_capture_execute.resolve_implementation_revision = _resolve_implementation_revision_round60


__all__ = ["HuggingFacePyTorchBackend"]
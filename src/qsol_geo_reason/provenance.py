"""Source-identity helpers for provenance-bound simulation runs."""

from __future__ import annotations

import importlib.util
import marshal
import os
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath
from typing import Any


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


def _git_run(root: Path, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run an identity-sensitive Git command with replacement objects disabled.

    Local ``refs/replace`` entries intentionally rewrite object lookup without changing
    the named commit ID. They are useful for repository surgery but cannot participate
    in canonical source identity: a clean replacement tree must never be attributed to
    the unreplaced commit SHA. Every Git query in this module therefore shares the same
    fail-closed environment rather than relying on individual callers to remember it.
    """
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", "-C", str(root), *args],
        env=environment,
        **kwargs,
    )


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


def _assert_tracked_importable_source_matches_head(root: Path, head: str) -> None:
    """Compare tracked package files with HEAD without trusting index stat flags.

    ``git status`` intentionally honors ``assume-unchanged`` and ``skip-worktree``.
    Those index hints are useful for normal development but cannot participate in a
    canonical OBSERVATION trust decision. Enumerate the committed package tree from
    HEAD itself, then hash each corresponding working-tree path directly. ``hash-object``
    reads the raw file bytes rather than the index or any clean/smudge filter, so a
    modified tracked source file cannot be hidden by index flags or Git attributes.
    Tracked symlinks or other non-regular package entries fail closed because their
    execution target is not represented by the ordinary file receipt used here.
    Replacement objects are disabled by ``_git_run`` for both the committed-tree
    lookup and working-tree hashing path.
    """
    package_root = "/".join(_IMPORTABLE_PACKAGE_ROOT)
    try:
        listing = _git_run(
            root,
            "ls-tree",
            "-r",
            "-z",
            head,
            "--",
            package_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceIdentityError(
            "unable to enumerate tracked importable source from Git HEAD"
        ) from exc

    records = tuple(record for record in listing.stdout.split("\0") if record)
    if not records:
        raise SourceIdentityError("Git HEAD contains no tracked qsol_geo_reason package source")

    for record in records:
        try:
            metadata, relative = record.split("\t", 1)
            mode, object_type, committed_oid = metadata.split(" ", 2)
        except ValueError as exc:
            raise SourceIdentityError("malformed Git HEAD package-tree entry") from exc

        normalized = relative.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        if (
            len(parts) < len(_IMPORTABLE_PACKAGE_ROOT) + 1
            or tuple(parts[: len(_IMPORTABLE_PACKAGE_ROOT)]) != _IMPORTABLE_PACKAGE_ROOT
        ):
            raise SourceIdentityError(
                f"tracked importable source escaped the canonical package root: {relative}"
            )
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise SourceIdentityError(
                f"canonical package source must be a regular tracked file: {relative}"
            )

        try:
            observed_oid = _git_run(
                root,
                "hash-object",
                "--no-filters",
                "--",
                normalized,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SourceIdentityError(
                f"unable to hash tracked importable source from the working tree: {relative}"
            ) from exc

        if not observed_oid or observed_oid != committed_oid:
            raise SourceIdentityError(
                "tracked importable source does not match Git HEAD independently of index flags: "
                f"{relative}"
            )


def _ignored_importable_bytecode(root: Path) -> tuple[str, ...]:
    """Return ignored bytecode that can participate in package imports."""
    try:
        result = _git_run(
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
        if cache_path.parent.name == "__pycache__":
            raise SourceIdentityError(
                f"importable bytecode cache has no canonical source mapping: {cache_path}"
            )
        return cache_path.with_suffix(".py")


def _stable_code_constant(value: object) -> object:
    """Normalize constants embedded in executable Python code."""
    if isinstance(value, types.CodeType):
        return ("code", _stable_code_identity(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_stable_code_constant(item) for item in value))
    if isinstance(value, frozenset):
        normalized = [_stable_code_constant(item) for item in value]
        return ("frozenset", tuple(sorted(normalized, key=repr)))
    if isinstance(value, bytes):
        return ("bytes", value)
    return (type(value).__qualname__, repr(value))


def _stable_code_identity(code: types.CodeType) -> tuple[object, ...]:
    """Return executable code identity without filename or adaptive runtime state."""
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_stable_code_constant(item) for item in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_linetable,
        code.co_exceptiontable,
    )


def _module_source_path(root: Path, module_name: str) -> Path:
    if module_name == "qsol_geo_reason":
        candidate = root / "src" / "qsol_geo_reason" / "__init__.py"
    elif module_name.startswith("qsol_geo_reason."):
        parts = module_name.split(".")
        base = root / "src"
        module_path = base.joinpath(*parts)
        file_candidate = module_path.with_suffix(".py")
        package_candidate = module_path / "__init__.py"
        candidate = file_candidate if file_candidate.is_file() else package_candidate
    else:
        raise SourceIdentityError(
            f"loaded callable is outside the canonical package: {module_name}"
        )
    if not candidate.is_file():
        raise SourceIdentityError(
            f"loaded canonical module has no tracked source file: {module_name}"
        )
    return candidate.resolve()


def _compiled_source_code_index(source_path: Path) -> dict[tuple[str, int], types.CodeType]:
    try:
        module_code = compile(
            source_path.read_bytes(),
            str(source_path),
            "exec",
            dont_inherit=True,
        )
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        raise SourceIdentityError(
            f"unable to compile clean tracked source for loaded-callable authentication: {source_path}"
        ) from exc

    index: dict[tuple[str, int], types.CodeType] = {}

    def visit(code: types.CodeType) -> None:
        key = (code.co_qualname, code.co_firstlineno)
        if key in index:
            raise SourceIdentityError(
                f"clean tracked source has an ambiguous executable identity: {source_path}:{code.co_qualname}"
            )
        index[key] = code
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                visit(constant)

    visit(module_code)
    return index


def _loaded_qsol_python_functions() -> tuple[types.FunctionType, ...]:
    """Collect loaded QSOL Python functions without invoking mutable descriptors."""
    seen: set[int] = set()
    functions: list[types.FunctionType] = []
    references = 0
    max_references = 32768

    def visit(value: object) -> None:
        nonlocal references
        references += 1
        if references > max_references:
            raise SourceIdentityError(
                "loaded canonical callable inspection exceeded its reference limit"
            )
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (staticmethod, classmethod)):
            visit(value.__func__)
            return
        if isinstance(value, types.FunctionType):
            if value.__module__.startswith("qsol_geo_reason"):
                functions.append(value)
                for cell in value.__closure__ or ():
                    try:
                        child = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(child, (types.FunctionType, dict, list, tuple, set, frozenset)):
                        visit(child)
            return
        if isinstance(value, type):
            if getattr(value, "__module__", "").startswith("qsol_geo_reason"):
                for child in vars(value).values():
                    if isinstance(child, (types.FunctionType, staticmethod, classmethod)):
                        visit(child)
            return
        if isinstance(value, dict):
            for child in value.values():
                if isinstance(child, (types.FunctionType, dict, list, tuple, set, frozenset)):
                    visit(child)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                if isinstance(child, (types.FunctionType, dict, list, tuple, set, frozenset)):
                    visit(child)

    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("qsol_geo_reason") or not isinstance(module, types.ModuleType):
            continue
        for value in vars(module).values():
            if isinstance(value, (types.FunctionType, type)):
                visit(value)
    return tuple(functions)


def _assert_loaded_importable_callables_match_source(root: Path) -> None:
    """Bind already-imported source-backed QSOL callables to clean tracked bytes."""
    source_cache: dict[str, tuple[Path, dict[tuple[str, int], types.CodeType]]] = {}
    authenticated = 0
    package_root = (root / "src" / "qsol_geo_reason").resolve()
    for function in _loaded_qsol_python_functions():
        code = function.__code__
        filename = code.co_filename
        if not filename or filename.startswith("<"):
            continue
        try:
            observed_path = Path(filename).resolve()
            observed_path.relative_to(package_root)
        except ValueError:
            # Provenance unit tests intentionally bind synthetic temporary Git
            # checkouts while the test runner has the real package loaded. Only
            # code actually loaded from the checkout being authenticated belongs
            # to this receipt. A real dirty-before-import package callable has a
            # co_filename inside this root and remains mandatory.
            continue
        except OSError as exc:
            raise SourceIdentityError(
                f"unable to resolve loaded callable source path: {function.__module__}.{function.__qualname__}"
            ) from exc

        module_name = function.__module__
        if module_name not in source_cache:
            source_path = _module_source_path(root, module_name)
            source_cache[module_name] = (
                source_path,
                _compiled_source_code_index(source_path),
            )
        source_path, index = source_cache[module_name]
        if observed_path != source_path:
            continue
        expected = index.get((code.co_qualname, code.co_firstlineno))
        if expected is None or _stable_code_identity(code) != _stable_code_identity(expected):
            raise SourceIdentityError(
                "loaded canonical callable does not match the clean tracked source: "
                f"{module_name}.{function.__qualname__}"
            )
        authenticated += 1

    # A synthetic repository used only for source-provenance tests may have no
    # callables loaded from its package root. A real canonical checkout does: the
    # capture API is already imported before OBSERVATION reaches this boundary.
    if authenticated == 0 and package_root == (Path(__file__).resolve().parents[2] / "src" / "qsol_geo_reason").resolve():
        raise SourceIdentityError(
            "unable to authenticate any loaded qsol_geo_reason callables against tracked source"
        )


def _authenticate_importable_bytecode(root: Path, paths: tuple[str, ...]) -> None:
    """Accept only caches whose executable code exactly matches tracked source.

    The normal editable-install CLI imports this package before the final source
    provenance check, so CPython may create ignored ``__pycache__`` entries during
    that same trusted invocation. Blanket rejection would make the canonical CLI
    reject itself. Instead, every ignored importable cache is mapped to a tracked
    package source file and its executable code structure is compared with a fresh
    compilation of the clean checkout source under the cache's optimization lane.
    Source filenames are authenticated separately through the canonical path and
    Git tracking checks, so they are intentionally excluded from the code receipt.
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
            _git_run(
                root,
                "ls-files",
                "--error-unmatch",
                "--",
                source_relative,
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

        if _stable_code_identity(observed) != _stable_code_identity(expected):
            raise SourceIdentityError(
                "importable bytecode cache does not match the clean tracked source: "
                f"{relative}"
            )


def git_source_revision(
    *, require_clean: bool = True, reject_importable_bytecode: bool = False
) -> str | None:
    """Return HEAD for the source checkout, or None when not running from Git.

    Ordinary provenance tolerates disposable interpreter caches. Canonical
    OBSERVATION callers additionally authenticate tracked package files directly
    against the HEAD tree, authenticate Git-ignored package bytecode against that
    source, and bind already-loaded package callables to a fresh compilation of the
    same clean tracked bytes. The direct file hashes do not trust index
    ``assume-unchanged`` or ``skip-worktree`` hints, and all Git object lookups ignore
    local replacement refs.
    """
    root = source_repo_root()
    try:
        probe = _git_run(
            root,
            "rev-parse",
            "--show-toplevel",
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
        status = _git_run(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            check=True,
            capture_output=True,
            text=True,
        )
        if _status_has_source_changes(status.stdout):
            raise SourceIdentityError(
                "source checkout is dirty; commit or stash source-relevant changes before binding an implementation revision"
            )

    head = _git_run(
        root,
        "rev-parse",
        "HEAD",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head:
        raise SourceIdentityError("git HEAD is empty")

    if require_clean and reject_importable_bytecode:
        _assert_tracked_importable_source_matches_head(root, head)
        _authenticate_importable_bytecode(root, _ignored_importable_bytecode(root))
        _assert_loaded_importable_callables_match_source(root)

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

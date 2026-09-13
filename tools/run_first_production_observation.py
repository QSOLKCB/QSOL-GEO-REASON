"""Second-stage authenticated launcher for GEO-CAP-001-EXP-001.

Canonical production entry uses the authenticated Git-blob ``qsol_first_observation``
bootstrap documented in ``experiments/GEO-CAP-001-EXP-001.md``. The working-tree
launcher is not itself a trust root and must not be executed directly for evidence.
The bootstrap resolves one immutable commit before reading the launcher blob and
injects both that commit and the exact launcher-blob SHA-256 into this committed
program. This launcher reauthenticates that same blob, binds all tracked package
source to the same commit, and forces that commit through the evidence-producing
facade. It also validates the original launcher pathname and checkout-local ancestors
without following symlinks, removes native/Python loader, Git-redirection, and network
transport trust overrides, and execs a second ``-I -S -B`` interpreter. Before either
process imports ``qsol_geo_reason``, the launcher rejects import shadows/bytecode and
directly authenticates every tracked package file against the bootstrap-bound Git
revision independently of index flags. The second interpreter re-hashes the
authenticated source manifest immediately before prepending ``src`` to ``sys.path``.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import stat
import subprocess as process
import sys
import sysconfig
from pathlib import Path, PurePosixPath
from typing import Any


def _repository_root_from_launcher_path(path: str | os.PathLike[str]) -> Path:
    """Derive the checkout root without following launcher or checkout-local symlinks."""
    requested = Path(os.path.abspath(os.fspath(path)))
    if (
        requested.name != "run_first_production_observation.py"
        or requested.parent.name != "tools"
    ):
        raise RuntimeError(
            "canonical production launcher pathname is not tools/run_first_production_observation.py"
        )

    repository_root = requested.parent.parent
    checks = (
        (requested, stat.S_ISREG, "launcher file"),
        (requested.parent, stat.S_ISDIR, "launcher tools directory"),
        (repository_root, stat.S_ISDIR, "launcher repository root"),
    )
    for candidate, expected_type, label in checks:
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"canonical production launcher cannot inspect {label}: {candidate}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(
                f"canonical production launcher rejects symlinked {label}: {candidate}"
            )
        if not expected_type(info.st_mode):
            raise RuntimeError(
                f"canonical production launcher requires a valid {label}: {candidate}"
            )
    return repository_root


ROOT = _repository_root_from_launcher_path(__file__)
_ORCHESTRATOR_MODULE = "qsol_geo_reason.first_production_observation"
_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_LOADER_ENV_NAMES = frozenset({"GLIBC_TUNABLES", "LIBPATH", "SHLIB_PATH"})
_TRANSPORT_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HF_HUB_DISABLE_SSL_VERIFY",
        "HF_HUB_DISABLE_SSL_VERIFICATION",
    }
)
_NATIVE_EXTENSION_SUFFIXES = tuple(
    sorted(
        {
            suffix.lower()
            for suffix in (*importlib.machinery.EXTENSION_SUFFIXES, ".so", ".pyd")
            if suffix
        },
        key=len,
        reverse=True,
    )
)
_IMPORTABLE_FILE_SUFFIXES = tuple(
    sorted(
        {
            suffix.lower()
            for suffix in (
                *importlib.machinery.SOURCE_SUFFIXES,
                *importlib.machinery.BYTECODE_SUFFIXES,
                *importlib.machinery.EXTENSION_SUFFIXES,
                ".py",
                ".pyc",
                ".pyo",
                ".so",
                ".pyd",
            )
            if suffix
        },
        key=len,
        reverse=True,
    )
)
_BYTECODE_SUFFIXES = tuple(
    sorted(
        {
            suffix.lower()
            for suffix in (*importlib.machinery.BYTECODE_SUFFIXES, ".pyc", ".pyo")
            if suffix
        },
        key=len,
        reverse=True,
    )
)
_PACKAGE_GIT_ROOT = "src/qsol_geo_reason"
_LAUNCHER_GIT_PATH = "tools/run_first_production_observation.py"
_ORCHESTRATOR_BOOTSTRAP = (
    "import hashlib,importlib.machinery,json,pathlib,sys;"
    "src=sys.argv[1];"
    "revision=sys.argv[2];"
    "source_manifest=json.loads(sys.argv[3]);"
    "sep=sys.argv.index('--');"
    "paths=sys.argv[4:sep];"
    "args=sys.argv[sep+1:];"
    "srcroot=pathlib.Path(src);"
    "pkg=srcroot/'qsol_geo_reason';"
    "(srcroot.is_symlink() or pkg.is_symlink()) and (_ for _ in ()).throw(RuntimeError('canonical production launcher rejects symlinked src/package roots'));"
    "suffixes=tuple(sorted({s.lower() for s in (*importlib.machinery.EXTENSION_SUFFIXES,'.so','.pyd') if s},key=len,reverse=True));"
    "importsfx=tuple(sorted({s.lower() for s in (*importlib.machinery.SOURCE_SUFFIXES,*importlib.machinery.BYTECODE_SUFFIXES,*importlib.machinery.EXTENSION_SUFFIXES,'.py','.pyc','.pyo','.so','.pyd') if s},key=len,reverse=True));"
    "bytecodesfx=tuple(sorted({s.lower() for s in (*importlib.machinery.BYTECODE_SUFFIXES,'.pyc','.pyo') if s},key=len,reverse=True));"
    "native=sorted(str(p) for p in pkg.rglob('*') if p.is_file() and p.name.lower().endswith(suffixes));"
    "native and (_ for _ in ()).throw(RuntimeError('canonical production launcher rejects native extension artifacts in the qsol_geo_reason source tree: '+','.join(native)));"
    "topmods=sorted(str(p) for p in srcroot.iterdir() if p!=pkg and p.is_file() and p.name.lower().endswith(importsfx));"
    "toppkgs=sorted(str(i) for p in srcroot.iterdir() if p!=pkg and p.is_dir() for i in (p/('__init__'+s) for s in importsfx) if i.is_file());"
    "bytecode=sorted(str(p) for p in pkg.rglob('*') if p.is_file() and p.name.lower().endswith(bytecodesfx));"
    "pkgdirs=sorted(str(i) for p in pkg.rglob('*') if p.is_dir() and p.name!='__pycache__' for i in (p/('__init__'+s) for s in importsfx) if i.is_file());"
    "pyshadows=sorted(set(topmods+toppkgs+bytecode+pkgdirs));"
    "pyshadows and (_ for _ in ()).throw(RuntimeError('canonical production launcher rejects pure-Python package shadows and other importable source shadows before src is trusted: '+','.join(pyshadows)));"
    "(not isinstance(source_manifest,dict) or not source_manifest) and (_ for _ in ()).throw(RuntimeError('canonical production launcher received an empty tracked-source manifest'));"
    "sourcebad=sorted(rel for rel,digest in source_manifest.items() if ((p:=pkg.joinpath(*pathlib.PurePosixPath(rel).parts)).is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest));"
    "sourcebad and (_ for _ in ()).throw(RuntimeError('canonical production launcher rejects tracked package source that does not match bound revision '+revision+': '+','.join(sourcebad)));"
    "sys.path.insert(0,src);"
    "[sys.path.append(p) for p in paths if p not in sys.path];"
    "sys.argv=['qsol_geo_reason.first_production_observation',*args,'--implementation-revision',revision];"
    "from qsol_geo_reason.first_production_observation import main;"
    "raise SystemExit(main())"
)


def _trusted_system_path() -> str:
    if os.name == "nt":
        return os.pathsep.join(
            (
                r"C:\Program Files\Git\cmd",
                r"C:\Program Files\Git\bin",
                r"C:\Windows\System32",
            )
        )
    return os.pathsep.join(("/usr/bin", "/bin", "/run/current-system/sw/bin"))


def _sanitized_orchestrator_environment() -> dict[str, str]:
    """Return an environment that cannot redirect code or canonical-Hub trust."""
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper == "PATH":
            continue
        if upper.startswith("PYTHON") or upper.startswith("GIT_"):
            continue
        if upper.startswith(_LOADER_ENV_PREFIXES) or upper in _LOADER_ENV_NAMES:
            continue
        if upper in _TRANSPORT_ENV_NAMES:
            continue
        environment[key] = value
    environment.update(
        {
            "PATH": _trusted_system_path(),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def _system_git_candidates() -> tuple[Path, ...]:
    if os.name == "nt":
        return (
            Path(r"C:\Program Files\Git\cmd\git.exe"),
            Path(r"C:\Program Files\Git\bin\git.exe"),
        )
    return (
        Path("/usr/bin/git"),
        Path("/bin/git"),
        Path("/run/current-system/sw/bin/git"),
    )


def _hash_trusted_git_executable(path: Path) -> tuple[Path, str]:
    try:
        resolved = Path(path).resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise RuntimeError(f"unable to resolve trusted Git executable {path}") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise RuntimeError(
            f"trusted Git executable is not an executable regular file: {resolved}"
        )
    if os.name == "posix" and (
        info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise RuntimeError(
            f"trusted Git executable is not root-owned and non-writable by group/other: {resolved}"
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"unable to hash trusted Git executable {resolved}") from exc
    return resolved, digest.hexdigest()


def _trusted_git_identity() -> tuple[Path, str]:
    last_error: BaseException | None = None
    for candidate in _system_git_candidates():
        try:
            return _hash_trusted_git_executable(candidate)
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(
        "canonical production launcher requires a trusted system Git executable"
    ) from last_error


def _trusted_git_environment() -> dict[str, str]:
    environment = _sanitized_orchestrator_environment()
    environment["LC_ALL"] = "C"
    return environment


def _git_identity_command(
    root: Path,
    git_path: Path,
    git_digest: str,
    *args: str,
    text: bool,
) -> process.CompletedProcess[Any]:
    completed = process.run(
        [str(git_path), "-C", str(root), *args],
        env=_trusted_git_environment(),
        check=True,
        capture_output=True,
        text=text,
    )
    observed_path, observed_digest = _hash_trusted_git_executable(git_path)
    if observed_path != git_path or observed_digest != git_digest:
        raise RuntimeError(
            "trusted Git executable changed during pre-import source authentication"
        )
    return completed


def _require_authenticated_bootstrap_identity() -> tuple[str, str]:
    revision = globals().get("_QSOL_AUTHENTICATED_BOOTSTRAP_REVISION")
    launcher_digest = globals().get("_QSOL_AUTHENTICATED_LAUNCHER_SHA256")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in "0123456789abcdef" for ch in revision)
    ):
        raise RuntimeError(
            "canonical production launcher requires the bootstrap-bound lowercase 40-hex revision"
        )
    if (
        not isinstance(launcher_digest, str)
        or len(launcher_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in launcher_digest)
    ):
        raise RuntimeError(
            "canonical production launcher requires the bootstrap-bound launcher blob SHA-256"
        )
    return revision, launcher_digest


def _authenticate_bootstrap_launcher_blob(
    *,
    revision: str,
    launcher_digest: str,
    git_path: Path,
    git_digest: str,
) -> None:
    """Require the bytes already executing to be the launcher blob from the bound commit."""
    try:
        root = ROOT.resolve(strict=True)
        bound = _git_identity_command(
            root,
            git_path,
            git_digest,
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
            text=True,
        ).stdout.strip()
        if bound != revision:
            raise RuntimeError(
                "authenticated launcher bootstrap revision is not a stable Git commit"
            )
        committed = _git_identity_command(
            root,
            git_path,
            git_digest,
            "cat-file",
            "blob",
            f"{revision}:{_LAUNCHER_GIT_PATH}",
            text=False,
        ).stdout
    except (OSError, process.CalledProcessError) as exc:
        raise RuntimeError(
            "canonical production launcher cannot authenticate its bootstrap-bound Git blob"
        ) from exc
    if hashlib.sha256(committed).hexdigest() != launcher_digest:
        raise RuntimeError(
            "canonical production launcher bytes do not match the bootstrap-bound launcher blob"
        )


def _authenticate_tracked_package_source(
    *,
    git_path: Path,
    git_digest: str,
    revision: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Bind all tracked package bytes to a commit before importing the package."""
    try:
        root = ROOT.resolve(strict=True)
        requested_revision = "HEAD" if revision is None else revision
        bound_revision = _git_identity_command(
            root,
            git_path,
            git_digest,
            "rev-parse",
            "--verify",
            f"{requested_revision}^{{commit}}",
            text=True,
        ).stdout.strip()
        if not bound_revision or len(bound_revision) != 40:
            raise RuntimeError("canonical production launcher could not bind a 40-hex Git commit")
        if revision is not None and bound_revision != revision:
            raise RuntimeError(
                "canonical production launcher Git revision changed during source authentication"
            )

        listing = _git_identity_command(
            root,
            git_path,
            git_digest,
            "ls-tree",
            "-r",
            "-z",
            bound_revision,
            "--",
            _PACKAGE_GIT_ROOT,
            text=True,
        ).stdout
    except (OSError, process.CalledProcessError) as exc:
        raise RuntimeError(
            "canonical production launcher cannot enumerate tracked package source"
        ) from exc

    records = tuple(record for record in listing.split("\0") if record)
    if not records:
        raise RuntimeError("bound Git revision contains no tracked qsol_geo_reason package files")

    manifest: dict[str, str] = {}
    package_prefix = f"{_PACKAGE_GIT_ROOT}/"
    for record in records:
        try:
            metadata, relative = record.split("\t", 1)
            mode, object_type, object_id = metadata.split(" ", 2)
        except ValueError as exc:
            raise RuntimeError("malformed tracked package entry in bound Git revision") from exc
        normalized = relative.replace("\\", "/")
        if not normalized.startswith(package_prefix):
            raise RuntimeError(
                f"tracked package source escaped canonical package root: {relative}"
            )
        package_relative = normalized[len(package_prefix) :]
        parts = PurePosixPath(package_relative).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise RuntimeError(f"invalid tracked package path: {relative}")
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeError(
                f"canonical package source must be a regular tracked file: {relative}"
            )

        requested = root.joinpath(*PurePosixPath(normalized).parts)
        try:
            before = requested.lstat()
        except OSError as exc:
            raise RuntimeError(f"tracked package source is missing: {relative}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"tracked package source is not a regular file: {relative}")

        try:
            committed = _git_identity_command(
                root,
                git_path,
                git_digest,
                "cat-file",
                "blob",
                object_id,
                text=False,
            ).stdout
            observed = requested.read_bytes()
            after = requested.lstat()
        except (OSError, process.CalledProcessError) as exc:
            raise RuntimeError(f"unable to authenticate tracked package source: {relative}") from exc
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise RuntimeError(
                f"tracked package source changed pathname identity during authentication: {relative}"
            )
        if observed != committed:
            raise RuntimeError(
                "canonical production launcher rejects tracked package source that does not match "
                f"bound revision independently of index flags: {relative}"
            )
        manifest[package_relative] = hashlib.sha256(committed).hexdigest()

    return bound_revision, manifest


def _literal_site_package_paths() -> list[str]:
    """Locate package directories without importing site or executing .pth files."""
    paths: list[str] = []
    executable = Path(sys.executable)
    venv_root = executable.parent.parent
    if (venv_root / "pyvenv.cfg").is_file():
        if os.name == "nt":
            candidates = [venv_root / "Lib" / "site-packages"]
        else:
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            candidates = [
                venv_root / "lib" / version / "site-packages",
                venv_root / "lib64" / version / "site-packages",
            ]
    else:
        candidates = []
        for key in ("purelib", "platlib"):
            value = sysconfig.get_paths().get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(Path(value))
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_dir() and str(resolved) not in paths:
            paths.append(str(resolved))
    if not paths:
        raise RuntimeError(
            "canonical production launcher cannot locate literal interpreter package directories"
        )
    return paths


def _assert_no_importable_native_extensions() -> None:
    """Reject native modules that could shadow the tracked pure-Python package."""
    package_root = ROOT / "src" / "qsol_geo_reason"
    try:
        native = sorted(
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(_NATIVE_EXTENSION_SUFFIXES)
        )
    except OSError as exc:
        raise RuntimeError(
            f"canonical production launcher cannot inspect native-extension boundary: {exc}"
        ) from exc
    if native:
        rendered = ", ".join(str(path) for path in native)
        raise RuntimeError(
            "canonical production launcher rejects native extension artifacts in the "
            f"qsol_geo_reason source tree: {rendered}"
        )


def _package_init_artifacts(directory: Path) -> list[Path]:
    return [
        candidate
        for suffix in _IMPORTABLE_FILE_SUFFIXES
        if (candidate := directory / f"__init__{suffix}").is_file()
    ]


def _assert_no_importable_python_shadows() -> None:
    """Reject every import candidate that can run before source authentication.

    The canonical source layout has one top-level package (``qsol_geo_reason``) and
    that package is intentionally flat. Prepending ``src`` must therefore not expose
    any other top-level module/package, any package bytecode (including cache-tagged
    ``__pycache__`` entries), or any importable subpackage that could execute before
    the tracked source has been authenticated. ``-B`` disables bytecode writes only;
    it does not prevent CPython from loading an existing valid cache-tagged ``.pyc``.
    """
    source_root = ROOT / "src"
    package_root = source_root / "qsol_geo_reason"
    try:
        if source_root.is_symlink() or package_root.is_symlink():
            raise RuntimeError(
                "canonical production launcher rejects symlinked src/package roots"
            )

        shadows: list[Path] = []
        for entry in source_root.iterdir():
            if entry == package_root:
                continue
            if entry.is_file() and entry.name.lower().endswith(_IMPORTABLE_FILE_SUFFIXES):
                shadows.append(entry)
            elif entry.is_dir():
                shadows.extend(_package_init_artifacts(entry))

        shadows.extend(
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(_BYTECODE_SUFFIXES)
        )
        for directory in package_root.rglob("*"):
            if directory.is_dir() and directory.name != "__pycache__":
                shadows.extend(_package_init_artifacts(directory))
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"canonical production launcher cannot inspect import-shadow boundary: {exc}"
        ) from exc

    unique = sorted(set(shadows))
    if unique:
        rendered = ", ".join(str(path) for path in unique)
        raise RuntimeError(
            "canonical production launcher rejects pure-Python package shadows and other "
            f"importable source shadows before src is trusted: {rendered}"
        )


def _assert_initial_launcher_boundary() -> None:
    if (
        not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.no_user_site
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError(
            "canonical production launcher must be invoked through the authenticated Git-blob "
            "qsol_first_observation bootstrap documented in GEO-CAP-001-EXP-001"
        )
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError(
            "canonical production launcher inherited a Python startup customization module"
        )


def main() -> int:
    _assert_initial_launcher_boundary()
    _assert_no_importable_native_extensions()
    _assert_no_importable_python_shadows()
    bootstrap_revision, bootstrap_launcher_digest = _require_authenticated_bootstrap_identity()
    git_path, git_digest = _trusted_git_identity()
    _authenticate_bootstrap_launcher_blob(
        revision=bootstrap_revision,
        launcher_digest=bootstrap_launcher_digest,
        git_path=git_path,
        git_digest=git_digest,
    )
    revision, source_manifest = _authenticate_tracked_package_source(
        git_path=git_path,
        git_digest=git_digest,
        revision=bootstrap_revision,
    )
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _ORCHESTRATOR_BOOTSTRAP,
        str((ROOT / "src").resolve()),
        revision,
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")),
        *_literal_site_package_paths(),
        "--",
        *sys.argv[1:],
    ]
    os.execve(
        sys.executable,
        argv,
        _sanitized_orchestrator_environment(),
    )
    return 2  # pragma: no cover - execve does not return on success.


if __name__ == "__main__":
    raise SystemExit(main())
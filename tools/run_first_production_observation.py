"""Launch the authenticated GEO-CAP-001-EXP-001 package orchestrator.

Canonical invocation is ``python -I -S -B tools/run_first_production_observation.py``.
The first interpreter must already suppress Python startup customization; this launcher
then removes native/Python loader, Git-redirection, and network transport trust
overrides before execing a second ``-I -S -B`` interpreter. Before either process
imports ``qsol_geo_reason``, the source tree is checked for every importable artifact
that could outrank authenticated modules when ``src`` is prepended to ``sys.path``.
"""
from __future__ import annotations

import importlib.machinery
import os
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
_ORCHESTRATOR_BOOTSTRAP = (
    "import importlib.machinery,pathlib,sys;"
    "src=sys.argv[1];"
    "sep=sys.argv.index('--');"
    "paths=sys.argv[2:sep];"
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
    "pkgdirs=sorted(str(i) for p in pkg.rglob('*') if p.is_dir() for i in (p/('__init__'+s) for s in importsfx) if i.is_file());"
    "pyshadows=sorted(set(topmods+toppkgs+bytecode+pkgdirs));"
    "pyshadows and (_ for _ in ()).throw(RuntimeError('canonical production launcher rejects pure-Python package shadows and other importable source shadows before src is trusted: '+','.join(pyshadows)));"
    "sys.path.insert(0,src);"
    "[sys.path.append(p) for p in paths if p not in sys.path];"
    "sys.argv=['qsol_geo_reason.first_production_observation',*args];"
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
    any other top-level module/package, any sourceless bytecode, or any importable
    subpackage that could outrank a tracked sibling module. This check deliberately
    does not depend on Git ignore/index state because it runs before package imports.
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
            if path.is_file() and path.name.lower().endswith(_BYTECODE_SUFFIXES)
        )
        for directory in package_root.rglob("*"):
            if directory.is_dir():
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
            "canonical production launcher must be invoked with: "
            "python -I -S -B tools/run_first_production_observation.py ..."
        )
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError(
            "canonical production launcher inherited a Python startup customization module"
        )


def main() -> int:
    _assert_initial_launcher_boundary()
    _assert_no_importable_native_extensions()
    _assert_no_importable_python_shadows()
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _ORCHESTRATOR_BOOTSTRAP,
        str((ROOT / "src").resolve()),
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

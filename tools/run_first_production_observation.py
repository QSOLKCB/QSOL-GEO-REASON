"""Launch the authenticated GEO-CAP-001-EXP-001 package orchestrator.

Canonical invocation is ``python -I -S -B tools/run_first_production_observation.py``.
The first interpreter must already suppress Python startup customization; this launcher
then removes native/Python loader, Git-redirection, and network transport trust
overrides before execing a second ``-I -S -B`` interpreter. The second interpreter
bootstraps only the checked-out ``src`` tree and literal interpreter package directories,
without executing ``.pth`` files or ``sitecustomize``.
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
_ORCHESTRATOR_BOOTSTRAP = (
    "import sys;"
    "src=sys.argv[1];"
    "sep=sys.argv.index('--');"
    "paths=sys.argv[2:sep];"
    "args=sys.argv[sep+1:];"
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

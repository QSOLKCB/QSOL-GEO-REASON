"""Launch the authenticated GEO-CAP-001-EXP-001 package orchestrator.

This file intentionally contains no evidence-producing logic. The isolated package
module is inside ``src/qsol_geo_reason`` and is directly authenticated against Git HEAD
before an OBSERVATION attempt is created. The launcher removes native/Python loader,
Git-redirection, and network transport trust overrides before the authenticated
interpreter starts.
"""
from __future__ import annotations

import os
import sys


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


def main() -> int:
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "qsol_geo_reason.first_production_observation",
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

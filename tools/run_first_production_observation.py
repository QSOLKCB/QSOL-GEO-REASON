"""Launch the authenticated GEO-CAP-001-EXP-001 package orchestrator.

This file intentionally contains no evidence-producing logic. The isolated package
module is inside ``src/qsol_geo_reason`` and is directly authenticated against Git HEAD
before an OBSERVATION attempt is created. The launcher removes native/Python loader
injection controls before the authenticated interpreter starts.
"""
from __future__ import annotations

import os
import sys


_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_LOADER_ENV_NAMES = frozenset({"GLIBC_TUNABLES", "LIBPATH", "SHLIB_PATH"})


def _sanitized_orchestrator_environment() -> dict[str, str]:
    """Return an environment that cannot inject code into the authenticated child."""
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("PYTHON"):
            continue
        if upper.startswith(_LOADER_ENV_PREFIXES) or upper in _LOADER_ENV_NAMES:
            continue
        environment[key] = value
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

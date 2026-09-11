"""Launch the authenticated GEO-CAP-001-EXP-001 package orchestrator.

This file intentionally contains no evidence-producing logic. The isolated package
module is inside ``src/qsol_geo_reason`` and is directly authenticated against Git HEAD
before an OBSERVATION attempt is created.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "qsol_geo_reason.first_production_observation",
        *sys.argv[1:],
    ]
    os.execv(sys.executable, argv)
    return 2  # pragma: no cover - execv does not return on success.


if __name__ == "__main__":
    raise SystemExit(main())

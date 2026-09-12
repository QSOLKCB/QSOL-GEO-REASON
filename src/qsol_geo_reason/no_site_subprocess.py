"""Shared no-site CPython bootstrap for authenticated package subprocesses.

Evidence-producing subprocesses must not execute ``sitecustomize``, ``usercustomize``,
or executable ``.pth`` files before the authenticated QSOL package code starts.  This
module constructs ``-I -S -B`` commands that add only the checked-out source tree and
literal interpreter package directories without importing ``site``.
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_ENTRYPOINTS = {
    "qsol_geo_reason.capture_cli": "main",
    "qsol_geo_reason.capture_worker": "main",
}


def literal_site_package_paths() -> list[str]:
    """Locate interpreter package directories without processing startup hooks."""
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
            "unable to locate literal interpreter package directories for no-site subprocess"
        )
    return paths


def isolated_package_command(module_name: str, argv: Sequence[str]) -> list[str]:
    """Build a fixed-entrypoint ``-I -S -B`` command without automatic site startup."""
    callable_name = _ALLOWED_ENTRYPOINTS.get(module_name)
    if callable_name is None:
        raise ValueError(f"unsupported authenticated package entrypoint {module_name!r}")
    bootstrap = (
        "import sys;"
        "src=sys.argv[1];"
        "sep=sys.argv.index('--');"
        "paths=sys.argv[2:sep];"
        "args=sys.argv[sep+1:];"
        "sys.path.insert(0,src);"
        "[sys.path.append(p) for p in paths if p not in sys.path];"
        f"sys.argv=[{module_name!r},*args];"
        f"from {module_name} import {callable_name};"
        f"raise SystemExit({callable_name}())"
    )
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        bootstrap,
        str((ROOT / "src").resolve()),
        *literal_site_package_paths(),
        "--",
        *[str(value) for value in argv],
    ]


__all__ = ["isolated_package_command", "literal_site_package_paths"]

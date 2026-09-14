"""Shared authenticated no-site CPython bootstrap for package subprocesses.

Evidence-producing subprocesses must not execute ``sitecustomize``, ``usercustomize``,
executable ``.pth`` files, or mutable checkout package code before the launcher-bound
QSOL source identity has been rechecked. This module carries the authenticated Git
revision and complete tracked-package SHA-256 manifest into each ``-I -S -B`` child.
The child rejects import shadows, bytecode, and package-local native extensions and
re-hashes every manifest entry before prepending ``src`` or importing any
``qsol_geo_reason`` module. It then installs the same verified identity in the child
so nested authenticated subprocesses retain the original trust root.
"""
from __future__ import annotations

import json
import os
import sys
import sysconfig
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_ENTRYPOINTS = {
    "qsol_geo_reason.capture_cli": "main",
    "qsol_geo_reason.capture_worker": "main",
    "qsol_geo_reason.capture_hub_prepare_worker": "main",
}
_AUTHENTICATED_SOURCE_REVISION: str | None = None
_AUTHENTICATED_SOURCE_MANIFEST: dict[str, str] | None = None


def _canonical_source_identity(
    revision: str,
    source_manifest: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in "0123456789abcdef" for ch in revision)
    ):
        raise RuntimeError(
            "authenticated package subprocess requires a lowercase 40-hex source revision"
        )
    if not isinstance(source_manifest, Mapping) or not source_manifest:
        raise RuntimeError(
            "authenticated package subprocess requires a non-empty tracked-source manifest"
        )
    canonical: dict[str, str] = {}
    for raw_path, digest in source_manifest.items():
        if not isinstance(raw_path, str):
            raise RuntimeError("tracked-source manifest paths must be strings")
        pure = PurePosixPath(raw_path)
        if (
            raw_path != pure.as_posix()
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise RuntimeError(
                f"tracked-source manifest contains a noncanonical package path: {raw_path!r}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise RuntimeError(
                f"tracked-source manifest contains an invalid SHA-256 for {raw_path}"
            )
        canonical[raw_path] = digest
    return revision, dict(sorted(canonical.items()))


def install_authenticated_source_manifest(
    revision: str,
    source_manifest: Mapping[str, str],
) -> None:
    """Install the launcher-bound package identity for later authenticated children."""
    canonical_revision, canonical_manifest = _canonical_source_identity(
        revision,
        source_manifest,
    )
    global _AUTHENTICATED_SOURCE_REVISION, _AUTHENTICATED_SOURCE_MANIFEST
    if _AUTHENTICATED_SOURCE_REVISION is not None:
        if (
            _AUTHENTICATED_SOURCE_REVISION != canonical_revision
            or _AUTHENTICATED_SOURCE_MANIFEST != canonical_manifest
        ):
            raise RuntimeError(
                "authenticated package source identity cannot be rebound within one process"
            )
        return
    _AUTHENTICATED_SOURCE_REVISION = canonical_revision
    _AUTHENTICATED_SOURCE_MANIFEST = canonical_manifest


def _require_authenticated_source_manifest() -> tuple[str, dict[str, str]]:
    revision = _AUTHENTICATED_SOURCE_REVISION
    manifest = _AUTHENTICATED_SOURCE_MANIFEST
    if revision is None or manifest is None:
        raise RuntimeError(
            "authenticated package subprocess requires the launcher-bound tracked-source manifest"
        )
    return revision, dict(manifest)


def literal_site_package_paths() -> list[str]:
    """Locate package directories without processing startup hooks or ``.pth`` files."""
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

    # A parent authenticated no-site bootstrap may have explicitly appended a user
    # site-packages directory after validating the editable install. Preserve those
    # literal directories for nested capture_cli -> capture_worker hops without ever
    # importing site or evaluating .pth files.
    for raw in sys.path:
        if not isinstance(raw, str) or not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.name.lower() == "site-packages":
            candidates.append(candidate)

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_dir() and str(resolved) not in paths:
            paths.append(str(resolved))
    if not paths:
        raise RuntimeError(
            "unable to locate literal interpreter package directories for no-site subprocess"
        )
    return paths


def _authenticated_bootstrap(module_name: str, callable_name: str) -> str:
    """Return a stdlib-only pre-import verifier for one fixed package entrypoint."""
    return (
        "import hashlib,importlib.machinery,json,pathlib,sys;"
        "src=sys.argv[1];"
        "revision=sys.argv[2];"
        "source_manifest=json.loads(sys.argv[3]);"
        "sep=sys.argv.index('--');"
        "paths=sys.argv[4:sep];"
        "args=sys.argv[sep+1:];"
        "srcroot=pathlib.Path(src);"
        "pkg=srcroot/'qsol_geo_reason';"
        "(srcroot.is_symlink() or pkg.is_symlink()) and (_ for _ in ()).throw(RuntimeError('authenticated package subprocess rejects symlinked src/package roots'));"
        "importsfx=tuple(sorted({s.lower() for s in (*importlib.machinery.SOURCE_SUFFIXES,*importlib.machinery.BYTECODE_SUFFIXES,*importlib.machinery.EXTENSION_SUFFIXES,'.py','.pyc','.pyo','.so','.pyd') if s},key=len,reverse=True));"
        "bytecodesfx=tuple(sorted({s.lower() for s in (*importlib.machinery.BYTECODE_SUFFIXES,'.pyc','.pyo') if s},key=len,reverse=True));"
        "nativesfx=tuple(sorted({s.lower() for s in (*importlib.machinery.EXTENSION_SUFFIXES,'.so','.pyd') if s},key=len,reverse=True));"
        "topmods=sorted(str(p) for p in srcroot.iterdir() if p!=pkg and p.is_file() and p.name.lower().endswith(importsfx));"
        "toppkgs=sorted(str(i) for p in srcroot.iterdir() if p!=pkg and p.is_dir() for i in (p/('__init__'+s) for s in importsfx) if i.is_file());"
        "bytecode=sorted(str(p) for p in pkg.rglob('*') if p.is_file() and p.name.lower().endswith(bytecodesfx));"
        "native=sorted(str(p) for p in pkg.rglob('*') if p.is_file() and p.name.lower().endswith(nativesfx));"
        "pkgdirs=sorted(str(i) for p in pkg.rglob('*') if p.is_dir() and p.name!='__pycache__' for i in (p/('__init__'+s) for s in importsfx) if i.is_file());"
        "shadows=sorted(set(topmods+toppkgs+bytecode+native+pkgdirs));"
        "shadows and (_ for _ in ()).throw(RuntimeError('authenticated package subprocess rejects importable source shadows before src is trusted: '+','.join(shadows)));"
        "(not isinstance(source_manifest,dict) or not source_manifest) and (_ for _ in ()).throw(RuntimeError('authenticated package subprocess received an empty tracked-source manifest'));"
        "sourcebad=sorted(rel for rel,digest in source_manifest.items() if ((p:=pkg.joinpath(*pathlib.PurePosixPath(rel).parts)).is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest));"
        "sourcebad and (_ for _ in ()).throw(RuntimeError('authenticated package subprocess rejects tracked package source that does not match bound revision '+revision+': '+','.join(sourcebad)));"
        "sys.path.insert(0,src);"
        "[sys.path.append(p) for p in paths if p not in sys.path];"
        "import qsol_geo_reason.no_site_subprocess as _qsol_ns;"
        "_qsol_ns.install_authenticated_source_manifest(revision,source_manifest);"
        f"sys.argv=[{module_name!r},*args];"
        f"from {module_name} import {callable_name};"
        f"raise SystemExit({callable_name}())"
    )


def isolated_package_command(module_name: str, argv: Sequence[str]) -> list[str]:
    """Build a source-authenticated ``-I -S -B`` fixed-entrypoint command."""
    callable_name = _ALLOWED_ENTRYPOINTS.get(module_name)
    if callable_name is None:
        raise ValueError(f"unsupported authenticated package entrypoint {module_name!r}")
    revision, source_manifest = _require_authenticated_source_manifest()
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _authenticated_bootstrap(module_name, callable_name),
        str((ROOT / "src").resolve()),
        revision,
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")),
        *literal_site_package_paths(),
        "--",
        *[str(value) for value in argv],
    ]


__all__ = [
    "install_authenticated_source_manifest",
    "isolated_package_command",
    "literal_site_package_paths",
]

#!/usr/bin/env python3
"""Verify a cached Lake dependency source graph against a frozen manifest.

The manifest SHA-256 is pinned by CI. Each git dependency must then be checked
out at the revision named by that manifest and have no tracked modifications.
Generated/untracked Lake build products are intentionally ignored here; they
are covered separately by the dependency-artifact receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(path: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed for {path}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lake-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"dependency source root not found: {args.root}")
    if args.root.is_symlink():
        raise SystemExit(f"symlinked dependency source root is not allowed: {args.root}")
    if not args.lake_manifest.is_file():
        raise SystemExit(f"lake manifest not found: {args.lake_manifest}")

    expected = args.expected_manifest_sha256.lower()
    actual = sha256_file(args.lake_manifest)
    if actual != expected:
        raise SystemExit(
            f"lake manifest SHA-256 mismatch: {actual} != {expected}"
        )

    data: dict[str, Any] = json.loads(args.lake_manifest.read_text(encoding="utf-8"))
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        raise SystemExit("lake manifest contains no packages")

    checked = 0
    for package in packages:
        if not isinstance(package, dict) or package.get("type") != "git":
            continue
        name = package.get("name")
        revision = package.get("rev")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"git package has invalid name: {package!r}")
        if not isinstance(revision, str) or len(revision) != 40:
            raise SystemExit(f"git package {name!r} has invalid revision: {revision!r}")

        path = args.root / name
        if not path.is_dir() or path.is_symlink():
            raise SystemExit(f"dependency package directory invalid: {path}")
        git_dir = path / ".git"
        if not git_dir.exists() or git_dir.is_symlink():
            raise SystemExit(f"dependency package lacks ordinary .git metadata: {path}")

        head = git(path, "rev-parse", "HEAD")
        if head.lower() != revision.lower():
            raise SystemExit(
                f"dependency revision mismatch for {name}: {head} != {revision}"
            )
        git(path, "cat-file", "-e", f"{revision}^{{commit}}")
        status = git(path, "status", "--porcelain=v1", "--untracked-files=no")
        if status:
            raise SystemExit(
                f"tracked dependency source modification detected for {name}:\n{status}"
            )
        checked += 1

    if checked == 0:
        raise SystemExit("no git dependency source packages were verified")

    print(
        "dependency-source-state verified "
        f"git_packages={checked} lake_manifest_sha256={actual}"
    )


if __name__ == "__main__":
    main()

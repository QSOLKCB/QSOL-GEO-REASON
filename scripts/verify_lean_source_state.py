#!/usr/bin/env python3
"""Verify a cached Lake dependency source graph against frozen declarations.

The verifier does not trust Git's working-tree status. For every dependency it:
* checks the exact manifest revision;
* rejects non-default index flags such as assume-unchanged/skip-worktree;
* compares tracked file bytes and executable/symlink modes against the pinned
  commit tree; and
* optionally creates/verifies a receipt binding the current lakefile.lean
  declaration to the frozen lake-manifest.json and dependency commit trees.

Generated Lake build products under .lake are outside this source-state receipt;
compiled dependency artifacts are verified separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "GEO-LEAN-SOURCE-RECEIPT-2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(path: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed for {path}: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def git_text(path: Path, *args: str) -> str:
    return run_git(path, *args).decode("utf-8", "strict").strip()


def safe_relpath(raw: bytes, package: str) -> PurePosixPath:
    try:
        text = raw.decode("utf-8", "surrogateescape")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"invalid tracked path in {package}: {exc}") from exc
    rel = PurePosixPath(text)
    if rel.is_absolute() or not rel.parts or any(part in ("", "..") for part in rel.parts):
        raise SystemExit(f"unsafe tracked path in {package}: {text!r}")
    return rel


def ensure_no_symlink_parents(root: Path, rel: PurePosixPath, checked: set[Path]) -> None:
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if current in checked:
            continue
        try:
            st = current.lstat()
        except FileNotFoundError:
            raise SystemExit(f"tracked parent directory missing: {current}")
        if stat.S_ISLNK(st.st_mode):
            raise SystemExit(f"symlinked tracked parent directory is not allowed: {current}")
        if not stat.S_ISDIR(st.st_mode):
            raise SystemExit(f"tracked parent is not a directory: {current}")
        checked.add(current)


def git_blob_oid(data: bytes, algorithm: str) -> str:
    payload = f"blob {len(data)}\0".encode("ascii") + data
    if algorithm == "sha1":
        return hashlib.sha1(payload).hexdigest()
    if algorithm == "sha256":
        return hashlib.sha256(payload).hexdigest()
    raise SystemExit(f"unsupported Git object format: {algorithm}")


def reject_index_flags(path: Path, package: str) -> None:
    # `git ls-files -v` emits normal cached entries as `H path`. Lowercase
    # letters expose assume-unchanged; `S` exposes skip-worktree. Reject every
    # non-H state rather than rely on Git status, which can honor those bits.
    raw = run_git(path, "ls-files", "-v", "-z")
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        if len(entry) < 3 or entry[1:2] != b" ":
            raise SystemExit(f"unexpected git ls-files record in {package}: {entry!r}")
        marker = entry[:1]
        if marker != b"H":
            tracked = entry[2:].decode("utf-8", "surrogateescape")
            raise SystemExit(
                f"non-default Git index flag/state in {package}: "
                f"marker={marker.decode('ascii', 'replace')!r} path={tracked!r}"
            )


def verify_commit_tree(path: Path, package: str, revision: str) -> str:
    reject_index_flags(path, package)

    object_format = git_text(path, "rev-parse", "--show-object-format")
    tree_oid = git_text(path, "rev-parse", f"{revision}^{{tree}}")
    raw = run_git(path, "ls-tree", "-r", "-z", "--full-tree", revision)

    checked_parents: set[Path] = set()
    tracked_paths: set[PurePosixPath] = set()
    tracked_files = 0

    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode_b, type_b, oid_b = header.split(b" ", 2)
        except ValueError as exc:
            raise SystemExit(f"malformed git tree record in {package}: {record!r}") from exc

        mode = mode_b.decode("ascii")
        kind = type_b.decode("ascii")
        oid = oid_b.decode("ascii").lower()
        rel = safe_relpath(raw_path, package)
        tracked_paths.add(rel)
        ensure_no_symlink_parents(path, rel, checked_parents)
        work = path.joinpath(*rel.parts)

        if kind == "commit" and mode == "160000":
            if not work.is_dir() or work.is_symlink():
                raise SystemExit(f"gitlink worktree missing/invalid in {package}: {rel}")
            sub_head = git_text(work, "rev-parse", "HEAD").lower()
            if sub_head != oid:
                raise SystemExit(
                    f"gitlink revision mismatch in {package}:{rel}: {sub_head} != {oid}"
                )
            tracked_files += 1
            continue

        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise SystemExit(
                f"unsupported tracked entry in {package}: mode={mode} type={kind} path={rel}"
            )

        try:
            st = work.lstat()
        except FileNotFoundError:
            raise SystemExit(f"tracked file missing in {package}: {rel}")

        if mode == "120000":
            if not stat.S_ISLNK(st.st_mode):
                raise SystemExit(f"tracked symlink replaced by non-symlink in {package}: {rel}")
            data = os.fsencode(os.readlink(work))
        else:
            if not stat.S_ISREG(st.st_mode):
                raise SystemExit(f"tracked regular file has wrong type in {package}: {rel}")
            actual_mode = "100755" if (st.st_mode & 0o111) else "100644"
            if actual_mode != mode:
                raise SystemExit(
                    f"tracked mode mismatch in {package}:{rel}: {actual_mode} != {mode}"
                )
            data = work.read_bytes()

        actual_oid = git_blob_oid(data, object_format)
        if actual_oid.lower() != oid:
            raise SystemExit(
                f"tracked bytes mismatch in {package}:{rel}: {actual_oid} != {oid}"
            )
        tracked_files += 1

    if tracked_files == 0:
        raise SystemExit(f"dependency commit tree contains no tracked entries: {package}")

    # A cache should not inject an untracked Lean/config source that could
    # shadow an import. Ignore Git/Lake metadata directories only.
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        rel_root = root_path.relative_to(path)
        dirs[:] = [d for d in dirs if d not in {".git", ".lake"}]
        for filename in files:
            rel = PurePosixPath(*(rel_root.parts + (filename,)))
            if rel in tracked_paths:
                continue
            if filename.endswith(".lean") or filename in {"lakefile.lean", "lean-toolchain"}:
                raise SystemExit(
                    f"untracked dependency source/config file is not allowed in {package}: {rel}"
                )

    return tree_oid.lower()


def load_manifest(path: Path, expected_sha256: str) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"lake manifest missing/invalid: {path}")
    actual = sha256_file(path)
    expected = expected_sha256.lower()
    if actual != expected:
        raise SystemExit(f"lake manifest SHA-256 mismatch: {actual} != {expected}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        raise SystemExit("lake manifest contains no packages")
    return actual, packages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lake-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--dependency-declaration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()

    if not args.root.is_dir() or args.root.is_symlink():
        raise SystemExit(f"dependency source root missing/invalid: {args.root}")
    if not args.dependency_declaration.is_file() or args.dependency_declaration.is_symlink():
        raise SystemExit(
            f"dependency declaration missing/invalid: {args.dependency_declaration}"
        )
    if args.write_receipt and args.receipt is None:
        raise SystemExit("--write-receipt requires --receipt")

    manifest_sha, packages = load_manifest(
        args.lake_manifest, args.expected_manifest_sha256
    )
    declaration_sha = sha256_file(args.dependency_declaration)

    summaries: list[dict[str, str]] = []
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

        head = git_text(path, "rev-parse", "HEAD").lower()
        if head != revision.lower():
            raise SystemExit(
                f"dependency revision mismatch for {name}: {head} != {revision}"
            )
        git_text(path, "cat-file", "-e", f"{revision}^{{commit}}")
        tree = verify_commit_tree(path, name, revision)
        summaries.append({"name": name, "revision": revision.lower(), "tree": tree})
        checked += 1

    if checked == 0:
        raise SystemExit("no git dependency source packages were verified")
    summaries.sort(key=lambda item: item["name"])

    snapshot = {
        "schema": SCHEMA,
        "dependency_declaration_sha256": declaration_sha,
        "lake_manifest_sha256": manifest_sha,
        "packages": summaries,
    }

    if args.receipt is not None:
        if args.write_receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            if not args.receipt.is_file() or args.receipt.is_symlink():
                raise SystemExit(f"source receipt missing/invalid: {args.receipt}")
            saved = json.loads(args.receipt.read_text(encoding="utf-8"))
            if saved != snapshot:
                raise SystemExit(
                    "dependency source receipt mismatch:\n"
                    f"saved={json.dumps(saved, sort_keys=True)}\n"
                    f"current={json.dumps(snapshot, sort_keys=True)}"
                )

    print(
        "dependency-source-state verified "
        f"git_packages={checked} "
        f"lakefile_sha256={declaration_sha} "
        f"lake_manifest_sha256={manifest_sha}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify a cached Lake dependency source graph against frozen declarations.

The verifier does not trust Git's working-tree status, cached repository
configuration, or mutable object redirection state. For every dependency it:
* sanitizes local Git configuration/hooks using filesystem operations only;
* checks the exact manifest revision with replacement processing disabled;
* rejects Git replacement refs/grafts/object alternates and non-default index
  flags such as assume-unchanged/skip-worktree;
* requires generated per-package Lake state to be purged before verification;
* compares tracked file bytes and executable/symlink modes against the pinned
  commit tree;
* requires the filesystem outside `.git` to be exactly the tracked commit-tree
  closure, rejecting every untracked file, directory, and symlink; and
* only after verification restores the frozen manifest-declared `origin` URL so
  Lake can reuse the authenticated checkout without re-cloning it.

Compiled dependency artifacts are verified separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "GEO-LEAN-SOURCE-RECEIPT-3"
SAFE_GIT_CONFIG = (
    b"[core]\n"
    b"\trepositoryformatversion = 0\n"
    b"\tfilemode = true\n"
    b"\tbare = false\n"
    b"\tlogallrefupdates = true\n"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_path_no_follow(path: Path) -> None:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def sanitize_git_metadata(path: Path, package: str) -> None:
    """Remove repository-configured executable surfaces before any Git call."""
    git_dir = path / ".git"
    if not git_dir.is_dir() or git_dir.is_symlink():
        raise SystemExit(f"dependency package lacks ordinary .git metadata: {path}")

    for forbidden in (
        git_dir / "commondir",
        git_dir / "objects" / "info" / "alternates",
        git_dir / "objects" / "info" / "http-alternates",
        git_dir / "modules",
    ):
        if forbidden.is_symlink() or forbidden.exists():
            raise SystemExit(
                f"forbidden Git metadata in {package}: {forbidden.relative_to(path)}"
            )

    replace_dir = git_dir / "refs" / "replace"
    if replace_dir.is_symlink() or replace_dir.exists():
        raise SystemExit(f"Git replacement-ref directory is not allowed in {package}")

    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_symlink():
        raise SystemExit(f"symlinked packed-refs is not allowed in {package}")
    if packed_refs.is_file():
        for raw_line in packed_refs.read_bytes().splitlines():
            line = raw_line.strip()
            if line and not line.startswith((b"#", b"^")) and b" refs/replace/" in line:
                raise SystemExit(f"packed Git replacement ref is not allowed in {package}")

    grafts = git_dir / "info" / "grafts"
    if grafts.is_symlink():
        raise SystemExit(f"symlinked Git graft metadata in {package}: {grafts}")
    if grafts.is_file() and grafts.read_bytes().strip():
        raise SystemExit(f"Git grafts are not allowed in {package}: {grafts}")

    remove_path_no_follow(git_dir / "hooks")
    remove_path_no_follow(git_dir / "config.worktree")
    remove_path_no_follow(git_dir / "config")

    config = git_dir / "config"
    fd = os.open(
        config,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, SAFE_GIT_CONFIG)
    finally:
        os.close(fd)


def assert_sanitized_git_metadata(path: Path, package: str) -> None:
    git_dir = path / ".git"
    config = git_dir / "config"
    if config.is_symlink() or not config.is_file():
        raise SystemExit(f"sanitized Git config missing/invalid in {package}")
    if config.read_bytes() != SAFE_GIT_CONFIG:
        raise SystemExit(f"sanitized Git config mismatch in {package}")
    for forbidden in (git_dir / "hooks", git_dir / "config.worktree"):
        if forbidden.is_symlink() or forbidden.exists():
            raise SystemExit(f"Git executable/config state survived sanitization: {forbidden}")


def validated_manifest_url(raw: Any, package: str) -> str:
    if not isinstance(raw, str) or not raw.startswith("https://github.com/"):
        raise SystemExit(f"git package {package!r} has invalid manifest URL: {raw!r}")
    if any(ch.isspace() for ch in raw) or "\x00" in raw:
        raise SystemExit(f"git package {package!r} has unsafe manifest URL: {raw!r}")
    return raw


def restore_manifest_remote(path: Path, package: str, url: str) -> None:
    """Restore only the frozen manifest-bound origin after source authentication."""
    assert_sanitized_git_metadata(path, package)
    config = path / ".git" / "config"
    remote = (
        SAFE_GIT_CONFIG
        + b"[remote \"origin\"]\n"
        + b"\turl = " + url.encode("utf-8") + b"\n"
        + b"\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
    )
    remove_path_no_follow(config)
    fd = os.open(
        config,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, remote)
    finally:
        os.close(fd)


def run_git(path: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "submodule.recurse=false",
        "-C",
        str(path),
        *args,
    ]
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
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
    text = raw.decode("utf-8", "surrogateescape")
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


def reject_object_redirection(path: Path, package: str) -> None:
    replace_refs = git_text(path, "for-each-ref", "--format=%(refname)", "refs/replace")
    if replace_refs:
        raise SystemExit(
            f"Git replacement refs are not allowed in {package}:\n{replace_refs}"
        )


def reject_index_flags(path: Path, package: str) -> None:
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


def add_parent_dirs(rel: PurePosixPath, tracked_dirs: set[PurePosixPath]) -> None:
    for count in range(1, len(rel.parts)):
        tracked_dirs.add(PurePosixPath(*rel.parts[:count]))


def verify_worktree_closure(
    path: Path,
    package: str,
    tracked_paths: set[PurePosixPath],
    tracked_dirs: set[PurePosixPath],
) -> None:
    stack: list[tuple[Path, PurePosixPath | None]] = [(path, None)]
    while stack:
        directory, rel_dir = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if rel_dir is None and entry.name == ".git":
                    continue
                rel = PurePosixPath(entry.name) if rel_dir is None else rel_dir / entry.name
                try:
                    st = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    raise SystemExit(f"worktree entry vanished during verification: {package}:{rel}")

                if stat.S_ISLNK(st.st_mode):
                    if rel not in tracked_paths:
                        raise SystemExit(f"untracked worktree symlink is not allowed in {package}: {rel}")
                    continue
                if stat.S_ISDIR(st.st_mode):
                    if rel not in tracked_dirs:
                        raise SystemExit(f"untracked worktree directory is not allowed in {package}: {rel}")
                    stack.append((Path(entry.path), rel))
                    continue
                if stat.S_ISREG(st.st_mode):
                    if rel not in tracked_paths:
                        raise SystemExit(f"untracked worktree file is not allowed in {package}: {rel}")
                    continue
                raise SystemExit(f"unsupported worktree entry type in {package}: {rel}")


def verify_commit_tree(path: Path, package: str, revision: str) -> str:
    assert_sanitized_git_metadata(path, package)
    reject_object_redirection(path, package)
    reject_index_flags(path, package)

    generated_lake = path / ".lake"
    if generated_lake.exists() or generated_lake.is_symlink():
        raise SystemExit(
            f"generated package Lake state must be purged before verification: {generated_lake}"
        )

    object_format = git_text(path, "rev-parse", "--show-object-format")
    tree_oid = git_text(path, "rev-parse", f"{revision}^{{tree}}")
    raw = run_git(path, "ls-tree", "-r", "-z", "--full-tree", revision)

    checked_parents: set[Path] = set()
    tracked_paths: set[PurePosixPath] = set()
    tracked_dirs: set[PurePosixPath] = set()
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
        add_parent_dirs(rel, tracked_dirs)
        ensure_no_symlink_parents(path, rel, checked_parents)
        work = path.joinpath(*rel.parts)

        if kind == "commit" and mode == "160000":
            raise SystemExit(f"gitlink/submodule is not allowed in {package}: {rel}")
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

    verify_worktree_closure(path, package, tracked_paths, tracked_dirs)
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
        raise SystemExit(f"dependency declaration missing/invalid: {args.dependency_declaration}")
    if args.write_receipt and args.receipt is None:
        raise SystemExit("--write-receipt requires --receipt")

    manifest_sha, packages = load_manifest(args.lake_manifest, args.expected_manifest_sha256)
    declaration_sha = sha256_file(args.dependency_declaration)

    summaries: list[dict[str, str]] = []
    verified_packages: list[tuple[Path, str, str]] = []
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
        url = validated_manifest_url(package.get("url"), name)

        path = args.root / name
        if not path.is_dir() or path.is_symlink():
            raise SystemExit(f"dependency package directory invalid: {path}")
        git_dir = path / ".git"
        if not git_dir.is_dir() or git_dir.is_symlink():
            raise SystemExit(f"dependency package lacks ordinary .git metadata: {path}")

        sanitize_git_metadata(path, name)

        head = git_text(path, "rev-parse", "HEAD").lower()
        if head != revision.lower():
            raise SystemExit(f"dependency revision mismatch for {name}: {head} != {revision}")
        git_text(path, "cat-file", "-e", f"{revision}^{{commit}}")
        tree = verify_commit_tree(path, name, revision)
        summaries.append({"name": name, "revision": revision.lower(), "tree": tree})
        verified_packages.append((path, name, url))
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

    # Only now, after commit-tree and receipt authentication, restore the exact
    # manifest-declared origin metadata Lake requires to reuse these checkouts.
    for path, name, url in verified_packages:
        restore_manifest_remote(path, name, url)

    print(
        "dependency-source-state verified "
        f"git_packages={checked} "
        f"lakefile_sha256={declaration_sha} "
        f"lake_manifest_sha256={manifest_sha} "
        "manifest_remotes=restored"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create or verify deterministic receipts for cached Lean dependency build artifacts.

The canonical SHA-256 receipt is authoritative for cache reuse. The XOR-fold
receipt is an order-independent regression signal only and is never treated as
cryptographic authority. Verification may additionally require an external
reviewed canonical digest, so the cache cannot authenticate itself merely by
shipping matching artifacts and receipt metadata together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "GEO-LEAN-DEPS-RECEIPT-1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"lake manifest not found/invalid: {path}")
    return sha256_file(path)


def collect_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"dependency root not found/invalid: {root}")

    records: list[dict[str, Any]] = []
    for package in sorted(root.iterdir(), key=lambda p: p.name):
        if package.is_symlink():
            raise SystemExit(f"symlinked dependency package is not allowed: {package}")
        if not package.is_dir():
            continue
        build = package / ".lake" / "build"
        if not build.exists():
            continue
        if build.is_symlink():
            raise SystemExit(f"symlinked dependency build tree is not allowed: {build}")
        for path in sorted(build.rglob("*"), key=lambda p: p.as_posix()):
            if path.is_symlink():
                raise SystemExit(f"symlink in dependency build tree is not allowed: {path}")
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            digest = sha256_file(path)
            records.append(
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "sha256": digest,
                }
            )

    if not records:
        raise SystemExit("no dependency build artifacts found")
    if not any(record["path"].endswith(".olean") for record in records):
        raise SystemExit("dependency receipt contains no .olean files")
    return records


def receipts(records: list[dict[str, Any]]) -> tuple[str, str]:
    canonical = hashlib.sha256()
    xor_fold = bytearray(32)

    for record in records:
        path_bytes = record["path"].encode("utf-8")
        size_bytes = str(record["size"]).encode("ascii")
        digest_bytes = bytes.fromhex(record["sha256"])

        canonical.update(path_bytes)
        canonical.update(b"\0")
        canonical.update(size_bytes)
        canonical.update(b"\0")
        canonical.update(record["sha256"].encode("ascii"))
        canonical.update(b"\n")

        # Path-bind the XOR contribution so moving identical bytes to a
        # different dependency path changes the regression receipt.
        contribution = hashlib.sha256(path_bytes + b"\0" + digest_bytes).digest()
        for index, value in enumerate(contribution):
            xor_fold[index] ^= value

    return canonical.hexdigest(), bytes(xor_fold).hex()


def expected_metadata(args: argparse.Namespace) -> dict[str, str]:
    return {
        "lean_version": args.lean_version,
        "lean_archive_sha256": args.lean_archive_sha256.lower(),
        "mathlib_commit": args.mathlib_commit.lower(),
        "lake_manifest_sha256": manifest_sha256(args.lake_manifest),
        "platform": args.platform,
    }


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    records = collect_records(args.root)
    canonical, xor_fold = receipts(records)
    return {
        "schema": SCHEMA,
        "metadata": expected_metadata(args),
        "artifact_count": len(records),
        "canonical_sha256": canonical,
        "xor_fold_sha256": xor_fold,
        "files": records,
    }


def require_external_anchor(args: argparse.Namespace, data: dict[str, Any]) -> None:
    if args.expected_canonical_sha256 is not None:
        expected = args.expected_canonical_sha256.lower()
        actual = str(data["canonical_sha256"]).lower()
        if actual != expected:
            raise SystemExit(
                f"dependency canonical SHA-256 mismatch against reviewed anchor: "
                f"{actual} != {expected}"
            )
    if args.expected_xor_fold_sha256 is not None:
        expected = args.expected_xor_fold_sha256.lower()
        actual = str(data["xor_fold_sha256"]).lower()
        if actual != expected:
            raise SystemExit(
                f"dependency XOR-fold mismatch against reviewed regression anchor: "
                f"{actual} != {expected}"
            )
    if args.expected_artifact_count is not None:
        actual = int(data["artifact_count"])
        if actual != args.expected_artifact_count:
            raise SystemExit(
                f"dependency artifact count mismatch against reviewed anchor: "
                f"{actual} != {args.expected_artifact_count}"
            )


def cmd_create(args: argparse.Namespace) -> None:
    data = snapshot(args)
    require_external_anchor(args, data)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "dependency-receipt "
        f"files={data['artifact_count']} "
        f"canonical_sha256={data['canonical_sha256']} "
        f"xor_fold_sha256={data['xor_fold_sha256']}"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    if not args.receipt.is_file() or args.receipt.is_symlink():
        raise SystemExit(f"dependency receipt not found/invalid: {args.receipt}")
    saved = json.loads(args.receipt.read_text(encoding="utf-8"))
    current = snapshot(args)

    if saved.get("schema") != SCHEMA:
        raise SystemExit(
            f"receipt schema mismatch: {saved.get('schema')!r} != {SCHEMA!r}"
        )
    if saved.get("metadata") != current["metadata"]:
        raise SystemExit(
            "dependency receipt metadata mismatch:\n"
            f"saved={json.dumps(saved.get('metadata'), sort_keys=True)}\n"
            f"current={json.dumps(current['metadata'], sort_keys=True)}"
        )

    for field in ("artifact_count", "canonical_sha256", "xor_fold_sha256"):
        if saved.get(field) != current[field]:
            raise SystemExit(
                f"dependency receipt {field} mismatch: "
                f"{saved.get(field)!r} != {current[field]!r}"
            )

    if saved.get("files") != current["files"]:
        saved_by_path = {
            record["path"]: record for record in saved.get("files", [])
            if isinstance(record, dict) and "path" in record
        }
        current_by_path = {record["path"]: record for record in current["files"]}
        changed = sorted(
            path
            for path in set(saved_by_path) | set(current_by_path)
            if saved_by_path.get(path) != current_by_path.get(path)
        )
        preview = ", ".join(changed[:10])
        raise SystemExit(
            f"dependency artifact manifest mismatch ({len(changed)} path(s)); "
            f"first differences: {preview}"
        )

    # This check is intentionally external to the cached receipt itself. A
    # self-consistent artifact+receipt pair is insufficient provenance.
    require_external_anchor(args, current)

    print(
        "dependency-receipt verified "
        f"files={current['artifact_count']} "
        f"canonical_sha256={current['canonical_sha256']} "
        f"xor_fold_sha256={current['xor_fold_sha256']}"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for command, handler in (("create", cmd_create), ("verify", cmd_verify)):
        q = sub.add_parser(command)
        q.add_argument("--root", type=Path, required=True)
        q.add_argument("--receipt", type=Path, required=True)
        q.add_argument("--lake-manifest", type=Path, required=True)
        q.add_argument("--lean-version", required=True)
        q.add_argument("--lean-archive-sha256", required=True)
        q.add_argument("--mathlib-commit", required=True)
        q.add_argument("--platform", default="linux-x86_64")
        q.add_argument("--expected-canonical-sha256")
        q.add_argument("--expected-xor-fold-sha256")
        q.add_argument("--expected-artifact-count", type=int)
        q.set_defaults(handler=handler)
    return p


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

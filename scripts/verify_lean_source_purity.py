#!/usr/bin/env python3
"""Enforce the declarative Lean subset used by the protected proof build.

The protected compiler accepts project modules only after this scanner proves
that the production source surface is closed and contains no project-defined
compile-time execution mechanism. Imported mathlib tactics remain part of the
pinned dependency trust base; project source may not add commands, elaborators,
initializers, unsafe declarations, foreign hooks, or filesystem/process IO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

SCHEMA = "GEO-LEAN-PRODUCTION-SOURCE-1"

EXPECTED_IMPORTS: dict[PurePosixPath, tuple[str, ...]] = {
    PurePosixPath("GeoReason.lean"): (
        "GeoReason.Trajectory",
        "GeoReason.Cosine",
        "GeoReason.Menger",
        "GeoReason.ArcLength",
    ),
    PurePosixPath("GeoReason/Trajectory.lean"): (
        "Lean.Elab.Tactic.Omega",
        "Mathlib.Analysis.InnerProductSpace.Basic",
        "Mathlib.Analysis.Normed.Operator.LinearIsometry",
        "Mathlib.Tactic.Abel",
        "Mathlib.Tactic.Linarith.Frontend",
    ),
    PurePosixPath("GeoReason/Cosine.lean"): (
        "Mathlib.Analysis.InnerProductSpace.Basic",
        "Mathlib.Analysis.InnerProductSpace.LinearMap",
        "GeoReason.Trajectory",
    ),
    PurePosixPath("GeoReason/Menger.lean"): (
        "Mathlib.Geometry.Euclidean.Circumcenter",
        "Mathlib.Tactic.Abel",
        "Mathlib.Tactic.FieldSimp",
    ),
    PurePosixPath("GeoReason/ArcLength.lean"): ("GeoReason.Trajectory",),
}

EXPECTED_ALL_LEAN_FILES = set(EXPECTED_IMPORTS) | {
    PurePosixPath("GeoReason/Audit.lean")
}

# Tokens capable of adding project-controlled elaboration, evaluator, native,
# initializer, foreign, or IO surfaces. Import lines are checked separately and
# removed before this token scan, so trusted pinned imports cannot mask a use in
# the project body.
FORBIDDEN_IDENTIFIERS = {
    "unsafe",
    "partial",
    "opaque",
    "axiom",
    "constant",
    "initialize",
    "builtin_initialize",
    "elab",
    "elab_rules",
    "macro",
    "macro_rules",
    "syntax",
    "declare_syntax_cat",
    "run_cmd",
    "run_tac",
    "set_option",
    "attribute",
    "include_str",
    "include_bytes",
    "native_decide",
    "implemented_by",
    "extern",
    "export",
    "deriving",
    "IO",
    "BaseIO",
    "EIO",
    "CoreM",
    "MetaM",
    "CommandElabM",
    "TermElabM",
    "TacticM",
    "MonadLift",
    "liftIO",
    "unsafeCast",
    "sorryAx",
    "ofReduceBool",
    "Process",
    "FilePath",
    "System",
    "Lean",
    "Elab",
    "Command",
    "Parser",
    "quote",
    "eval",
    "compile",
    "register_option",
    "register_builtin_option",
    "register_trace_class",
}

IMPORT_RE = re.compile(r"(?m)^[ \t]*import[ \t]+([A-Za-z0-9_.]+)[ \t]*$")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
ATTRIBUTE_RE = re.compile(r"@\[([^\]]*)\]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_comments_and_strings(text: str, source: str) -> str:
    """Blank nested comments and strings while preserving offsets/newlines."""
    out = list(text)
    i = 0
    block_depth = 0
    line_comment = False
    in_string = False
    escaped = False

    while i < len(text):
        if line_comment:
            if text[i] == "\n":
                line_comment = False
            else:
                out[i] = " "
            i += 1
            continue

        if block_depth:
            if text.startswith("/-", i):
                out[i] = out[i + 1] = " "
                block_depth += 1
                i += 2
                continue
            if text.startswith("-/", i):
                out[i] = out[i + 1] = " "
                block_depth -= 1
                i += 2
                continue
            if text[i] != "\n":
                out[i] = " "
            i += 1
            continue

        if in_string:
            if text[i] == "\n":
                escaped = False
                i += 1
                continue
            out[i] = " "
            if escaped:
                escaped = False
            elif text[i] == "\\":
                escaped = True
            elif text[i] == '"':
                in_string = False
            i += 1
            continue

        if text.startswith("--", i):
            out[i] = out[i + 1] = " "
            line_comment = True
            i += 2
            continue
        if text.startswith("/-", i):
            out[i] = out[i + 1] = " "
            block_depth = 1
            i += 2
            continue
        if text[i] == '"':
            out[i] = " "
            in_string = True
            i += 1
            continue
        i += 1

    if block_depth:
        raise SystemExit(f"unterminated block comment in {source}")
    if in_string:
        raise SystemExit(f"unterminated string literal in {source}")
    return "".join(out)


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def verify_attributes(clean: str, rel: PurePosixPath) -> None:
    starts = [match.start() for match in re.finditer(r"@\[", clean)]
    matches = list(ATTRIBUTE_RE.finditer(clean))
    if len(starts) != len(matches):
        raise SystemExit(f"malformed or nested attribute block in {rel}")
    for match in matches:
        normalized = "".join(match.group(1).split())
        if normalized != "simp":
            line = line_for_offset(clean, match.start())
            raise SystemExit(
                f"non-whitelisted Lean attribute in {rel}:{line}: @[{match.group(1).strip()}]"
            )


def verify_source(path: Path, rel: PurePosixPath) -> dict[str, object]:
    try:
        st = path.lstat()
    except FileNotFoundError:
        raise SystemExit(f"required production Lean source is missing: {rel}")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"production Lean source must be a regular non-symlink: {rel}")

    text = path.read_text(encoding="utf-8")
    clean = strip_comments_and_strings(text, str(rel))
    imports = tuple(IMPORT_RE.findall(clean))
    expected = EXPECTED_IMPORTS[rel]
    if imports != expected:
        raise SystemExit(
            f"production import surface mismatch in {rel}: imports={imports!r} expected={expected!r}"
        )

    verify_attributes(clean, rel)
    body = IMPORT_RE.sub("", clean)

    hash_pos = body.find("#")
    if hash_pos >= 0:
        raise SystemExit(
            f"Lean hash command is forbidden in production source {rel}:{line_for_offset(body, hash_pos)}"
        )
    quote_pos = body.find("`")
    if quote_pos >= 0:
        raise SystemExit(
            f"Lean syntax quotation is forbidden in production source {rel}:{line_for_offset(body, quote_pos)}"
        )

    for match in IDENT_RE.finditer(body):
        token = match.group(0)
        if token in FORBIDDEN_IDENTIFIERS:
            raise SystemExit(
                f"compile-time/executable Lean token {token!r} is forbidden in "
                f"production source {rel}:{line_for_offset(body, match.start())}"
            )

    return {
        "path": str(rel),
        "sha256": sha256_file(path),
        "imports": list(imports),
    }


def verify_tree(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"Lean source root must be an ordinary directory: {root}")

    actual: set[PurePosixPath] = set()
    for path in root.rglob("*.lean"):
        rel = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise SystemExit(f"symlinked Lean source is forbidden: {rel}")
        actual.add(rel)
    if actual != EXPECTED_ALL_LEAN_FILES:
        raise SystemExit(
            "Lean source surface is not closed: "
            f"missing={sorted(map(str, EXPECTED_ALL_LEAN_FILES - actual))} "
            f"unexpected={sorted(map(str, actual - EXPECTED_ALL_LEAN_FILES))}"
        )

    files = [
        verify_source(root / Path(*rel.parts), rel)
        for rel in sorted(EXPECTED_IMPORTS, key=str)
    ]
    return {"schema": SCHEMA, "production_files": files}


def run_self_test() -> None:
    safe = '''
/- outer run_cmd /- nested #eval -/ end -/
-- initialize unsafe IO
private theorem sample : True := by
  have text := "run_cmd #eval initialize"
  trivial
'''
    clean = strip_comments_and_strings(safe, "self-test-safe")
    body = IMPORT_RE.sub("", clean)
    assert not any(token in FORBIDDEN_IDENTIFIERS for token in IDENT_RE.findall(body))
    assert "#" not in body

    for unsafe_source, expected in (
        ("run_cmd IO.println \"bad\"", "run_cmd"),
        ("@[command_elab demo] def x := 1", "attribute"),
        ("#eval 1", "hash"),
        ("unsafe def x := 1", "unsafe"),
    ):
        clean = strip_comments_and_strings(unsafe_source, "self-test-unsafe")
        if expected == "hash":
            assert "#" in clean
        elif expected == "attribute":
            assert any(
                "".join(match.group(1).split()) != "simp"
                for match in ATTRIBUTE_RE.finditer(clean)
            )
        else:
            assert expected in IDENT_RE.findall(clean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Lean"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    if args.write_receipt and args.receipt is None:
        raise SystemExit("--write-receipt requires --receipt")

    snapshot = verify_tree(args.root)
    if args.receipt is not None:
        if args.write_receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            if args.receipt.is_symlink() or not args.receipt.is_file():
                raise SystemExit(f"production source receipt missing/invalid: {args.receipt}")
            saved = json.loads(args.receipt.read_text(encoding="utf-8"))
            if saved != snapshot:
                raise SystemExit("production Lean source changed after purity verification")

    print(
        "lean-production-source verified "
        f"files={len(snapshot['production_files'])} "
        "compile_time_execution=forbidden import_surface=closed"
    )


if __name__ == "__main__":
    main()

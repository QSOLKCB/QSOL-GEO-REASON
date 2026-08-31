# Lean Dependency Cache Policy

This document defines the performance-only dependency reuse lane for the Phase 1 Lean formalization.

## Two evidence lanes

The `lean-phase1` workflow deliberately separates two claims.

### Verified-cache PR lane

Routine pull-request checks may reuse pinned dependency source state and dependency build artifacts only when all of the following are true:

- the exact repository `lean-toolchain` bytes match SHA-256
  `3aac669c7a910ec2389f4e4f921b605adf6ebf2d1e0c9b9cd0be4d33f3f5db71`,
  which is the declaration `leanprover/lean4:v4.33.1`;
- the exact Lean Linux archive SHA-256 is
  `890afd185370f85666025b883914ab4f4b339136f8c96167b69cfb62aecaf235`;
- the exact mathlib commit is
  `0df444a360eaa60ab8c11dca51a86af692955474`;
- the reviewed `lakefile.lean` bytes and frozen `lake-manifest.json` SHA-256 are
  part of the source-cache identity;
- every cached git dependency is at the manifest revision and its tracked
  worktree bytes/modes match that commit tree with Git replacement processing
  disabled;
- replacement refs/grafts and non-default index states such as
  assume-unchanged/skip-worktree are rejected;
- generated per-package `.lake` state is purged before source verification and
  is not admitted as source evidence;
- the restored dependency build tree reproduces the reviewed external canonical
  SHA-256
  `91f7181f1657481a8a00a3f4fe67b8d5663951838b5a0a76ef2adbd8b54e66d3`;
- the build tree contains exactly `37,312` recorded artifacts; and
- the path-bound XOR-fold regression receipt matches
  `5140247acee0acb36de98fa8192602e09815d27207c38d62a83b818861a0a5a3`.

The canonical build digest is stored in the reviewed workflow, not learned from
the restored cache. A cache cannot therefore authenticate arbitrary proof
objects merely by carrying a matching receipt alongside them.

On a verified cache hit, the dependency objects are reused but the complete
GeoReason project is rebuilt from the current PR source. Source hygiene,
recursive sorry inspection, and the positive compiled-axiom allowlist are run
again on the resulting theorem declarations.

If the build cache is absent, the PR lane can reconstruct dependencies from the
verified pinned source state, but a reusable receipt is accepted only if the
result reproduces the reviewed external canonical anchor.

### Cold trust lane

The `workflow_dispatch` `cold-trust` job never restores either source or build
caches. It verifies the repository Lean toolchain declaration, resolves the
pinned source graph, purges generated per-package Lake state, verifies tracked
source bytes/modes against the frozen commit trees with replacement processing
disabled, rebuilds the imported dependency closure from source, rebuilds
GeoReason, and reruns the same proof and axiom audits.

This lane is the release-grade authority for a formal evidence freeze.

## Source-state receipt semantics

`scripts/verify_lean_source_state.py` verifies source identity independently of
compiled artifacts. Its source receipt binds:

- the reviewed dependency declaration SHA-256;
- the frozen Lake manifest SHA-256; and
- each dependency name, exact revision, and verified commit-tree ID.

The source verifier does not use `git status` as integrity evidence. It hashes
tracked worktree bytes back to their Git blob object IDs, verifies executable
and symlink modes, rejects suspicious index flags, disables and rejects Git
replacement mechanisms, and rejects untracked Lean/config source that could
shadow an import.

Generated package `.lake` state is intentionally purged before this check so
compiled Lake configuration cannot travel inside the source cache and influence
later builds without being regenerated from verified source.

## Build receipt semantics

`scripts/lean_dependency_receipt.py` emits two receipts over the dependency
build tree.

The **canonical receipt** hashes the sorted path, size, and SHA-256 of every
cached dependency artifact. It is the authoritative integrity check for cache
reuse, and the fast lane additionally requires the externally reviewed seed
digest above.

The **XOR-fold receipt** XOR-reduces path-bound SHA-256 contributions into one
order-independent 256-bit value. It is compact deterministic regression
evidence only. It is not cryptographic authority and cannot replace the
canonical SHA-256 path.

This mirrors the NEXUS/VORTEX evidence discipline: compact XOR-style receipts
are useful for fast regression detection, while stronger canonical evidence
remains authoritative.

## First audited build-cache seed

The first repository-produced dependency build cache was seeded by
`lean-phase1` run `#61` from PR head
`d01f21da6bb12194b6c5e5a66ba3b623d5362b3c`.

The seed run rebuilt the dependency closure from pinned source and then passed
the source-hygiene, recursive-sorry, and positive compiled-axiom gates before
the dependency cache was saved.

Measured cold-build receipt:

- cold source build wall time: `2501.52 s`;
- Lean threads: `4` on `4` runner CPUs;
- dependency-cache artifact records: `37,312`;
- canonical dependency SHA-256:
  `91f7181f1657481a8a00a3f4fe67b8d5663951838b5a0a76ef2adbd8b54e66d3`;
- path-bound XOR-fold regression receipt:
  `5140247acee0acb36de98fa8192602e09815d27207c38d62a83b818861a0a5a3`;
- generated `lake-manifest.json` SHA-256:
  `646d5b171d7b7200f4f85d887ff655c45ee7796019ada4aab7e4ca759f41602b`.

These values are now reviewed trust anchors for routine build-cache reuse rather
than merely descriptive documentation.

## Scientific and trust boundary

A green verified-cache PR run means:

> the current GeoReason source rebuilt and passed all formal audits while using
> dependency source/build state whose identities matched reviewed frozen source
> declarations and the externally anchored audited build-cache digest.

It does **not** mean:

> the dependency graph was reconstructed from source on this exact run.

Only a green `cold-trust` run licenses that second statement.

The cache optimization changes build latency and artifact transport. It does not
change any `GEO-LEAN-TGT-*` theorem, allowed axiom, mathematical definition, or
the formalization boundary documented in `LEAN-FORMALIZATION.md`.

# Lean Dependency Cache Policy

This document defines the performance-only dependency reuse lane for the Phase 1 Lean formalization.

## Two evidence lanes

The `lean-phase1` workflow deliberately separates two claims.

### Verified-cache PR lane

Routine pull-request checks may reuse dependency build artifacts only when all of the following are true:

- the exact Lean version is `4.33.1`;
- the exact Lean Linux archive SHA-256 is
  `890afd185370f85666025b883914ab4f4b339136f8c96167b69cfb62aecaf235`;
- the exact mathlib commit is
  `0df444a360eaa60ab8c11dca51a86af692955474`;
- the current `lake-manifest.json` SHA-256 matches the cache identity;
- the cached dependency tree matches its canonical per-file SHA-256 manifest;
- the compact XOR-fold receipt also matches as a regression signal.

On a verified cache hit, the dependency objects are reused but the complete
GeoReason project is rebuilt from the current PR source. Source hygiene,
recursive sorry inspection, and the positive compiled-axiom allowlist are run
again on the resulting theorem declarations.

If the exact cache is absent, the PR lane reconstructs dependencies from pinned
source, runs the full proof/audit gates, creates the receipts, and only then
saves that dependency state for later PR runs.

### Cold trust lane

The `workflow_dispatch` `cold-trust` job never restores the dependency cache.
It resolves the pinned source graph, deletes every dependency build tree,
rejects surviving `.olean` / `.ilean` files, rebuilds the imported dependency
closure from source, rebuilds GeoReason, and reruns the same proof and axiom
audits.

This lane is the release-grade authority for a formal evidence freeze.

## Receipt semantics

`scripts/lean_dependency_receipt.py` emits two receipts over the dependency
build tree.

The **canonical receipt** hashes the sorted path, size, and SHA-256 of every
cached dependency artifact. It is the authoritative integrity check for cache
reuse.

The **XOR-fold receipt** XOR-reduces path-bound SHA-256 contributions into one
order-independent 256-bit value. It is compact deterministic regression
evidence only. It is not a cryptographic authority and cannot replace the
canonical SHA-256 manifest.

This mirrors the NEXUS/VORTEX evidence discipline: compact XOR-style receipts
are useful for fast regression detection, while stronger canonical evidence
remains authoritative.

## First audited cache seed

The first repository-produced dependency cache was seeded by `lean-phase1`
run `#61` from PR head
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

These values document the seed event. Routine reuse remains governed by the
live receipt verifier rather than by manually comparing this documentation.

## Scientific and trust boundary

A green verified-cache PR run means:

> the current GeoReason source rebuilt and passed all formal audits while using
> dependency artifacts whose exact identity matched a previously audited cold
> reconstruction under the same pinned proof environment.

It does **not** mean:

> the dependency graph was reconstructed from source on this exact run.

Only a green `cold-trust` run licenses that second statement.

The cache optimization changes build latency and artifact transport. It does not
change any `GEO-LEAN-TGT-*` theorem, allowed axiom, mathematical definition, or
the formalization boundary documented in `LEAN-FORMALIZATION.md`.

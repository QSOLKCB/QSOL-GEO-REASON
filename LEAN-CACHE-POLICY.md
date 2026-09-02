# Lean Dependency Cache and Trust Policy

This document defines the evidence meaning of the Phase 1 Lean workflows. The repository keeps a fast pull-request lane and a separate isolated cold-reconstruction authority. They are intentionally not interchangeable.

## Workflow authority map

### `lean-phase1`: verified-cache pull-request lane

`.github/workflows/lean-phase1.yml` runs only for pull requests. It is a performance and regression lane, not a release-grade trust authority.

It may reuse pinned dependency source state and dependency build artifacts only when all of the following hold:

- the repository `lean-toolchain` bytes match SHA-256 `3aac669c7a910ec2389f4e4f921b605adf6ebf2d1e0c9b9cd0be4d33f3f5db71`, corresponding to `leanprover/lean4:v4.33.1`;
- the Lean Linux archive matches SHA-256 `890afd185370f85666025b883914ab4f4b339136f8c96167b69cfb62aecaf235`;
- mathlib is pinned to commit `0df444a360eaa60ab8c11dca51a86af692955474`;
- the reviewed `lakefile.lean` and frozen `lake-manifest.json` identities are part of the source-cache key;
- every cached Git dependency is at the manifest revision and its tracked bytes and modes match that commit tree with replacement processing disabled;
- replacement refs, grafts, suspicious index flags, generated package Lake state, untracked entries, and source-shadowing paths are rejected;
- the restored dependency build tree reproduces canonical SHA-256 `91f7181f1657481a8a00a3f4fe67b8d5663951838b5a0a76ef2adbd8b54e66d3` over exactly `37,312` artifact records; and
- the path-bound XOR-fold regression receipt matches `5140247acee0acb36de98fa8192602e09815d27207c38d62a83b818861a0a5a3`.

The canonical digest is stored in reviewed workflow source rather than learned from the restored cache. A cache therefore cannot authenticate arbitrary proof objects merely by carrying a matching receipt beside them.

On a verified hit, dependencies may be reused, but the current GeoReason source is rebuilt and audited again. On a cache miss, this workflow may perform a reconstruction to seed performance artifacts, but that run still does not acquire release-grade authority.

The former manually dispatched `lean-phase1 / cold-trust` job has been removed. `lean-phase1` is no longer manually dispatchable, eliminating the duplicate unisolated authority path.

### `lean-isolated-audit / isolated-audit`: protected pull-request gate

The isolated pull-request job restores only externally anchored dependency state, freezes reviewed inputs, builds through the unprivileged `qsolbuild` identity, and then independently recompiles every reviewed GeoReason module from the frozen source using the hash-pinned Lean binary under `qsolcompile`.

Each project module receives a fresh writable compiler root. That root is transferred immediately to `root:root`, made read-only, and copied with SHA-256 equality checking into one root-owned assembled `GeoReason` package tree. The final theorem audit never loads the project `.lake/build` objects produced by `qsolbuild`.

Before project outputs are frozen, every process owned by `qsolbuild` is terminated and absence is verified. This closes inherited writable file descriptors before ownership and mode changes become the freeze boundary. The externally anchored dependency receipt is copied into the protected audit directory and the complete dependency artifact closure is recomputed after protected project compilation and before the final audit.

### `lean-isolated-audit / isolated-cold-trust`: sole release-grade authority

The manually dispatched `isolated-cold-trust` job is the only workflow authorized to support a release-grade cold-reconstruction statement.

It never restores the dependency source or build caches. Before the first project Lake evaluation it:

1. installs and hash-verifies the pinned Lean distribution;
2. copies the audit runner to a root-owned read-only location;
3. freezes `Lean/`, `scripts/`, `lakefile.lean`, and `lean-toolchain`;
4. makes the checkout root non-writable to `qsolbuild`; and
5. grants the resolver write access only to the pre-created Lake and manifest output surfaces.

Dependency resolution and compilation run under `qsolbuild`. After compilation, the job terminates every `qsolbuild` process before freezing `.lake/packages` and `.lake/build`, creates an externally anchored receipt, and installs that receipt root-owned and read-only. Protected GeoReason recompilation then runs under the separate `qsolcompile` identity. The dependency closure is recomputed against the protected receipt after that compilation and immediately before the theorem audit.

Only a green `lean-isolated-audit / isolated-cold-trust` run licenses the statement:

> the dependency graph was reconstructed from pinned source under the protected cold boundary on this exact run.

## Non-initializing theorem audit

`Lean/GeoReason/Audit.lean` is an executable audit runner. It does not import `GeoReason` in its source header. Its `main` function loads the protected module graph through `Lean.withImportModules` with the API default `loadExts := false`, so imported project `initialize` actions are not executed in the audit process.

The runner checks all twelve `GEO_LEAN_TGT_*` declarations directly in the imported environment:

- every named declaration must exist;
- every named declaration must be a theorem; and
- every transitive axiom dependency must belong to the positive allowlist `propext`, `Classical.choice`, and `Quot.sound`.

Because `sorryAx` is outside that allowlist, the same gate establishes sorry-freedom. Only after all twelve checks succeed does the runner emit the exact completion record:

```text
QSOL_PROTECTED_AUDIT_COMPLETE targets=12 theorem_kinds=verified axiom_allowlist=verified project_initializers=not_executed
```

Workflows require an exact full-line match for that record. Output printed by project code cannot substitute for it because project initializers are never run by the protected importer.

## Source-state receipt semantics

`scripts/verify_lean_source_state.py` verifies source identity independently of compiled artifacts. Its receipt binds:

- the reviewed dependency declaration SHA-256;
- the frozen Lake manifest SHA-256; and
- each dependency name, exact revision, and verified commit-tree identity.

The verifier does not use `git status` as integrity evidence. It hashes tracked worktree bytes back to their Git object identities, verifies executable and symlink modes, rejects suspicious index state, disables and rejects replacement mechanisms, sanitizes repository configuration before running Git, and rejects untracked or alternate worktree entries.

Generated package `.lake` state is purged before source verification, so compiled Lake configuration cannot travel inside the source cache and influence later builds without regeneration from verified source.

## Build receipt semantics

`scripts/lean_dependency_receipt.py` records the sorted path, size, and SHA-256 of every dependency build artifact and derives:

- a canonical aggregate SHA-256, which is authoritative for cache integrity; and
- a path-bound XOR fold, which is compact deterministic regression evidence only.

The XOR fold is never treated as cryptographic authority. The protected workflows retain the receipt outside all unprivileged writable surfaces and recompute the closure after compilation identities have been terminated.

## First repository-produced build-cache seed

The first dependency build cache was seeded by `lean-phase1` run `#61` from PR head `d01f21da6bb12194b6c5e5a66ba3b623d5362b3c`.

Recorded values:

- cold source build wall time: `2501.52 s`;
- Lean threads: `4` on `4` runner CPUs;
- dependency artifact records: `37,312`;
- canonical dependency SHA-256: `91f7181f1657481a8a00a3f4fe67b8d5663951838b5a0a76ef2adbd8b54e66d3`;
- path-bound XOR-fold regression value: `5140247acee0acb36de98fa8192602e09815d27207c38d62a83b818861a0a5a3`; and
- generated `lake-manifest.json` SHA-256: `646d5b171d7b7200f4f85d887ff655c45ee7796019ada4aab7e4ca759f41602b`.

These values are reviewed anchors for routine reuse. They do not turn the fast lane into a claim that dependencies were rebuilt on each run.

## Scientific boundary

The cache and isolation machinery changes build latency, artifact transport, and trust evidence. It does not change any `GEO-LEAN-TGT-*` theorem, mathematical definition, permitted axiom, numerical fixture, or evidence class.

A green formal workflow proves only the exact Lean statements under its documented trust boundary. It does not by itself prove CPython or IEEE-754 execution, JSON canonicalization, SHA-256 implementations, GitHub infrastructure, serving equivalence, hidden-state extraction, or semantic and mechanistic claims about reasoning.

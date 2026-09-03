# Lean 4 formalization of the frozen Phase 1 mathematics

This document records the Phase 1 exact-math evidence layer built against the immutable numerical release. It is historical/provenance documentation for the frozen formal layer and must not be read as claiming that Lean proves CPython, IEEE-754, GitHub Actions, or LLM hidden-state behavior.

## Frozen target identity

The first Lean formalization targets immutable numerical release `v0.1.0` at commit:

`1b5ab8b4543b20cdb6d439f7ad215c08e698188f`

The bound numerical implementation / mathematical kernel remains:

`5f45b5e69bcab890a757fffa491cf787f92a5bea`

The separate Lean evidence layer was merged in PR #3 and published as immutable release `v0.2.0` at merge commit:

`ec3312dcc102d859819c764a881e2d020662e880`

The formalization is additive evidence. It does not rewrite either frozen identity.

## Pinned proof environment

The reviewed proof environment is bound to:

- Lean `4.33.1`;
- Lean Linux archive SHA-256 `890afd185370f85666025b883914ab4f4b339136f8c96167b69cfb62aecaf235`;
- mathlib commit `0df444a360eaa60ab8c11dca51a86af692955474`;
- generated `lake-manifest.json` SHA-256 `646d5b171d7b7200f4f85d887ff655c45ee7796019ada4aab7e4ca759f41602b`;
- `lean-toolchain` SHA-256 `3aac669c7a910ec2389f4e4f921b605adf6ebf2d1e0c9b9cd0be4d33f3f5db71`;
- canonical dependency artifact SHA-256 `91f7181f1657481a8a00a3f4fe67b8d5663951838b5a0a76ef2adbd8b54e66d3`;
- path-bound XOR-fold regression value `5140247acee0acb36de98fa8192602e09815d27207c38d62a83b818861a0a5a3`; and
- dependency artifact count `37,312`.

The canonical artifact digest is the authoritative external build-object anchor. The XOR fold is compact regression evidence only.

## Formal theorem surface

The proof layer implements the twelve frozen targets `GEO-LEAN-TGT-001` through `GEO-LEAN-TGT-012` under namespace `GeoReason`.

The theorem surface covers the exact-real Phase 1 objects described in `MATH-SPEC.md`, including trajectory finite differences/path length, project cosine, Menger curvature, and normalized arc length.

The Menger implementation uses the frozen exact quantity `4A/(abc)` and proves the reciprocal-circumradius bridge on the nondegenerate branch. The proof does not substitute a different definition into the immutable numerical record.

## Nonempty trajectory representation

The Lean trajectory object carries a head plus tail:

```lean
structure Trajectory (V : Type*) where
  head : V
  tail : List V
```

An empty source trajectory is therefore unrepresentable. Internal higher-order finite-difference lists may still exhaust where mathematically appropriate.

## Positive theorem audit

`Lean/GeoReason/Audit.lean` is an executable audit runner. It loads the protected `GeoReason` module graph through `Lean.withImportModules` with extension loading disabled, rather than using a source-level `import GeoReason` that could run project initializers.

For all twelve target names the audit requires:

- the declaration exists;
- it is a theorem declaration; and
- its transitive axiom dependencies are confined to:
  - `propext`;
  - `Classical.choice`; and
  - `Quot.sound`.

Only after all checks succeed does it emit the exact protected completion marker used by CI.

## Declarative production-source boundary

`scripts/verify_lean_source_purity.py` closes the reviewed project-controlled Lean source surface before protected recompilation.

The accepted source requires:

- exactly the reviewed production module set;
- an exact import list for every production module;
- ordinary non-symlink source files;
- only the explicitly permitted `@[simp]` attribute; and
- no project-defined compile-time or foreign execution surface, including `run_cmd`, `run_tac`, initializers, custom syntax or elaborators, unsafe or partial declarations, native evaluation, foreign hooks, or IO/process/filesystem APIs.

The scanner removes nested comments and string literals before token inspection and carries self-tests for accepted and rejected examples. Before protected recompilation it writes a root-owned source receipt containing every production source digest and import list. That receipt is checked before and after every module compilation. The `qsolcompile` identity is terminated before an emitted object is inspected or frozen.

This boundary permits tactics supplied by the exact pinned dependency graph while preventing reviewed project source from spawning a competing writer for the object being authenticated.

## Workflow roles

### Verified-cache regression and cache-maintenance lane

`.github/workflows/lean-phase1.yml` is a **non-authoritative verified-cache lane** with three entry modes:

- pull-request runs validate the current candidate;
- relevant pushes to `main` may populate the default-branch cache scope used by later PRs; and
- explicit `workflow_dispatch` runs may refresh evicted or expired authenticated caches.

Every mode verifies the declarative project source surface, pinned dependency source and artifact identities, rebuilds the current project, reruns source checks, and executes the theorem audit. A cache miss may reconstruct dependency objects and publish them under the reviewed cache identity after the canonical dependency receipt matches the frozen external anchors.

Neither a `main` cache-maintenance run nor a manual `lean-phase1` dispatch is release-grade cold-reconstruction authority. These executions exist to maintain verified performance artifacts and eliminate producer/consumer races for later PRs.

### Isolated pull-request gate

`.github/workflows/lean-isolated-audit.yml` runs an isolated PR job that:

1. verifies the closed declarative production-source surface;
2. installs the hash-pinned Lean distribution;
3. restores and verifies declaration-bound dependency source state when available;
4. resolves and verifies the pinned dependency graph itself on a source-cache miss under a dedicated `qsolresolve` identity;
5. terminates every `qsolresolve` process before any resolved source or manifest byte is accepted;
6. freezes authenticated dependency source root-owned and read-only before dependency artifact transport begins;
7. restores only externally anchored dependency objects and verifies their receipt;
8. freezes reviewed theorem, audit, configuration, and verifier inputs;
9. builds through the unprivileged `qsolbuild` identity and terminates all descendants before output freeze;
10. recompiles each reviewed GeoReason module from the source receipt under the separate `qsolcompile` identity;
11. assembles the frozen project objects into one root-owned read-only package tree;
12. recomputes the dependency closure against a protected receipt after compilation; and
13. executes the non-initializing theorem audit under the read-only `qsolaudit` identity.

Dependency build-artifact cache availability is a performance prerequisite for this protected PR lane, not a source of trust. The corresponding verified-cache maintenance lane produces that cache under the same externally anchored identity. A restored object closure is accepted only after its canonical receipt is recomputed and matches the reviewed anchors.

The project `.lake/build` objects produced by `qsolbuild` are not eligible for the final protected import path.

### Sole release-grade cold authority

The manually dispatched `lean-isolated-audit / isolated-cold-trust` job is the **only** release-grade cold-reconstruction authority. It restores no dependency source or build cache and freezes reviewed inputs before the first project Lake evaluation.

A green `isolated-cold-trust` run is required for any claim that the pinned dependency graph was reconstructed from source under the protected cold boundary on that exact run. Neither `lean-phase1` cache maintenance nor the protected PR cache-consumer lane licenses that statement.

## Evidence boundary

The Lean layer proves the exact mathematical statements encoded in Lean under this documented trust boundary.

It does not by itself prove:

- CPython or IEEE-754 execution;
- JSON canonicalization;
- SHA-256 implementations;
- Git or GitHub infrastructure;
- local-model hidden-state extraction;
- serving equivalence;
- benchmark reasoning;
- carrier invariance;
- perturbation causality; or
- a mechanism of reasoning.

Those remain separate numerical, systems, or empirical evidence layers under `SCIENTIFIC-CONTRACT.md`.

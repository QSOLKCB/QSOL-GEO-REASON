# Phase 1 Lean 4 Formalization

This document records the formal proof layer for the exact mathematical contract frozen by the immutable QSOL-GEO-REASON Phase 1 release.

## Status

The numerical Phase 1 release is already immutable. Pull request #3 implements all twelve named Lean theorem targets as a new evidence layer against that frozen identity. The formalization remains a pull-request candidate until its exact head has no unresolved correctness findings, all required checks are green, and the PR is merged.

This status does not reopen or rewrite the release tag.

## Frozen source target

The formalization targets the repository state published as:

- release: `v0.1.0 — Phase 1 Mathematical Kernel`;
- release commit: `1b5ab8b4543b20cdb6d439f7ad215c08e698188f`;
- release immutability: enabled before this formalization began;
- numerical implementation kernel recorded by that release: `5f45b5e69bcab890a757fffa491cf787f92a5bea`; and
- frozen Phase 1 simulation result SHA-256: `c542bce987d31350b4904122e5ec02ef026715f51a1fe21ee184a452cc67a583`.

The Lean files were added after that immutable release. They do not rewrite the release and do not retroactively upgrade its `SIMULATION` evidence class.

## Pinned proof environment

The proof environment is intentionally stationary:

- Lean: `v4.33.1`;
- Lean Linux distribution: `lean-4.33.1-linux.tar.zst`;
- Lean distribution SHA-256: `890afd185370f85666025b883914ab4f4b339136f8c96167b69cfb62aecaf235`;
- mathlib release lineage: `v4.33.1`;
- exact mathlib commit: `0df444a360eaa60ab8c11dca51a86af692955474`;
- checkout action: `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`; and
- GitHub runner family: `ubuntu-24.04`, x86_64.

The trust workflow does not execute a moving `elan` installer. It downloads the exact Lean archive, checks the frozen digest before extraction, transfers the installed distribution to root ownership, removes write access, and invokes the verified `lean` and `lake` binaries directly.

`lakefile.lean` pins mathlib by exact commit. `lean-toolchain` remains the repository declaration of the intended compiler and is independently hash-checked by CI.

## Formal source model

`GEO-MATH-001` requires a finite nonempty trajectory. Lean encodes that requirement directly:

```lean
structure Trajectory (V : Type*) where
  head : V
  tail : List V
```

An empty source trajectory is therefore unrepresentable. Higher-order finite differences use ordinary lists internally because repeated differencing may legitimately exhaust all samples.

## Frozen theorem targets

Pull request #3 implements all twelve targets named by `MATH-SPEC.md` in immutable `v0.1.0`:

1. `GEO_LEAN_TGT_001` — order-`k` forward finite-difference length.
2. `GEO_LEAN_TGT_002` — positive-order finite differences cancel global translation.
3. `GEO_LEAN_TGT_003` — piecewise-linear path length is translation invariant.
4. `GEO_LEAN_TGT_004` — path length is invariant under Euclidean linear isometry plus translation.
5. `GEO_LEAN_TGT_005` — zero path length iff all points of a nonempty trajectory are equal.
6. `GEO_LEAN_TGT_006` — the project extended-cosine convention lies in `[-1, 1]`.
7. `GEO_LEAN_TGT_007` — extended cosine is invariant under Euclidean linear isometries.
8. `GEO_LEAN_TGT_008` — Menger curvature is translation invariant.
9. `GEO_LEAN_TGT_009` — Menger curvature is invariant under Euclidean linear isometry plus translation.
10. `GEO_LEAN_TGT_010` — nonzero uniform scaling obeys `κ(sP) = (1 / |s|) κ(P)`.
11. `GEO_LEAN_TGT_011` — a nondegenerate Euclidean-circle triple of radius `r > 0` has curvature `1 / r`.
12. `GEO_LEAN_TGT_012` — normalized piecewise-linear arc-length parameterization preserves the exact first and last endpoints at progress `0` and `1` for nonzero-length trajectories.

Implementation is complete on the PR branch. Acceptance remains conditional on final exact-head review and merge.

## Menger formalization

For a triple `p : Fin 3 → V`, the exact Lean definition follows the frozen Phase 1 convention:

- affinely dependent triples, including repeated and collinear triples, have curvature `0`; and
- an affinely independent triple has curvature equal to the reciprocal of its Euclidean circumradius.

For nondegenerate triples this is the quantity frozen in `MATH-SPEC.md`:

`κ = 4A / (abc) = 1 / R`.

The scaling theorem proves that the circumradius scales by `|s|` under a nonzero scalar map and then takes reciprocals.

## Workflow roles

### Routine verified-cache lane

`.github/workflows/lean-phase1.yml` is pull-request-only. It verifies pinned source and artifact identities, rebuilds the current project, and runs the theorem audit as a fast regression gate. It is not manually dispatchable and is not release-grade cold-reconstruction authority.

### Isolated pull-request gate

`.github/workflows/lean-isolated-audit.yml` runs an isolated PR job that:

1. installs the hash-pinned Lean distribution;
2. verifies declaration-bound dependency source state and externally anchored dependency objects;
3. freezes the reviewed theorem, audit, configuration, verifier, and dependency inputs;
4. builds through the unprivileged `qsolbuild` identity;
5. terminates all `qsolbuild` descendants before freezing project outputs;
6. recompiles each reviewed GeoReason module directly from frozen source under the separate `qsolcompile` identity;
7. assembles the frozen project objects into one root-owned read-only package tree;
8. recomputes the dependency closure against a protected receipt after compilation; and
9. executes the non-initializing theorem audit under the read-only `qsolaudit` identity.

The project `.lake/build` objects produced by `qsolbuild` are not eligible for the final protected import path.

### Sole release-grade cold authority

The manually dispatched `lean-isolated-audit / isolated-cold-trust` job is the only release-grade cold-reconstruction authority. It restores no dependency caches and freezes reviewed inputs before the first project Lake evaluation.

After dependency compilation, every `qsolbuild` process is terminated before dependency objects are transferred to root ownership and made read-only. The job records the complete dependency artifact closure in a protected receipt, performs the source-bound project recompilation, and verifies the closure again before the theorem audit. This order prevents a descendant process from retaining a writable file descriptor across the freeze boundary.

The removed legacy `lean-phase1 / cold-trust` job has no remaining evidentiary authority.

## Non-initializing protected audit

`Lean/GeoReason/Audit.lean` is an executable audit runner rather than a command file that imports `GeoReason` in its header.

Its `main` function uses `Lean.withImportModules` to load the protected `GeoReason` object graph with the API default `loadExts := false`. Imported project `initialize` actions are therefore not executed in the audit process. The importer uses trust level `0`, and the runner inspects the resulting environment directly.

For every `GEO_LEAN_TGT_001` through `GEO_LEAN_TGT_012`, the runner requires:

- the declaration exists;
- the declaration is a theorem; and
- `Lean.collectAxioms` returns no name outside:
  - `propext`;
  - `Classical.choice`; and
  - `Quot.sound`.

`Lean.sorryAx`, `Lean.ofReduceBool`, project-local constants, and every other unexpected axiom are rejected by the positive allowlist.

Only after all twelve declarations pass does the runner print:

```text
QSOL_PROTECTED_AUDIT_COMPLETE targets=12 theorem_kinds=verified axiom_allowlist=verified project_initializers=not_executed
```

CI requires an exact full-line match. A project initializer cannot forge the record because project initializers are never executed by the protected importer.

## Dependency reconstruction and cache policy

The exact source, cache, receipt, process-lifetime, and workflow-authority semantics are normative in `LEAN-CACHE-POLICY.md`.

In summary:

- the fast lane may reuse only externally anchored dependency artifacts;
- the isolated PR lane independently reconstructs the current project theorem graph from reviewed source;
- the isolated cold lane reconstructs dependencies without cache restore;
- writable build identities are terminated before object freeze; and
- protected dependency receipts are verified after project compilation and before audit.

The independent `phase1-reference` workflow remains a separate gate and must continue reproducing the immutable numerical Phase 1 evidence unchanged.

## Formalization boundary

This Lean development proves statements in exact mathematics over real normed and inner-product spaces. It does not, without a separate refinement proof, establish correctness of:

- CPython execution;
- IEEE-754 or binary64 behavior;
- `math.hypot`, `math.fsum`, or libm implementations;
- JSON canonicalization;
- SHA-256 implementations;
- Git, GitHub, hosted-runner, operating-system, or hardware provenance;
- the deterministic simulation noise generator;
- serving/runtime equivalence;
- hidden-state extraction from an LLM; or
- semantic, cognitive, or mechanistic claims about reasoning.

The intended evidence relation remains:

```text
immutable MATH-SPEC exact contract
        |
        +--> Lean 4 proof layer
        |
        +--> Python numerical/conformance layer
                  |
                  +--> frozen GEO-SIM-001 evidence
```

Agreement between formal and numerical layers is an explicit conformance question, not an identity assumed by documentation.

## Change policy

The formalization targets immutable `v0.1.0`. If a later proof exposes a defect in the frozen mathematical contract, `v0.1.0` remains unchanged. The correction receives a new repository and release identity, and affected theorem targets are updated or reproved against that new target.

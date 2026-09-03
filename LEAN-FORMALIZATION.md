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

For a triple `p : Fin 3 → V`, the Lean definition now begins with the exact quantity frozen by `GEO-MATH-006` rather than taking a downstream characterization as its definition.

Let

```text
u = p₁ - p₀
v = p₂ - p₀
w = p₂ - p₁
D = ‖u‖² ‖v‖² - inner(u,v)²
A = sqrt(D) / 2
abc = ‖u‖ ‖w‖ ‖v‖
```

Then:

- an affinely dependent triple, including repeated or collinear points, has curvature `0`; and
- an affinely independent triple has curvature `κ = 4A / abc`.

The formal bridge to the circumradius presentation is proved rather than assumed. For an affinely independent triple, the proof places the circumcenter displacement in the span of `u` and `v`, derives the Gram identity

```text
4 D R² = ‖u‖² ‖w‖² ‖v‖²,
```

uses positivity to obtain

```text
2 sqrt(D) R = ‖u‖ ‖w‖ ‖v‖,
```

and concludes

```text
4A / abc = 1 / R.
```

The isometry, translation, uniform-scaling, and circle theorems then use that proved bridge. The formal target therefore applies to the same area-and-side-length quantity named by the frozen exact contract and numerical kernel.

## Declarative production-source boundary

`scripts/verify_lean_source_purity.py` defines the project-controlled production source surface accepted by the protected compiler.

It requires:

- exactly the reviewed production module set;
- an exact import list for every production module;
- ordinary non-symlink source files;
- only the explicitly permitted `@[simp]` attribute; and
- no project-defined compile-time or foreign execution surface, including `run_cmd`, `run_tac`, initializers, custom syntax or elaborators, unsafe or partial declarations, native evaluation, foreign hooks, or IO/process/filesystem APIs.

The scanner removes nested comments and string literals before token inspection and carries self-tests for accepted and rejected examples. Before protected recompilation it writes a root-owned source receipt containing every production source digest and import list. That receipt is checked before and after every module compilation. The `qsolcompile` identity is also terminated before an emitted object is inspected or frozen.

This boundary deliberately permits tactics supplied by the exact pinned dependency graph while preventing reviewed project source from spawning a competing writer for the object being authenticated.

## Workflow roles

### Routine verified-cache lane

`.github/workflows/lean-phase1.yml` is pull-request-only. It verifies the declarative project source surface, pinned dependency source and artifact identities, rebuilds the current project, reruns the source check, and executes the theorem audit as a fast regression gate. It is not manually dispatchable and is not release-grade cold-reconstruction authority.

### Isolated pull-request gate

`.github/workflows/lean-isolated-audit.yml` runs an isolated PR job that:

1. verifies the closed declarative production-source surface;
2. installs the hash-pinned Lean distribution;
3. restores and verifies declaration-bound dependency source state when available;
4. resolves and verifies the pinned dependency graph itself on a source-cache miss under a dedicated `qsolresolve` identity;
5. terminates every `qsolresolve` process before any resolved source or manifest byte is accepted;
6. freezes authenticated dependency source root-owned and read-only before dependency artifact transport begins;
7. verifies externally anchored dependency objects;
8. freezes reviewed theorem, audit, configuration, and verifier inputs;
9. builds through the unprivileged `qsolbuild` identity and terminates all descendants before output freeze;
10. recompiles each reviewed GeoReason module from the source receipt under the separate `qsolcompile` identity;
11. assembles the frozen project objects into one root-owned read-only package tree;
12. recomputes the dependency closure against a protected receipt after compilation; and
13. executes the non-initializing theorem audit under the read-only `qsolaudit` identity.

The isolated job therefore does not depend on a competing workflow winning a source-cache race. The project `.lake/build` objects produced by `qsolbuild` are not eligible for the final protected import path.

### Sole release-grade cold authority

The manually dispatched `lean-isolated-audit / isolated-cold-trust` job is the only release-grade cold-reconstruction authority. It restores no dependency caches and freezes reviewed inputs before the first project Lake evaluation.

Dependency resolution runs under the dedicated `qsolresolve` identity. Every resolver process is terminated before the manifest or dependency source is verified or transferred to root ownership. Only after that source boundary is closed does the separate `qsolbuild` identity receive generated Lake/build output surfaces.

After dependency compilation, every `qsolbuild` process is terminated before dependency objects are transferred to root ownership and made read-only. The job records the complete dependency artifact closure in a protected receipt, performs the declarative source-bound project recompilation, and verifies the closure again before the theorem audit. This order prevents resolver, build, or compiler descendants from retaining a writable file descriptor across an accepted freeze boundary.

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

CI requires an exact full-line match. A project initializer cannot forge the record because project initializers are never executed by the protected importer and production project modules are rejected if they introduce initializer or compile-time execution surfaces.

## Dependency reconstruction and cache policy

The exact source, cache, receipt, process-lifetime, and workflow-authority semantics are normative in `LEAN-CACHE-POLICY.md`.

In summary:

- the fast lane may reuse only externally anchored dependency artifacts;
- the isolated PR lane can verify a source-cache hit or securely resolve pinned source on a miss;
- dependency source is frozen before artifact restoration begins;
- the isolated cold lane reconstructs dependencies without any cache restore;
- resolver, build, and compiler identities are terminated before their outputs are accepted or frozen; and
- protected source and dependency receipts are verified after project compilation and before audit.

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

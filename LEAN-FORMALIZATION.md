# Phase 1 Lean 4 Formalization

This document records the formal proof layer for the exact mathematical contract frozen by the immutable QSOL-GEO-REASON Phase 1 release.

## Frozen source target

The formalization targets the repository state published as:

- release: `v0.1.0 — Phase 1 Mathematical Kernel`
- release commit: `1b5ab8b4543b20cdb6d439f7ad215c08e698188f`
- release immutability: enabled before this formalization began
- numerical implementation kernel recorded by that release: `5f45b5e69bcab890a757fffa491cf787f92a5bea`
- frozen Phase 1 simulation result SHA-256: `c542bce987d31350b4904122e5ec02ef026715f51a1fe21ee184a452cc67a583`

The Lean files are a new formal evidence layer built after that immutable release. They do not rewrite the release and do not upgrade its `SIMULATION` evidence class retroactively.

## Pinned proof environment

The proof environment is intentionally stationary:

- Lean: `leanprover/lean4:v4.33.1`
- mathlib release lineage: `v4.33.1`
- exact pinned mathlib commit: `0df444a360eaa60ab8c11dca51a86af692955474`

`lakefile.lean` uses the exact mathlib commit rather than a moving tag. `lean-toolchain` pins the Lean toolchain.

## Formal source model

`GEO-MATH-001` requires a finite nonempty trajectory. The Lean representation encodes that requirement directly:

```lean
structure Trajectory (V : Type*) where
  head : V
  tail : List V
```

An empty source trajectory is therefore unrepresentable. Higher-order finite differences use ordinary lists internally because repeated differencing may legitimately exhaust all samples.

This distinction was added after Codex identified that a raw `List V` source type could incorrectly certify `[]` as valid zero-valued geometry.

## Frozen theorem targets

The Lean library implements all twelve theorem targets named by `MATH-SPEC.md` in `v0.1.0`:

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
11. `GEO_LEAN_TGT_011` — a nondegenerate triple on a Euclidean circle of radius `r > 0` has curvature `1 / r`.
12. `GEO_LEAN_TGT_012` — normalized piecewise-linear arc-length parameterization preserves the exact first and last endpoints at progress `0` and `1` for nonzero-length trajectories.

## Menger formalization

For a triple `p : Fin 3 → V`, the exact Lean definition follows the frozen Phase 1 convention:

- affinely dependent triples, including repeated and collinear triples, have curvature `0`;
- an affinely independent triple has curvature equal to the reciprocal of its Euclidean circumradius.

For nondegenerate triples this is the same mathematical quantity frozen in `MATH-SPEC.md`:

`κ = 4A / (abc) = 1 / R`.

The scaling theorem is derived by proving that the circumradius scales by `|s|` under a nonzero scalar map and then taking reciprocals.

## Verification gates

The `lean-phase1` workflow must pass all of the following before the formalization is accepted:

1. resolve the exact pinned dependencies;
2. compile the complete `GeoReason` library with `lake build --wfail`;
3. reject source-level `sorry`, `admit`, or local `axiom` declarations;
4. execute `Lean/GeoReason/Audit.lean` against the compiled environment;
5. require mathlib's recursive `#print sorries` audit to report the twelve targets sorry-free;
6. reject compiled dependencies on `sorryAx` or `Lean.ofReduceBool`.

The existing `phase1-reference` workflow remains a separate gate and must continue reproducing the immutable numerical Phase 1 evidence unchanged.

## Formalization boundary

This Lean development proves statements in exact mathematics over real normed/inner-product spaces. It does not, without a separate refinement proof, establish correctness of:

- CPython execution;
- IEEE-754 or binary64 behavior;
- `math.hypot`, `math.fsum`, or libm implementations;
- JSON canonicalization;
- SHA-256 implementations;
- Git or GitHub provenance machinery;
- the deterministic simulation noise generator;
- serving/runtime equivalence;
- hidden-state extraction from an LLM;
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

The formalization targets immutable `v0.1.0`. If a later proof attempt exposes a defect in the frozen mathematical contract, `v0.1.0` must remain unchanged. The correction must receive a new repository/release identity and the affected Lean targets must be updated or reproved against that new target.

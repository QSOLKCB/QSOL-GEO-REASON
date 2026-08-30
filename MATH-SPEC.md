# QSOL-GEO-REASON Mathematical Specification

Status: **Phase 1 exact-mathematics contract**

This document freezes the mathematical semantics that the Phase 1 reference implementation approximates numerically. It is normative for the meaning of the Phase 1 geometry primitives, subject to the higher-level epistemic rules in `INVARIANTS.md` and `SCIENTIFIC-CONTRACT.md`.

The purpose of this separation is deliberate:

- this document specifies mathematics over exact real vector spaces;
- `src/qsol_geo_reason/` implements a finite-precision reference instrument;
- `GEO-SIM-001` tests implementation conformance on frozen synthetic cases;
- a future Lean 4 development should formalize this exact layer against an immutable release/tag;
- a Lean proof of this specification will **not** by itself prove the Python/IEEE-754 implementation correct.

## 1. Ambient space and trajectories

### GEO-MATH-001 — Finite Euclidean trajectory

Fix a dimension `d >= 1`. A trajectory is a non-empty finite ordered sequence

\[
Z = (z_0, z_1, \ldots, z_{n-1}), \qquad z_i \in \mathbb{R}^d,\ n \ge 1.
\]

All points in a trajectory have the same dimension.

The order of the sequence is part of the object. A trajectory is not identified with its set of points.

## 2. Finite differences

### GEO-MATH-002 — Forward finite difference

Define the order-zero finite difference by

\[
\Delta^0 Z = Z.
\]

For `k >= 0`, if

\[
\Delta^k Z = (x_0,\ldots,x_{m-1})
\]

with `m >= 2`, then

\[
\Delta^{k+1} Z
=
(x_1-x_0,\ x_2-x_1,\ldots,x_{m-1}-x_{m-2}).
\]

Therefore `Delta^k Z` contains `n-k` samples when `0 <= k < n`, and is empty for `k >= n`.

Terms such as *velocity* and *acceleration* are shorthand for finite-difference quantities only.

## 3. Path length

### GEO-MATH-003 — Piecewise-linear path length

For

\[
Z=(z_0,\ldots,z_{n-1}),
\]

define

\[
L(Z)=\sum_{i=0}^{n-2}\|z_{i+1}-z_i\|_2.
\]

For a one-point trajectory, `L(Z)=0`.

Consequences to preserve:

- `L(Z) >= 0`;
- `L(Z)=0` iff every consecutive displacement is zero;
- translations preserve path length;
- Euclidean isometries preserve path length;
- uniform scaling by scalar `s` multiplies path length by `|s|`.

## 4. Cosine alignment

### GEO-MATH-004 — Extended cosine convention

For nonzero vectors `u,v in R^d`,

\[
\operatorname{cos}(u,v)
=
\frac{\langle u,v\rangle}{\|u\|_2\|v\|_2}.
\]

QSOL-GEO-REASON extends the ordinary cosine with explicit exact-zero conventions:

\[
C(0,0)=1,
\qquad
C(0,v)=C(u,0)=0
\]

for nonzero `u` and `v`.

These conventions apply only to **exact** zero vectors. No absolute magnitude threshold may reclassify a nonzero vector as zero.

For two aligned trajectories whose order-`k` finite-difference sequences contain `m >= 1` samples, mean cosine alignment is

\[
A_k(X,Y)
=
\frac{1}{m}
\sum_{i=0}^{m-1}
C((\Delta^k X)_i,(\Delta^k Y)_i).
\]

If no finite-difference samples remain, the mean is **undefined**. It must not be encoded as `0`.

## 5. Alignment

### GEO-MATH-005 — Pair alignment policies

Three alignment policies are defined.

**`error`** requires equal sample counts and otherwise rejects the comparison.

**`truncate`** retains the common prefix of length `min(n_X,n_Y)`.

**`arclength`** treats each trajectory as a piecewise-linear curve and samples it at equally spaced fractions of its total arc length.

For nonzero-length paths, arc-length resampling must preserve the exact first and last geometric endpoints.

For exactly zero-length paths, every resampled point is the common geometric location.

Small but nonzero segments are not mathematically discardable merely because other segments are much larger.

## 6. Menger curvature

### GEO-MATH-006 — Consecutive-triple Menger curvature

For three consecutive points `p0,p1,p2`, let their triangle have side lengths `a,b,c` and area `A`.

For three distinct non-collinear points,

\[
\kappa(p_0,p_1,p_2)
=
\frac{4A}{abc}
=
\frac{1}{R},
\]

where `R` is the circumradius.

For repeated-point or collinear triples, QSOL-GEO-REASON defines

\[
\kappa = 0.
\]

Equivalently, for consecutive displacements

\[
u=p_1-p_0,\qquad v=p_2-p_1
\]

and chord `c = ||p2-p0||`, when the triple is nondegenerate,

\[
\kappa
=
\frac{2\sin\theta}{c},
\]

where `theta` is the angle between `u` and `v`.

No absolute curvature threshold may replace a small nonzero curvature by zero.

## 7. Geometric transformation laws

### GEO-MATH-007 — Translation laws

For a fixed `t in R^d`, define

\[
(T_t Z)_i=z_i+t.
\]

Then:

- `Delta^0(T_t Z)` is generally not equal to `Delta^0 Z`;
- for every `k >= 1`, `Delta^k(T_t Z)=Delta^k Z`;
- `L(T_t Z)=L(Z)`;
- corresponding Menger curvatures are unchanged;
- any defined finite-difference cosine comparison at order `k >= 1` is unchanged when the same translation is applied.

This is the exact mathematical basis of the Phase 1 carrier/control translation analogues. It does not imply semantic equivalence in an LLM.

### GEO-MATH-008 — Euclidean isometry laws

Let

\[
F(x)=Qx+t
\]

where `Q` is orthogonal and `t` is a translation. Signed coordinate permutations used in Phase 1 are a restricted exact instance.

Then:

- displacement vectors transform as `Q Delta z`;
- path length is preserved;
- Menger curvature is preserved;
- cosine between corresponding nonzero displacement vectors is preserved;
- exact zero displacement remains exact zero.

### GEO-MATH-009 — Uniform scaling laws

For nonzero scalar `s`:

- `Delta^k(sZ)=s Delta^k Z`;
- `L(sZ)=|s|L(Z)`;
- corresponding cosine alignment is unchanged when both operands receive the same scaling;
- Menger curvature scales as `kappa(sZ)=kappa(Z)/|s|`.

This scale law is one reason the numerical implementation must not use fixed absolute thresholds to classify nonzero geometry as zero.

## 8. Domain and undefined cases

### GEO-MATH-010 — Undefined is not zero

An unavailable mathematical quantity must not be silently encoded as a valid zero.

In particular:

- mean cosine alignment is undefined when the selected finite-difference order leaves no samples;
- a dimension mismatch is invalid input;
- a requested exact-length alignment with different lengths is invalid input;
- non-finite implementation values are outside the Phase 1 binary64 conformance domain.

The implementation must reject these cases or represent them explicitly as unavailable.

## 9. Numerical representation boundary

### GEO-MATH-011 — Serialization normalization is not mathematics

The exact definitions above do not contain decimal rounding, binary64, JSON, SHA-256, or a numerical epsilon.

The Python reference implementation uses IEEE-754 binary64 arithmetic and normalizes result floats to a fixed number of **significant decimal digits** before canonical serialization. This is a reproducibility convention only.

Normalization must be scale-aware in the following minimal sense:

- it must not impose a fixed absolute floor that turns every sufficiently small nonzero value into zero;
- exact negative zero may be canonicalized to positive zero;
- non-finite outputs are rejected.

Hash identity is evidence provenance, not a mathematical invariant.

## 10. Phase 1 theorem targets

The following stable theorem targets are intended for a post-release Lean 4 formalization. Names are descriptive placeholders; the mathematical statements are the contract.

### GEO-LEAN-TGT-001 — Finite-difference length

For `0 <= k < n`, `Delta^k Z` has length `n-k`.

### GEO-LEAN-TGT-002 — Translation cancellation

For every `k >= 1`,

\[
\Delta^k(T_t Z)=\Delta^k Z.
\]

### GEO-LEAN-TGT-003 — Path-length translation invariance

\[
L(T_t Z)=L(Z).
\]

### GEO-LEAN-TGT-004 — Path-length isometry invariance

For orthogonal `Q`,

\[
L(QZ+t)=L(Z).
\]

### GEO-LEAN-TGT-005 — Zero-length characterization

\[
L(Z)=0
\]

iff all points of `Z` are equal.

### GEO-LEAN-TGT-006 — Cosine bounds

Whenever the extended cosine is defined by `GEO-MATH-004`,

\[
-1 \le C(u,v) \le 1.
\]

### GEO-LEAN-TGT-007 — Cosine isometry invariance

For orthogonal `Q`,

\[
C(Qu,Qv)=C(u,v).
\]

### GEO-LEAN-TGT-008 — Menger translation invariance

\[
\kappa(p_0+t,p_1+t,p_2+t)
=
\kappa(p_0,p_1,p_2).
\]

### GEO-LEAN-TGT-009 — Menger isometry invariance

For orthogonal `Q`,

\[
\kappa(Qp_0+t,Qp_1+t,Qp_2+t)
=
\kappa(p_0,p_1,p_2).
\]

### GEO-LEAN-TGT-010 — Menger scaling law

For `s != 0`,

\[
\kappa(sp_0,sp_1,sp_2)
=
\frac{1}{|s|}
\kappa(p_0,p_1,p_2).
\]

### GEO-LEAN-TGT-011 — Circle curvature

Three distinct non-collinear points on a Euclidean circle of radius `r > 0` have Menger curvature `1/r`.

### GEO-LEAN-TGT-012 — Arc-length endpoint preservation

For a nonzero-length piecewise-linear path, normalized arc-length parameterization evaluates to the original first point at progress `0` and the original last point at progress `1`.

## 11. Formalization boundary

The planned Lean development should formalize the exact layer using real Euclidean vector spaces, finite sequences, and explicit domain hypotheses.

It must not claim, without a separate verified refinement argument, to prove:

- CPython floating-point execution;
- `math.hypot`, `math.fsum`, or libm behavior;
- JSON canonicalization;
- SHA-256 implementation correctness;
- Git provenance checks;
- the deterministic noise generator;
- semantic or cognitive claims about LLM reasoning.

The intended evidence chain is therefore:

```text
MATH-SPEC exact definitions
        |
        +--> Lean 4 proofs of exact theorems
        |
        +--> Python conformance implementation
                  |
                  +--> GEO-SIM-001 frozen numerical fixtures
```

Agreement between those layers must be tested and documented rather than assumed.

## 12. Freeze policy

Changes to a `GEO-MATH-*` definition after the first immutable Phase 1 release are mathematical-contract changes.

Such a change should:

1. name every affected `GEO-MATH-*` and `GEO-LEAN-TGT-*` identifier;
2. state whether it is a clarification or a semantic change;
3. regenerate affected Python fixtures;
4. create a new release identity rather than rewriting frozen evidence;
5. update or reprove affected Lean theorems against the new release.

The first Lean formalization should target the exact immutable commit/tag produced after Phase 1 receives a green exact-head review and is merged.

# GEO-SIM-001 — Deterministic Geometry Reference Simulation

Status: **Phase 1 conformance protocol**

Evidence class: **`SIMULATION`**

This protocol validates QSOL-GEO-REASON's geometry instrumentation against synthetic trajectories whose relevant properties are known by construction. It does not test a language model and cannot support a claim about model reasoning.

The exact mathematical semantics are frozen in [`MATH-SPEC.md`](../MATH-SPEC.md). This protocol tests a finite-precision reference implementation of that specification.

## Purpose

The protocol asks whether the implementation can generate deterministic reference trajectories, recover expected finite-difference/path/alignment/curvature properties, distinguish a geometry-preserving control from a deliberately geometry-changing suffix perturbation, and regenerate the complete result byte-for-byte for a fixed recipe and implementation revision.

## Frozen recipe

`recipes/reference-suite.json`, recipe ID `GEO-SIM-REF-001`, contains straight, circular, branching, deterministic-noise, null, carrier-translation, control-translation, and suffix-shift trajectories. Synthetic carrier labels are analogues only and do not encode natural-language semantics.

## Measurement definitions

For `z_0,...,z_T`, order-1 finite difference is `Delta z_t = z_(t+1)-z_t`; higher orders repeat that operator. Path length is `sum_t ||z_(t+1)-z_t||_2`.

Cosine alignment uses an explicit alignment policy (`error`, `truncate`, or `arclength`), applies the selected finite-difference order, then reports mean pointwise cosine similarity. Two **exact** zero displacement vectors have alignment 1 by convention; exact zero versus non-zero has alignment 0. Nonzero vectors are not reclassified as zero by an absolute threshold.

If the selected finite-difference order leaves no samples, cosine alignment is undefined and the comparison is rejected. It is never emitted as a valid-looking score of `0.0`.

For three consecutive distinct points with side lengths `a,b,c` and Euclidean triangle area `A`, Menger curvature is `kappa = 4A/(abc)`. The reference implementation computes the equivalent squared expression over exact dyadic-rational representations of the stored binary64 displacement vectors and converts only the final positive square root back to binary64. Repeated-point or exactly collinear represented triples have curvature 0 by convention. A non-collinear represented triple is never turned into zero merely because unit-vector subtraction rounded away its angle; an unrepresentably small nonzero binary64 result is rejected instead.

## Alignment

`error` requires equal lengths. `truncate` retains the common prefix.

For a pair with sample counts `n_left` and `n_right`, `arclength` chooses the common resampling count

`m = max(n_left, n_right)`.

Each trajectory is resampled to exactly `m` samples. For `m >= 2`, the samples occur at normalized arc-length fractions `j/(m-1)` for `j = 0,...,m-1`; for `m = 1`, the sole point is retained.

For nonzero paths, first and final endpoints are preserved exactly. Segment-length accumulation must not erase a short segment merely because it follows a much larger segment.

No interpolation rule is treated as cognitively privileged.

## Deterministic noise

The noisy generator uses a local integer linear-congruential construction keyed by recipe seed, point index, and axis. It does not consume process-global random state and is not intended as a statistical model of neural noise.

## Recipe identity and validation

Runtime recipe validation follows the published top-level trajectory/comparison object shapes, rejects unknown trajectory or comparison fields, requires unique trajectory and comparison IDs, and requires derived trajectories to reference a strictly earlier trajectory. A branch recipe must leave at least one generated step after its branch point.

## Canonical emitted trajectory representation

The evidence record must be internally self-consistent: serialized points and serialized metrics describe the same numerical trajectory.

To make significant-digit normalization stable under large absolute coordinate offsets, each generated trajectory is canonicalized relative to its first raw point:

1. normalize the first point to the frozen significant-digit precision;
2. compute each raw point's displacement from that raw first point;
3. normalize each displacement independently to the same significant-digit precision;
4. reconstruct the emitted absolute coordinates as canonical origin plus canonical displacement;
5. compute trajectory hashes, finite differences, path length, Menger curvature, and all cross-trajectory comparisons from those emitted coordinates.

This rule preserves a representable local displacement such as `2` on top of an absolute coordinate near `1e16`; the result may not serialize a null coordinate path while retaining non-null derived geometry.

## Reference expectations

The frozen suite requires: straight curvature 0; radius-2 circle curvature 0.5 at every interior point within absolute tolerance `1e-12`; null path length 0; carrier translation changes positions while preserving order-1 differences and curvature; carrier/control order-1 cosine alignment 1; suffix-shift alignment strictly below 1; and byte-identical replay for fixed recipe plus implementation revision.

Unit tests additionally cover repeated points, short sequences, exact zero-length paths, tiny nonzero paths and vectors, very large vectors, near-collinear nonzero curvature, late small arc-length segments, large absolute coordinate offsets with small local displacements, dimension mismatch, arbitrary finite-difference order, undefined empty comparisons, all Phase-1 alignment policies, recipe identity errors, and frozen-metadata identity enforcement.

## Floating-output identity

Machine-readable finite values are canonicalized to **14 significant decimal digits** for the Phase 1 release candidate. Significant-digit normalization is scale-aware and does not impose an absolute floor that collapses sufficiently small nonzero values to zero.

Coordinates use the first-point-plus-displacement canonicalization above before being emitted. Metrics are then derived from those exact emitted coordinates and normalized for serialization.

Fourteen significant digits are intentionally coarser than binary64's full printed precision so insignificant libm/runtime tail differences do not become evidence identity across the supported Python/Linux CI matrix while retaining the preregistered radius-2 curvature tolerance.

This normalization is a serialization/reproducibility convention, not part of the exact mathematics. Exact negative zero may canonicalize to positive zero. Non-finite outputs are rejected.

## Hash and provenance binding

Every result records `protocol_id`, `recipe_sha256`, `implementation_revision`, per-trajectory `trajectory_sha256`, and top-level `artifact_sha256`. Each trajectory hash covers the exact emitted point array from which its metrics are derived. The artifact hash covers the canonical payload before the hash field itself is added. Frozen metadata records protocol, recipe, implementation, recipe hash, artifact hash, evidence class, replication status, and schema version.

`tools/verify_reference.py` verifies every one of those frozen metadata identities against the recipe, regenerated record, and frozen result before accepting byte identity.

When a CLI or reference-generation command derives or verifies a Git implementation revision from the source checkout, tracked changes and source-relevant untracked files make the checkout dirty and revision binding is rejected. Untracked interpreter/build artifacts such as `__pycache__`, bytecode, editable-install metadata, and ordinary build caches are not source identity.

## Falsifier

Phase 1 fails if the frozen implementation cannot recover the known synthetic properties within the stated tolerances, cannot preserve the specified numerical-domain distinctions, if serialized points and metrics disagree about the represented trajectory, if frozen provenance identities disagree, or if it cannot regenerate its frozen fixture byte-for-byte.

## Claim ceiling

Success permits only an instrumentation claim: under GEO-SIM-001, the implementation deterministically recovers the preregistered synthetic geometry. It does not permit claims about real LLM reasoning, hidden-state smoothness, logical equivalence, or training benefit.

## Formalization handoff

After this mathematical kernel receives a green exact-head review and is merged, the intended next step is to freeze an immutable release/tag. Lean 4 formalization should then target the `GEO-MATH-*` definitions and `GEO-LEAN-TGT-*` theorem targets in `MATH-SPEC.md` against that exact frozen release.

A Lean proof is a new formal evidence layer. It must not be described as retroactively proving the Python floating-point implementation unless a separate refinement argument is established.
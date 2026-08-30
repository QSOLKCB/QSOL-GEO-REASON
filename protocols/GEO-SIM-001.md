# GEO-SIM-001 — Deterministic Geometry Reference Simulation

Status: **Phase 1 conformance protocol**

Evidence class: **`SIMULATION`**

This protocol validates QSOL-GEO-REASON's geometry instrumentation against synthetic trajectories whose relevant properties are known by construction. It does not test a language model and cannot support a claim about model reasoning.

## Purpose

The protocol asks whether the implementation can generate deterministic reference trajectories, recover expected finite-difference/path/alignment/curvature properties, distinguish a geometry-preserving control from a deliberately geometry-changing suffix perturbation, and regenerate the complete result byte-for-byte for a fixed recipe and implementation revision.

## Frozen recipe

`recipes/reference-suite.json`, recipe ID `GEO-SIM-REF-001`, contains straight, circular, branching, deterministic-noise, null, carrier-translation, control-translation, and suffix-shift trajectories. Synthetic carrier labels are analogues only and do not encode natural-language semantics.

## Measurement definitions

For `z_0,...,z_T`, order-1 finite difference is `Delta z_t = z_(t+1)-z_t`; higher orders repeat that operator. Path length is `sum_t ||z_(t+1)-z_t||_2`.

Cosine alignment uses an explicit alignment policy (`error`, `truncate`, or `arclength`), applies the selected finite-difference order, then reports mean pointwise cosine similarity. Two **exactly zero** displacement vectors have alignment 1 by convention; exactly zero versus non-zero has alignment 0. Small nonzero displacements retain their ordinary cosine direction and are not coerced to the zero-vector convention.

For three consecutive distinct points with side lengths `a,b,c` and Euclidean triangle area `A`, Menger curvature is `kappa = 4A/(abc)`. Area is obtained from the Gram determinant. Exactly repeated-point or collinear triples have curvature 0 by convention; small nonzero geometry is not clipped solely because of scale.

## Alignment

`error` requires equal lengths. `truncate` retains the common prefix. `arclength` linearly resamples to equally spaced cumulative-arc-length targets; an **exactly zero-length** path repeats its single geometric location. Small nonzero paths preserve their endpoints and are resampled normally. No interpolation rule is treated as cognitively privileged.

## Deterministic noise

The noisy generator uses a local integer linear-congruential construction keyed by recipe seed, point index, and axis. It does not consume process-global random state and is not intended as a statistical model of neural noise.

## Recipe identity and validation

Runtime recipe validation follows the published top-level trajectory/comparison object shapes, rejects unknown trajectory or comparison fields, requires unique trajectory and comparison IDs, and requires derived trajectories to reference a strictly earlier trajectory. A branch recipe must leave at least one generated step after its branch point.

## Reference expectations

The frozen suite requires: straight curvature 0; radius-2 circle curvature 0.5 at every interior point within absolute tolerance `1e-12`; null path length 0; carrier translation changes positions while preserving order-1 differences and curvature; carrier/control order-1 cosine alignment 1; suffix-shift alignment strictly below 1; and byte-identical replay for fixed recipe plus implementation revision.

Unit tests additionally cover repeated points, short sequences, zero-length and tiny-nonzero paths, dimension mismatch, arbitrary finite-difference order, tiny nonzero vectors/curvature, all Phase-1 alignment policies, recipe identity errors, and frozen-metadata identity enforcement.

## Floating-output identity

Machine-readable result floats are rounded to 15 decimal digits before hashing so insignificant libm tail bits do not become artifact identity across supported Python/Linux CI environments.

## Hash and provenance binding

Every result records `protocol_id`, `recipe_sha256`, `implementation_revision`, per-trajectory `trajectory_sha256`, and top-level `artifact_sha256`. The artifact hash covers the canonical payload before the hash field itself is added. Frozen metadata records protocol, recipe, implementation, recipe hash, artifact hash, evidence class, replication status, and schema version.

`tools/verify_reference.py` verifies every one of those frozen metadata identities against the recipe, regenerated record, and frozen result before accepting byte identity.

When a CLI or reference-generation command derives or verifies a Git implementation revision from the source checkout, the checkout must be clean. A dirty working tree cannot be silently attributed to `HEAD`.

## Falsifier

Phase 1 fails if the frozen implementation cannot recover the known synthetic properties within the stated tolerances, if frozen provenance identities disagree, or if it cannot regenerate its frozen fixture byte-for-byte.

## Claim ceiling

Success permits only an instrumentation claim: under GEO-SIM-001, the implementation deterministically recovers the preregistered synthetic geometry. It does not permit claims about real LLM reasoning, hidden-state smoothness, logical equivalence, or training benefit.

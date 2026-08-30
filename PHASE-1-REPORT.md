# Phase 1 Report — GEO-SIM-001

Status: **PASS (synthetic instrument conformance)**

Evidence class: **`SIMULATION`**

Replication status: **`not_attempted`**

This report records the frozen Phase 1 reference result. It is not evidence about any language model.

## Bound identities

- Protocol: `GEO-SIM-001`
- Recipe: `GEO-SIM-REF-001`
- Implementation / mathematical-kernel revision: `952d20bb4c3506b4ddda5db54628e5b029d0eadc`
- Recipe SHA-256: `763edeb96a1eec8d87a90d200f8c03a3e2131ec924b558e26492640a342dbbeb`
- Result artifact SHA-256: `3e890ccbd349ae6e0e7be330752c340d6e74f88ee33c78ed963935b15085dad4`

The implementation revision contains the exact mathematical specification in [`MATH-SPEC.md`](MATH-SPEC.md) and the complete code changes that produce the frozen numerical artifact. The frozen artifact is an implementation-conformance record, not a substitute for the exact mathematics.

## Frozen checks

| Check | Expected | Frozen result | Status |
| --- | ---: | ---: | --- |
| Straight path length | `5.0` | `5.0` | PASS |
| Straight curvature | all `0` | all `0` | PASS |
| Radius-2 arc curvature | `0.5 ± 1e-12` | within tolerance | PASS |
| Null path length | `0.0` | `0.0` | PASS |
| Carrier order-1 alignment | `1.0` | `1.0` | PASS |
| Control order-1 alignment | `1.0` | `1.0` | PASS |
| Causal suffix alignment | `< 1.0` | `0.96641005886757` | PASS |
| Null self-alignment | `1.0` | `1.0` | PASS |

The carrier analogue differs in absolute position while preserving its order-1 differences and exact-zero Menger-curvature sequence. The control translation preserves the same order-1 flow. The suffix perturbation deliberately changes the synthetic trajectory and lowers the preregistered order-1 alignment.

## Mathematical freeze

`MATH-SPEC.md` freezes stable `GEO-MATH-001` through `GEO-MATH-011` definitions covering finite trajectories, finite differences, path length, the project cosine convention, alignment, Menger curvature, transformation laws, undefined cases, and the exact/numerical boundary.

`GEO-MATH-005` fixes the pairwise arc-length rule: both operands are resampled to `m = max(n_X, n_Y)` samples, at fractions `j/(m-1)` when `m >= 2`.

It also records `GEO-LEAN-TGT-001` through `GEO-LEAN-TGT-012` as theorem targets for a post-release Lean 4 formalization.

The formalization boundary is explicit: Lean is intended to prove the exact-real mathematical layer. It does not automatically prove CPython floating-point execution, serialization, hashes, Git provenance, or any LLM semantic claim.

## Review hardening

Four Codex review rounds reduced from **9 findings → 6 findings → 3 findings → 2 findings**, for **20 findings total**. The current kernel fixes and regression-tests all of them.

The hardening includes:

- exact-zero rather than absolute-threshold handling for cosine vectors;
- overflow-safe cosine normalization for very large vectors;
- immediate rejection of finite-difference subtraction overflow instead of allowing a non-finite vector into downstream metrics;
- preservation of tiny nonzero paths during arc-length resampling;
- preservation of late small path segments after much larger segments;
- exact dyadic-rational Menger `kappa^2` evaluation on represented binary64 displacements, preserving both exact collinearity and genuinely non-collinear near-collinearity before the final binary64 square root;
- rejection rather than silent zero/infinity if a nonzero represented curvature cannot be expressed as a positive finite binary64 result;
- scale-aware **14-significant-digit** ordinary output normalization;
- translation-aware point canonicalization using canonical local displacements, with adaptive **17-significant-digit round-trip origin precision** when ordinary origin rounding would erase a nonzero ULP-sized displacement;
- explicit rejection if a canonical nonzero displacement still cannot survive absolute-coordinate reconstruction;
- derivation of all trajectory metrics and comparisons from the exact emitted coordinate arrays, preventing serialized points from disagreeing with serialized metrics;
- rejection of undefined empty finite-difference comparisons rather than emitting `0.0`;
- strict branch recipes with a required post-branch segment;
- strict trajectory/comparison object shapes and unique comparison IDs;
- rejection of self-referential derived trajectories;
- structured result-schema definitions for trajectory and comparison records;
- source-relevant dirty-checkout rejection for Git-derived implementation identity while ignoring generated interpreter/build artifacts;
- verification of every frozen metadata identity, including protocol, recipe, hashes, evidence class, and replication status;
- consistent roadmap/report/fixture binding to the exact implementation revision.

## Edge-case coverage

The Phase 1 unit suite covers order-0/order-1/order-2/higher finite differences, path length, known-circle Menger curvature, axis and diagonal collinearity, repeated points, short sequences, exact zero-length and tiny nonzero paths, tiny opposing vectors, `1e100`-scale cosine vectors, finite-difference overflow from `-1e308` to `+1e308`, large-scale Menger curvature, Codex's exact near-collinear nonzero-curvature counterexample, late small arc-length segments, large absolute coordinates with small representable displacements, the ULP spacing-boundary step `9007199254740991 -> 9007199254740992`, serialized-point/metric consistency, undefined empty comparisons, dimension mismatch, truncate/error/arc-length alignment, deterministic replay, implementation-revision hash binding, strict recipe object shapes, duplicate comparison IDs, invalid forward/self references, branch bounds, source-provenance behavior, result-schema shapes, the frozen arc-length count rule, and frozen-metadata identities.

The hardened suite contains **47 unit tests**. GitHub Actions independently reruns the suite on Python 3.11, 3.12, and 3.13; all three jobs pass the tests, frozen metadata verification, and byte-for-byte result identity for the current frozen candidate.

## Frozen artifacts

- `MATH-SPEC.md`
- `recipes/reference-suite.json`
- `fixtures/reference-metadata.json`
- `fixtures/reference-result.json`
- `protocols/GEO-SIM-001.md`

`tools/verify_reference.py` regenerates the result from the frozen recipe and bound implementation revision, verifies every frozen provenance identity, and then requires byte identity.

## Release/formalization handoff

After PR #2 receives a green exact-head Codex review and is merged, the intended next step is to create an immutable Phase 1 release/tag. Lean 4 formalization should target that exact frozen release rather than a moving branch.

Future mathematical changes must produce a new release identity and update/reprove affected `GEO-LEAN-TGT-*` theorems rather than rewriting the frozen target under the Lean development.

## Scientific boundary

Passing Phase 1 establishes only that the current instrumentation recovers known synthetic geometry under the frozen protocol. It does not establish that LLM representations contain reasoning flows, that geometry causes reasoning, or that any geometric intervention improves a model.

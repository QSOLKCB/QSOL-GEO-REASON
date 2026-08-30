# Phase 1 Report — GEO-SIM-001

Status: **PASS (synthetic instrument conformance)**

Evidence class: **`SIMULATION`**

Replication status: **`not_attempted`**

This report records the frozen Phase 1 reference result. It is not evidence about any language model.

## Bound identities

- Protocol: `GEO-SIM-001`
- Recipe: `GEO-SIM-REF-001`
- Implementation revision: `03a3103a04d19f53331d37aa486db563ecca31f8`
- Recipe SHA-256: `763edeb96a1eec8d87a90d200f8c03a3e2131ec924b558e26492640a342dbbeb`
- Result artifact SHA-256: `01a0cba29d45b7859243fa12c5ae178cf91c0551b4e92dfa60fbf79387fe7d28`

The implementation revision includes the exact mathematical specification in [`MATH-SPEC.md`](MATH-SPEC.md). The frozen numerical artifact is an implementation-conformance record, not a substitute for the exact mathematics.

## Frozen checks

| Check | Expected | Frozen result | Status |
| --- | ---: | ---: | --- |
| Straight path length | `5.0` | `5.0` | PASS |
| Straight curvature | all `0` | all `0` | PASS |
| Radius-2 arc curvature | `0.5 ± 1e-12` | all `0.5` | PASS |
| Null path length | `0.0` | `0.0` | PASS |
| Carrier order-1 alignment | `1.0` | `1.0` | PASS |
| Control order-1 alignment | `1.0` | `1.0` | PASS |
| Causal suffix alignment | `< 1.0` | `0.9664100588676` | PASS |
| Null self-alignment | `1.0` | `1.0` | PASS |

The carrier analogue differs in absolute position while preserving its order-1 differences and exact-zero Menger-curvature sequence. The control translation preserves the same order-1 flow. The suffix perturbation deliberately changes the synthetic trajectory and lowers the preregistered order-1 alignment.

## Mathematical freeze

`MATH-SPEC.md` freezes stable `GEO-MATH-001` through `GEO-MATH-011` definitions covering finite trajectories, finite differences, path length, the project cosine convention, alignment, Menger curvature, transformation laws, undefined cases, and the exact/numerical boundary.

It also records `GEO-LEAN-TGT-001` through `GEO-LEAN-TGT-012` as theorem targets for a post-release Lean 4 formalization.

The formalization boundary is explicit: Lean is intended to prove the exact-real mathematical layer. It does not automatically prove CPython floating-point execution, serialization, hashes, Git provenance, or any LLM semantic claim.

## Review hardening

The two Codex review rounds and proactive math-freeze pass fix and regression-test:

- exact-zero rather than absolute-threshold handling for cosine vectors;
- overflow-safe cosine normalization for very large vectors;
- preservation of tiny nonzero paths during arc-length resampling;
- preservation of late small path segments after much larger segments;
- exact dyadic-rational collinearity detection before floating Menger estimation;
- preservation of tiny nonzero Menger curvature;
- scale-aware 13-significant-digit result normalization that preserves tiny nonzero values while stabilizing the supported runtime matrix;
- rejection of undefined empty finite-difference comparisons rather than emitting `0.0`;
- strict branch recipes with a required post-branch segment;
- strict trajectory/comparison object shapes and unique comparison IDs;
- rejection of self-referential derived trajectories;
- structured result-schema definitions for trajectory and comparison records;
- source-relevant dirty-checkout rejection for Git-derived implementation identity while ignoring generated interpreter/build artifacts;
- verification of every frozen metadata identity, including protocol, recipe, hashes, evidence class, and replication status;
- consistent roadmap/report/fixture binding to the exact implementation revision.

## Edge-case coverage

The Phase 1 unit suite covers order-0/order-1/order-2/higher finite differences, path length, known-circle Menger curvature, axis and diagonal collinearity, repeated points, short sequences, exact zero-length and tiny nonzero paths, tiny opposing vectors, `1e100`-scale cosine vectors, large-scale Menger curvature, late small arc-length segments, undefined empty comparisons, dimension mismatch, truncate/error/arc-length alignment, deterministic replay, implementation-revision hash binding, strict recipe object shapes, duplicate comparison IDs, invalid forward/self references, branch bounds, source-provenance behavior, result-schema shapes, and frozen-metadata identities.

The hardened suite contains **42 unit tests**. GitHub Actions independently reruns the suite on Python 3.11, 3.12, and 3.13 and verifies both frozen metadata identities and byte-for-byte result identity.

## Frozen artifacts

- `MATH-SPEC.md`
- `recipes/reference-suite.json`
- `fixtures/reference-metadata.json`
- `fixtures/reference-result.json`
- `protocols/GEO-SIM-001.md`

`tools/verify_reference.py` regenerates the result from the frozen recipe and bound implementation revision, verifies every frozen provenance identity, and then requires byte identity.

## Release/formalization handoff

After PR #2 receives a green exact-head review and is merged, the intended next step is to create an immutable Phase 1 release/tag. Lean 4 formalization should target that exact frozen release rather than a moving branch.

Future mathematical changes must produce a new release identity and update/reprove affected `GEO-LEAN-TGT-*` theorems rather than rewriting the frozen target under the Lean development.

## Scientific boundary

Passing Phase 1 establishes only that the current instrumentation recovers known synthetic geometry under the frozen protocol. It does not establish that LLM representations contain reasoning flows, that geometry causes reasoning, or that any geometric intervention improves a model.

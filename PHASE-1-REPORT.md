# Phase 1 Report — GEO-SIM-001

Status: **PASS (synthetic instrument conformance)**

Evidence class: **`SIMULATION`**

Replication status: **`not_attempted`**

This report records the frozen Phase 1 reference result. It is not evidence about any language model.

## Bound identities

- Protocol: `GEO-SIM-001`
- Recipe: `GEO-SIM-REF-001`
- Implementation revision: `e23c057e283c22cfec9de11b623e8f4d2173da75`
- Recipe SHA-256: `763edeb96a1eec8d87a90d200f8c03a3e2131ec924b558e26492640a342dbbeb`
- Result artifact SHA-256: `c2c93399b4352958e6a95e4336d4cd7b5800eb16937126173805763c661ebd37`

## Frozen checks

| Check | Expected | Frozen result | Status |
| --- | ---: | ---: | --- |
| Straight path length | `5.0` | `5.0` | PASS |
| Straight curvature | all `0` | all `0` | PASS |
| Radius-2 arc curvature | `0.5 ± 1e-12` | within tolerance | PASS |
| Null path length | `0.0` | `0.0` | PASS |
| Carrier order-1 alignment | `1.0` | `1.0` | PASS |
| Control order-1 alignment | `1.0` | `1.0` | PASS |
| Causal suffix alignment | `< 1.0` | `0.966410058867569` | PASS |
| Null self-alignment | `1.0` | `1.0` | PASS |

The carrier analogue differs in absolute position while preserving its order-1 differences and Menger-curvature sequence. The control translation preserves the same order-1 flow. The suffix perturbation deliberately changes the synthetic trajectory and lowers the preregistered order-1 alignment.

## Review hardening

The Codex review hardening revision fixes and regression-tests:

- exact-zero rather than absolute-threshold handling for cosine vectors;
- preservation of tiny nonzero paths during arc-length resampling;
- preservation of tiny nonzero Menger curvature;
- strict branch recipes with a required post-branch segment;
- strict trajectory/comparison object shapes and unique comparison IDs;
- rejection of self-referential derived trajectories;
- structured result-schema definitions for trajectory and comparison records;
- dirty-checkout rejection for Git-derived implementation identity;
- verification of every frozen metadata identity, including protocol, recipe, hashes, evidence class, and replication status.

## Edge-case coverage

The Phase 1 unit suite covers order-0/order-1/order-2/higher finite differences, path length, known-circle Menger curvature, collinear triples, repeated points, short sequences, exact zero-length and tiny nonzero paths, tiny opposing vectors, tiny nonzero curvature, dimension mismatch, truncate/error/arc-length alignment, deterministic replay, implementation-revision hash binding, strict recipe object shapes, duplicate comparison IDs, invalid forward/self references, branch bounds, source-provenance behavior, result-schema shapes, and frozen-metadata identities.

The hardened pre-CI suite contains **33 unit tests**. GitHub Actions independently reruns the suite on Python 3.11, 3.12, and 3.13 and verifies both frozen metadata identities and byte-for-byte result identity.

## Frozen artifacts

- `recipes/reference-suite.json`
- `fixtures/reference-metadata.json`
- `fixtures/reference-result.json`
- `protocols/GEO-SIM-001.md`

`tools/verify_reference.py` regenerates the result from the frozen recipe and bound implementation revision, verifies every frozen provenance identity, and then requires byte identity.

## Scientific boundary

Passing Phase 1 establishes only that the current instrumentation recovers known synthetic geometry under the frozen protocol. It does not establish that LLM representations contain reasoning flows, that geometry causes reasoning, or that any geometric intervention improves a model.

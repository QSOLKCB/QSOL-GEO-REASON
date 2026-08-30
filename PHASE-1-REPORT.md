# Phase 1 Report — GEO-SIM-001

Status: **PASS (synthetic instrument conformance)**

Evidence class: **`SIMULATION`**

Replication status: **`not_attempted`**

This report records the frozen Phase 1 reference result. It is not evidence about any language model.

## Bound identities

- Protocol: `GEO-SIM-001`
- Recipe: `GEO-SIM-REF-001`
- Implementation revision: `4850d985c9844d361c053b8cc37f98e402f1f450`
- Recipe SHA-256: `763edeb96a1eec8d87a90d200f8c03a3e2131ec924b558e26492640a342dbbeb`
- Result artifact SHA-256: `4124db87a775dc2d1f7ae83418dbe8e9e0f3b26f1fcc5c2b2556781a93bd25e1`

## Frozen checks

| Check | Expected | Frozen result | Status |
| --- | ---: | ---: | --- |
| Straight path length | `5.0` | `5.0` | PASS |
| Straight curvature | all `0` | all `0` | PASS |
| Radius-2 arc curvature | `0.5 ± 1e-12` | all `0.5` | PASS |
| Null path length | `0.0` | `0.0` | PASS |
| Carrier order-1 alignment | `1.0` | `1.0` | PASS |
| Control order-1 alignment | `1.0` | `1.0` | PASS |
| Causal suffix alignment | `< 1.0` | `0.966410058867569` | PASS |
| Null self-alignment | `1.0` | `1.0` | PASS |

The carrier analogue differs in absolute position while preserving its order-1 differences and Menger-curvature sequence. The control translation preserves the same order-1 flow. The suffix perturbation deliberately changes the synthetic trajectory and lowers the preregistered order-1 alignment.

## Edge-case coverage

The Phase 1 unit suite covers order-0/order-1/order-2/higher finite differences, path length, known-circle Menger curvature, collinear triples, repeated points, short sequences, zero-length paths, dimension mismatch, truncate/error/arc-length alignment, deterministic replay, implementation-revision hash binding, unknown recipe fields, and invalid forward references.

The local pre-commit conformance run executed **17 unit tests successfully**. GitHub Actions independently reruns the suite on Python 3.11, 3.12, and 3.13 and byte-compares the frozen result.

## Frozen artifacts

- `recipes/reference-suite.json`
- `fixtures/reference-metadata.json`
- `fixtures/reference-result.json`
- `protocols/GEO-SIM-001.md`

`tools/verify_reference.py` regenerates the result from the frozen recipe and bound implementation revision and requires byte identity.

## Scientific boundary

Passing Phase 1 establishes only that the current instrumentation recovers known synthetic geometry under the frozen protocol. It does not establish that LLM representations contain reasoning flows, that geometry causes reasoning, or that any geometric intervention improves a model.

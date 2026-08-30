# QSOL-GEO-REASON

**Experimental research framework for measuring, perturbing, simulating, and eventually training geometric reasoning flows in local language-model representation spaces.**

> Status: **Phase 1 complete / mathematical kernel freeze candidate**. The repository contains a deterministic geometry reference instrument, an exact mathematical specification, and frozen `SIMULATION` evidence. It still makes **no empirical claim about any language model**.

## Research question

Can reasoning capability in local language models be identified, measured, perturbed, and eventually induced as reproducible structure in representation-space trajectories?

A working trajectory notation is:

\[
\gamma = (z_0, z_1, \ldots, z_T), \qquad \Delta z_t = z_{t+1}-z_t
\]

with higher finite differences, path geometry, and curvature-like quantities treated as **measurements of a chosen representation**, not automatically as mechanisms of reasoning.

The project is especially interested in whether models preserve logical structure across different semantic carriers, whether load-bearing premise changes produce distinguishable trajectory changes, and whether geometric training objectives can improve a small model without simply hiding extra capability in additional parameters or evaluation leakage.

## Core discipline

QSOL-GEO-REASON is designed to be able to falsify its motivating hypothesis.

The project therefore treats the following distinctions as foundational:

- semantic content is not logical form;
- geometric similarity is not logical equivalence;
- correlation is not mechanism;
- visualization is not evidence;
- simulation is not an empirical local-model result;
- replication status is not an evidence class;
- matching output tokens is not proof of hidden-state equivalence;
- exact mathematical semantics are not the same thing as one floating-point implementation;
- a benchmark win is not automatically a reasoning improvement;
- a smaller parameter count is not, by itself, evidence of greater reasoning efficiency.

The primary normative rules are frozen in [`INVARIANTS.md`](INVARIANTS.md) and [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md). [`MATH-SPEC.md`](MATH-SPEC.md) is normative for the Phase 1 exact mathematical semantics, subject to those higher-level contracts.

## Documentation map

| File | Audience | Purpose |
| --- | --- | --- |
| [`README.md`](README.md) | Human | Project overview and scientific boundaries |
| [`README4AI.md`](README4AI.md) | AI / agents | Machine-oriented project context and terminology |
| [`AGENTS.md`](AGENTS.md) | AI / agents | Repository operating rules |
| [`INVARIANTS.md`](INVARIANTS.md) | Human + AI | Non-negotiable epistemic and experimental invariants |
| [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md) | Human + AI | Evidence classes, replication status, operational definitions, provenance, and experiment rules |
| [`MATH-SPEC.md`](MATH-SPEC.md) | Human + AI + Lean | Exact Phase 1 definitions, transformation laws, numerical boundary, and formal theorem targets |
| [`ROADMAP.md`](ROADMAP.md) | Human + AI | Staged research programme |
| [`protocols/GEO-SIM-001.md`](protocols/GEO-SIM-001.md) | Human + AI | Phase 1 synthetic conformance protocol |
| [`PHASE-1-REPORT.md`](PHASE-1-REPORT.md) | Human + AI | Frozen Phase 1 evidence summary and hashes |

## Initial research context

This project was motivated in part by work studying reasoning as trajectories in representation space, especially:

- Yufa Zhou, Yixiao Wang, Xunjian Yin, Shuyan Zhou, and Anru R. Zhang, **“The Geometry of Reasoning: Flowing Logics in Representation Space”**, ICLR 2026: <https://github.com/MasterZhou1/Reasoning-Flow>
- Paper: <https://arxiv.org/abs/2510.09782>
- Julian D. Michels profile and related geometric-reasoning discussion: <https://philpeople.org/profiles/julian-michels>

External work is treated as motivation and prior art, not as evidence for claims made by this repository. Reproduction and extension experiments must satisfy this repository's own contract.

### Source boundary

Zhou et al. study **post-hoc representation geometry in fixed, trained models**. Their principal construction is a context-cumulative trajectory of representation states, with finite differences and Menger curvature used to compare logical structure across semantic carriers. Their paper explicitly limits its central scope to natural-language understanding and does not claim to explain training dynamics, generation behaviour, or a causal mechanism of reasoning.

QSOL-GEO-REASON therefore distinguishes:

- **reproduction**: testing whether the reported geometric patterns can be recovered under a frozen local-model protocol;
- **extension**: adding controls, perturbations, cross-model exposure audits, output-behaviour comparisons, training interventions, serving-equivalence studies, or mechanistic tests that go beyond the cited work.

An extension result must not be attributed to the cited paper unless that result is actually established there.

### Serving-system context

The project is also interested in local serving research such as **FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution**: <https://arxiv.org/abs/2608.16157>.

Serving ideas such as hardware profiling, phase-aware prefill/decode execution, prefix/state reuse, elastic memory, expert caching, and CPU/GPU co-execution may later improve local-model practicality. They are treated as serving-system prior art, not as evidence about geometric reasoning.

For representation research, the serving backend is itself part of the instrument. An optimized backend must either pass a serving-equivalence protocol against the canonical capture path or remain an explicit experimental variable.

## Phase 1 mathematical kernel

Phase 1 validates the measurement machinery **before any model hidden state is touched**.

The exact semantics are defined in [`MATH-SPEC.md`](MATH-SPEC.md). Stable identifiers `GEO-MATH-001` through `GEO-MATH-011` define trajectories, finite differences, path length, the project cosine convention, alignment, Menger curvature, transformation laws, undefined cases, and the exact/numerical boundary.

`GEO-LEAN-TGT-001` through `GEO-LEAN-TGT-012` record theorem targets for a later Lean 4 formalization against an immutable release.

The frozen numerical protocol is [`GEO-SIM-001`](protocols/GEO-SIM-001.md). It provides:

- deterministic straight, circular, branching, noisy, and null trajectory generators;
- same-flow/different-carrier synthetic analogues by rigid translation;
- geometry-preserving control translation and a geometry-changing suffix perturbation;
- order-0 through configurable order-k finite differences with immediate rejection of non-finite subtraction overflow;
- Euclidean path length and mean cosine alignment;
- dimension-independent Menger curvature using exact dyadic-rational `kappa^2` on represented binary64 displacements before the final binary64 square root;
- deterministic truncate/error/arc-length alignment and resampling, with pairwise `arclength` count fixed at `max(n_left, n_right)`;
- translation-aware coordinate canonicalization using 14-digit origin-relative and consecutive local displacements with an adaptive 17-digit round-trip origin whenever binary64 spacing would erase any nonzero canonical step;
- derivation of metrics and comparisons from the exact emitted coordinate arrays;
- scale-aware **14-significant-digit** ordinary output normalization;
- canonical JSON and SHA-256 evidence binding;
- frozen reference fixtures with full metadata-identity and byte-for-byte replay verification;
- CI on Python 3.11, 3.12, and 3.13.

Frozen identities:

- protocol: `GEO-SIM-001`
- implementation / mathematical-kernel revision: `5f45b5e69bcab890a757fffa491cf787f92a5bea`
- recipe SHA-256: `763edeb96a1eec8d87a90d200f8c03a3e2131ec924b558e26492640a342dbbeb`
- result artifact SHA-256: `c542bce987d31350b4904122e5ec02ef026715f51a1fe21ee184a452cc67a583`

The frozen checks recover straight-line curvature `0`, radius-2 circular curvature within `0.5 ± 1e-12`, null path length `0`, carrier/control order-1 alignment `1.0`, and a lower order-1 alignment for the deliberate suffix perturbation. The hardened suite preserves tiny nonzero geometry, avoids large-vector cosine overflow, preserves late small arc-length segments, preserves genuine near-collinear curvature, keeps small local displacements visible on large absolute coordinate offsets and at later spacing boundaries, makes exact stored collinearity exactly zero without an epsilon, rejects overflowing finite differences, and rejects undefined empty comparisons. See [`PHASE-1-REPORT.md`](PHASE-1-REPORT.md).

### Running the reference simulation

```bash
python -m pip install -e .
python -m qsol_geo_reason recipes/reference-suite.json \
  --output /tmp/reference-result.json
```

The CLI uses `HEAD` only when the source checkout is clean with respect to source-relevant changes. Generated interpreter/build artifacts such as `__pycache__` do not make an otherwise clean checkout dirty.

Verify the frozen evidence artifact:

```bash
python -m unittest discover -s tests -v
python tools/verify_reference.py
```

The hardened suite contains **48 unit tests** before the cross-version CI matrix is applied.

## Release and Lean handoff

The intended Phase 1 closure is:

```text
exact-head Codex approval
        ↓
merge PR #2
        ↓
freeze immutable Phase 1 release/tag
        ↓
formalize MATH-SPEC in Lean 4 against that frozen identity
```

Lean is intended to prove the exact-real mathematics, not to retroactively certify CPython floating point, JSON, SHA-256, Git provenance, or LLM semantics. A formal implementation-refinement proof would be a separate future result.

Once a release is frozen, later mathematical changes must create a new release identity rather than rewriting the frozen target under the Lean development.

## Evidence classes and replication

The repository uses six evidence classes defined normatively in [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md):

1. **`SIMULATION`**: validate measurement code against known synthetic geometry; no real-model claim is permitted.
2. **`OBSERVATION`**: measure a property in one or more frozen model runs under a specified extraction protocol.
3. **`ASSOCIATION`**: establish a statistical relationship between a geometric quantity and another measured variable under the frozen analysis.
4. **`PERTURBATION`**: test reproducible differential response to controlled input changes against matched controls.
5. **`INTERVENTION`**: intentionally alter a model, training process, or representation property and test downstream change against a controlled baseline.
6. **`MECHANISM`**: identify a specific internal process that survives targeted intervention, ablation, prediction, and alternative-explanation tests.

**Replication is orthogonal to that sequence.** A result separately records whether the underlying evidence has been replicated, failed replication, produced mixed replications, or has not yet been replicated.

These are claim ceilings, not an automatic ladder: completing a later experiment does not silently grant every stronger interpretation.

See [`ROADMAP.md`](ROADMAP.md) for the full programme.

## Non-claims

Phase 1 does **not** claim that:

- latent or representation geometry is the mechanism of reasoning;
- any LLM has been observed to produce the synthetic structures in the fixture;
- smooth trajectories imply correct reasoning;
- curvature, velocity, or other geometric quantities have a unique cognitive interpretation;
- matching outputs across serving engines imply matching hidden-state geometry;
- a future Lean proof of the exact mathematical contract automatically proves the Python implementation;
- geometric reasoning makes small models equivalent to larger models;
- any cited external performance claim has been independently reproduced here.

## License

Licensed under the [Apache License 2.0](LICENSE).

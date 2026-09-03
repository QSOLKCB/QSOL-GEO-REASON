# QSOL-GEO-REASON

**Experimental research framework for measuring, perturbing, simulating, and eventually training geometric reasoning flows in local language-model representation spaces.**

> Status: **Phase 1 is frozen in two immutable evidence layers: numerical kernel `v0.1.0` and Lean 4 formal layer `v0.2.0`. PR #4 begins Phase 2A canonical local hidden-state capture. The repository still makes no empirical geometric-reasoning claim about any language model.**

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
- a benchmark win is not automatically a reasoning improvement; and
- a smaller parameter count is not, by itself, evidence of greater reasoning efficiency.

The primary normative rules are [`INVARIANTS.md`](INVARIANTS.md) and [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md). [`MATH-SPEC.md`](MATH-SPEC.md) is normative for the Phase 1 exact mathematical semantics, subject to those higher-level contracts.

## Documentation map

| File | Audience | Purpose |
| --- | --- | --- |
| [`README.md`](README.md) | Human | Project overview and scientific boundaries |
| [`README4AI.md`](README4AI.md) | AI / agents | Machine-oriented project context and terminology |
| [`AGENTS.md`](AGENTS.md) | AI / agents | Repository operating rules |
| [`INVARIANTS.md`](INVARIANTS.md) | Human + AI | Non-negotiable epistemic and experimental invariants |
| [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md) | Human + AI | Evidence classes, replication status, operational definitions, provenance, and experiment rules |
| [`MATH-SPEC.md`](MATH-SPEC.md) | Human + AI + Lean | Exact Phase 1 definitions, transformation laws, numerical boundary, and formal theorem targets |
| [`LEAN-FORMALIZATION.md`](LEAN-FORMALIZATION.md) | Human + AI + Lean | Frozen release target, theorem coverage, proof environment, and formal evidence boundary |
| [`LEAN-CACHE-POLICY.md`](LEAN-CACHE-POLICY.md) | Human + AI | Workflow authority, cache integrity, process isolation, and cold-reconstruction semantics |
| [`ROADMAP.md`](ROADMAP.md) | Human + AI | Staged research programme and current phase status |
| [`protocols/GEO-SIM-001.md`](protocols/GEO-SIM-001.md) | Human + AI | Phase 1 synthetic conformance protocol |
| [`protocols/GEO-CAP-001.md`](protocols/GEO-CAP-001.md) | Human + AI | Phase 2A canonical local hidden-state capture protocol |
| [`RESEARCH-HYPOTHESIS-MAP.md`](RESEARCH-HYPOTHESIS-MAP.md) | Human + AI | Non-normative theory-to-test registry; sources are hypotheses, not evidence |
| [`PHASE-1-REPORT.md`](PHASE-1-REPORT.md) | Human + AI | Frozen Phase 1 numerical evidence summary and hashes |

## Initial research context

This project was motivated in part by work studying reasoning as trajectories in representation space, especially:

- Yufa Zhou, Yixiao Wang, Xunjian Yin, Shuyan Zhou, and Anru R. Zhang, **“The Geometry of Reasoning: Flowing Logics in Representation Space”**, ICLR 2026: <https://github.com/MasterZhou1/Reasoning-Flow>
- Paper: <https://arxiv.org/abs/2510.09782>
- Julian D. Michels profile and related geometric-reasoning discussion: <https://philpeople.org/profiles/julian-michels>

External work is treated as motivation and prior art, not as evidence for claims made by this repository. Reproduction and extension experiments must satisfy this repository's own contract.

### Source boundary

Zhou et al. study **post-hoc representation geometry in fixed, trained models**. Their principal construction is a context-cumulative trajectory of representation states, with finite differences and Menger curvature used to compare logical structure across semantic carriers. Their central scope does not establish training dynamics, generation behaviour, or a causal mechanism of reasoning.

QSOL-GEO-REASON therefore distinguishes:

- **reproduction**: testing whether reported geometric patterns can be recovered under a frozen local-model protocol; and
- **extension**: adding controls, perturbations, cross-model exposure audits, output-behaviour comparisons, training interventions, serving-equivalence studies, or mechanistic tests that go beyond the cited work.

An extension result must not be attributed to cited prior work unless that result is actually established there.

The broader theory corpus registered in [`RESEARCH-HYPOTHESIS-MAP.md`](RESEARCH-HYPOTHESIS-MAP.md) is handled even more conservatively: it generates candidate measurements and falsifiable hypotheses. It does not change the evidence state of this repository. Conflicting theoretical predictions remain conflicting hypotheses until an experiment distinguishes them.

### Serving-system context

The project is also interested in local serving research such as **FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution**: <https://arxiv.org/abs/2608.16157>.

Serving ideas such as hardware profiling, phase-aware prefill/decode execution, prefix/state reuse, elastic memory, expert caching, and CPU/GPU co-execution may later improve local-model practicality. They are serving-system prior art, not evidence about geometric reasoning.

For representation research, the serving backend is part of the instrument. An optimized backend must either pass a serving-equivalence protocol against the canonical capture path or remain an explicit experimental variable.

## Phase 1 mathematical kernel

Phase 1 validated the measurement machinery **before any model hidden state was touched**.

The exact semantics are defined in [`MATH-SPEC.md`](MATH-SPEC.md). Stable identifiers `GEO-MATH-001` through `GEO-MATH-011` define trajectories, finite differences, path length, the project cosine convention, alignment, Menger curvature, transformation laws, undefined cases, and the exact/numerical boundary.

The exact mathematical and numerical kernel was published as immutable release `v0.1.0 — Phase 1 Mathematical Kernel` at commit:

`1b5ab8b4543b20cdb6d439f7ad215c08e698188f`

PR #3 then implemented all twelve frozen `GEO-LEAN-TGT-*` targets against that stationary target. The merged formal evidence layer was published separately as immutable `v0.2.0 — Phase 1 Lean 4 Formal Evidence Layer` at merge commit:

`ec3312dcc102d859819c764a881e2d020662e880`

The two releases deliberately preserve different evidence objects. `v0.2.0` does not rewrite or retroactively upgrade the simulation artifact in `v0.1.0`.

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
- frozen reference fixtures with full metadata-identity and byte-for-byte replay verification; and
- CI on Python 3.11, 3.12, and 3.13.

Frozen numerical identities:

- release: `v0.1.0 — Phase 1 Mathematical Kernel`
- release commit: `1b5ab8b4543b20cdb6d439f7ad215c08e698188f`
- protocol: `GEO-SIM-001`
- implementation / mathematical-kernel revision: `5f45b5e69bcab890a757fffa491cf787f92a5bea`
- recipe SHA-256: `763edeb96a1eec8d87a90d200f8c03a3e2131ec924b558e26492640a342dbbeb`
- result artifact SHA-256: `c542bce987d31350b4904122e5ec02ef026715f51a1fe21ee184a452cc67a583`

The frozen checks recover straight-line curvature `0`, radius-2 circular curvature within `0.5 ± 1e-12`, null path length `0`, carrier/control order-1 alignment `1.0`, and a lower order-1 alignment for the deliberate suffix perturbation. See [`PHASE-1-REPORT.md`](PHASE-1-REPORT.md).

### Running the reference simulation

```bash
python -m pip install -e .
python -m qsol_geo_reason recipes/reference-suite.json \
  --output /tmp/reference-result.json
```

Verify the frozen numerical evidence artifact:

```bash
python -m unittest discover -s tests -v
python tools/verify_reference.py
```

The frozen Phase 1 numerical suite contains **48 unit tests** before the cross-version CI matrix is applied. Later tests extend repository coverage without changing that historical count.

## Phase 1 release and Lean handoff

The completed provenance chain is:

```text
PR #2 exact-head review and merge
        ↓
immutable v0.1.0 numerical kernel at 1b5ab8b…
        ↓
PR #3 Lean 4 formalization of all twelve frozen targets
        ↓
merge commit ec3312d…
        ↓
immutable v0.2.0 formal evidence layer
        ↓
Phase 2A canonical hidden-state capture instrument
```

The sole release-grade cold proof authority is the manually dispatched `lean-isolated-audit / isolated-cold-trust` job. `lean-phase1` is a verified-cache regression/cache-maintenance lane: pull-request runs validate candidates, while relevant pushes to `main` and explicit maintenance dispatches may seed authenticated caches for later pull requests. Those cache-maintenance executions carry no competing cold-trust authority. The protected theorem audit imports the source-bound GeoReason object graph without executing project initializers and emits its completion record only after all twelve theorem-kind and axiom-allowlist checks pass.

Lean proves the exact-real mathematics, not CPython floating point, JSON, SHA-256, Git provenance, serving behaviour, or LLM semantics. An implementation-refinement proof would be a separate result.

## Phase 2A canonical local capture

[`GEO-CAP-001`](protocols/GEO-CAP-001.md) introduces the first empirical **instrument**, not the first empirical conclusion.

The canonical production path is intentionally boring in the best possible way:

- direct Hugging Face/PyTorch replay;
- model and tokenizer pinned to full 40-hex Hugging Face commit identities;
- local files only;
- no remote code;
- no quantization in the canonical lane;
- `use_cache=false`;
- explicit hidden-state tuple indices captured selectively with hooks rather than retaining every layer output;
- requested spans moved to CPU before bounded float64 pooling, including MPS captures;
- exact input IDs and token-span provenance;
- explicit cumulative/isolated context mode;
- explicit pooling;
- recorded runtime/device/dtype/backend metadata; and
- content-addressed request, manifest, vector, and trajectory identities.

Install optional model-capture dependencies with:

```bash
python -m pip install -e '.[capture]'
```

Then copy and edit the request template:

```bash
cp examples/GEO-CAP-001.example.json /tmp/capture-request.json
```

Replace the placeholder zero revisions with the exact model and tokenizer commit identities already available locally, freeze the step segmentation, then run:

```bash
qsol-geo-capture /tmp/capture-request.json \
  --output-dir /tmp/GEO-CAP-001-run
```

The example request is not itself valid empirical evidence. Production model selection, first frozen capture, and replay evidence remain open roadmap items.

CI exercises the capture contract with a deterministic fake backend. That fixture is explicitly `SIMULATION`; it cannot close an empirical Phase 2A milestone.

## Evidence classes and replication

The repository uses six evidence classes defined normatively in [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md):

1. **`SIMULATION`**: validate measurement code against known synthetic geometry; no real-model claim is permitted.
2. **`OBSERVATION`**: measure a property in one or more frozen model runs under a specified extraction protocol.
3. **`ASSOCIATION`**: establish a statistical relationship between a geometric quantity and another measured variable under the frozen analysis.
4. **`PERTURBATION`**: test reproducible differential response to controlled input changes against matched controls.
5. **`INTERVENTION`**: intentionally alter a model, training process, or representation property and test downstream change against a controlled baseline.
6. **`MECHANISM`**: identify a specific internal process that survives targeted intervention, ablation, prediction, and alternative-explanation tests.

**Replication is orthogonal to that sequence.** A result separately records whether the underlying evidence has been replicated, failed replication, produced mixed replications, or has not yet been replicated.

These are claim ceilings, not an automatic ladder.

## Non-claims

Phase 1, its Lean proof layer, and the Phase 2A capture implementation do **not** claim that:

- latent or representation geometry is the mechanism of reasoning;
- any LLM has yet been shown by this repository to produce the synthetic structures in the Phase 1 fixture;
- smooth trajectories imply correct reasoning;
- curvature, velocity, holonomy, entropy, gauge, or other geometric/physical vocabulary has a unique cognitive interpretation;
- matching outputs across serving engines imply matching hidden-state geometry;
- the exact Lean proof layer automatically proves the Python/IEEE-754 implementation;
- a software-only fake capture fixture is a model observation;
- geometric reasoning makes small models equivalent to larger models; or
- any cited external performance or theoretical claim has been independently reproduced here.

## License

Licensed under the [Apache License 2.0](LICENSE).

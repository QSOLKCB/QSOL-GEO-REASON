# QSOL-GEO-REASON

**Experimental research framework for measuring, perturbing, simulating, and eventually training geometric reasoning flows in local language-model representation spaces.**

> Status: **foundation / pre-experiment**. This repository currently defines the scientific contract, invariants, documentation, and research roadmap. It does **not** yet claim empirical evidence that geometric structure causes, explains, or improves reasoning.

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
- a benchmark win is not automatically a reasoning improvement;
- a smaller parameter count is not, by itself, evidence of greater reasoning efficiency.

The normative rules are frozen in [`INVARIANTS.md`](INVARIANTS.md) and [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md).

## Documentation map

| File | Audience | Purpose |
| --- | --- | --- |
| [`README.md`](README.md) | Human | Project overview and scientific boundaries |
| [`README4AI.md`](README4AI.md) | AI / agents | Machine-oriented project context and terminology |
| [`AGENTS.md`](AGENTS.md) | AI / agents | Repository operating rules |
| [`INVARIANTS.md`](INVARIANTS.md) | Human + AI | Non-negotiable epistemic and experimental invariants |
| [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md) | Human + AI | Claim classes, operational definitions, provenance, and experiment rules |
| [`ROADMAP.md`](ROADMAP.md) | Human + AI | Staged research programme |

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
- **extension**: adding controls, perturbations, cross-model exposure audits, output-behaviour comparisons, training interventions, or mechanistic tests that go beyond the cited work.

An extension result must not be attributed to the cited paper unless that result is actually established there.

## Evidence and claim sequence

The repository deliberately separates the seven claim classes defined normatively in [`SCIENTIFIC-CONTRACT.md`](SCIENTIFIC-CONTRACT.md):

1. **`SIMULATION`**: validate measurement code against known synthetic geometry; no real-model claim is permitted.
2. **`OBSERVATION`**: measure a property in one or more frozen model runs under a specified extraction protocol.
3. **`ASSOCIATION`**: establish a statistical relationship between a geometric quantity and another measured variable under the frozen analysis.
4. **`PERTURBATION`**: test reproducible differential response to controlled input changes against matched controls.
5. **`REPLICATION`**: reproduce a previously defined result under a materially new model, seed set, carrier set, dataset split, or implementation while preserving the relevant contract.
6. **`INTERVENTION`**: intentionally alter a model, training process, or representation property and test downstream change against a controlled baseline.
7. **`MECHANISM`**: identify a specific internal process that survives targeted intervention, ablation, prediction, and alternative-explanation tests.

These are claim ceilings, not an automatic ladder: completing a later experiment does not silently grant every stronger interpretation.

See [`ROADMAP.md`](ROADMAP.md) for the full programme.

## Non-claims

At the foundation stage, QSOL-GEO-REASON does **not** claim that:

- latent or representation geometry is the mechanism of reasoning;
- smooth trajectories imply correct reasoning;
- curvature, velocity, or other geometric quantities have a unique cognitive interpretation;
- geometric reasoning makes small models equivalent to larger models;
- any cited external performance claim has been independently reproduced here.

## License

Licensed under the [Apache License 2.0](LICENSE).

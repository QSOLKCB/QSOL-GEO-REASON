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

## Intended research sequence

The repository deliberately separates five evidence levels:

1. **Reference simulation**: validate measurement code against known synthetic geometry.
2. **Observation**: measure hidden-state trajectories in frozen local models.
3. **Controlled perturbation**: test sensitivity to causal versus surface changes.
4. **Replication**: repeat results across model sizes, families, layers, and seeds.
5. **Intervention**: train geometric objectives and compare against controlled baselines.

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

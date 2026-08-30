# QSOL-GEO-REASON Invariants

These invariants define the non-negotiable epistemic and experimental boundaries of the project.

They apply to humans, AI agents, code, datasets, simulations, experiments, plots, reports, releases, and downstream claims.

An invariant may be changed only by an explicit reviewed change that names the invariant, explains the scientific reason for changing it, and records any compatibility break.

## Epistemic invariants

### GEO-INV-001 — Observation is not mechanism

A reproducible geometric pattern in model representations does not, by itself, establish that the pattern causes or implements reasoning.

**Prohibited upgrade:** `observed geometry -> mechanism` without intervention and mechanistic evidence.

### GEO-INV-002 — Visualization is not evidence

PCA, UMAP, t-SNE, trajectory plots, heatmaps, and similar visualizations may illustrate results but may not serve as the sole evidentiary basis for a claim.

Every evidentiary claim must have a machine-readable numerical counterpart.

### GEO-INV-003 — Semantic carrier is not logical form

Surface topic, vocabulary, language, names, and narrative domain must remain analytically separable from the formal reasoning structure they carry.

Carrier-invariant experiments require explicit identifiers for both carrier and logic.

### GEO-INV-004 — Representation state is not a truth state

A hidden-state vector or pooled embedding is an extracted representation under a specified protocol. It must not be described as a belief, proof state, semantic truth state, or cognitive state without separate evidence.

### GEO-INV-005 — Geometric similarity is not logical equivalence

Similarity in position, direction, curvature, path length, or any other geometric statistic does not automatically imply semantic, logical, causal, or functional equivalence.

### GEO-INV-006 — Correctness is not geometry

A correct answer does not demonstrate a particular geometric reasoning process, and a geometrically regular trajectory does not demonstrate a correct answer.

Correctness and geometry must be measured separately.

### GEO-INV-007 — Benchmark performance is not sufficient evidence of reasoning

A benchmark score may measure task performance. Claims about reasoning require additional controlled evidence appropriate to the claim class.

## Experimental invariants

### GEO-INV-008 — Causal perturbations require matched controls

Any experiment claiming sensitivity to a load-bearing premise must include control perturbations that alter comparable surface content without changing the relevant logical consequence whenever feasible.

The target comparison is not merely `before vs after`; it is the differential response to causal and non-causal changes.

### GEO-INV-009 — Analysis choices must be provenance-bound

Layer selection, pooling, token spans, normalization, alignment, finite-difference order, metrics, projections, thresholds, and subsets must be recorded.

Choices made after inspecting target-labelled results must be marked exploratory.

### GEO-INV-010 — Native-space evidence precedes projection-space interpretation

Claims about representation geometry must be evaluated in the native or explicitly transformed analysis space. Low-dimensional projections cannot substitute for native-space measurements.

### GEO-INV-011 — Simulation is not empirical LLM evidence

Synthetic/reference trajectories validate analysis machinery only.

Simulation outputs must be labelled as synthetic, and successful recovery of known geometry must not be presented as evidence that a real language model reasons geometrically.

### GEO-INV-012 — Model comparisons expose material confounders

Comparisons across models must record material differences including architecture/family, tokenizer, parameter count, quantization, dtype, prompting, generation settings, context, training/evaluation budget, serving backend, and dataset-exposure or contamination risk where known or plausibly different.

Exposure assessments must distinguish the assessed outcome from confidence in that assessment. Unknown exposure is itself a documented confounder, and absence of evidence is not evidence of non-exposure.

If material differences are not controlled, causal wording is prohibited.

### GEO-INV-013 — Training interventions require controlled baselines

A claim that a geometric objective improves reasoning must compare against an appropriate baseline with the same base model and, where practical, matched data, optimization budget, evaluation protocol, and parameter accounting.

Any added trainable parameters must be reported.

### GEO-INV-014 — Parameter count is not capability normalization

A smaller model outperforming a larger model does not, by itself, establish superior reasoning efficiency, geometric density, or parameter efficiency.

Any such claim requires a defined normalization and controlled evaluation.

### GEO-INV-015 — Deterministic provenance where possible; explicit stochastic provenance otherwise

Runs must preserve enough information to reproduce or meaningfully audit them.

Deterministic components should be byte- or hash-verifiable where practical. Stochastic components must record seeds and relevant runtime settings.

### GEO-INV-016 — Null and negative results are first-class results

Evidence that fails to support the motivating geometric hypothesis must be retained and reported under the same provenance standards as positive results.

Do not redefine metrics, subsets, or hypotheses solely to erase a null result.

### GEO-INV-017 — No target leakage into controls or evaluation

Carrier generation, perturbation generation, training, threshold selection, and evaluation must be designed to prevent answer labels or target outcomes from leaking into model inputs or analysis decisions.

Known or suspected contamination must be disclosed.

### GEO-INV-018 — Claims bind to exact evidence artifacts

Every published experimental claim must be traceable to the exact code revision, protocol identity, model revision, dataset revision, run ID/run manifest, serving backend where applicable, and result artifacts that support it.

A later run may strengthen or weaken a claim, but must not silently replace the evidence identity of an earlier claim.

## Research-program invariant

### GEO-INV-019 — The repository must remain capable of falsifying its motivating hypothesis

The project must preserve tests capable of producing a scientifically meaningful negative conclusion.

An experiment design that can only ever be interpreted as support for geometric reasoning is not an adequate test of geometric reasoning.

## Terminology invariant

### GEO-INV-020 — Geometric Reasoning is not Geometric Unity

QSOL-GEO-REASON's use of **geometric reasoning** refers to the measurement, analysis, perturbation, simulation, or training of geometric structure in machine-learning representation spaces.

It does **not** refer to, derive from, endorse, instantiate, validate, or imply any relationship with Eric Weinstein's **Geometric Unity** proposal. The shared word *geometric* establishes no mathematical, scientific, philosophical, evidentiary, or attributional equivalence.

In short:

`Geometric Reasoning != Geometric Unity`

This invariant exists to prevent terminology collision from being mistaken for conceptual inheritance.

## Evidence-identity invariant

### GEO-INV-021 — Replication status is not an evidence class

Replication describes whether a previously defined result survives a materially varied repetition. It does not replace the evidence class of the experiment being replicated.

For example, a controlled perturbation that reproduces on another model remains `PERTURBATION` evidence with a separate replication status.

Result schemas and prose must preserve both dimensions independently.

## Serving invariant

### GEO-INV-022 — Serving output equivalence is not representation equivalence

Two serving backends producing the same generated tokens, final answer, or benchmark score are not thereby shown to produce equivalent hidden states or reasoning-flow geometry.

Serving backend, precision, kernels, device placement, offloading, cache/state reuse, and related runtime choices remain measurement variables until a serving-equivalence protocol demonstrates otherwise for the claim being made.

Resource optimizations may change performance; they must not silently redefine the scientific object under measurement.

## Change policy

Changes to this file are high-impact scientific-contract changes.

Any pull request modifying an invariant SHOULD:

1. identify the affected invariant IDs in the PR description;
2. explain the motivating failure mode or new evidence;
3. state whether prior experiments remain comparable;
4. update `SCIENTIFIC-CONTRACT.md`, `README4AI.md`, `AGENTS.md`, and `ROADMAP.md` where necessary;
5. avoid retroactively relabelling old evidence unless the original record is preserved.

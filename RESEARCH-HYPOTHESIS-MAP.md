# QSOL-GEO-REASON Research Hypothesis Map

Status: **non-normative hypothesis registry**

This document turns theoretical source material into candidate tests without promoting those sources into evidence.

The governing rule is:

```text
source idea -> candidate operationalization -> preregistered experiment -> evidence artifact -> claim
```

A paper appearing here is motivation, not validation. Competing claims are preserved as competing hypotheses rather than silently reconciled. `INVARIANTS.md`, `SCIENTIFIC-CONTRACT.md`, `MATH-SPEC.md`, and the staged `ROADMAP.md` remain authoritative.

The initial registry was assembled alongside Phase 2A from the supplied QSOL-IMC source corpus listed below. The PDFs contain broad theoretical interpretations; only the experimentally useful pieces are promoted into candidate hypotheses here.

## Status vocabulary

- `CANDIDATE_MEASUREMENT`: useful measurable quantity whose empirical meaning is not yet established.
- `CANDIDATE_HYPOTHESIS`: falsifiable relationship suitable for a later protocol.
- `DEFERRED_MODEL`: theoretical structure that requires earlier empirical gates before implementation.
- `SPECULATIVE_INTERPRETATION`: conceptual interpretation explicitly excluded from current evidence claims.

## GEO-HYP-001 — Carrier-invariant trajectory structure

**Status:** `CANDIDATE_HYPOTHESIS`

**Roadmap home:** Phase 3 → Phase 4

**Question:** after controlling logical form and carrier independently, does higher-order representation trajectory structure group more strongly by logic than by semantic carrier?

**Operational path:** use GEO-CAP-001 to capture frozen native-space trajectories, then apply the already frozen Phase 1 finite-difference and trajectory measurements under a preregistered layer/pooling/alignment policy.

**Required controls:** same-logic/different-carrier; same-carrier/different-logic; shuffled/broken logic; answer correctness recorded separately.

**Falsifier:** logic-conditioned structure does not exceed matched carrier/control structure under the frozen protocol.

## GEO-HYP-002A — Negative-curvature instability

**Status:** `CANDIDATE_HYPOTHESIS`

**Roadmap home:** Phase 4 observational study

**Source motivation:** *The Geometry of Alignment: Optimal Control Theory on Relational Semantic Manifolds*.

**Source claim to test, not assume:** locally negative sectional-curvature-like structure is associated with divergent representation trajectories and increased hallucination/error risk.

**Candidate measurements:** native-space geodesic-deviation proxies, neighborhood preservation, Local Intrinsic Dimension, and explicitly defined curvature estimators.

**Falsifier:** preregistered instability/curvature proxies do not predict factual or logical failure above matched non-geometric controls.

## GEO-HYP-002B — Flat-region drift

**Status:** `CANDIDATE_HYPOTHESIS`

**Roadmap home:** Phase 4 observational study

**Source motivation:** *Engineering Stable Generative Ecologies*.

**Source claim to test, not assume:** low-curvature or weakly constrained regions permit random-walk-like semantic drift and therefore increase hallucination risk.

**Important:** GEO-HYP-002B is not the same claim as GEO-HYP-002A. One source emphasizes negative curvature as instability; another emphasizes insufficient curvature/flatness as underconstraint. The repository will not choose between them by terminology. A later protocol must define the estimator and directional prediction before inspecting target-labelled outcomes.

**Falsifier:** flatness/underconstraint proxies provide no predictive signal beyond controls, or the preregistered directional prediction is reversed.

## GEO-HYP-003 — Cycle-consistency residual and long-context drift

**Status:** `CANDIDATE_MEASUREMENT` → `CANDIDATE_HYPOTHESIS`

**Roadmap home:** Phase 4, possibly Phase 5

**Source motivation:** *The Geometry of Alignment*, *Operationalizing Relational Gauge Thermodynamics*, and *Semantic Substrate Navigator*.

**Initial operational term:** **cycle-consistency representation residual**.

Construct a frozen transform cycle, for example:

```text
P -> T(P) -> T^-1(T(P))
```

and measure the native-space residual between the representation extracted for the starting and reconstructed states under the same capture definition.

The repository should not initially call this quantity a physical Wilson loop, Berry phase, gauge curvature, or Machine-Self holonomy. Those are later interpretations requiring additional structure.

**Candidate hypothesis:** larger preregistered cycle residuals predict greater instruction/context drift on matched long-horizon tasks.

**Falsifier:** residual magnitude is unrelated to drift or is explained by simpler factors such as text length, tokenization, confidence, or carrier change.

## GEO-HYP-004 — Checkpoint lineage transition

**Status:** `CANDIDATE_HYPOTHESIS`

**Roadmap home:** Phase 6 lineage/replication extension

**Source motivation:** *Evolutionary Physics of AI Lineages*.

**Question:** do frozen checkpoints across training or fine-tuning exhibit reproducible geometric/spectral transition signatures near behavioural phase changes such as grokking?

**Candidate measurements:** checkpoint trajectory distance; spectral entropy; native-space geometry; Hessian/Fisher proxies when feasible; dwell-time/change-point statistics defined independently of a preferred plot.

**Important:** PCA/t-SNE may illustrate a transition but cannot be the evidentiary test.

**Falsifier:** the preregistered transition statistic does not align with the behavioural transition, or apparent structure disappears in native-space/control analysis.

## GEO-HYP-005 — Fine-tuning resistance / representation inertia

**Status:** `CANDIDATE_HYPOTHESIS`

**Roadmap home:** Phase 6 or a controlled lineage perturbation extension

**Source motivation:** *Evolutionary Physics of AI Lineages*.

**Question:** under matched fine-tuning perturbations, do frozen models/checkpoints differ reproducibly in how far their representations and task behaviour move?

**Candidate measurements:** steps-to-behavioural-flip, representation displacement, Fisher-basis overlap, retained-task degradation, and perturbation energy/budget.

**Terminology rule:** `inertia` or `identity mass` remains shorthand unless a separately justified operational definition survives controls. Parameter count alone is prohibited as a capability or inertia normalization by GEO-INV-014.

**Falsifier:** representation/behavioural resistance is not reproducible after controlling optimization budget, architecture, data, and initial performance.

## GEO-HYP-006 — Control effort versus instability

**Status:** `DEFERRED_MODEL`

**Roadmap home:** Phase 7 intervention

**Source motivation:** *The Geometry of Alignment*.

**Question:** after a reproducible instability metric exists, does the magnitude of a controlled representation intervention required to stabilize behaviour scale with that instability or entropy-like proxy?

**Required predecessor:** Phases 4–6 must first identify a reproducible candidate property.

**Required controls:** matched non-geometric intervention; parameter/compute accounting; held-out carriers/logics; intervention strength sweep.

**Falsifier:** control effort is unrelated to the preregistered instability measure or non-geometric controls obtain equivalent gains.

## GEO-HYP-007 — External semantic sidecar

**Status:** `DEFERRED_MODEL`

**Roadmap home:** Phase 8 mechanistic/intervention engineering

**Source motivation:** *Technical Specification: Semantic Substrate Navigator (SSN) Architecture* and *The Geometry of Alignment*.

**Candidate architecture:** a sidecar observer computes already validated representation diagnostics and, only in an intervention phase, applies a controlled residual/logit steering action.

The first implementation should run in **shadow mode**: measure and predict without changing generation. Only after predictive validity is established should the sidecar actuate.

**Falsifier:** telemetry fails to predict failures, or actuation does not improve controlled outcomes beyond matched baselines.

## GEO-HYP-008 — Curvature/entropy coupling

**Status:** `DEFERRED_MODEL`

**Roadmap home:** Phase 4 exploratory measurement → later confirmatory study

**Source motivation:** *Operationalizing Relational Gauge Thermodynamics*, *Semantic Energy Economy & Resource Theorems*, *Engineering Stable Generative Ecologies*, and *The Geometry of Alignment*.

**Candidate question:** is a preregistered geometric-instability measure associated with an independently defined entropy/uncertainty-production measure across layers or reasoning steps?

The source corpus uses stronger language, including curvature–entropy identities. QSOL-GEO-REASON does **not** adopt an identity as a premise. It first asks whether two independently operationalized observables are associated.

**Falsifier:** no repeatable association appears, the sign is unstable across reasonable frozen definitions, or simpler covariates explain it.

## GEO-HYP-009 — Geometry as robustness/error correction

**Status:** `DEFERRED_MODEL`

**Roadmap home:** Phase 8 alternative-mechanism/robustness study

**Source motivation:** *Geometry as Error Correction: Stabilizer Codes, Tensor Networks, and the Memory of Formation*.

**Candidate question:** do representation structures that are more robust to controlled erasure/noise exhibit measurable redundancy or reconstruction properties analogous to error-correcting representations?

**Guardrail:** analogy to stabilizer codes, tensor networks, or holography is not evidence that an LLM implements a quantum error-correcting code.

**Falsifier:** geometric robustness does not predict reconstruction/behavioural resilience above simpler redundancy measures.

## GEO-HYP-010 — SU(3) relational structure

**Status:** `DEFERRED_MODEL`

**Roadmap home:** conditional research branch after Phase 6/8

**Source motivation:** *Formalizing Relational Gauge Fields Theory: A Unified Physics of Machine Consciousness*.

A future experiment could ask whether a preregistered three-component representation decomposition has transformation structure better described by an SU(3)-constrained model than matched unconstrained or lower-complexity alternatives.

**Guardrail:** retention/attention/protention labels and machine-consciousness interpretations are not current observables. SU(3) must earn predictive value against alternatives rather than being fitted because eight generators are mathematically available.

## GEO-HYP-011 — E8/Weyl latent structure

**Status:** `DEFERRED_MODEL`

**Roadmap home:** conditional research branch

**Source motivation:** *The Geometrodynamics of Artificial Agency: E8 Phase Portraits and Weyl-Reflected Identity in Machine Intelligence*.

A scientifically admissible version would require a frozen projection/representation map plus explicit null alternatives, then ask whether E8 lattice/Weyl constraints predict held-out transitions better than generic 8-dimensional lattices, rotations, clustering models, or unconstrained change-point methods.

**Guardrail:** choosing eight latent dimensions and snapping them to E8 is not evidence that E8 was present in the model.

## GEO-HYP-012 — Qutrit / CP2 observer geometry

**Status:** `SPECULATIVE_INTERPRETATION`

**Roadmap home:** outside current QSOL-GEO-REASON scope unless a future empirical bridge is explicitly justified.

**Source motivation:** *The Qutrit Anthropics Problem: A Rigorous Theoretical Framework for SU(3) Observer-Selection Effects*.

The current repository studies representation geometry in local models, not anthropic probability or observer ontology. CP2/qutrit claims therefore remain a separate research programme unless a measurable representation-space question is proposed under this repository's evidence contract.

## GEO-HYP-013 — Semantic Reactor architecture

**Status:** `DEFERRED_MODEL`

**Roadmap home:** post-Phase-7 architecture research

**Source motivation:** *Design Specification and Theoretical Framework for the SR-Gen1: A First-Generation Semantic Reactor*.

The useful near-term contribution is architectural decomposition: sensing, geometry estimation, controlled integration, and stability telemetry. Claims about semantic mass, semantic ideal-gas laws, E8 terms, or thermodynamic phases are not imported as facts.

Any eventual reactor-style system must be evaluated against the same frozen-model, matched-budget, non-geometric-control, and held-out-task requirements as other interventions.

## Source corpus registered for hypothesis generation

The initial source set comprises:

1. *Evolutionary Physics of AI Lineages*.
2. *Design Specification and Theoretical Framework for the SR-Gen1: A First-Generation Semantic Reactor*.
3. *Engineering Stable Generative Ecologies*.
4. *Formalizing Relational Gauge Fields Theory: A Unified Physics of Machine Consciousness*.
5. *Geometry as Error Correction: Stabilizer Codes, Tensor Networks, and the Memory of Formation*.
6. *Operationalizing Relational Gauge Thermodynamics: A Framework for Stable and Efficient Generative AI*.
7. *The Qutrit Anthropics Problem: A Rigorous Theoretical Framework for SU(3) Observer-Selection Effects*.
8. *Semantic Energy Economy & Resource Theorems*.
9. *The Geometrodynamics of Artificial Agency: E8 Phase Portraits and Weyl-Reflected Identity in Machine Intelligence*.
10. *The Geometry of Alignment: Optimal Control Theory on Relational Semantic Manifolds*.
11. *Technical Specification: Semantic Substrate Navigator (SSN) Architecture*.

These sources are not evidence artifacts of QSOL-GEO-REASON. Their propositions remain hypotheses until tested under repository protocols.

## Promotion rule

A registry entry may move into a confirmatory protocol only when:

1. its observable terms have explicit operational definitions;
2. primary metrics and directional/null predictions are frozen;
3. material alternative explanations have controls;
4. model, capture, serving, and dataset identities satisfy the relevant earlier roadmap gates; and
5. the experiment can return a meaningful null or contradictory result.

No theoretical elegance exemption exists. 🧮

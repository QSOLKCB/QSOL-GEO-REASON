# AGENTS.md

Machine-oriented repository instructions for QSOL-GEO-REASON.

## Mandatory read order

Before editing research code, datasets, protocols, results, or claims:

1. `INVARIANTS.md`
2. `SCIENTIFIC-CONTRACT.md`
3. `MATH-SPEC.md` when geometry/math semantics are involved
4. `ROADMAP.md`
5. `README4AI.md`

## Authority

`INVARIANTS.md` and `SCIENTIFIC-CONTRACT.md` are the primary normative scientific contracts.

`MATH-SPEC.md` is normative for the exact mathematical meaning of Phase 1 geometry primitives and future formalization targets. It is subordinate to the primary scientific contracts and must not be interpreted to weaken them.

An agent MUST NOT weaken, bypass, reinterpret, or silently contradict these contracts to make an experiment easier, a result stronger, or a benchmark look better.

If a proposed task requires changing a normative rule or a `GEO-MATH-*` definition, make that change explicit in the pull request and explain:

- which invariant, contract clause, or mathematical ID changes;
- why the prior rule is insufficient;
- whether the change is semantic or clarifying;
- what new failure modes the change introduces;
- how backward comparability is preserved or intentionally broken;
- which frozen fixtures and future Lean theorem targets are affected.

## Mathematical discipline

Changes to `src/qsol_geo_reason/geometry.py`, numerical normalization, or alignment semantics MUST preserve `MATH-SPEC.md` or explicitly revise the affected mathematical contract.

Do not conflate exact-real theorems with floating-point implementation behavior. Future Lean 4 proofs are intended to establish the exact mathematical layer, not CPython, IEEE-754, JSON, SHA-256, Git, or LLM semantics without separate refinement evidence.

Undefined mathematical quantities must not be encoded as valid-looking zeros.

## Epistemic discipline

Agents MUST preserve these distinctions in code, docs, schemas, fixtures, and prose:

- simulation vs empirical model observation;
- observation vs association;
- association vs controlled perturbation;
- perturbation vs training intervention;
- intervention vs mechanistic explanation;
- evidence class vs replication status;
- semantic carrier vs logical structure;
- representation position vs finite-difference geometry;
- projected visualization vs native-space metric;
- benchmark accuracy vs demonstrated reasoning behaviour;
- output-token equivalence vs representation-space equivalence;
- exact mathematical semantics vs numerical implementation.

Never promote a claim beyond the evidence class actually produced. Replication status must be recorded separately and must not replace the evidence class.

## Experiment changes

For any experiment implementation, an agent SHOULD provide or update:

- a protocol;
- immutable/frozen inputs where practical;
- repository commit, protocol ID/version, run ID, and run-manifest identity;
- deterministic or explicitly stochastic run metadata;
- schemas for machine-readable results;
- positive and negative controls;
- tests for analysis primitives;
- provenance sufficient to reproduce the run;
- explicit primary and exploratory metrics.

If a metric, threshold, layer, subset, or visualization was selected after viewing target results, label it exploratory.

## Simulations

Synthetic/reference simulations are instrument-conformance tests.

Agents MUST NOT:

- present simulated trajectories as hidden states from a real model;
- infer local-model capability from successful synthetic recovery;
- use simulation success to close empirical roadmap items.

## Visualizations

Plots are explanatory surfaces, not evidentiary gates.

A result MUST remain expressible numerically without requiring visual interpretation. Projection methods such as PCA, UMAP, and t-SNE must be labelled with their transform and must not replace native-space analysis.

## Model comparisons

Record material confounders. At minimum, consider:

- model family;
- architecture;
- parameter count;
- tokenizer;
- quantization;
- dtype;
- prompting/template;
- context length;
- generation settings;
- serving backend/runtime;
- dataset exposure risk;
- compute/training budget.

Dataset exposure assessments must record separate outcome, confidence, and supporting basis. `unknown` must not be converted to `unexposed` merely because no exposure evidence was found.

If material differences are not controlled, describe the comparison as confounded rather than causal.

## Serving backends

For hidden-state or geometry experiments, the serving runtime is part of the instrument.

Agents MUST NOT treat matching generated tokens or final answers across runtimes as proof that hidden states or trajectory geometry are equivalent.

Establish a canonical capture backend first. Optimized, quantized, hybrid CPU/GPU, cached, or otherwise adaptive serving paths may replace it only after a serving-equivalence protocol supports that substitution for the claim being made. Otherwise, backend identity remains an explicit experimental variable.

Record material serving choices including backend/version, kernels or attention implementation, dtype/quantization, device placement/offloading, prefix/KV/recurrent-state reuse, and prefill/decode/replayed-prefix phase where relevant.

## Results and roadmap

Do not mark a roadmap research milestone complete merely because code exists. Experimental milestones require the evidence artifact named by the roadmap or protocol.

Negative and null results are first-class outputs and must not be removed solely because they weaken the motivating hypothesis.

A frozen release/tag must never be rewritten to retrofit later mathematics or evidence. Formalization against a frozen release is a new evidence layer, not a retroactive modification of the release.

## Scope of PR #1

The initial foundation PR is documentation/contract only. It establishes the rules under which later simulation and local-model experiments will operate. It makes no empirical geometric-reasoning claim.

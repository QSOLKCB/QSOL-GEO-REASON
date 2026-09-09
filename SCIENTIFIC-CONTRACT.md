# Scientific Contract

This document defines how QSOL-GEO-REASON converts experiments into claims.

It is normative together with `INVARIANTS.md`.

## 1. Research posture

QSOL-GEO-REASON investigates whether local language-model reasoning exhibits reproducible geometric structure in representation space and whether controlled geometric interventions can improve reasoning.

The repository does not assume that the motivating hypothesis is true.

The minimum acceptable research outcome is a reproducible result that narrows the hypothesis space, including a null result.

## 2. Evidence classes and replication status

Every substantive result SHOULD be assigned the strongest **evidence class** actually supported by the experiment that produced it.

Replication is recorded on a separate axis. It is not a stronger evidence class and must not erase the experimental type of the underlying result.

### Evidence classes

#### `SIMULATION`

A property was demonstrated in synthetic/reference data generated from a known construction.

Permits claims about analysis-tool correctness or recovery of known geometry.

Does not permit claims about real LLM behaviour.

#### `OBSERVATION`

A property was measured in one or more frozen model runs under a specified extraction protocol.

Permits descriptive claims such as “order-1 trajectory similarity was higher within logic class than within carrier class under protocol P.”

Does not establish causation.

#### `ASSOCIATION`

A statistical relationship was demonstrated between a geometric quantity and another variable such as logic class, correctness, model size, or layer.

Requires an explicit comparison statistic and uncertainty or repeatability analysis appropriate to the design.

Does not establish mechanism.

#### `PERTURBATION`

A controlled input intervention produced a reproducible differential response compared with matched controls.

This supports sensitivity claims about the tested intervention under the tested protocol.

It does not, by itself, establish that the observed geometry causes reasoning.

#### `INTERVENTION`

A model/training/representation intervention intentionally changed a geometric property and produced a controlled downstream change relative to a matched baseline.

This may support causal claims about the intervention under the tested conditions.

It does not automatically establish a complete mechanism of reasoning.

#### `MECHANISM`

A mechanistic claim requires converging evidence that identifies a specific internal structure or process, predicts behaviour under intervention, survives relevant ablations, and outperforms plausible alternative explanations.

This is deliberately a high bar.

### Replication status

A result record SHOULD separately carry `replication_status` using one of:

- `not_attempted`: no independent or materially varied replication has yet been evaluated;
- `replicated`: the preregistered result reproduced under the declared replication scope;
- `failed`: the preregistered result failed to reproduce under the declared replication scope;
- `mixed`: materially varied replications produced inconsistent outcomes.

A replication record must identify what changed, for example model, model size, seed set, carrier set, dataset split, serving implementation, or independent implementation, while preserving the relevant protocol contract.

Example: a controlled perturbation reproduced on another model remains `evidence_class: PERTURBATION` and may additionally carry `replication_status: replicated`.

Material deviations from the original protocol must be documented rather than hidden inside the replication label.

## 3. Operational definitions

### Representation state

A representation state `z_t` is the vector obtained by an extraction function:

`z_t = E(model, layer, token_span, context, pooling, transform)`

All arguments that materially affect `E` must be provenance-recorded.

### Trajectory

A trajectory is an ordered sequence:

`gamma = (z_0, z_1, ..., z_T)`

The step index must correspond to an explicit unit such as reasoning step, accumulated text segment, token block, or synthetic time step.

### Order-k finite difference

Order 0 is the original trajectory. Higher orders are repeated finite differences under a specified convention.

Terms such as “velocity” and “acceleration” are permitted as shorthand only when the document makes clear that these are finite-difference analogues in representation space.

### Curvature

Any curvature statistic must name its definition, dimensional assumptions, numerical handling, and alignment procedure. No unique cognitive interpretation is assumed.

### Carrier-invariant logic item

An item belongs to a logic class and a semantic carrier class separately. A valid carrier-invariance study must be able to vary one without silently redefining the other.

### Load-bearing perturbation

An intervention that changes a premise/dependency whose change should alter the correct downstream conclusion according to the formal item specification.

### Matched surface control

A perturbation designed to change similar amounts of lexical or semantic surface material while preserving the target logical consequence.

## 4. Primary comparison pattern

A core perturbation experiment SHOULD estimate whether geometry responds more strongly or differently to a load-bearing change than to a matched non-load-bearing change.

For a trajectory distance or response function `D`:

`R_causal = D(gamma_base, gamma_causal)`

`R_control = D(gamma_base, gamma_control)`

A motivating contrast may be:

`Delta_response = R_causal - R_control`

The sign, scale, null hypothesis, aggregation, and uncertainty procedure must be defined by the protocol rather than inferred from a preferred plot.

## 5. Confirmatory versus exploratory analysis

### Confirmatory

A confirmatory result requires the relevant protocol choices to be frozen before inspecting the target-labelled outcome.

At minimum, freeze or version:

- dataset split and dataset provenance;
- model/revision;
- dataset-exposure assessment when the claim compares models, families, scales, or training interventions;
- extraction layers;
- pooling / token-span rules;
- geometry metrics;
- alignment/resampling rule;
- primary comparison statistic;
- inclusion/exclusion criteria;
- multiple-comparison handling where relevant.

### Exploratory

Post-hoc layer searches, metric searches, projections, subgroup discoveries, and threshold tuning are allowed and encouraged, but must be labelled exploratory.

Exploratory findings may generate a later frozen confirmatory protocol.

## 6. Controls

Experiments SHOULD include the strongest practical controls for the claim being tested.

Possible controls include:

- same logic, different carrier;
- same carrier, different logic;
- causal perturbation, matched surface perturbation;
- correct and incorrect reasoning traces;
- shuffled step order;
- random-vector or synthetic null trajectories;
- alternative pooling rules;
- alternative layers;
- prompt/template controls;
- model-size or family controls;
- newly generated or otherwise exposure-resistant held-out evaluation material;
- serving-backend equivalence controls;
- label permutation tests.

Controls are not interchangeable. A protocol must explain what failure mode each control addresses.

## 7. Projection policy

Low-dimensional projections are for visualization and exploratory structure discovery.

Any projection figure must record:

- method;
- fitted data scope;
- component/dimension count;
- random seed where applicable;
- whether labels influenced selection or presentation.

A projection may not substitute for the native-space statistic supporting the claim.

## 8. Model comparison and dataset-exposure contract

Cross-model comparisons must record known material differences.

A strong parameter-efficiency comparison SHOULD use the same:

- model family when possible;
- evaluation data;
- prompts/templates;
- generation policy;
- scoring implementation;
- hidden-state extraction definition where architecture permits;
- quantization policy or an explicit quantization ablation.

For every model/dataset pair used in a cross-model claim, the protocol SHOULD assess evaluation exposure through pretraining, post-training, fine-tuning, benchmark use, or other known ingestion routes.

Exposure must be represented with separate fields:

- `exposure_outcome`: `exposed`, `unexposed`, or `unknown`;
- `exposure_confidence`: `verified`, `high`, `medium`, `low`, or `unknown`;
- `exposure_basis`: citations, model-card statements, training-corpus records, benchmark disclosures, or other provenance supporting the assessment.

`unexposed` is always scoped to the named model revision, dataset/version, and assessed ingestion routes. It must not be used as an absolute claim that no semantically related material ever appeared in training. If affirmative evidence for scoped non-exposure is unavailable, use `unknown` rather than inferring `unexposed` from silence.

Differential or uncontrolled exposure is a confounder and must limit claims about scale, family generality, parameter efficiency, or geometric superiority.

Training-intervention comparisons SHOULD additionally match data and optimization budgets and report all added trainable parameters.

Uncontrolled differences are confounders, not footnotes.

## 9. Serving and backend equivalence contract

The serving stack is part of the measurement instrument whenever hidden states or representation geometry are studied.

Matching generated tokens, final answers, or benchmark scores across two runtimes does **not** establish representation-space equivalence.

A canonical reference capture SHOULD therefore be established using the simplest sufficiently transparent backend before optimized serving paths are admitted as measurement-equivalent substitutes.

A serving-equivalence study SHOULD compare the canonical backend and candidate backend under the same model revision and frozen inputs, including where applicable:

- representation positions `z_t`;
- order-1 and higher finite differences;
- preregistered curvature statistics;
- generated-token equality or divergence;
- layer and token-span identity;
- numeric tolerances and uncertainty appropriate to dtype and hardware.

Every empirical run must record material serving variables, including where applicable:

- serving backend and version;
- attention/kernel implementation;
- dtype and quantization;
- CPU/GPU/device placement and offloading policy;
- KV/prefix/recurrent-state reuse policy;
- prefill versus decode phase;
- cache/state reuse after prompt edits;
- hardware and measured resource characteristics relevant to execution.

Until equivalence is demonstrated for the claim being made, a changed serving backend remains an experimental factor or confounder.

Resource adaptation may change latency, throughput, memory residency, or scheduling. It must not silently redefine the scientific object being measured.

For MoE or hybrid serving, exact-output claims about CPU/GPU split execution, expert caching, or other heterogeneous scheduling must be verified rather than inferred from architecture alone.

## 10. Provenance contract

Every empirical result artifact SHOULD be traceable to:

- repository commit;
- protocol ID/version;
- run ID and run-manifest identity;
- dataset ID/version/hash;
- dataset-generation provenance;
- dataset-exposure assessment where relevant;
- model identifier and immutable revision where available;
- tokenizer identifier/revision;
- serving backend/version and material runtime policy;
- environment/runtime versions;
- hardware/device metadata;
- quantization and dtype;
- random seeds;
- generation parameters;
- extraction settings;
- analysis settings;
- output hashes.

The project should prefer content-addressed or hash-bound artifacts where practical.

## 11. Result records

A future machine-readable result record SHOULD contain:

- `evidence_class`;
- `replication_status`;
- `replication_scope` when replication is attempted;
- `replicates_result_ids` when applicable;
- `protocol_id`;
- `run_id`;
- `run_manifest_id`;
- `repository_commit`;
- `evidence_artifacts`;
- `primary_metrics`;
- `control_metrics`;
- `uncertainty`;
- `dataset_exposure_assessment` when relevant, containing outcome, confidence, and basis;
- `serving_backend` and material serving settings for empirical model runs;
- `result_status` (`supports`, `null`, `contradicts`, `inconclusive`);
- `limitations`;
- `provenance`.

A result record must not omit a null or contradictory primary outcome merely because an exploratory secondary metric is positive.

## 12. Simulation contract

Reference simulations exist to answer questions such as:

- Does the implementation recover a known first-order alignment?
- Does the curvature implementation recover a known curved path?
- Does resampling preserve the expected synthetic comparison?
- Can a negative control produce the expected null?
- Are result records deterministic under a fixed synthetic recipe?

They do not answer whether a local model reasons geometrically.

Synthetic artifacts must be visibly labelled `SIMULATION`.

A software-only Phase 2 capture-contract fixture driven by a fake backend is also `SIMULATION` evidence about instrumentation behaviour, never `OBSERVATION` evidence about an LLM.

## 13. Falsification contract

Each major hypothesis must eventually have a stated observation that would count against it.

Examples:

- logical classes fail to predict trajectory geometry above matched carrier/control structure;
- apparent geometry disappears under native-space tests after projection artefacts are removed;
- causal premise perturbations are not distinguishable from matched surface perturbations;
- geometric training changes geometry without improving controlled reasoning outcomes;
- apparent small-model gains vanish under matched parameter, data, compute, or exposure accounting;
- a claimed serving-equivalent backend materially changes preregistered representation geometry beyond its tolerance.

The repository must preserve these outcomes if observed.

## 14. Current evidence status

Phase 1 is complete and frozen without changing the empirical claim ceiling:

- immutable numerical release: `v0.1.0`, evidence class `SIMULATION`;
- immutable Lean formal evidence release: `v0.2.0`, a separate exact-mathematics proof layer rather than an empirical LLM evidence class;
- all twelve frozen Lean theorem targets: implemented, reviewed, merged, and frozen in `v0.2.0`;
- empirical local-model runs: none at the start of Phase 2A;
- empirical `OBSERVATION` results: none until a production `GEO-CAP-001` run is executed;
- `ASSOCIATION` results: none;
- `PERTURBATION` results: none;
- `INTERVENTION` results: none;
- supported `MECHANISM` claims: none;
- serving-equivalence studies: none;
- geometric training interventions: none.

PR #4 introduces the canonical Phase 2A capture instrument, schemas, and software-only contract fixture. Their existence does not satisfy the empirical Phase 2A evidence gate. The repository still assumes no geometric, gauge, thermodynamic, SU(3), E8, qutrit, or other mechanistic theory of LLM reasoning.

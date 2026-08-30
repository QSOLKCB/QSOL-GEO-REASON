# Scientific Contract

This document defines how QSOL-GEO-REASON converts experiments into claims.

It is normative together with `INVARIANTS.md`.

## 1. Research posture

QSOL-GEO-REASON investigates whether local language-model reasoning exhibits reproducible geometric structure in representation space and whether controlled geometric interventions can improve reasoning.

The repository does not assume that the motivating hypothesis is true.

The minimum acceptable research outcome is a reproducible result that narrows the hypothesis space, including a null result.

## 2. Claim classes

Every substantive result SHOULD be assigned the strongest claim class actually supported by its evidence.

### `SIMULATION`

A property was demonstrated in synthetic/reference data generated from a known construction.

Permits claims about analysis-tool correctness or recovery of known geometry.

Does not permit claims about real LLM behaviour.

### `OBSERVATION`

A property was measured in one or more frozen model runs under a specified extraction protocol.

Permits descriptive claims such as “order-1 trajectory similarity was higher within logic class than within carrier class under protocol P.”

Does not establish causation.

### `ASSOCIATION`

A statistical relationship was demonstrated between a geometric quantity and another variable such as logic class, correctness, model size, or layer.

Requires an explicit comparison statistic and uncertainty or repeatability analysis appropriate to the design.

Does not establish mechanism.

### `PERTURBATION`

A controlled input intervention produced a reproducible differential response compared with matched controls.

This supports sensitivity claims about the tested intervention under the tested protocol.

It does not, by itself, establish that the observed geometry causes reasoning.

### `REPLICATION`

A previously defined result reproduced under a new model, model size, seed set, carrier set, dataset split, or implementation while preserving the relevant protocol contract.

Material deviations must be documented.

### `INTERVENTION`

A model/training/representation intervention intentionally changed a geometric property and produced a controlled downstream change relative to a matched baseline.

This may support causal claims about the intervention under the tested conditions.

It does not automatically establish a complete mechanism of reasoning.

### `MECHANISM`

A mechanistic claim requires converging evidence that identifies a specific internal structure or process, predicts behaviour under intervention, survives relevant ablations, and outperforms plausible alternative explanations.

This is deliberately a high bar.

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

## 8. Model comparison contract

Cross-model comparisons must record known material differences.

A strong parameter-efficiency comparison SHOULD use the same:

- model family when possible;
- evaluation data;
- prompts/templates;
- generation policy;
- scoring implementation;
- hidden-state extraction definition where architecture permits;
- quantization policy or an explicit quantization ablation.

For every model/dataset pair used in a cross-model claim, the protocol SHOULD assess evaluation exposure through pretraining, post-training, fine-tuning, benchmark use, or other known ingestion routes. Exposure status should be recorded as `known`, `plausible`, `unlikely`, or `unknown`, together with supporting provenance where available.

`unknown` does not mean `unexposed`. Differential or uncontrolled exposure is a confounder and must limit claims about scale, family generality, parameter efficiency, or geometric superiority.

Training-intervention comparisons SHOULD additionally match data and optimization budgets and report all added trainable parameters.

Uncontrolled differences are confounders, not footnotes.

## 9. Provenance contract

Every empirical result artifact SHOULD be traceable to:

- repository commit;
- protocol ID/version;
- dataset ID/version/hash;
- dataset-generation provenance;
- dataset-exposure assessment where relevant;
- model identifier and immutable revision where available;
- tokenizer identifier/revision;
- environment/runtime versions;
- hardware/device metadata;
- quantization and dtype;
- random seeds;
- generation parameters;
- extraction settings;
- analysis settings;
- output hashes.

The project should prefer content-addressed or hash-bound artifacts where practical.

## 10. Result records

A future machine-readable result record SHOULD contain:

- `claim_class`;
- `protocol_id`;
- `run_id`;
- `evidence_artifacts`;
- `primary_metrics`;
- `control_metrics`;
- `uncertainty`;
- `dataset_exposure_assessment` when relevant;
- `result_status` (`supports`, `null`, `contradicts`, `inconclusive`);
- `limitations`;
- `provenance`.

A result record must not omit a null or contradictory primary outcome merely because an exploratory secondary metric is positive.

## 11. Simulation contract

Reference simulations exist to answer questions such as:

- Does the implementation recover a known first-order alignment?
- Does the curvature implementation recover a known curved path?
- Does resampling preserve the expected synthetic comparison?
- Can a negative control produce the expected null?
- Are result records deterministic under a fixed synthetic recipe?

They do not answer whether a local model reasons geometrically.

Synthetic artifacts must be visibly labelled `SIMULATION`.

## 12. Falsification contract

Each major hypothesis must eventually have a stated observation that would count against it.

Examples:

- logical classes fail to predict trajectory geometry above matched carrier/control structure;
- apparent geometry disappears under native-space tests after projection artefacts are removed;
- causal premise perturbations are not distinguishable from matched surface perturbations;
- geometric training changes geometry without improving controlled reasoning outcomes;
- apparent small-model gains vanish under matched parameter, data, compute, or exposure accounting.

The repository must preserve these outcomes if observed.

## 13. Current evidence status

At PR #1 / foundation stage:

- claim class: none;
- empirical runs: none;
- reference simulations: none;
- geometric training interventions: none;
- supported mechanism claims: none.

The repository currently establishes only the rules under which later evidence will be generated and interpreted.

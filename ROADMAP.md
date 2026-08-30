# QSOL-GEO-REASON Research Roadmap

This roadmap is staged to prevent the project from making stronger claims than its instrumentation and controls support.

The governing rule is:

`contract -> validate instruments -> observe -> perturb -> replicate -> intervene -> ablate -> claim`

A later phase may prototype early, but its scientific claims must not outrun the evidence gates of the earlier phases.

---

## Phase 0 — Foundation and invariants

**Goal:** lock the epistemic boundary before generating attractive results.

- [x] Establish human-facing `README.md`.
- [x] Establish AI-facing `README4AI.md`.
- [x] Establish machine operating rules in `AGENTS.md`.
- [x] Define stable invariant IDs in `INVARIANTS.md`.
- [x] Define claim classes and experiment rules in `SCIENTIFIC-CONTRACT.md`.
- [x] Explicitly separate simulation from empirical evidence.
- [x] Explicitly separate visualization from evidentiary metrics.
- [x] Define falsification as a first-class requirement.
- [x] Establish the research roadmap.

**Evidence gate:** none. Phase 0 creates no empirical claim.

**Candidate delivery:** PR #1.

---

## Phase 1 — Deterministic geometry reference simulation

**Goal:** prove that the measurement machinery can recover known geometry before touching model hidden states.

- [ ] Define a machine-readable synthetic trajectory recipe schema.
- [ ] Implement deterministic generators for straight, curved, branching, noisy, and null trajectories.
- [ ] Implement known same-logic/different-carrier analogue trajectories by construction.
- [ ] Implement known causal/control perturbation analogues by construction.
- [ ] Implement order-0, order-1, order-2, and configurable order-k finite differences.
- [ ] Implement path length and cosine alignment primitives.
- [ ] Implement at least one explicitly defined curvature statistic, initially Menger curvature if retained.
- [ ] Implement deterministic alignment/resampling with edge-case tests.
- [ ] Add degenerate-path, zero-length, repeated-point, short-sequence, and dimension-mismatch tests.
- [ ] Produce frozen reference fixtures with expected numerical outputs.
- [ ] Hash-bind simulation recipe, implementation revision, and result artifacts.
- [ ] Mark every generated result record `SIMULATION`.

**Evidence gate:** numerical recovery of known synthetic properties within preregistered tolerances.

**Non-claim:** successful completion says nothing about real LLM reasoning.

**Candidate delivery:** PR #2.

---

## Phase 2 — Local-model hidden-state capture harness

**Goal:** obtain auditable representation trajectories from local models without yet claiming geometric structure.

- [ ] Select an initial fully local Hugging Face-compatible model small enough for routine workstation runs.
- [ ] Freeze model identifier and immutable revision/hash where available.
- [ ] Record tokenizer revision.
- [ ] Implement hidden-state capture by layer.
- [ ] Define step segmentation independently of analysis outcome.
- [ ] Implement pooling modes such as step mean, last token, context mean, and explicitly bounded context-aware pooling.
- [ ] Record cumulative versus isolated context mode.
- [ ] Record dtype and quantization.
- [ ] Record generation parameters and seeds.
- [ ] Record runtime/library/device metadata.
- [ ] Define run-manifest and captured-trajectory schemas.
- [ ] Verify deterministic replay where the backend permits it.
- [ ] Explicitly record irreducible nondeterminism where it does not.
- [ ] Add small frozen capture fixtures for CI that do not require shipping restricted model weights.

**Evidence gate:** capture provenance is sufficient to identify exactly what vector each trajectory point represents.

**Claim ceiling:** `OBSERVATION` only after an actual model run is performed.

---

## Phase 3 — Carrier-invariant reasoning dataset

**Goal:** separate logical structure from semantic surface form.

- [ ] Define formal logic-template schema.
- [ ] Define semantic-carrier schema.
- [ ] Define independent `logic_id` and `carrier_id` identities.
- [ ] Build initial logic families using formally checkable inference structures.
- [ ] Instantiate each logic family across several semantically distant carriers.
- [ ] Add synthetic-symbol carriers with minimal world knowledge.
- [ ] Add same-carrier/different-logic negative controls.
- [ ] Add same-logic/different-carrier positive structural controls.
- [ ] Add answer-correctness labels independently of geometry.
- [ ] Add shuffled-step and broken-logic controls.
- [ ] Audit accidental lexical cues that reveal logic class.
- [ ] Record dataset-generation provenance, including any model-assisted generation or rewriting.
- [ ] Assess whether proposed evaluation material may already be present in candidate models' pretraining, post-training, benchmark, or fine-tuning exposure.
- [ ] Prefer newly generated or otherwise exposure-resistant confirmatory items where feasible, while retaining reproducible public baselines separately.
- [ ] Freeze a first dataset version before confirmatory analysis.

**Evidence gate:** logic and carrier can be varied independently enough to support the planned comparisons, with material exposure risks documented.

---

## Phase 4 — Observational geometry study

**Goal:** determine whether logical structure predicts trajectory geometry beyond carrier effects in frozen local models.

- [ ] Preregister primary layers or a layer-aggregation strategy.
- [ ] Preregister primary pooling rule.
- [ ] Preregister primary order(s) and similarity metric(s).
- [ ] Compare clustering/similarity by logic class.
- [ ] Compare clustering/similarity by carrier class.
- [ ] Compare geometry by answer correctness.
- [ ] Perform native-space tests before viewing projection-space narratives.
- [ ] Generate PCA or other projections only as explanatory artifacts.
- [ ] Use label permutation or equivalent null analysis where appropriate.
- [ ] Quantify uncertainty/repeatability across item families and seeds.
- [ ] Record both positive and null primary outcomes.

**Primary motivating hypothesis:** higher-order trajectory structure may align more strongly by logic than absolute representation position does across semantic carriers.

**Falsifier:** logic-conditioned geometry does not exceed matched carrier/control structure under the frozen protocol.

**Claim ceiling:** `OBSERVATION` / `ASSOCIATION`.

---

## Phase 5 — Controlled perturbation study

**Goal:** test whether trajectory geometry is selectively sensitive to load-bearing reasoning changes rather than generic textual changes.

- [ ] Define formal load-bearing premise perturbations.
- [ ] Define matched surface/distractor perturbations.
- [ ] Validate that causal perturbations actually change the correct conclusion.
- [ ] Validate that control perturbations preserve the relevant conclusion.
- [ ] Freeze a trajectory-distance/response statistic.
- [ ] Estimate `R_causal` and `R_control`.
- [ ] Estimate the differential response `Delta_response = R_causal - R_control` or a preregistered alternative.
- [ ] Repeat across logic families and carriers.
- [ ] Test whether geometric response predicts answer updating.
- [ ] Retain cases where the model updates its answer without the predicted geometry and vice versa.

**Falsifier:** causal premise changes are not reliably distinguishable from matched surface changes in the preregistered geometric response.

**Claim ceiling:** `PERTURBATION`.

---

## Phase 6 — Cross-model replication and scaling

**Goal:** determine what aspects of the observed geometry generalize across models and scale.

- [ ] Select multiple sizes within one model family where feasible.
- [ ] Add at least one different model family.
- [ ] Match prompting and evaluation policy as closely as architecture permits.
- [ ] Record tokenizer and architecture differences explicitly.
- [ ] Audit dataset-exposure risk separately for every compared model, including known or plausible pretraining, post-training, fine-tuning, and benchmark exposure.
- [ ] Classify exposure status for each model/dataset pair as `known`, `plausible`, `unlikely`, or `unknown`, with supporting provenance where available.
- [ ] Treat uncontrolled or unknown differential exposure as a comparison confounder rather than evidence of scale, family, or geometric superiority.
- [ ] Where feasible, repeat key comparisons on newly generated or exposure-resistant held-out logic/carrier material.
- [ ] Run quantization sensitivity/ablation where practical.
- [ ] Compare logic/carrier separation across layers and sizes.
- [ ] Compare perturbation sensitivity across sizes.
- [ ] Estimate whether any metric changes monotonically with model scale.
- [ ] Identify geometry that appears family-specific rather than general.
- [ ] Re-run frozen primary analyses without retuning thresholds per model.

**Falsifier:** apparent geometric signatures fail to replicate outside the original model/protocol configuration, or an apparent cross-model advantage cannot be separated from material exposure differences.

**Claim ceiling:** `REPLICATION` plus the supported lower classes.

---

## Phase 7 — Geometric training intervention

**Goal:** test whether deliberately shaping representation trajectories improves controlled reasoning behaviour.

Only begin confirmatory work after Phases 4–6 identify a reproducible candidate geometric property.

- [ ] Define a baseline training/fine-tuning protocol.
- [ ] Define one geometric auxiliary objective at a time before combining objectives.
- [ ] Candidate: align order-1 trajectory directions across carrier-equivalent logic items.
- [ ] Candidate: separate geometry for formally different logic under matched carriers.
- [ ] Candidate: increase differential sensitivity to load-bearing versus control perturbations.
- [ ] Report every added trainable parameter.
- [ ] Match data and optimization budget against baseline where practical.
- [ ] Evaluate on held-out logic/carrier combinations.
- [ ] Evaluate standard reasoning performance separately from geometry metrics.
- [ ] Test whether geometry changes without reasoning improvement.
- [ ] Test whether reasoning improves without the targeted geometry change.
- [ ] Compare against a non-geometric auxiliary-loss control.

**Falsifiers include:**

- targeted geometry changes but controlled reasoning does not improve;
- reasoning gains are matched by a non-geometric control;
- gains disappear under held-out carriers/logics;
- gains are explained by added parameters or compute.

**Claim ceiling:** `INTERVENTION`.

---

## Phase 8 — Mechanistic ablation and alternative explanations

**Goal:** distinguish useful correlates from candidate internal mechanisms.

- [ ] Identify the strongest candidate internal geometric structure from prior phases.
- [ ] Develop targeted activation/representation interventions where technically justified.
- [ ] Test necessity via ablation or disruption.
- [ ] Test sufficiency or partial sufficiency via controlled steering/intervention where possible.
- [ ] Compare predictions against simpler alternatives such as lexical similarity, confidence, token position, sequence length, and generic activation magnitude.
- [ ] Test whether the proposed mechanism predicts failures, not only successes.
- [ ] Document unresolved alternative explanations.

**Claim ceiling:** `MECHANISM` only if the high bar in `SCIENTIFIC-CONTRACT.md` is satisfied.

---

## Phase 9 — Reproducible local reference run and release

**Goal:** freeze a complete end-to-end research artifact after the methodology has survived the preceding stages.

- [ ] Select a stable reference local model and immutable revision.
- [ ] Freeze dataset, protocol, code, and environment manifest.
- [ ] Execute a complete local reference run.
- [ ] Store machine-readable results and artifact hashes.
- [ ] Generate human-readable report from the exact machine-readable evidence.
- [ ] Verify that every headline claim resolves to an evidence artifact.
- [ ] Publish limitations and null results alongside positive results.
- [ ] Tag a release only after exact-head review and reproducibility checks.

---

## Deferred / conditional research branches

These are deliberately not assumed to be useful until earlier evidence warrants them:

- [ ] intrinsic-dimension estimation of reasoning trajectories;
- [ ] topology/manifold-learning analysis beyond simple trajectory metrics;
- [ ] tangent-space transport across semantic carriers;
- [ ] attractor-state analysis;
- [ ] information-geometric metrics;
- [ ] geometric regularization during pretraining rather than fine-tuning;
- [ ] latent-step reasoning without explicit textual intermediate steps;
- [ ] formal verification of analysis invariants where it reduces real ambiguity;
- [ ] integration with other QSOL repositories only after this repository has stable interfaces and evidence classes.

## Roadmap completion rule

A checkbox representing an experimental result is complete only when its required evidence artifact exists and satisfies the current scientific contract.

Code existence alone does not complete an empirical milestone.

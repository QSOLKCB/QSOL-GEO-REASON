# QSOL-GEO-REASON Research Roadmap

This roadmap is staged to prevent the project from making stronger claims than its instrumentation and controls support.

The governing research workflow is:

`contract -> exact math -> validate instruments -> canonical capture -> validate serving -> observe -> perturb -> replicate -> intervene -> ablate -> claim`

This is a workflow, not a claim ladder. Replication status remains orthogonal to evidence class.

A later phase may prototype early, but its scientific claims must not outrun the evidence gates of the earlier phases.

---

## Phase 0 — Foundation and invariants

**Goal:** lock the epistemic boundary before generating attractive results.

- [x] Establish human-facing `README.md`.
- [x] Establish AI-facing `README4AI.md`.
- [x] Establish machine operating rules in `AGENTS.md`.
- [x] Define stable invariant IDs in `INVARIANTS.md`.
- [x] Define evidence classes, replication status, and experiment rules in `SCIENTIFIC-CONTRACT.md`.
- [x] Explicitly separate simulation from empirical evidence.
- [x] Explicitly separate visualization from evidentiary metrics.
- [x] Explicitly separate replication status from evidence class.
- [x] Explicitly treat serving backend as part of the measurement instrument.
- [x] Define falsification as a first-class requirement.
- [x] Establish the research roadmap.

**Evidence gate:** none. Phase 0 creates no empirical claim.

**Delivery:** PR #1.

---

## Phase 1 — Deterministic geometry reference simulation and exact formal layer

**Status:** **complete and frozen in two immutable releases: `v0.1.0` for the numerical mathematical kernel and `v0.2.0` for the separate Lean 4 formal evidence layer. Phase 2A instrumentation begins in PR #4; no empirical local-model result exists yet.**

**Goal:** prove that the measurement machinery can recover known geometry before touching model hidden states, while freezing the exact mathematical semantics before later model work depends on them.

- [x] Define the exact Phase 1 mathematical semantics and stable theorem targets in `MATH-SPEC.md`.
- [x] Define a machine-readable synthetic trajectory recipe schema.
- [x] Implement deterministic generators for straight, curved, branching, noisy, and null trajectories.
- [x] Implement known same-logic/different-carrier analogue trajectories by construction.
- [x] Implement known causal/control perturbation analogues by construction.
- [x] Implement order-0, order-1, order-2, and configurable order-k finite differences.
- [x] Implement path length and cosine alignment primitives.
- [x] Implement an explicitly defined curvature statistic using Menger curvature.
- [x] Implement deterministic alignment/resampling with edge-case tests.
- [x] Freeze the pairwise arc-length count rule as `m = max(n_X, n_Y)` in the exact mathematical contract.
- [x] Add degenerate-path, zero-length, repeated-point, short-sequence, and dimension-mismatch tests.
- [x] Add tiny/large-scale numerical regressions so absolute magnitude alone cannot erase valid geometry.
- [x] Evaluate represented binary64 Menger `kappa^2` with exact dyadic-rational arithmetic so exact collinearity and genuinely near-collinear nonzero curvature remain distinguishable without an epsilon.
- [x] Canonicalize emitted coordinates relative to each trajectory origin so representable local geometry survives large absolute coordinate offsets.
- [x] Preserve ULP-sized canonical offsets at binary64 spacing boundaries by escalating the origin to 17-digit round-trip precision only when ordinary evidence precision would erase a nonzero displacement.
- [x] Verify every nonzero canonical consecutive displacement survives reconstruction, including later steps that do not involve the origin.
- [x] Derive trajectory metrics and cross-trajectory comparisons from the exact emitted coordinate arrays so serialized points and metrics cannot describe different trajectories.
- [x] Reject finite-difference subtraction that overflows to a non-finite binary64 value before it reaches downstream metrics.
- [x] Reject undefined empty finite-difference comparisons rather than encoding them as zero.
- [x] Produce frozen reference fixtures with expected numerical outputs.
- [x] Hash-bind simulation recipe, implementation revision, and result artifacts.
- [x] Mark every generated Phase 1 result record `evidence_class: SIMULATION`.

**Numerical evidence gate:** **PASS on immutable release `v0.1.0`** — numerical recovery of known synthetic properties within the preregistered tolerances in `GEO-SIM-001`, plus metadata-consistent byte-identical frozen replay across Python 3.11, 3.12, and 3.13.

**Immutable numerical release:** `v0.1.0 — Phase 1 Mathematical Kernel`.

**Immutable numerical release commit:** `1b5ab8b4543b20cdb6d439f7ad215c08e698188f`.

**Bound implementation / mathematical kernel:** `5f45b5e69bcab890a757fffa491cf787f92a5bea`.

**Bound result artifact:** `c542bce987d31350b4904122e5ec02ef026715f51a1fe21ee184a452cc67a583`.

**Regression suite:** **48 tests** before the cross-version matrix.

**Non-claim:** successful completion says nothing about real LLM reasoning.

### Phase 1 release/formalization gate

The numerical kernel is frozen. The Lean work is additive evidence and does not alter that release identity.

- [x] Obtain a fresh Codex review on the exact PR #2 head with no unresolved correctness findings.
- [x] Confirm the Python 3.11/3.12/3.13 conformance matrix on the latest frozen numerical candidate.
- [x] Merge PR #2 without changing the reviewed mathematical semantics.
- [x] Create immutable Phase 1 release `v0.1.0` and bind it to commit `1b5ab8b4543b20cdb6d439f7ad215c08e698188f`.
- [x] Treat that immutable release as the target for the first Lean 4 formalization.
- [x] Implement all twelve `GEO-LEAN-TGT-*` theorem targets in PR #3 without silently strengthening the release claims.
- [x] Keep Lean proofs as a separate formal evidence layer; do not claim they automatically prove CPython/IEEE-754 execution.
- [x] Obtain final exact-head review on PR #3 with no unresolved correctness findings.
- [x] Pass the exact-head routine, protected isolated, and immutable numerical reference checks for PR #3.
- [x] Merge PR #3 without rewriting or retagging immutable `v0.1.0`.
- [x] Publish the separate formal evidence release `v0.2.0` at merge commit `ec3312dcc102d859819c764a881e2d020662e880`.
- [x] Make `v0.2.0` immutable.

**Immutable formal release:** `v0.2.0 — Phase 1 Lean 4 Formal Evidence Layer`.

**Formal release commit:** `ec3312dcc102d859819c764a881e2d020662e880`.

The sole release-grade cold proof authority remains `lean-isolated-audit / isolated-cold-trust`. `lean-phase1` is a verified-cache regression/cache-maintenance lane: PR runs validate candidates, while relevant `main` pushes or explicit maintenance dispatches may seed authenticated caches for later PRs. Those cache-maintenance runs are not a competing cold-trust authority.

If later work discovers a mathematical defect in the frozen specification, preserve both immutable releases as historical evidence, amend the mathematical contract in a new release, and reprove affected theorem targets against the new identity.

---

## Phase 2 — Local-model capture and serving instrument

**Goal:** obtain auditable representation trajectories from local models and establish when optimized serving paths preserve the scientific object under measurement.

### Phase 2A — Canonical hidden-state capture

**Status:** **instrument implementation introduced in PR #4 under `GEO-CAP-001`; empirical evidence gate remains open until a real frozen local-model capture is executed and reviewed.**

Start with the simplest sufficiently transparent local capture path before introducing serving optimization.

- [ ] Select an initial fully local Hugging Face-compatible model small enough for routine workstation runs.
- [ ] Freeze the selected production model identifier and immutable revision/hash.
- [ ] Freeze the selected production tokenizer identifier and immutable revision/hash.
- [x] Define the canonical reference backend as a direct local Hugging Face/PyTorch replay path in `GEO-CAP-001`.
- [x] Require full 40-hex Hugging Face model/tokenizer commit identities for canonical production requests.
- [x] Require `local_files_only=true`, `trust_remote_code=false`, and no quantization in the canonical capture lane.
- [x] Implement hidden-state capture by explicit layer index.
- [x] Define step segmentation independently of analysis outcome.
- [x] Record exact token IDs and derive each changed token span from the longest common token-ID prefix, exposing tokenizer boundary retokenization.
- [x] Implement `step_mean`, `last_token`, `context_mean`, and explicitly bounded context pooling.
- [x] Record cumulative versus isolated context mode.
- [x] Record dtype and quantization policy.
- [x] Record generation parameters and seeds even when generation itself is not used by replay capture.
- [x] Record repository commit, protocol ID/version, run ID, and content-addressed run-manifest identity.
- [x] Record runtime/library/device, attention implementation, cache, offloading, and determinism metadata where available.
- [x] Define strict capture-request, run-manifest, and captured-trajectory schemas.
- [x] Make `replayed_prefix` an explicit capture phase and distinguish it from future prefill/decode/cache-reuse measurements.
- [x] Record per-vector SHA-256, trajectory SHA-256, request SHA-256, run-manifest identity, and manifest SHA-256.
- [x] Add a software-only deterministic capture-contract fixture for CI without shipping or downloading model weights.
- [x] Label that test-double fixture `SIMULATION` and forbid interpreting it as a model observation.
- [ ] Execute the first production `GEO-CAP-001` local-model capture under a frozen request.
- [ ] Verify deterministic replay for the selected backend where the backend permits it.
- [ ] Explicitly record irreducible nondeterminism if deterministic replay cannot be established.
- [ ] Freeze and review the first production capture evidence artifact.

**Evidence gate:** capture provenance is sufficient to identify exactly what vector each trajectory point represents **and an actual frozen local-model capture artifact exists**.

**Claim ceiling:** `OBSERVATION` only after an actual model run is performed. Code, schemas, and fake-backend fixtures do not create empirical evidence.

**Protocol:** `protocols/GEO-CAP-001.md`.

### Phase 2B — Serving-equivalence study

Treat serving optimization as a measurement variable until equivalence is demonstrated for the intended geometry claims.

- [ ] Define a serving-equivalence protocol against the canonical backend.
- [ ] Freeze identical model revision, tokenizer, prompts, step segmentation, and extraction definitions across compared backends.
- [ ] Record serving backend/version, attention/kernel implementation, dtype/quantization, device placement, and offloading policy.
- [ ] Record KV/prefix/recurrent-state reuse policy and context-edit behaviour.
- [ ] Compare generated-token equality separately from hidden-state equality.
- [ ] Compare representation positions `z_t` under preregistered tolerances.
- [ ] Compare order-1 and selected higher finite differences.
- [ ] Compare preregistered curvature statistics.
- [ ] Test whether backend-induced differences are smaller than the experimental effects later used for scientific claims.
- [ ] Record backend divergence as a result rather than tuning it away.
- [ ] Treat a backend that fails equivalence as an explicit experimental factor, not a transparent substitute.

**Falsifier:** a candidate serving backend produces material representation or trajectory-geometry differences beyond the preregistered equivalence tolerance despite matching output tokens or answers.

**Evidence class:** normally `OBSERVATION` or `ASSOCIATION`, depending on the comparison design. Replication status is recorded separately.

### Phase 2C — Adaptive local serving

Only promote serving optimizations into the default research path after Phase 2B establishes the relevant equivalence or the backend is intentionally treated as an experimental factor.

Candidate serving ideas include those motivated by edge-serving systems such as FreeToken:

- [ ] Profile actual host-memory and host-to-device bandwidth on the deployed machine rather than relying solely on specification sheets.
- [ ] Record available VRAM and material runtime resource changes.
- [ ] Treat prefill and decode as distinct serving regimes.
- [ ] Evaluate prefix/KV/recurrent-state reuse at explicit semantic boundaries where the model/runtime supports it.
- [ ] Permit elastic memory allocation when it changes performance without silently changing the captured scientific object.
- [ ] For MoE models, evaluate expert caching based on observed routing locality rather than assuming static placement is optimal.
- [ ] For MoE or hybrid models, evaluate CPU/GPU cooperative execution using measured machine bandwidth where technically justified.
- [ ] Verify exact or tolerance-bounded representation consequences of any hybrid execution path used for scientific capture.
- [ ] Keep a source-of-truth model/weight identity independent of transient cache residency.
- [ ] Preserve the canonical backend as a regression oracle even after an optimized path is adopted.

**Serving principle:** resource adaptation may alter latency, throughput, memory residency, and scheduling; it must not silently redefine the representation being measured.

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
- [ ] Consider `GEO-HYP-002A` and `GEO-HYP-002B` as explicit competing curvature/instability hypotheses rather than merging their directional claims.
- [ ] Evaluate `GEO-HYP-003` cycle-consistency residual only after its transform cycle and simpler confound controls are frozen.

**Primary motivating hypothesis (`GEO-HYP-001`):** higher-order trajectory structure may align more strongly by logic than absolute representation position does across semantic carriers.

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

## Phase 6 — Cross-model replication, scaling, and lineage extension

**Goal:** determine what aspects of the observed geometry generalize across models, scale, and eventually frozen model lineages.

- [ ] Select multiple sizes within one model family where feasible.
- [ ] Add at least one different model family.
- [ ] Match prompting and evaluation policy as closely as architecture permits.
- [ ] Record tokenizer, architecture, and serving-backend differences explicitly.
- [ ] Audit dataset-exposure risk separately for every compared model, including pretraining, post-training, fine-tuning, and benchmark exposure where evidence is available.
- [ ] Record `exposure_outcome` separately as `exposed`, `unexposed`, or `unknown`.
- [ ] Record `exposure_confidence` separately as `verified`, `high`, `medium`, `low`, or `unknown`.
- [ ] Record `exposure_basis` with supporting provenance.
- [ ] Scope any `unexposed` assessment to the named model revision, dataset/version, and assessed ingestion routes; absence of exposure evidence remains `unknown`.
- [ ] Treat uncontrolled or unknown differential exposure as a comparison confounder rather than evidence of scale, family, or geometric superiority.
- [ ] Where feasible, repeat key comparisons on newly generated or exposure-resistant held-out logic/carrier material.
- [ ] Run quantization sensitivity/ablation where practical.
- [ ] Compare logic/carrier separation across layers and sizes.
- [ ] Compare perturbation sensitivity across sizes.
- [ ] Estimate whether any metric changes monotonically with model scale.
- [ ] Identify geometry that appears family-specific rather than general.
- [ ] Re-run frozen primary analyses without retuning thresholds per model.
- [ ] Preserve the evidence class of the experiment being replicated.
- [ ] Record replication status as `replicated`, `failed`, or `mixed` according to the declared replication scope.
- [ ] If earlier evidence warrants it, preregister the checkpoint-lineage transition study in `GEO-HYP-004`.
- [ ] If earlier evidence warrants it, preregister the matched fine-tuning resistance study in `GEO-HYP-005`.

**Falsifier:** apparent geometric signatures fail to replicate outside the original model/protocol configuration, or an apparent cross-model advantage cannot be separated from material exposure or serving differences.

**Evidence rule:** replication does not create a new evidence class. A replicated `ASSOCIATION`, `PERTURBATION`, or `INTERVENTION` remains that evidence class with separate replication status.

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
- [ ] Consider `GEO-HYP-006` control-effort scaling only after a reproducible observational instability metric exists.

**Falsifiers include:**

- targeted geometry changes but controlled reasoning does not improve;
- reasoning gains are matched by a non-geometric control;
- gains disappear under held-out carriers/logics; and
- gains are explained by added parameters or compute.

**Claim ceiling:** `INTERVENTION`.

---

## Phase 8 — Mechanistic ablation and alternative explanations

**Goal:** distinguish useful correlates from candidate internal mechanisms.

- [ ] Identify the strongest candidate internal geometric structure from prior phases.
- [ ] Develop targeted activation/representation interventions where technically justified.
- [ ] Test necessity via ablation or disruption.
- [ ] Test sufficiency or partial sufficiency via controlled steering/intervention where possible.
- [ ] Compare predictions against simpler alternatives such as lexical similarity, confidence, token position, sequence length, generic activation magnitude, and serving-backend artefacts.
- [ ] Test whether the proposed mechanism predicts failures, not only successes.
- [ ] Document unresolved alternative explanations.
- [ ] Consider `GEO-HYP-007` sidecar control first in prediction-only shadow mode before any actuation.
- [ ] Consider `GEO-HYP-009` error-correction/robustness tests without assuming a quantum-code mechanism.

**Claim ceiling:** `MECHANISM` only if the high bar in `SCIENTIFIC-CONTRACT.md` is satisfied.

---

## Phase 9 — Reproducible local reference run and release

**Goal:** freeze a complete end-to-end research artifact after the methodology has survived the preceding stages.

- [ ] Select a stable reference local model and immutable revision.
- [ ] Freeze dataset, protocol, code, serving backend, and environment manifest.
- [ ] Execute a complete local reference run.
- [ ] Store machine-readable results and artifact hashes.
- [ ] Record evidence class and replication status separately for every headline result.
- [ ] Generate human-readable report from the exact machine-readable evidence.
- [ ] Verify that every headline claim resolves to an evidence artifact.
- [ ] Publish limitations and null results alongside positive results.
- [ ] Tag a release only after exact-head review and reproducibility checks.

---

## Deferred / conditional research branches

These are deliberately not assumed to be useful until earlier evidence warrants them. Candidate theory-derived branches are registered in `RESEARCH-HYPOTHESIS-MAP.md`, which is non-normative and cannot promote a source idea into evidence.

- [ ] intrinsic-dimension estimation of reasoning trajectories;
- [ ] topology/manifold-learning analysis beyond simple trajectory metrics;
- [ ] tangent-space transport across semantic carriers;
- [ ] attractor-state analysis;
- [ ] information-geometric metrics;
- [ ] curvature/entropy coupling (`GEO-HYP-008`) only after both observables are independently defined;
- [ ] SU(3) relational structure (`GEO-HYP-010`) only against explicit lower-complexity alternatives;
- [ ] E8/Weyl latent structure (`GEO-HYP-011`) only against generic lattice/projection/null alternatives;
- [ ] qutrit/CP2 observer geometry (`GEO-HYP-012`) remains outside current repository scope absent an explicit empirical bridge;
- [ ] semantic-reactor architecture (`GEO-HYP-013`) only after intervention evidence warrants a new architecture;
- [ ] geometric regularization during pretraining rather than fine-tuning;
- [ ] latent-step reasoning without explicit textual intermediate steps;
- [ ] formal verification beyond the Phase 1 exact-math kernel where it reduces real ambiguity;
- [ ] advanced MoE expert-paging or bandwidth-adaptive serving beyond what Phase 2 equivalence testing justifies; and
- [ ] integration with other QSOL repositories only after this repository has stable interfaces and evidence classes.

## Roadmap completion rule

A checkbox representing an experimental result is complete only when its required evidence artifact exists and satisfies the current scientific contract.

Code existence alone does not complete an empirical milestone.

# README4AI — QSOL-GEO-REASON

## Repository purpose

QSOL-GEO-REASON is a research repository for testing whether reasoning in local language models exhibits reproducible geometric structure in hidden-state / representation-space trajectories, whether that structure responds to controlled perturbations, and whether explicitly training geometric objectives can improve reasoning under controlled comparisons.

The repository is intentionally **hypothesis-neutral**. A null result is a valid result.

## Current project state

Phase 1 is complete and frozen in two immutable releases:

- `v0.1.0 — Phase 1 Mathematical Kernel`, commit `1b5ab8b4543b20cdb6d439f7ad215c08e698188f`, containing the frozen numerical/simulation evidence; and
- `v0.2.0 — Phase 1 Lean 4 Formal Evidence Layer`, merge commit `ec3312dcc102d859819c764a881e2d020662e880`, containing the separate formal proof layer for all twelve frozen theorem targets.

PR #4 begins Phase 2A by implementing `GEO-CAP-001`, the canonical local hidden-state capture instrument. Code, schemas, and software-only fixtures do **not** constitute an empirical local-model result. The first production model selection, frozen capture, and replay evidence remain open roadmap items.

## Read order for AI agents

Before proposing or modifying experiments, read in this order:

1. `INVARIANTS.md`
2. `SCIENTIFIC-CONTRACT.md`
3. `MATH-SPEC.md` for Phase 1 geometry and mathematical semantics
4. `ROADMAP.md`
5. `protocols/GEO-CAP-001.md` for Phase 2A capture work
6. `RESEARCH-HYPOTHESIS-MAP.md` when theory-derived hypotheses are relevant
7. `LEAN-FORMALIZATION.md` and `LEAN-CACHE-POLICY.md` for the frozen Phase 1 proof layer
8. this file
9. `README.md`
10. relevant implementation, protocol, schema, fixture, and result files

`INVARIANTS.md` and `SCIENTIFIC-CONTRACT.md` are the primary normative files. `MATH-SPEC.md` is normative for the immutable Phase 1 exact mathematical semantics and cannot be weakened by later capture or theory work.

## Exact mathematics versus implementation

`MATH-SPEC.md` defines exact-real mathematical objects. The Python geometry implementation is a finite-precision conformance instrument. The Lean 4 development proves the exact mathematical statements only.

Do not claim that the Lean proof layer proves the Python/IEEE-754 implementation, JSON serialization, SHA-256, Git provenance, serving behaviour, hidden-state extraction, or LLM behaviour. Those require separate evidence or an explicit refinement proof.

Stable IDs `GEO-MATH-*` and `GEO-LEAN-TGT-*` must be preserved when discussing the frozen Phase 1 layer.

## Core objects

### Representation state

`z_t` denotes a vector derived from a specified model hidden state at a specified layer, token span, context, pooling rule, and transform.

It is not automatically a semantic state, belief state, proof state, gauge field, thermodynamic state, or mechanistic reasoning state.

### Trajectory

`gamma = (z_0, ..., z_T)` is an ordered sequence of representation states produced under a fully specified extraction protocol.

### Finite differences

- order 0: `z_t`
- order 1: `Delta z_t = z_(t+1) - z_t`
- order 2: `Delta^2 z_t`
- higher orders: repeated finite differences under an explicitly stated convention

These are analysis constructs. Terms such as “velocity” and “acceleration” are shorthand and must not be interpreted as physical quantities without additional evidence.

A comparison whose selected order leaves no finite-difference samples is undefined and must not be emitted as alignment `0`.

### Semantic carrier

A semantic carrier is the surface domain or vocabulary used to instantiate an underlying reasoning structure, for example mathematics, software, biology, ordinary-language objects, or synthetic symbols.

### Logical structure

A logical structure is the formal dependency / inference pattern intended to remain invariant across carriers. It must be encoded independently of the carrier text whenever feasible.

### Causal perturbation

A perturbation that changes a load-bearing premise or dependency such that the correct downstream conclusion should change.

### Surface / distractor perturbation

A perturbation intended not to change the relevant logical consequence while modifying comparable surface content.

### Reference simulation

A synthetic trajectory experiment where the generating geometry is known by construction. Reference simulations validate instruments and analysis code only. They are not empirical evidence about LLM reasoning.

## Phase 2A capture contract

The first canonical empirical instrument is `GEO-CAP-001`.

Production requests require:

- model identifier plus immutable full 40-hex Hugging Face commit;
- tokenizer identifier plus immutable full 40-hex Hugging Face commit;
- `revision_kind: hf_commit` for both identities;
- direct `huggingface-pytorch` replay;
- `local_files_only: true`;
- `trust_remote_code: false`;
- `quantization: none`;
- explicit device and dtype;
- explicit `replayed_prefix` phase with `use_cache=false`;
- explicit cumulative or isolated context mode;
- explicit hidden-state tuple layer indices;
- explicit pooling and step segmentation; and
- seed/determinism policy.

The instrument records exact input token IDs for every step. The changed token span is derived from the longest common token-ID prefix between the baseline and current rendered input so tokenizer boundary retokenization is visible rather than hidden.

Supported pooling modes are:

- `last_token`;
- `step_mean`;
- `context_mean`;
- `bounded_context_mean` with explicit positive `window_tokens`.

The backend reports observed model/tokenizer commits after loading. A mismatch with the frozen request invalidates the capture.

### Capture provenance

A production bundle contains:

- `capture-request.json`;
- `run-manifest.json`;
- `captured-trajectory.json`.

The bundle records repository revision, protocol/run identities, model/tokenizer identities, backend request, observed backend/runtime/device metadata, extraction definition, exact token IDs/spans, vectors, and content hashes.

The captured trajectory record uses `evidence_class: OBSERVATION` because it is a raw empirical measurement. It does not say that any geometry hypothesis is supported.

### Software fixture boundary

`fixtures/capture-contract-request.json` is exercised through a deterministic fake backend in CI. Its fixture metadata is explicitly `evidence_class: SIMULATION`.

Never cite the fake-backend vectors, hashes, or passing tests as an LLM observation. They prove software-contract behaviour only.

## Evidence classes and replication

Use the evidence classes defined in `SCIENTIFIC-CONTRACT.md`:

- `SIMULATION`
- `OBSERVATION`
- `ASSOCIATION`
- `PERTURBATION`
- `INTERVENTION`
- `MECHANISM`

Replication is separate. Record `replication_status` as `not_attempted`, `replicated`, `failed`, or `mixed` and preserve the underlying evidence class.

Never silently upgrade an evidence class or replace it with replication status.

## Required experiment metadata

Every empirical run must record, at minimum:

- repository commit;
- protocol ID/version;
- run ID and run-manifest identity;
- model identifier and immutable revision/hash when available;
- model family and parameter count where relevant;
- tokenizer identifier/revision;
- serving backend and version;
- runtime/library versions;
- quantization and dtype;
- device information;
- prompts / dataset revision;
- dataset-generation provenance;
- dataset-exposure / contamination assessment when relevant;
- random seeds;
- generation parameters;
- hidden-state layer(s);
- token/span selection;
- pooling method;
- context accumulation rule;
- normalization and projection transforms;
- geometric metrics;
- comparison/alignment procedure;
- preregistered primary outcomes when confirmatory; and
- output artifact hashes.

### Exposure assessment shape

Keep outcome and confidence separate:

- `exposure_outcome`: `exposed`, `unexposed`, or `unknown`;
- `exposure_confidence`: `verified`, `high`, `medium`, `low`, or `unknown`;
- `exposure_basis`: supporting provenance.

`unexposed` is scoped to the named model revision, dataset/version, and assessed ingestion routes. Lack of evidence of exposure remains `unknown`.

### Serving metadata

When hidden states or geometry are measured, record material serving choices including backend/version, attention or kernel implementation, dtype/quantization, device placement/offloading, cache or state reuse, and whether the captured state occurred during prefill, decode, or replayed-prefix execution.

Matching output tokens across runtimes does not prove hidden-state equivalence.

## Theory-derived hypothesis policy

`RESEARCH-HYPOTHESIS-MAP.md` is intentionally non-normative. It translates theoretical source material into candidate measurements, hypotheses, deferred models, and speculative interpretations.

Do not import equations or physical interpretations from those sources as facts about an LLM.

In particular:

- conflicting curvature/hallucination predictions remain competing hypotheses;
- cycle-consistency residual is initially a representation residual, not automatically a Wilson loop or physical holonomy;
- “identity mass,” “semantic inertia,” “entropy,” and related terms require independent operational definitions before confirmatory use;
- SU(3), E8, CP2/qutrit, gauge-field, reactor, or error-correction descriptions must outperform explicit simpler alternatives before receiving mechanistic interpretation.

## Scientific guardrails

Do not:

- infer mechanism from PCA, UMAP, t-SNE, or another projection;
- select layers or metrics after inspecting target labels and present them as preregistered;
- describe synthetic outputs as model observations;
- mark Phase 2A empirically complete merely because `GEO-CAP-001` code exists;
- treat placeholder example revisions as valid evidence identities;
- compare differently quantized or differently prompted models as though parameter count were the only difference;
- treat unknown dataset exposure as evidence that a model was unexposed;
- attribute a cross-model advantage to scale or geometry when differential exposure remains a material confounder;
- treat identical generated tokens from two serving backends as proof of representation-space equivalence;
- substitute an optimized backend for the canonical capture backend without a serving-equivalence result or explicit experimental-factor treatment;
- use fixed absolute zero thresholds where `MATH-SPEC.md` specifies exact zero;
- turn an undefined metric into an ordinary numerical zero;
- treat a reasoning benchmark score alone as proof of geometric reasoning;
- hide negative results;
- collapse carrier similarity, logical similarity, answer correctness, and trajectory similarity into one label; or
- use the term “causal” for ordinary observational correlations.

## Lean workflow authority

`lean-phase1` is a pull-request-only verified-cache regression lane. It is not a release-grade cold authority.

The sole release-grade cold proof authority is the manually dispatched `lean-isolated-audit / isolated-cold-trust` job. The protected audit imports `GeoReason` without executing project initializers, verifies all twelve target declarations are theorems, and restricts transitive axioms to `propext`, `Classical.choice`, and `Quot.sound`.

That proof layer is now frozen in immutable `v0.2.0`. Later empirical work must not modify its meaning or claim that it proves hidden-state capture.

## Expected repository evolution

The intended overall progression is:

`contract -> exact math -> simulation -> formal proof -> canonical instrumentation -> serving-equivalence validation -> observational dataset -> perturbation -> cross-model replication -> geometric training intervention -> ablation / falsification -> release`

Phase 1 numerical and formal evidence are complete and immutable. Phase 2A now builds the canonical capture instrument, while its empirical gate remains open until a real frozen local-model run exists.

This is a research workflow, not a claim ladder, and replication remains orthogonal to evidence class.

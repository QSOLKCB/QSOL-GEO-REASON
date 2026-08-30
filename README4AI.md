# README4AI — QSOL-GEO-REASON

## Repository purpose

QSOL-GEO-REASON is a research repository for testing whether reasoning in local language models exhibits reproducible geometric structure in hidden-state / representation-space trajectories, whether that structure responds to controlled perturbations, and whether explicitly training geometric objectives can improve reasoning under controlled comparisons.

The repository is intentionally **hypothesis-neutral**. A null result is a valid result.

## Read order for AI agents

Before proposing or modifying experiments, read in this order:

1. `INVARIANTS.md`
2. `SCIENTIFIC-CONTRACT.md`
3. `ROADMAP.md`
4. this file
5. `README.md`
6. relevant implementation, protocol, schema, fixture, and result files

`INVARIANTS.md` and `SCIENTIFIC-CONTRACT.md` are normative. If another document conflicts with them, the normative files win unless a pull request explicitly updates the contract and explains why.

## Core objects

### Representation state

`z_t` denotes a vector derived from a specified model hidden state at a specified layer, token span, pooling rule, and accumulation context.

It is not automatically a semantic state, belief state, proof state, or mechanistic reasoning state.

### Trajectory

`gamma = (z_0, ..., z_T)` is an ordered sequence of representation states produced under a fully specified extraction protocol.

### Finite differences

- order 0: `z_t`
- order 1: `Delta z_t = z_(t+1) - z_t`
- order 2: `Delta^2 z_t`
- higher orders: repeated finite differences under an explicitly stated convention

These are analysis constructs. Terms such as “velocity” and “acceleration” are shorthand and must not be interpreted as physical quantities without additional justification.

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
- model family and parameter count;
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
- preregistered primary outcomes when confirmatory;
- output artifact hashes.

### Exposure assessment shape

Keep outcome and confidence separate:

- `exposure_outcome`: `exposed`, `unexposed`, or `unknown`;
- `exposure_confidence`: `verified`, `high`, `medium`, `low`, or `unknown`;
- `exposure_basis`: supporting provenance.

`unexposed` is scoped to the named model revision, dataset/version, and assessed ingestion routes. Lack of evidence of exposure must remain `unknown`, not be silently converted to `unexposed`.

### Serving metadata

When hidden states or geometry are measured, record material serving choices including backend/version, attention or kernel implementation, dtype/quantization, device placement/offloading, cache or state reuse, and whether the captured state occurred during prefill, decode, or replayed-prefix execution.

Matching output tokens across runtimes does not prove hidden-state equivalence.

## Scientific guardrails

Do not:

- infer mechanism from a PCA, UMAP, t-SNE, or other projection;
- select layers or metrics after inspecting target labels and present them as preregistered;
- describe synthetic outputs as model observations;
- compare differently quantized or differently prompted models as though parameter count were the only difference;
- treat unknown dataset exposure as evidence that a model was unexposed;
- attribute a cross-model advantage to scale or geometry when differential exposure remains a material confounder;
- treat identical generated tokens from two serving backends as proof of representation-space equivalence;
- substitute an optimized backend for the canonical capture backend without either a serving-equivalence result or explicit treatment of backend as an experimental variable;
- treat a reasoning benchmark score alone as proof of geometric reasoning;
- hide negative results;
- collapse carrier similarity, logical similarity, answer correctness, and trajectory similarity into one label;
- use the term “causal” for ordinary observational correlations.

## Expected repository evolution

The intended progression is:

`contract -> simulation -> canonical instrumentation -> serving-equivalence validation -> observational dataset -> perturbation -> cross-model replication -> geometric training intervention -> ablation / falsification -> release`

This is a research workflow, not a claim ladder. Replication remains orthogonal to evidence class.

Do not skip directly from the foundation stage to capability claims.

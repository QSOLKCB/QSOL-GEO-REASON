# GEO-CAP-001-EXP-001 — First production observation

Status: **preregistered production observation; execution pending**

Protocol: `GEO-CAP-001`

Evidence ceiling after execution: **`OBSERVATION`**

This experiment is the first real-model use of the canonical Phase 2A capture instrument. It is deliberately an instrument-validation observation, not a confirmatory test of `GEO-HYP-001` or any later reasoning-geometry hypothesis.

## Frozen model and tokenizer

The selected reference model and tokenizer are both:

- Hugging Face repository: `openai-community/gpt2`
- immutable revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- revision kind: `hf_commit`

The same immutable repository revision is used for the model and tokenizer. The production lane remains `local_files_only=true`, `trust_remote_code=false`, and `quantization=none`.

GPT-2 was chosen for this first observation because it is small enough for routine local CPU replay, uses a long-established Transformers architecture supported by the canonical decoder-layout resolver, and does not require remote model code. The selection is about auditability and repeatability, not state-of-the-art reasoning capability.

## Frozen capture definition

The preregistered request template is `GEO-CAP-001-EXP-001.request.template.json`.

It freezes before observation:

- backend device: `cpu`;
- requested dtype: `float32`;
- determinism mode: `required`;
- seed: `20260912`;
- context mode: `cumulative`;
- phase: `replayed_prefix`;
- sampled hidden-state indices: `0, 3, 6, 9, 12`;
- pooling: `step_mean`;
- exact prefix text, step joiner, ordered step IDs, and ordered step text; and
- `generation_used=false`.

The five text steps form a simple monotonic implication chain. They exist to create a nontrivial multi-point representation trajectory while exercising the real instrument. This single trajectory is not a carrier-invariance dataset, a benchmark, a perturbation study, or evidence that the model is performing a geometric reasoning mechanism.

## Why the committed file is a template

Canonical `OBSERVATION` execution additionally requires two authenticated Hub tree-receipt SHA-256 values. Those values bind the trusted online Hugging Face commit-tree metadata cached on the machine that warms the exact immutable revision.

They are intentionally not guessed or copied into the preregistered template. The only permitted post-preregistration mutation is addition of:

- `model.revision_tree_sha256`; and
- `model.tokenizer_revision_tree_sha256`.

`tools/run_first_production_observation.py` enforces that boundary by comparing the final request against the committed template after removing exactly those two fields.

## Execution procedure

Install the pinned capture lane in a clean checkout:

```bash
python -m pip install -e '.[capture]'
```

### 1. Online trusted preparation

Run once while the machine is online:

```bash
python tools/run_first_production_observation.py prepare \
  --output /tmp/GEO-CAP-001-EXP-001.request.json
```

This warms the exact model/tokenizer commit, creates the authenticated QSOL Hub tree artifacts, injects their receipts into the frozen request, writes the final request without overwriting any existing artifact, and prints the canonical request SHA-256.

The final request is the request that must be preserved with the evidence bundle.

### 2. Offline dual observation and replay check

Disconnect network access if practical, or otherwise rely on the tool's forced Hugging Face/Transformers offline environment, then run:

```bash
python tools/run_first_production_observation.py observe \
  --request /tmp/GEO-CAP-001-EXP-001.request.json \
  --output-root /tmp/GEO-CAP-001-EXP-001
```

The canonical capture CLI launches a fresh isolated worker for each run. The experiment runner executes the exact request twice and creates:

```text
GEO-CAP-001-EXP-001/
├── run-a/
│   ├── capture-request.json
│   ├── run-manifest.json
│   └── captured-trajectory.json
├── run-b/
│   ├── capture-request.json
│   ├── run-manifest.json
│   └── captured-trajectory.json
└── replay-verdict.json
```

Both observation bundles are independently verified with the canonical semantic verifier. The replay verdict records byte-level equality for each canonical bundle file, manifest-receipt equality, trajectory-hash equality, and the observed replay outcome.

A divergence is retained. The tool writes `replay_outcome: "diverged"` and exits nonzero rather than tuning the request, deleting the differing run, or silently weakening the determinism requirement.

## Evidence boundary

Before the production command is actually executed, this experiment contributes **no empirical model evidence**.

After a valid run exists, the captured hidden-state trajectory is an `OBSERVATION` of this exact model/request/backend/runtime configuration. Even a byte-identical replay does not establish:

- that GPT-2 reasons geometrically;
- that the trajectory encodes logical truth;
- that curvature or higher-order differences are mechanisms of reasoning;
- that the same representation is preserved by a different serving backend;
- that the result generalizes to another prompt, carrier, model, model size, device, dtype, or runtime; or
- that any Phase 3–8 hypothesis has passed.

The Phase 2A empirical gate closes only after the real artifacts and replay result are reviewed and frozen as evidence. Phase 2B must not treat an optimized serving backend as transparent before that happens.

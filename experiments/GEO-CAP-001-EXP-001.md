# GEO-CAP-001-EXP-001 — First production observation

Status: **preregistered production observation; execution pending**

Protocol: `GEO-CAP-001`

Evidence ceiling after execution: **`OBSERVATION`**

This experiment is the first real-model use of the canonical Phase 2A capture instrument. It is deliberately an instrument-validation observation, not a confirmatory test of `GEO-HYP-001` or any later reasoning-geometry hypothesis.

## Frozen model and tokenizer

The selected reference model and tokenizer are both:

- Hugging Face repository: `Qwen/Qwen2.5-0.5B`
- immutable revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- revision kind: `hf_commit`

The same immutable repository revision is used for the model and tokenizer. The production lane remains `local_files_only=true`, `trust_remote_code=false`, and `quantization=none`.

Qwen2.5-0.5B was chosen for this first observation because the frozen repository is compact enough for routine local warm-up, stores the model weights as Safetensors, uses the `qwen2` Transformers architecture supported by the canonical decoder-layout resolver, and requires no remote model code. The repository's documented Transformers floor is 4.37; the project's pinned reference lane uses Transformers 4.40.2. The selection is about auditability and repeatability, not model ranking or a claim about reasoning capability.

The frozen model has 24 decoder layers, so the request samples the canonical hidden-state sequence at the input, quarter-depth, half-depth, three-quarter-depth, and final base-model state.

## Frozen capture definition

The preregistered request template is `GEO-CAP-001-EXP-001.request.template.json`.

It freezes before observation:

- backend device: `cpu`;
- requested dtype: `float32`;
- determinism mode: `required`;
- seed: `20260912`;
- context mode: `cumulative`;
- phase: `replayed_prefix`;
- sampled hidden-state indices: `0, 6, 12, 18, 24`;
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

This warms the exact model/tokenizer commit, creates the authenticated QSOL Hub tree artifacts, injects their receipts into the frozen request, and prints the canonical request SHA-256. The final request is first written and fsynced to a sibling temporary file, then published with no-replace semantics, so a failed write cannot leave a partial destination that blocks a clean retry.

The final request is the request that must be preserved with the evidence bundle.

### 2. Offline dual observation and replay check

Disconnect network access if practical, or otherwise rely on the tool's forced Hugging Face/Transformers offline environment, then run:

```bash
python tools/run_first_production_observation.py observe \
  --request /tmp/GEO-CAP-001-EXP-001.request.json \
  --output-root /tmp/GEO-CAP-001-EXP-001
```

Before launching either worker, the runner validates the final request and publishes an immutable canonical snapshot named `validated-request.json`. Both workers consume that staged snapshot rather than reopening the caller's original request path. The replay verifier then requires each bundle's `capture-request.json` to equal the validated snapshot exactly, including both authenticated tree-receipt hashes.

The canonical capture CLI launches a fresh isolated worker for each run. A successful experiment directory is:

```text
GEO-CAP-001-EXP-001/
├── validated-request.json
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

Both observation bundles are independently verified with the canonical semantic verifier. `replay-verdict.json` is itself a machine-readable evidence artifact governed by `schemas/replay-verdict.schema.json` and the semantic verifier `qsol_geo_reason.capture_replay.verify_replay_verdict`. The verifier recomputes the validated-request artifact hash, every canonical bundle-file hash, manifest-receipt equality, trajectory-hash equality, repository-commit equality, and the resulting replay outcome before accepting the verdict.

A completed divergence is retained. The tool writes `replay_outcome: "diverged"` and exits nonzero rather than tuning the request, deleting the differing run, or silently weakening the determinism requirement.

If a worker, verifier, or filesystem operation fails **before** a replay verdict exists, the incomplete attempt is not erased. The runner writes `execution-failure.json` where possible and moves the whole attempt to a uniquely named sibling such as:

```text
GEO-CAP-001-EXP-001.failed-<pid>-<nonce>/
```

That preserves partial/failure evidence while freeing the requested `/tmp/GEO-CAP-001-EXP-001` path for an explicit retry. A failed-attempt directory is never a successful replay result and must not be substituted for the final experiment.

## Evidence boundary

Before the production command is actually executed, this experiment contributes **no empirical model evidence**.

After a valid run exists, the captured hidden-state trajectory is an `OBSERVATION` of this exact model/request/backend/runtime configuration. Even a byte-identical replay does not establish:

- that Qwen2.5-0.5B reasons geometrically;
- that the trajectory encodes logical truth;
- that curvature or higher-order differences are mechanisms of reasoning;
- that the same representation is preserved by a different serving backend;
- that the result generalizes to another prompt, carrier, model, model size, device, dtype, or runtime; or
- that any Phase 3–8 hypothesis has passed.

The Phase 2A empirical gate closes only after the real artifacts and replay result are reviewed and frozen as evidence. Phase 2B must not treat an optimized serving backend as transparent before that happens.

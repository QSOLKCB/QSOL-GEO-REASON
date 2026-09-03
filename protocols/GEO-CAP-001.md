# GEO-CAP-001 — Canonical Local Hidden-State Capture

Status: **Phase 2A instrumentation protocol**

Production artifact evidence class: **`OBSERVATION`**

Current empirical status: **no production local-model capture is established merely by implementing this protocol.**

GEO-CAP-001 defines the first canonical path for extracting representation trajectories from a frozen local language model. Its job is deliberately narrow: identify exactly which hidden-state vector was captured, under which immutable model/tokenizer/runtime choices, and bind that capture to reproducible provenance before later geometric interpretation begins.

This protocol does **not** claim that a captured vector is a belief, proof state, semantic truth state, gauge field, cognitive state, or mechanism of reasoning. It implements the `representation state` object defined by `SCIENTIFIC-CONTRACT.md`.

## 1. Scope

The canonical Phase 2A instrument uses a direct Hugging Face / PyTorch forward pass with:

- a fully local model and tokenizer;
- immutable 40-hex Hugging Face commit revisions for both model and tokenizer;
- `local_files_only=true`;
- `trust_remote_code=false`;
- no quantization;
- an explicitly recorded dtype and device;
- `output_hidden_states=true`;
- `use_cache=false`; and
- replayed-prefix capture rather than optimized serving.

Optimized, quantized, cached, hybrid, remote, or otherwise substituted serving paths are **not** measurement-equivalent merely because they emit the same text. They belong in the later serving-equivalence study defined by Phase 2B.

## 2. Source identity

A production request MUST provide:

- `model.identifier`;
- `model.revision_kind = "hf_commit"`;
- `model.revision` as a full 40-hex Git commit;
- `model.tokenizer_identifier`;
- `model.tokenizer_revision_kind = "hf_commit"`; and
- `model.tokenizer_revision` as a full 40-hex Git commit.

Aliases such as `main`, moving tags, branches, abbreviated SHAs, or unversioned local directories are not canonical production identities for GEO-CAP-001.

After loading, the backend records the model and tokenizer commit identities observed from the Hugging Face objects. The run is rejected if either observed commit differs from the request.

This check is provenance evidence, not a proof of the complete supply chain of the downloaded weights. The production model must already be available locally; GEO-CAP-001 does not download model artifacts during capture.

## 3. Step construction

A request contains an ordered non-empty `steps` array. Each step has an immutable `step_id` and text segment.

The step segmentation is selected before geometric analysis. Capture code does not inspect correctness labels, geometry, hypothesis labels, or target outcomes when defining the segments.

Two context modes are supported:

### `cumulative`

At step `t`, the rendered input is:

```text
prefix_text + step_joiner + step_0 + ... + step_t
```

with separators omitted only where implied by the exact configured strings.

### `isolated`

At each step, the rendered input is:

```text
prefix_text + step_joiner + step_t
```

The two modes are distinct experimental objects and must not be compared as though the extraction function were unchanged.

## 4. Token boundary semantics

Text boundaries are not assumed to be tokenizer boundaries.

For every captured step the instrument records the exact `input_ids` used for the forward pass. The new/changed token span begins at the **longest common token-ID prefix** between:

- the previous cumulative context and current cumulative context in `cumulative` mode; or
- the tokenized prefix and the current isolated input in `isolated` mode.

This deliberately exposes boundary retokenization. A step whose current tokenization adds no changed token span is rejected rather than receiving a fabricated pooled vector.

The record stores:

- exact input token IDs;
- token count;
- changed token span `[start, end)`;
- SHA-256 of the input-ID array; and
- SHA-256 of the rendered text.

## 5. Layer semantics

`capture.layers` contains explicit unique non-negative indices into the Hugging Face `outputs.hidden_states` tuple.

For ordinary causal-language-model implementations that follow the Transformers contract, tuple index `0` is the embedding output before the first transformer block, followed by block outputs. Because architectures can differ, the captured record stores this indexing convention rather than silently relabelling it as a human layer number.

A requested layer outside the observed tuple is rejected.

The per-token vector dimension must be nonzero and stable for each requested layer across all captured steps. Non-finite values are rejected.

## 6. Pooling

GEO-CAP-001 supports four explicit pooling modes:

- `last_token`: use the final token representation;
- `step_mean`: arithmetic mean over the changed token span;
- `context_mean`: arithmetic mean over the entire rendered context;
- `bounded_context_mean`: arithmetic mean over the final `window_tokens` tokens.

The exact `[start, end)` pool span is recorded for every layer and step. `bounded_context_mean` requires a strictly positive `window_tokens`; that parameter is prohibited for the other modes.

Pooling is performed by protocol code rather than by the backend adapter so its semantics are independently testable.

## 7. Capture phase

The initial canonical phase is:

`replayed_prefix`

Each step is evaluated as a complete forward pass with `use_cache=false`.

This is intentionally distinct from:

- optimized prefill capture;
- live autoregressive decode capture;
- KV/prefix/recurrent-state reuse; and
- state reuse after prompt edits.

Those execution modes may produce a materially different measurement object. They remain explicit Phase 2B/2C variables until equivalence is demonstrated.

## 8. Determinism

A request records:

- `determinism.mode`: `required` or `best_effort`;
- `determinism.seed`.

The Hugging Face / PyTorch adapter seeds CPU and CUDA RNGs. In `required` mode it enables PyTorch deterministic algorithms.

This does not turn every hardware/kernel stack into a mathematical guarantee of byte-identical floating-point execution. Runtime versions, device information, attention implementation, dtype, and determinism state are recorded so replay behaviour can be tested rather than presumed.

A production Phase 2A evidence gate remains open until deterministic replay is actually evaluated for the selected frozen model/backend where the backend permits it, or irreducible nondeterminism is explicitly recorded.

## 9. Runtime provenance

The run manifest binds the request to the executing repository revision and observed backend metadata. The canonical backend records, where available:

- Python version;
- platform;
- PyTorch version;
- Transformers version;
- Tokenizers version;
- Hugging Face Hub version;
- loaded model class;
- loaded tokenizer class;
- observed model commit;
- observed tokenizer commit;
- attention implementation;
- device and CUDA device name;
- dtype;
- quantization state;
- offloading state;
- cache policy;
- capture phase; and
- deterministic-algorithm state.

The repository revision is resolved using the existing source-provenance rules: a clean source checkout is required when `HEAD` supplies the implementation identity.

## 10. Output bundle

A production invocation writes three canonical JSON files:

```text
capture-request.json
run-manifest.json
captured-trajectory.json
```

The manifest contains a content-addressed `run_manifest_id`, the request SHA-256, the trajectory SHA-256, and a final manifest SHA-256.

Each captured layer vector carries its own SHA-256. The trajectory record carries:

- `evidence_class: OBSERVATION`;
- `replication_status: not_attempted`;
- protocol and run identities;
- repository commit;
- the frozen representation definition; and
- every captured step, token span, layer vector, and vector hash.

Raw capture is observation only. The capture program emits no statement that a particular geometric hypothesis is supported.

## 11. Software-only contract fixture

CI uses a deterministic test-double backend to exercise request validation, token-span semantics, pooling, provenance binding, and content hashes without shipping or downloading model weights.

That fixture is explicitly labelled:

`evidence_class: SIMULATION`

It is not a hidden-state observation from an LLM, even though it exercises the same protocol engine used by production capture.

A fake backend, mocked tensor, synthetic vector, or frozen software fixture must never be promoted to `OBSERVATION` evidence.

## 12. Running a production capture

Install the optional capture dependencies:

```bash
python -m pip install -e '.[capture]'
```

Copy `examples/GEO-CAP-001.example.json`, replace every placeholder model/tokenizer revision with the exact immutable 40-hex Hugging Face commits already present in the local cache, choose the frozen step segmentation, then run:

```bash
qsol-geo-capture capture-request.json --output-dir results/GEO-CAP-001/<run-id>
```

No result directory produced from placeholder identities is valid production evidence.

## 13. Failure conditions

The capture is invalid if, among other cases:

- model or tokenizer revision is not a full immutable commit identity;
- remote code or network loading is enabled;
- quantization is silently introduced into the canonical lane;
- the loaded model/tokenizer reports a different commit from the frozen request;
- a step tokenizes to no changed token span;
- a requested hidden-state layer does not exist;
- token count and hidden-state row count disagree;
- vector dimensions change unexpectedly;
- any captured or pooled value is non-finite;
- a material runtime/capture choice is absent from provenance; or
- an optimized serving path is substituted without an equivalence result or explicit experimental-factor treatment.

## 14. Phase 2A evidence gate

Merging the capture implementation does **not** complete Phase 2A.

The Phase 2A evidence gate requires, at minimum:

1. selection of a sufficiently transparent fully local model;
2. immutable model and tokenizer identities;
3. a frozen production GEO-CAP-001 request;
4. a real local-model capture bundle;
5. provenance sufficient to identify exactly what each vector represents;
6. deterministic replay evidence where supported, or explicit nondeterminism records; and
7. review of the resulting evidence artifact without post-hoc alteration of the frozen extraction choices.

Until then, the repository has a capture **instrument**, not an empirical geometric result.

## 15. Claim ceiling

A successful production GEO-CAP-001 run supports only an `OBSERVATION` claim that specified representation vectors were extracted from the named local model under the named protocol.

It does not establish:

- that the vectors encode reasoning;
- carrier invariance;
- a curvature/hallucination relationship;
- gauge dynamics or holonomy in a physical sense;
- a thermodynamic law of cognition;
- a causal relationship between geometry and correctness; or
- a mechanism of reasoning.

Those are later hypotheses with their own controls, falsifiers, and evidence gates.

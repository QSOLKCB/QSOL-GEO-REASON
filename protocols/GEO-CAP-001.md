# GEO-CAP-001 — Canonical Local Hidden-State Capture

Status: **Phase 2A instrumentation protocol**

Production artifact evidence class: **`OBSERVATION`**

Current empirical status: **no production local-model capture is established merely by implementing this protocol.**

GEO-CAP-001 defines the first canonical path for extracting representation trajectories from a frozen local language model. Its job is deliberately narrow: identify exactly which hidden-state vector was captured, under which immutable model/tokenizer/runtime choices, and bind that capture to reproducible provenance before later geometric interpretation begins.

This protocol does **not** claim that a captured vector is a belief, proof state, semantic truth state, gauge field, cognitive state, or mechanism of reasoning. It implements the `representation state` object defined by `SCIENTIFIC-CONTRACT.md`.

## 1. Scope

The canonical Phase 2A instrument uses a direct Hugging Face / PyTorch forward pass with:

- a fully local model and tokenizer;
- canonical Hugging Face Hub repository identifiers plus immutable 40-hex commits for both model and tokenizer;
- `local_files_only=true`;
- `trust_remote_code=false`;
- no declared or checkpoint-embedded quantization;
- an explicitly requested dtype and device;
- the actual dtype observed at each selected hidden-state tensor recorded before conversion;
- an exact checkpoint loading report with no missing, unexpected, mismatched, or errored parameters;
- `output_hidden_states=true`;
- `use_cache=false`; and
- replayed-prefix capture rather than optimized serving.

Optimized, quantized, cached, hybrid, remote, or otherwise substituted serving paths are **not** measurement-equivalent merely because they emit the same text. They belong in the later serving-equivalence study defined by Phase 2B.

## 2. Source identity

A production request MUST provide:

- `model.identifier` in canonical `namespace/repository` Hub-ID form;
- `model.revision_kind = "hf_commit"`;
- `model.revision` as a full 40-hex Git commit;
- `model.tokenizer_identifier` in canonical `namespace/repository` Hub-ID form;
- `model.tokenizer_revision_kind = "hf_commit"`; and
- `model.tokenizer_revision` as a full 40-hex Git commit.

Aliases such as `main`, moving tags, branches, abbreviated SHAs, single-component aliases, filesystem paths, snapshot-directory paths, or unversioned local directories are not canonical production identities for GEO-CAP-001. The production model and tokenizer bytes may already exist in the local Hugging Face cache, but the request identifies them through their Hub repository identity and immutable commit, not through a mutable local path.

After loading, the backend records the model and tokenizer commit identities observed from the Hugging Face objects. The run is rejected if either observed commit differs from the request.

The checkpoint is loaded with Transformers loading information enabled. A canonical observation is rejected if loading reports any missing keys, unexpected keys, mismatched keys, or load errors. Parameters silently initialized at load time are therefore outside the production capture contract.

The backend also inspects the loaded model and configuration for quantization signals. A model carrying `quantization_config`, reporting `is_quantized`, or exposing an active Hugging Face quantizer is rejected even when the request itself says `quantization: none`.

These checks are provenance evidence, not a proof of the complete supply chain of the downloaded weights. GEO-CAP-001 does not download model artifacts during capture.

## 3. Step construction and generation provenance

A request contains an ordered non-empty `steps` array. Each step has an immutable `step_id` and text segment.

Step segmentation is selected before geometric analysis. Capture code does not inspect correctness labels, geometry, hypothesis labels, or target outcomes when defining segments.

GEO-CAP-001 currently **replays supplied text segments**. It does not generate those segments. Therefore every canonical request must explicitly contain:

```json
"generation_parameters": {
  "generation_used": false
}
```

Additional generation-setting fields may be retained only as explicit `null` placeholders. A non-null temperature, top-p, top-k, sampling policy, or other generation setting is incompatible with `generation_used: false` and is rejected. A future protocol that captures externally or internally generated steps must define and validate that provenance separately rather than hiding it inside this replay contract.

Two context modes are supported.

### `cumulative`

At step `t`, the rendered input is formed from:

```text
prefix_text + step_0 + ... + step_t
```

with `step_joiner` inserted exactly between included components.

### `isolated`

At each step, the rendered input contains only:

```text
prefix_text + step_t
```

again using the exact configured `step_joiner`.

The two modes are distinct experimental objects and must not be compared as though the extraction function were unchanged.

## 4. Token boundary semantics

Text boundaries are not assumed to be tokenizer boundaries.

For every captured step the instrument records the exact `input_ids` used for the forward pass. The new/changed token span begins at the **longest common token-ID prefix** between:

- the previous cumulative context and current cumulative context in `cumulative` mode; or
- the tokenized prefix and current isolated input in `isolated` mode.

This deliberately exposes boundary retokenization. A step whose current tokenization adds no changed token span is rejected rather than receiving a fabricated pooled vector.

The record stores:

- exact input token IDs;
- token count;
- changed token span `[start, end)`;
- SHA-256 of the input-ID array; and
- SHA-256 of the rendered text.

## 5. Layer semantics, dtype, and memory boundary

`capture.layers` contains explicit unique non-negative indices into the Hugging Face `outputs.hidden_states` tuple.

For ordinary causal-language-model implementations that follow the Transformers contract, tuple index `0` is the embedding output before the first transformer block, followed by block outputs. Because architectures can differ, the captured record stores this indexing convention rather than silently relabelling it as a human layer number.

A requested layer outside the observed tuple is rejected.

Transformers may internally return the complete hidden-state tuple, but the canonical adapter never expands the full sequence for a selected layer into Python objects. For each requested layer it:

1. inspects the tensor shape and **records the tensor's actual dtype before conversion**;
2. selects the already-determined token pool span on the tensor;
3. reduces that span to one vector using float64 accumulation when a mean is required;
4. validates that the pooled tensor is finite; and
5. only then transfers the single pooled vector to CPU/Python form.

Unrequested layers are not materialized, and requested layers do not acquire Python-object memory proportional to the full context length. This closes both the model-depth and sequence-length materialization hazards.

The vector dimension must be nonzero and stable for each requested layer across all captured steps. Every layer record stores `observed_dtype` separately from the requested backend dtype so an architecture-internal upcast is visible rather than silently rewritten as the requested precision.

## 6. Pooling

GEO-CAP-001 supports four explicit pooling modes:

- `last_token`: use the final token representation;
- `step_mean`: arithmetic mean over the changed token span;
- `context_mean`: arithmetic mean over the entire rendered context;
- `bounded_context_mean`: arithmetic mean over the final `window_tokens` tokens.

The protocol core determines the exact `[start, end)` pool span independently of backend results. `bounded_context_mean` requires a strictly positive `window_tokens`; that parameter is prohibited for the other modes.

For the production PyTorch adapter, the span is then applied and reduced on the tensor before Python conversion. The adapter records `pool_accumulation_dtype: float64`. The software-only test double implements the same span semantics independently in ordinary Python so the protocol's span-selection behavior remains unit-testable without model weights.

Every layer record stores the exact pool span used.

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

This does not turn every hardware/kernel stack into a mathematical guarantee of byte-identical floating-point execution. Runtime versions, device information, attention implementation, requested dtype, observed hidden-state dtype, thread configuration, and determinism state are recorded so replay behaviour can be tested rather than presumed.

A production Phase 2A evidence gate remains open until deterministic replay is actually evaluated for the selected frozen model/backend where the backend permits it, or irreducible nondeterminism is explicitly recorded.

## 9. Runtime provenance

The run manifest binds the request to the executing repository revision and observed backend metadata. The canonical backend records, where available:

- Python version and platform;
- PyTorch version;
- Transformers version;
- Tokenizers version;
- Hugging Face Hub version;
- loaded model and tokenizer classes;
- observed model and tokenizer commits;
- clean checkpoint-loading status;
- quantization-detection state;
- attention implementation;
- requested device and dtype;
- actual observed hidden-state dtypes by selected layer;
- float64 pooling accumulation policy;
- CPU machine/processor identity;
- CPU instruction flags/features when exposed by the operating system;
- PyTorch intra-op and inter-op thread counts;
- `OMP_NUM_THREADS` and `MKL_NUM_THREADS` when set;
- CUDA device name/capability;
- PyTorch CUDA build version;
- cuDNN version;
- NVIDIA driver version reported by `nvidia-smi` for CUDA captures;
- quantization and offloading state;
- cache policy;
- capture phase; and
- deterministic-algorithm state.

Unavailable runtime fields are recorded as unavailable/null rather than guessed.

The repository revision is resolved using the existing source-provenance rules: a clean source checkout is required when `HEAD` supplies the implementation identity.

## 10. Evidence-class boundary

The backend-independent protocol engine defaults to:

`evidence_class: SIMULATION`

This is the only evidence class available to arbitrary test doubles or third-party adapters through the generic path.

`OBSERVATION` requires the concrete `HuggingFacePyTorchBackend` used by the canonical CLI. A backend cannot obtain empirical status merely by returning metadata whose `name` string says `huggingface-pytorch`.

The captured-trajectory schema permits both explicitly labelled `SIMULATION` software-contract trajectories and `OBSERVATION` production trajectories. The classification is therefore preserved in the machine-readable artifact rather than contradicted by its schema.

A real production capture is still only an observation of specified representation vectors. It does not state that any geometric theory is supported.

## 11. Output bundle, leaf identities, and atomic publication

A production invocation publishes three canonical JSON files as one new directory:

```text
capture-request.json
run-manifest.json
captured-trajectory.json
```

Before publication, the bundle verifier recomputes and cross-checks:

- normalized request SHA-256;
- every input-ID-array SHA-256 and token count;
- rendered-text SHA-256 values derived from the frozen request;
- every selected layer identity;
- every vector dimension and finite-value constraint;
- every recorded pool span against the frozen pooling rule;
- every per-vector SHA-256;
- stored trajectory SHA-256;
- stored manifest SHA-256;
- content-addressed `run_manifest_id`;
- request hash embedded in the manifest artifact table;
- trajectory hash embedded in the manifest artifact table;
- protocol, schema, run, repository, and manifest identities shared by manifest and trajectory;
- request-bound model/backend/capture/determinism/generation settings; and
- trajectory representation semantics.

Recomputing only the outer trajectory/manifest hashes therefore cannot bless stale nested hashes or contradictory dimensions, counts, spans, or layer identities.

The writer does not update an existing capture directory in place. The destination must not already exist. It writes and fsyncs all three files inside a temporary sibling directory, fsyncs that directory, then publishes the complete directory with one same-filesystem rename and fsyncs the parent. If publication fails, the staging directory is removed. A previous bundle can therefore never be partially overwritten into a mixture of runs by this writer.

Published canonical bundle directories are treated as immutable records. A rerun receives a new output directory/run identity.

## 12. Software-only contract fixture

CI uses a deterministic test-double backend to exercise request validation, token-span semantics, requested-layer selection, pooling, provenance binding, bundle leaf checks, atomic publication behavior, and content hashes without shipping or downloading model weights.

That fixture is explicitly labelled:

`evidence_class: SIMULATION`

It is not a hidden-state observation from an LLM, even though it exercises the same protocol core used by production capture.

A fake backend, mocked tensor, synthetic vector, or frozen software fixture must never be promoted to `OBSERVATION` evidence.

## 13. Running a production capture

Install the optional capture dependencies:

```bash
python -m pip install -e '.[capture]'
```

Keep the edited request and run outputs **outside the source checkout** so the repository provenance check remains clean:

```bash
cp examples/GEO-CAP-001.example.json /tmp/GEO-CAP-001-request.json
```

Edit `/tmp/GEO-CAP-001-request.json`, replacing the placeholder Hub repository IDs and every zero revision with exact canonical `namespace/repository` identities and immutable 40-hex commits already present in the local Hugging Face cache. Freeze the step segmentation before inspecting target-labelled geometry.

Run to a **new, nonexistent** bundle directory:

```bash
qsol-geo-capture /tmp/GEO-CAP-001-request.json \
  --output-dir /tmp/GEO-CAP-001/run-001
```

For an archival result, copy the verified bundle into the repository or external evidence store only through an explicit later commit/artifact-record workflow that preserves the implementation revision used for capture.

No output produced from placeholder identities is valid production evidence.

## 14. Failure conditions

The capture is invalid if, among other cases:

- the request JSON root is not an object;
- model or tokenizer identity is a filesystem path rather than a canonical Hub repository ID;
- model or tokenizer revision is not a full immutable commit identity;
- `generation_parameters.generation_used` is absent or not explicitly false for this replay-only protocol;
- non-null generation settings are supplied while generation is declared unused;
- remote code or network loading is enabled;
- declared or checkpoint-embedded quantization is present in the canonical lane;
- checkpoint loading reports missing, unexpected, mismatched, or errored parameters;
- the loaded model/tokenizer reports a different commit from the frozen request;
- a step tokenizes to no changed token span;
- a requested hidden-state layer does not exist;
- the backend returns layers other than exactly the requested set;
- vector dimensions change unexpectedly;
- any captured or pooled value is non-finite;
- a material runtime/capture choice is absent from provenance;
- per-leaf hashes, dimensions, counts, spans, layer identities, bundle hashes, or cross-artifact identities disagree;
- an existing output directory would be overwritten; or
- an optimized serving path is substituted without an equivalence result or explicit experimental-factor treatment.

## 15. Phase 2A evidence gate

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

## 16. Claim ceiling

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

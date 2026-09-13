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

The authenticated package orchestrator enforces that boundary by comparing the final request against the committed template after removing exactly those two fields.

## Execution procedure

Use the frozen Python 3.11 capture-reference lane from a clean checkout. The virtual environment must be created **outside the repository checkout** so environment files cannot dirty the source tree that preparation later authenticates. Create the external environment, install the exact CPU PyTorch build under the complete runtime lock, install the capture extra under that same lock, then verify the installed closure before preparation:

```bash
python3.11 -m venv /tmp/qsol-geo-reason-capture-py311
. /tmp/qsol-geo-reason-capture-py311/bin/activate
python -m pip install \
  -c constraints/capture-reference-py311.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.2.2+cpu'
python -m pip install \
  -c constraints/capture-reference-py311.txt \
  -e '.[capture]' \
  'jsonschema==4.23.0'
python -B tools/verify_capture_reference_environment.py
python -m pip check
python --version  # must report Python 3.11.x
```

`constraints/capture-reference-py311.txt` is the authoritative complete resolved runtime lock for this Python 3.11 Linux x86_64 CPU reference lane. It pins the direct capture packages **and their transitive runtime closure**, including Torch dependencies, Transformers/Hugging Face dependencies, Requests/TLS dependencies, and the jsonschema validator closure. `tools/verify_capture_reference_environment.py` rejects a missing pin, version mismatch, unexpected non-bootstrap runtime distribution, non-CPython interpreter, non-Linux/x86_64 platform, or any interpreter outside Python 3.11. The `python --version` line is only a human-readable sanity check; the machine verifier enforces the interpreter requirement itself.

The standalone verifier is a preflight, not the provenance authority. `prepare` independently measures the same complete closure and the exact CPython patch version, then stores the self-hashed result **embedded in the preparation receipt**. For each content-bound locked distribution, the environment receipt combines the authenticated Python import surface with the distribution's installed runtime-file inventory, including sibling wheel payloads outside the import package such as `numpy.libs/*.so`. This covers bundled native dependencies such as NumPy's OpenBLAS/libgfortran/libquadmath payload when they are owned by the installed wheel, rather than assuming the `numpy/` directory alone is the executable identity. The same stronger distribution-owned receipt applies to `huggingface_hub`, the Requests/TLS chain, and the remaining locked transitive capture stack including NumPy, regex, PyYAML, filelock, fsspec, Jinja2, SymPy, NetworkX, packaging, referencing, rpds, tqdm, and typing-extensions. Torch, Transformers, tokenizers, and Safetensors retain their existing canonical capture-time package/runtime receipts. The union covers the full frozen 28-distribution executable reference closure. `observe` independently measures its live environment and requires exact equality with the embedded preparation receipt before any observation attempt and again before replay-verdict publication. A different compatible dependency resolution or modified distribution-owned runtime file must not be described as the selected reference environment.

### Authenticated launcher bootstrap

The working-tree `tools/run_first_production_observation.py` file is **not itself a trust root**. Do not execute it directly. A file hidden with Git `skip-worktree` or `assume-unchanged` could otherwise run top-level code before it had a chance to authenticate itself.

Define the following shell function once from the repository root. Its inline `-I -S -B -c` program is independent of the checkout: it resolves fixed system Git, requires a root-owned/non-group-or-world-writable executable on POSIX, hashes that executable before and after use, invokes Git with a minimal redirection-free environment, resolves one immutable `HEAD^{commit}` **before** reading the launcher, reads `<that-commit>:tools/run_first_production_observation.py`, computes the SHA-256 of those exact committed launcher bytes, and injects both the resolved commit and launcher digest into the committed launcher namespace. The working-tree launcher bytes are never loaded by this first boundary, and a later movement of `HEAD` cannot change the revision attributed to the already-running launcher.

```bash
QSOL_FIRST_OBSERVATION_BOOTSTRAP='import hashlib,os,pathlib,stat,subprocess,sys;root=pathlib.Path.cwd().resolve();path=root/"tools"/"run_first_production_observation.py";git=pathlib.Path("/usr/bin/git").resolve(strict=True);info=git.stat();(not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or bool(info.st_mode & (stat.S_IWGRP|stat.S_IWOTH))) and (_ for _ in ()).throw(RuntimeError("untrusted system Git executable"));before=hashlib.sha256(git.read_bytes()).hexdigest();env={"PATH":"/usr/bin:/bin","LC_ALL":"C","GIT_NO_REPLACE_OBJECTS":"1","GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":os.devnull,"GIT_OPTIONAL_LOCKS":"0","GIT_TERMINAL_PROMPT":"0"};commit=subprocess.run([str(git),"-C",str(root),"rev-parse","--verify","HEAD^{commit}"],env=env,check=True,capture_output=True).stdout.decode().strip();(len(commit)!=40 or any(c not in "0123456789abcdef" for c in commit)) and (_ for _ in ()).throw(RuntimeError("invalid authenticated launcher commit"));src=subprocess.run([str(git),"-C",str(root),"cat-file","blob",commit+":tools/run_first_production_observation.py"],env=env,check=True,capture_output=True).stdout;launcher_sha=hashlib.sha256(src).hexdigest();hashlib.sha256(git.read_bytes()).hexdigest()!=before and (_ for _ in ()).throw(RuntimeError("trusted Git executable changed during launcher authentication"));ns={"__name__":"__main__","__file__":str(path),"__package__":None,"_QSOL_AUTHENTICATED_BOOTSTRAP_REVISION":commit,"_QSOL_AUTHENTICATED_LAUNCHER_SHA256":launcher_sha};sys.argv=[str(path),*sys.argv[1:]];exec(compile(src,str(path),"exec"),ns,ns)'
qsol_first_observation() {
  python -I -S -B -c "$QSOL_FIRST_OBSERVATION_BOOTSTRAP" "$@"
}
```

The committed launcher re-reads `<bootstrap-commit>:tools/run_first_production_observation.py` through its independently trusted Git path and requires its SHA-256 to equal the bootstrap-supplied launcher digest. It then authenticates the complete tracked `src/qsol_geo_reason` tree against **that same bootstrap commit**, not ambient `HEAD`, re-hashes that source manifest again in the second no-site interpreter immediately before prepending `src`, and forces `--implementation-revision <bootstrap-commit>` through the package facade. If the checkout moves from commit A to B after the bootstrap starts, the A launcher cannot silently attribute B package bytes to itself: the working-tree/package checks fail closed unless they still equal A.

### 1. Online trusted preparation

Run once while the machine is online. The prepared request and its sidecar must remain **outside the source checkout**; the tool rejects checkout-contained output paths before revision resolution or Hub contact.

```bash
qsol_first_observation prepare \
  --output /tmp/GEO-CAP-001-EXP-001.request.json
```

Preparation authenticates the clean QSOL-GEO-REASON checkout **before** contacting the Hub. It also directly authenticates the working launcher pathname and the reference lock against the bootstrap-bound Git revision, so Git index hints such as `assume-unchanged` or `skip-worktree` cannot substitute either evidence-critical file.

The authenticated Git-blob bootstrap executes the committed launcher bytes in an initial `-I -S -B` interpreter. That launcher sanitizes Python/native-loader/Git/transport environment controls and execs a second `-I -S -B` interpreter. The second interpreter adds only the checked-out `src` tree and literal interpreter package directories without processing `.pth` files, authenticates every tracked package source against the bootstrap-bound commit before import, and re-hashes the authenticated source manifest immediately before `sys.path` exposure. Canonical Hub work then runs in a separate fresh `-I -S -B` preparation child. The child requires `huggingface_hub` and the Requests/TLS transport chain to be absent before their authenticated imports, content-hashes their distribution-owned runtime payloads plus import surfaces before import, verifies the same receipts after import and again after Hub work, pins the canonical public Hugging Face endpoint, warms the exact model/tokenizer commit, and creates the authenticated QSOL Hub tree artifacts.

A successful preparation publishes two immutable files:

```text
/tmp/GEO-CAP-001-EXP-001.request.json
/tmp/GEO-CAP-001-EXP-001.request.preparation-receipt.json
```

The preparation receipt is governed by `schemas/capture-preparation-receipt.schema.json`. It records the clean repository commit that executed preparation, the canonical Hugging Face endpoint, the finalized request SHA-256, the frozen model/tokenizer repository and commit identities, both Hub tree receipts, the complete self-hashed reference-environment receipt (CPython version, Linux/x86_64 lane, complete 28-distribution lock, lock SHA-256, Hub/transport package receipts, and capture-transitive package receipts), and its own self-hash. The final request is not considered a complete prepared input without this receipt.

The preparation receipt is published first and the finalized request second, both with no-replace durable JSON publication. The receipt is deliberately monotonic: if request publication fails, **the receipt is retained rather than deleted**. A receipt-only state is a defined recoverable state, and the next `prepare` can authenticate it and publish the missing request without contacting the Hub again. This also prevents a failing concurrent publisher from deleting provenance already used by another process to recover the request.

Preserve **both** preparation files together.

### 2. Offline dual observation and replay check

Disconnect network access if practical, or otherwise rely on the tool's forced Hugging Face/Transformers offline environment, then run:

```bash
qsol_first_observation observe \
  --request /tmp/GEO-CAP-001-EXP-001.request.json \
  --preparation-receipt /tmp/GEO-CAP-001-EXP-001.request.preparation-receipt.json \
  --output-root /tmp/GEO-CAP-001-EXP-001
```

`--preparation-receipt` may be omitted when the canonical sidecar name above has not changed; the orchestrator derives it from `--request`.

Before creating an observation attempt, the orchestrator verifies that the preparation receipt is self-consistent and matches the exact finalized request. It authenticates the current clean repository revision, launcher, frozen experiment template, and complete reference lock, then requires the live Python/runtime environment to equal the environment receipt embedded by `prepare`. Environment drift therefore fails before an evidence directory is created. The same environment relation is checked again during the observation and before replay-verdict publication.

The preparation revision and observation revision may be reviewed independently if the checkout legitimately advanced **between separate authenticated phase invocations**, but each invocation is internally bound to the single commit selected by its bootstrap. The experiment template and reference lock must authenticate at the executing revision and the observation runtime must remain identical to the prepared reference environment.

The orchestrator publishes immutable snapshots named `validated-request.json` and `preparation-receipt.json` inside the experiment directory. Both capture subprocesses consume the staged request snapshot. The intermediate capture CLI and fresh worker remain isolated no-site processes.

Each physical replay execution receives a distinct occurrence identity **before** capture. The planned run-a and run-b IDs are passed into replay-verdict construction and persisted-verdict verification. Each execution receipt must match its assigned planned slot exactly; merely having two distinct IDs is insufficient. Swapping the two receipt files, or substituting a receipt from another deterministic attempt with byte-identical bundle contents, therefore fails closed.

A successful experiment directory is:

```text
GEO-CAP-001-EXP-001/
├── validated-request.json
├── preparation-receipt.json
├── run-a/
│   ├── capture-request.json
│   ├── run-manifest.json
│   └── captured-trajectory.json
├── run-a-execution-receipt.json
├── run-b/
│   ├── capture-request.json
│   ├── run-manifest.json
│   └── captured-trajectory.json
├── run-b-execution-receipt.json
└── replay-verdict.json
```

The complete environment receipt is nested inside `preparation-receipt.json`; no loose environment sidecar is required. Both observation bundles are independently verified with the canonical semantic verifier. Each execution receipt is self-hashed and bound to its exact bundle and assigned replay occurrence, and `replay-verdict.json` binds both distinct execution identities and receipt-file hashes while evaluating deterministic replay from the three canonical scientific bundle files.

`replay-verdict.json` is governed by `schemas/replay-verdict.schema.json` and the semantic verifier `qsol_geo_reason.capture_replay.verify_replay_verdict`. The verifier recomputes the validated-request artifact hash, preparation receipt semantic/raw hashes, every canonical bundle-file hash, role-specific execution-receipt binding, manifest-receipt equality, trajectory-hash equality, repository-commit equality, and resulting replay outcome before accepting the verdict.

A completed divergence is retained. The tool writes `replay_outcome: "diverged"` and exits nonzero rather than tuning the request, deleting the differing run, or silently weakening the determinism requirement.

If a worker, verifier, or filesystem operation fails before a replay verdict exists, the incomplete attempt is not erased. The runner writes `execution-failure.json` where possible and moves the whole attempt to a uniquely named sibling such as:

```text
GEO-CAP-001-EXP-001.failed-<pid>-<nonce>/
```

The failure marker records both the preparation revision and the observation revision, together with the planned execution identities. The archived preparation receipt also preserves the complete prepared reference-environment evidence.

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
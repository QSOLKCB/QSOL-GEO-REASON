# GEO-CAP-001 Threat Model

Status: Phase 2A canonical capture security/reproducibility boundary.

This document defines what GEO-CAP-001 does and does not claim to defend against. It is normative for interpreting the production capture implementation and its provenance receipts.

## 1. Chosen architecture

The canonical user-facing production path is a **trusted local CLI that launches fresh isolated Python workers** for evidence-producing dependency work and each `OBSERVATION`.

`tools/run_first_production_observation.py` is a minimal environment-sanitizing launcher. The authenticated package orchestrator directly compares that launcher's working-tree bytes with the launcher blob at the bound Git revision before preparation or observation artifacts can be produced; Git index hints such as `assume-unchanged` and `skip-worktree` are not trusted for that check. The complete reference lock is authenticated the same way.

Online Hugging Face preparation does not import Hub code in the long-lived orchestrator. It launches a fresh `-I -S -B` preparation child. `-S` suppresses automatic `site` startup processing, so `sitecustomize`, `usercustomize`, and executable `.pth` files are not evaluated in that child. The bootstrap inserts only the checked-out `src` tree and literal interpreter site-package directories without processing `.pth` files. The child requires `huggingface_hub` to be absent before import, content-hashes its package tree before import, verifies identical package content after import, performs the canonical-endpoint warm-up, and verifies that package content again before returning the two Hub-tree receipts **plus the Hub package file count and package-content SHA-256 receipt**. The parent independently content-measures the same Hub package as part of the reference-environment receipt and rejects preparation unless the child's package provenance exactly matches that parent measurement.

For an actual observation, `qsol-geo-capture` launches `qsol_geo_reason.capture_worker` with CPython isolated mode (`-I`) and bytecode writes disabled (`-B`). The parent strips Python-path/startup controls and common native-loader injection variables before launch, forces Hugging Face and Transformers offline mode, and passes only the frozen request path, output path, occurrence identity, receipt destination, and implementation revision.

The observation worker rejects execution unless it is in isolated mode and unless Torch, Transformers, Hugging Face Hub, Tokenizers, and Safetensors are absent from `sys.modules` before the canonical backend stack is imported.

These fresh-process boundaries are the normal production architecture. The library API remains available for software tests and trusted embedding, but arbitrary hostile mutation inside an already-running embedding process is **not** the security boundary claimed by GEO-CAP-001.

## 2. Trusted components and assumptions

GEO-CAP-001 assumes the following are trusted for the duration of one observation:

- the operating-system kernel and process isolation;
- the account invoking the capture and its filesystem permissions;
- the selected CPython executable and its system installation;
- the QSOL-GEO-REASON checkout being authenticated;
- the system Git executable selected by the source-provenance implementation;
- the directly authenticated canonical launcher and reference-lock bytes;
- the local Hugging Face cache after its commit-tree receipts have been frozen;
- hardware and firmware below the observable runtime interfaces;
- the administrator/user not to grant a concurrent attacker equal or greater OS privileges during capture.

A root/administrator attacker, kernel compromise, debugger with process-write capability, DMA attack, malicious hypervisor, or arbitrary native code executing with equivalent privileges can invalidate assumptions below the Python layer and is outside this protocol's threat model.

## 3. Threats the implementation actively detects or rejects

Within the trusted-host model, the instrument fails closed on many accidental or lower-privilege mutations, including:

- dirty, mismatched, ignored, or substituted importable QSOL source;
- replacement Git objects and untrusted Git execution paths;
- a modified canonical production launcher or reference lock hidden by Git index stat hints;
- preloaded production Python dependencies at the fresh observation-import boundary;
- Python startup customization and preloaded Hub code at the no-site preparation boundary;
- changed PyTorch, Transformers, Hugging Face Hub, Tokenizers, and Safetensors execution dependencies where receipts are defined;
- online Hub preparation whose content-bound `huggingface_hub` package receipt differs from the parent reference-environment measurement;
- a Python runtime or installed distribution closure that differs from the prepared complete reference-environment receipt;
- snapshot bytes that do not match the frozen QSOL Hub commit-tree receipt;
- transient package/snapshot/runtime-library replacement evidence covered by stat/change-time and mapped-image stability receipts;
- registered Python/PyTorch execution hooks and supported execution-surface substitutions;
- drift in deterministic, CPU, CUDA, MPS, SDPA, thread, denormal, and related policies represented by the canonical backend;
- mapped CPU, CUDA, and MPS runtime-library drift where the protocol defines receipts;
- post-capture artifact mutation detected by canonical hashes and cross-artifact verification.

These checks are defense in depth and reproducibility evidence. They do not transform Python introspection into an OS sandbox.

## 4. Embedded-process mutation is outside the canonical security boundary

Python objects and module dictionaries are mutable by design. A caller that already has arbitrary code execution in the same interpreter can replace functions, descriptors, globals, import hooks, builtins, or native extension state. Historical Round hardening tests intentionally probe many such cases and remain valuable regression tests, but GEO-CAP-001 no longer claims that Python-level closure binding is a complete adversarial-process isolation mechanism.

For that reason, production documentation must distinguish:

- **canonical CLI observation** — fresh isolated worker plus provenance checks; and
- **direct in-process library use** — trusted embedding/test surface with defense-in-depth mutation detection, but no claim of protection against an adversary already executing arbitrary code in that interpreter.

## 5. Filesystem concurrency

The implementation authenticates package, snapshot, source, launcher, lock, and mapped-runtime state at multiple points and uses file identity/change-time evidence to detect many write/restore races. This is intended to catch accidental changes and concurrent mutations visible to the process.

Preparation publishes provenance receipt first and final request second using no-replace durable publication. A durable receipt-only state is intentionally recoverable. A publisher that later loses a concurrent race to publish the final request never deletes the shared receipt, because another process may already have used it to recover the request. This makes receipt-first publication monotonic across both abrupt crashes and concurrent recovery.

It is not a proof against an administrator/root attacker able to manipulate filesystem, kernel, mount namespace, process memory, or clock/stat semantics beneath those measurements. Such an attacker is outside scope.

## 6. What receipts prove

A receipt proves only the bytes/state actually covered by its documented measurement and the consistency relations checked by the canonical verifier. In particular:

- repository commit provenance binds the authenticated checkout under the source-identity rules;
- direct tracked-artifact checks bind the frozen experiment template, launcher, and complete reference lock to the executing revision independently of Git index hints;
- the preparation receipt binds the canonical Hub endpoint, immutable model/tokenizer tree receipts, the exact content receipt of the `huggingface_hub` package used for online preparation, and the complete measured reference environment;
- package receipts bind enumerated package artifacts;
- Hub tree receipts bind the exported commit-tree metadata and snapshot contents checked against it;
- runtime receipts bind the mapped runtime libraries selected by each platform-specific enumerator;
- manifest and trajectory hashes bind the canonical JSON artifacts;
- the replay verdict binds both the semantic preparation-receipt hash and its raw archived file hash, which transitively binds the nested reference-environment and online-Hub-package receipts into successful replay evidence.

A receipt does **not** prove that upstream maintainers, package indexes, compilers, hardware, firmware, the operating system, or the complete software supply chain are trustworthy.

## 7. Dependency/reference environment

The repository publishes `constraints/capture-reference-py311.txt` as the Phase 2A Python 3.11 Linux x86_64 CPU reference runtime lock. It contains the complete resolved runtime closure used by the reference lane, not only the top-level capture packages. CI applies that lock to the CPU PyTorch installation and to the editable capture installation, then runs `tools/verify_capture_reference_environment.py`, which rejects missing pins, version mismatches, unexpected non-bootstrap runtime distributions, non-CPython interpreters, interpreters outside Python 3.11, platforms outside Linux x86_64, and malformed Hub package-content provenance before the real backend integration begins.

The standalone verifier is a convenience preflight. Canonical `prepare` independently measures the exact live environment and embeds a self-hashed `reference_environment` object in `preparation-receipt.json`, including the CPython patch version, complete normalized distribution/version map, distribution count, lock SHA-256, and content-bound `huggingface_hub` package file count and SHA-256 receipt. The no-site preparation child must report the same Hub package provenance before its tree receipts are accepted. Canonical `observe` requires its live environment to exactly equal that prepared receipt before creating an observation attempt and rechecks the equality before replay-verdict publication. The replay verdict's existing preparation-receipt hashes therefore bind this complete environment and online-Hub-code evidence without a separate loose sidecar.

CI then builds a tiny local GPT-2-style model and fast tokenizer without downloading model weights, performs two real canonical CLI observations, validates their JSON schemas and semantic bundle verification, and requires deterministic equality. The integration request additionally binds the SHA-256 of the complete lock file in its `notes` field.

The complete runtime lock strengthens reproducibility of the selected reference lane; it does not by itself establish trust in package indexes, upstream releases, the Python interpreter, the operating system, or the wider software supply chain.

## 8. Historical Round modules

Numbered `capture_backend_round*` modules are retained temporarily as implementation history and regression strata. Public capture code no longer composes them directly; `capture_backend_canonical.py` is the single composition point. These modules are not separate protocol layers and should not be cited as the security boundary.

Future maintenance should consolidate behavior into cohesive non-numbered modules while preserving behavioral regressions. The merge criterion for this Phase 2A PR is that the public runtime composition is explicit, the canonical CLI process boundary is explicit, and the real production path is exercised by CI; historical regression files may remain until a follow-up mechanical consolidation can be reviewed independently.

## 9. Claim ceiling

GEO-CAP-001 is an auditable measurement instrument, not a sandbox, remote-attestation system, formal proof of runtime integrity, or proof of scientific interpretation. A valid `OBSERVATION` establishes that the declared capture procedure produced the recorded vectors under the measured conditions. It does not establish that those vectors represent reasoning, truth, cognition, geometry, gauge structure, or any other downstream hypothesis.

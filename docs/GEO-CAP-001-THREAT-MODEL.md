# GEO-CAP-001 Threat Model

Status: Phase 2A canonical capture security/reproducibility boundary.

This document defines what GEO-CAP-001 does and does not claim to defend against. It is normative for interpreting the production capture implementation and its provenance receipts.

## 1. Chosen architecture

The canonical user-facing production path is a **trusted local CLI that launches a fresh isolated Python worker** for each `OBSERVATION`.

`qsol-geo-capture` performs request validation and optional online Hub-tree preparation in its command process. For an actual observation it launches `qsol_geo_reason.capture_worker` with CPython isolated mode (`-I`) and bytecode writes disabled (`-B`). The parent strips Python-path/startup controls and common native-loader injection variables before launch, forces Hugging Face and Transformers offline mode, and passes only the frozen request path, output path, and optional implementation revision.

The worker rejects execution unless it is in isolated mode and unless Torch, Transformers, Hugging Face Hub, Tokenizers, and Safetensors are absent from `sys.modules` before the canonical backend stack is imported.

This fresh-process boundary is the normal production architecture. The library API remains available for software tests and trusted embedding, but arbitrary hostile mutation inside an already-running embedding process is **not** the security boundary claimed by GEO-CAP-001.

## 2. Trusted components and assumptions

GEO-CAP-001 assumes the following are trusted for the duration of one observation:

- the operating-system kernel and process isolation;
- the account invoking the capture and its filesystem permissions;
- the selected CPython executable and its system installation;
- the QSOL-GEO-REASON checkout being authenticated;
- the system Git executable selected by the source-provenance implementation;
- the local Hugging Face cache after its commit-tree receipts have been frozen;
- hardware and firmware below the observable runtime interfaces;
- the administrator/user not to grant a concurrent attacker equal or greater OS privileges during capture.

A root/administrator attacker, kernel compromise, debugger with process-write capability, DMA attack, malicious hypervisor, or arbitrary native code executing with equivalent privileges can invalidate assumptions below the Python layer and is outside this protocol's threat model.

## 3. Threats the implementation actively detects or rejects

Within the trusted-host model, the instrument fails closed on many accidental or lower-privilege mutations, including:

- dirty, mismatched, ignored, or substituted importable QSOL source;
- replacement Git objects and untrusted Git execution paths;
- preloaded production Python dependencies at the fresh-import boundary;
- changed PyTorch, Transformers, Hugging Face Hub, Tokenizers, and Safetensors execution dependencies where receipts are defined;
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

The implementation authenticates package, snapshot, source, and mapped-runtime state at multiple points and uses file identity/change-time evidence to detect many write/restore races. This is intended to catch accidental changes and concurrent mutations visible to the process.

It is not a proof against an administrator/root attacker able to manipulate filesystem, kernel, mount namespace, process memory, or clock/stat semantics beneath those measurements. Such an attacker is outside scope.

## 6. What receipts prove

A receipt proves only the bytes/state actually covered by its documented measurement and the consistency relations checked by the canonical verifier. In particular:

- repository commit provenance binds the authenticated checkout under the source-identity rules;
- package receipts bind enumerated package artifacts;
- Hub tree receipts bind the exported commit-tree metadata and snapshot contents checked against it;
- runtime receipts bind the mapped runtime libraries selected by each platform-specific enumerator;
- manifest and trajectory hashes bind the canonical JSON artifacts.

A receipt does **not** prove that upstream maintainers, package indexes, compilers, hardware, firmware, the operating system, or the complete software supply chain are trustworthy.

## 7. Dependency/reference environment

The repository publishes `constraints/capture-reference-py311.txt` as the Phase 2A Python 3.11 Linux x86_64 CPU reference runtime lock. It contains the complete resolved runtime closure used by the reference lane, not only the top-level capture packages. CI applies that lock to the CPU PyTorch installation and to the editable capture installation, then runs `tools/verify_capture_reference_environment.py`, which rejects missing pins, version mismatches, and unexpected non-bootstrap runtime distributions before the real backend integration begins.

CI then builds a tiny local GPT-2-style model and fast tokenizer without downloading model weights, performs two real canonical CLI observations, validates their JSON schemas and semantic bundle verification, and requires deterministic equality. The integration request binds the SHA-256 of the complete lock file in its `notes` field, while production manifests continue to record exact versions and content receipts for the principal execution packages.

The complete runtime lock strengthens reproducibility of the selected reference lane; it does not by itself establish trust in package indexes, upstream releases, the Python interpreter, the operating system, or the wider software supply chain.

## 8. Historical Round modules

Numbered `capture_backend_round*` modules are retained temporarily as implementation history and regression strata. Public capture code no longer composes them directly; `capture_backend_canonical.py` is the single composition point. These modules are not separate protocol layers and should not be cited as the security boundary.

Future maintenance should consolidate behavior into cohesive non-numbered modules while preserving behavioral regressions. The merge criterion for this Phase 2A PR is that the public runtime composition is explicit, the canonical CLI process boundary is explicit, and the real production path is exercised by CI; historical regression files may remain until a follow-up mechanical consolidation can be reviewed independently.

## 9. Claim ceiling

GEO-CAP-001 is an auditable measurement instrument, not a sandbox, remote-attestation system, formal proof of runtime integrity, or proof of scientific interpretation. A valid `OBSERVATION` establishes that the declared capture procedure produced the recorded vectors under the measured conditions. It does not establish that those vectors represent reasoning, truth, cognition, geometry, gauge structure, or any other downstream hypothesis.

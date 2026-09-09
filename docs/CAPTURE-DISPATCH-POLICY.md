# GEO-CAP-001 dispatch-policy implementation notes

These implementation notes supplement `protocols/GEO-CAP-001.md`. They do not
change any scientific invariant, exact-mathematics definition, evidence class,
frozen Phase 1 fixture, or empirical roadmap gate.

## Security and process boundary

The normative security/reproducibility assumptions are defined in
`docs/GEO-CAP-001-THREAT-MODEL.md`. The canonical user-facing `OBSERVATION` path is a
trusted local CLI that launches a fresh CPython isolated worker for each capture.
Python-level execution-surface checks described below are defense in depth inside that
worker; they are not presented as a sandbox against an adversary who already has
arbitrary code execution in the same interpreter.

The public capture facade imports one explicit backend composition point,
`capture_backend_canonical.py`. Historical `capture_backend_round*` files remain as
reviewed regression strata during Phase 2A migration but are not separate public
backends or security boundaries.

## CUDA FP16 accumulation

Canonical CUDA execution disables `torch.backends.cuda.matmul.allow_fp16_accumulation`
when that API exists, independently of the FP16/BF16 reduced-precision-reduction
switches. The policy is verified after each forward and before metadata emission.
Construction and observation cleanup restore the caller's original setting.
`cuda_matmul_allow_fp16_accumulation` is false when the control exists and null
when the installed build does not expose it or the request is not CUDA. An exposed
non-boolean control or a changed availability state fails closed.

## Effective CPU dispatch

`cpu_aten_capability` is the effective result of
`torch.backends.cpu.get_cpu_capability()`, not an inference from processor flags.
The runtime probe must succeed and return a supported dispatch identity. Its
value and the current override environment are checked for drift throughout
capture. Different effective dispatch identities therefore produce different
content-addressed manifests even when hardware and build metadata agree.

When construction owns the first PyTorch import, `aten_cpu_capability_env_known`
is true and `aten_cpu_capability_env` records the initialization-time override;
null in that case means the override was absent. When PyTorch was already
imported, the historical override is not observable: the known flag is false
and the historical field is null. The effective runtime probe is still mandatory.
A later environment string is never presented as a known initialization-time
setting. These three fields are null outside CPU observations.

The independent oneDNN/MKL initialization controls `ONEDNN_MAX_CPU_ISA`,
`DNNL_MAX_CPU_ISA`, and `MKL_CBWR` are frozen on the same boundary. When the
capture owns the first PyTorch import their exact values (including absence) are
recorded and checked for drift. If PyTorch was already imported, their effective
historical values are not recoverable: `cpu_math_dispatch_env_known` is false and
the three value fields remain null. All four fields are null outside CPU
observations.

## SDPA on CPU, CUDA, and MPS

Every canonical SDPA capture uses the same math-only selector policy. Flash,
memory-efficient, and available cuDNN SDPA are disabled, and math-SDPA reduced
FP16/BF16 reductions are disabled. The existing CUDA guards retain their role.
CPU/MPS base-model calls now force and verify that policy at the actual call
boundary. The observation's ambient-state cleanup restores selector settings,
including on forward failure. Missing required selector APIs fail closed.

The four `sdpa_*_enabled` provenance fields describe this enforced policy on all
three devices; they are null only for non-SDPA attention. Matching conditions are
encoded in the run-manifest schema and canonical verifier.

## Referenced Python execution dependencies

Executable authentication binds forwards, self-resolved Python helper methods,
instance overrides, and their referenced global and statically qualified callable
dependencies. Python function dependencies are traversed transitively with cycle
and work limits. Callable registries and closure bindings are included; function
identity and complete marshalled code distinguish constant-only code changes.
No model tensor storage is read by this dependency traversal.

This is an in-process mutation detector for the declared Python execution surface,
not an OS/native-code sandbox or a proof that arbitrary dynamic Python lookup,
external native libraries, or the complete software supply chain are immutable.
The canonical CLI avoids inheriting an application's existing Python dependency state
by launching the isolated worker described above. Direct in-process library use is a
trusted embedding/test surface and does not inherit the stronger fresh-process claim.
Tests use software fixtures unless explicitly identified as real production integration
checks. They do not constitute an empirical language-model observation.

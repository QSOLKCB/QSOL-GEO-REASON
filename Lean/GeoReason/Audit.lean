import Lean.CoreM
import Lean.Environment
import Lean.Util.CollectAxioms

/-!
# Phase 1 protected formal-proof audit

This file is an executable audit runner. It deliberately does **not** import
`GeoReason` in its source header. Instead, `main` loads the protected
`GeoReason.olean` graph with `Lean.withImportModules`, whose `loadExts := false`
default does not execute imported project `initialize` actions.

The runner then checks all twelve theorem declarations directly in the imported
kernel environment. Every target must exist as a theorem and its transitive
axiom set must be contained in the explicit classical trust-base allowlist:

- `propext`
- `Classical.choice`
- `Quot.sound`

Because `sorryAx` is not allowed, the same positive allowlist also establishes
sorry-freedom. The completion marker is emitted only after every target passes.
-/

open Lean

private def auditedTargets : Array Name := #[
  `GeoReason.GEO_LEAN_TGT_001,
  `GeoReason.GEO_LEAN_TGT_002,
  `GeoReason.GEO_LEAN_TGT_003,
  `GeoReason.GEO_LEAN_TGT_004,
  `GeoReason.GEO_LEAN_TGT_005,
  `GeoReason.GEO_LEAN_TGT_006,
  `GeoReason.GEO_LEAN_TGT_007,
  `GeoReason.GEO_LEAN_TGT_008,
  `GeoReason.GEO_LEAN_TGT_009,
  `GeoReason.GEO_LEAN_TGT_010,
  `GeoReason.GEO_LEAN_TGT_011,
  `GeoReason.GEO_LEAN_TGT_012
]

private def allowedAxioms : Array Name := #[
  `propext,
  `Classical.choice,
  `Quot.sound
]

private def auditTargets (env : Environment) : IO Unit :=
  Core.CoreM.toIO'
    (ctx := { fileName := "protected-audit", fileMap := default })
    (s := { env }) do
      unless auditedTargets.size == 12 do
        throwError "protected audit target table must contain exactly 12 declarations"

      for target in auditedTargets do
        let some info := (← getEnv).find? target
          | throwError "missing protected audit declaration {target}"
        match info with
        | .thmInfo _ => pure ()
        | _ => throwError "protected audit declaration {target} is not a theorem"

        let axioms ← Lean.collectAxioms target
        for axiom in axioms do
          unless allowedAxioms.contains axiom do
            throwError "{target} depends on disallowed axiom {axiom}"

        IO.println s!"'{target}' depends on axioms: {axioms}"

unsafe def main : IO Unit := do
  Lean.withImportModules #[{ module := `GeoReason : Lean.Import }] {} fun env => do
    auditTargets env
    IO.println
      "QSOL_PROTECTED_AUDIT_COMPLETE targets=12 theorem_kinds=verified axiom_allowlist=verified project_initializers=not_executed"

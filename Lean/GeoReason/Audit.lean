import GeoReason
import Mathlib.Util.PrintSorries
import Lean.Util.CollectAxioms

/-!
# Phase 1 formal-proof audit

This file is executed directly by CI after `lake build --wfail`.
It audits the twelve theorem targets frozen by immutable release `v0.1.0`.

The audit has two independent compiled-environment checks:

1. `#print sorries` recursively checks the declaration graph for `sorryAx`.
2. `#audit_axioms` collects the transitive kernel axioms of every frozen
   theorem and permits only the explicit classical Lean trust base:
   `propext`, `Classical.choice`, and `Quot.sound`.

The second check is an allowlist, not a denylist. Therefore a project axiom
introduced through `constant`, a macro, or any other declaration mechanism is
rejected if any frozen theorem depends on it, even when its source spelling
would evade a textual `axiom` scan.
-/

open Lean Elab Command in
elab "#audit_axioms " ids:ident+ : command => do
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  for id in ids do
    let name ← liftCoreM <| Lean.Elab.realizeGlobalConstNoOverloadWithInfo id
    let axioms ← Lean.collectAxioms name
    for ax in axioms do
      unless allowed.contains ax do
        throwError "{name} depends on disallowed axiom {ax}"

#print sorries
  GeoReason.GEO_LEAN_TGT_001
  GeoReason.GEO_LEAN_TGT_002
  GeoReason.GEO_LEAN_TGT_003
  GeoReason.GEO_LEAN_TGT_004
  GeoReason.GEO_LEAN_TGT_005
  GeoReason.GEO_LEAN_TGT_006
  GeoReason.GEO_LEAN_TGT_007
  GeoReason.GEO_LEAN_TGT_008
  GeoReason.GEO_LEAN_TGT_009
  GeoReason.GEO_LEAN_TGT_010
  GeoReason.GEO_LEAN_TGT_011
  GeoReason.GEO_LEAN_TGT_012

#audit_axioms
  GeoReason.GEO_LEAN_TGT_001
  GeoReason.GEO_LEAN_TGT_002
  GeoReason.GEO_LEAN_TGT_003
  GeoReason.GEO_LEAN_TGT_004
  GeoReason.GEO_LEAN_TGT_005
  GeoReason.GEO_LEAN_TGT_006
  GeoReason.GEO_LEAN_TGT_007
  GeoReason.GEO_LEAN_TGT_008
  GeoReason.GEO_LEAN_TGT_009
  GeoReason.GEO_LEAN_TGT_010
  GeoReason.GEO_LEAN_TGT_011
  GeoReason.GEO_LEAN_TGT_012

/- Human-readable kernel dependency log retained for review evidence. -/
#print axioms GeoReason.GEO_LEAN_TGT_001
#print axioms GeoReason.GEO_LEAN_TGT_002
#print axioms GeoReason.GEO_LEAN_TGT_003
#print axioms GeoReason.GEO_LEAN_TGT_004
#print axioms GeoReason.GEO_LEAN_TGT_005
#print axioms GeoReason.GEO_LEAN_TGT_006
#print axioms GeoReason.GEO_LEAN_TGT_007
#print axioms GeoReason.GEO_LEAN_TGT_008
#print axioms GeoReason.GEO_LEAN_TGT_009
#print axioms GeoReason.GEO_LEAN_TGT_010
#print axioms GeoReason.GEO_LEAN_TGT_011
#print axioms GeoReason.GEO_LEAN_TGT_012

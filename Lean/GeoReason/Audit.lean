import GeoReason
import Mathlib.Util.PrintSorries

/-!
# Phase 1 formal-proof audit

This file is executed directly by CI after `lake build --wfail`.
It audits the twelve theorem targets frozen by immutable release `v0.1.0`.

`#print sorries` recursively checks the compiled declaration graph for
`sorryAx`. `#print axioms` records the kernel axiom dependencies so CI can
reject proof escapes outside the explicitly allowed Lean/mathlib trust base.
-/

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

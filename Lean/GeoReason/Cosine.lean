import Mathlib
import GeoReason.Trajectory

namespace GeoReason

noncomputable section

open scoped RealInnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Exact project cosine convention from GEO-MATH-004. -/
def extendedCosine (u v : V) : ℝ :=
  if u = 0 then
    if v = 0 then 1 else 0
  else if v = 0 then
    0
  else
    ⟪u, v⟫_ℝ / (‖u‖ * ‖v‖)

/-- GEO-LEAN-TGT-006 — cosine bounds for the exact project convention. -/
theorem GEO_LEAN_TGT_006 (u v : V) :
    -1 ≤ extendedCosine u v ∧ extendedCosine u v ≤ 1 := by
  by_cases hu : u = 0
  · subst u
    by_cases hv : v = 0 <;> simp [extendedCosine, hv]
  · by_cases hv : v = 0
    · subst v
      simp [extendedCosine, hu]
    · have h := abs_real_inner_div_norm_mul_norm_le_one u v
      exact abs_le.mp (by simpa [extendedCosine, hu, hv] using h)

/-- GEO-LEAN-TGT-007 — Euclidean linear isometries preserve the project cosine. -/
theorem GEO_LEAN_TGT_007 (Q : V ≃ₗᵢ[ℝ] V) (u v : V) :
    extendedCosine (Q u) (Q v) = extendedCosine u v := by
  by_cases hu : u = 0
  · subst u
    simp [extendedCosine]
  · by_cases hv : v = 0
    · subst v
      simp [extendedCosine, hu]
    · have hQu : Q u ≠ 0 := by
        intro h
        apply hu
        apply Q.injective
        simpa using h
      have hQv : Q v ≠ 0 := by
        intro h
        apply hv
        apply Q.injective
        simpa using h
      simp [extendedCosine, hu, hv, hQu, hQv]

end

end GeoReason

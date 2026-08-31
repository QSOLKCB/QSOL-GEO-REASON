import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.LinearMap
import GeoReason.Trajectory

namespace GeoReason

noncomputable section

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Exact project cosine convention from GEO-MATH-004. -/
noncomputable def extendedCosine (u v : V) : ℝ := by
  classical
  exact
    if u = 0 then
      if v = 0 then 1 else 0
    else if v = 0 then
      0
    else
      inner ℝ u v / (‖u‖ * ‖v‖)

/-- GEO-LEAN-TGT-006 — cosine bounds for the exact project convention. -/
theorem GEO_LEAN_TGT_006 (u v : V) :
    -1 ≤ extendedCosine u v ∧ extendedCosine u v ≤ 1 := by
  classical
  by_cases hu : u = 0
  · subst u
    by_cases hv : v = 0 <;> simp [extendedCosine, hv]
  · by_cases hv : v = 0
    · subst v
      simp [extendedCosine, hu]
    · have habs : |inner ℝ u v / (‖u‖ * ‖v‖)| ≤ 1 := by
        simpa using abs_real_inner_div_norm_mul_norm_le_one u v
      simpa [extendedCosine, hu, hv] using (abs_le.mp habs)

/-- GEO-LEAN-TGT-007 — Euclidean linear isometries preserve the project cosine. -/
theorem GEO_LEAN_TGT_007 (Q : V ≃ₗᵢ[ℝ] V) (u v : V) :
    extendedCosine (Q u) (Q v) = extendedCosine u v := by
  classical
  by_cases hu : u = 0
  · subst u
    simp [extendedCosine]
  · by_cases hv : v = 0
    · subst v
      simp [extendedCosine, hu]
    · simp [extendedCosine, hu, hv]

end

end GeoReason

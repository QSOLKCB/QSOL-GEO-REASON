import Mathlib
import GeoReason.Trajectory

namespace GeoReason

noncomputable section

open Affine

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Exact Menger curvature for an ordered triple, matching GEO-MATH-006.

For an affinely independent triple this is the reciprocal of its Euclidean
circumradius. Affinely dependent triples, including repeated or collinear
points, use the project convention `κ = 0`.
-/
noncomputable def mengerCurvature (p : Fin 3 → V) : ℝ := by
  classical
  exact if h : AffineIndependent ℝ p then
    1 / (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumradius
  else
    0

/-- Affine Euclidean isometries preserve the exact project Menger curvature. -/
theorem mengerCurvature_affineIsometry (f : V →ᵃⁱ[ℝ] V) (p : Fin 3 → V) :
    mengerCurvature (fun i => f (p i)) = mengerCurvature p := by
  classical
  by_cases h : AffineIndependent ℝ p
  · have hfp : AffineIndependent ℝ (fun i => f (p i)) := by
      simpa [Function.comp_def] using h.map' f.toAffineMap f.injective
    unfold mengerCurvature
    rw [dif_pos hfp, dif_pos h]
    let s : Affine.Simplex ℝ V 2 := ⟨p, h⟩
    have hs :
        (⟨fun i => f (p i), hfp⟩ : Affine.Simplex ℝ V 2) =
          s.map f.toAffineMap f.injective := by
      apply Affine.Simplex.ext
      intro i
      rfl
    rw [hs, Affine.Simplex.circumradius_map]
  · have hfp : ¬ AffineIndependent ℝ (fun i => f (p i)) := by
      intro hbad
      apply h
      exact AffineIndependent.of_comp f.toAffineMap (by
        simpa [Function.comp_def] using hbad)
    unfold mengerCurvature
    rw [dif_neg hfp, dif_neg h]

/-- The affine Euclidean isometry `x ↦ Q x + t`. -/
def linearTranslateIsometry (Q : V ≃ₗᵢ[ℝ] V) (t : V) : V →ᵃⁱ[ℝ] V :=
  (AffineIsometryEquiv.mk' (fun x => Q x + t) Q 0 (by
    intro x
    simp)).toAffineIsometry

@[simp]
theorem linearTranslateIsometry_apply (Q : V ≃ₗᵢ[ℝ] V) (t x : V) :
    linearTranslateIsometry Q t x = Q x + t :=
  rfl

/-- GEO-LEAN-TGT-009 — Menger curvature is invariant under `x ↦ Qx+t`. -/
theorem GEO_LEAN_TGT_009 (Q : V ≃ₗᵢ[ℝ] V) (t : V) (p : Fin 3 → V) :
    mengerCurvature (fun i => Q (p i) + t) = mengerCurvature p := by
  simpa using mengerCurvature_affineIsometry (linearTranslateIsometry Q t) p

/-- GEO-LEAN-TGT-008 — Menger curvature is translation invariant. -/
theorem GEO_LEAN_TGT_008 (t : V) (p : Fin 3 → V) :
    mengerCurvature (fun i => p i + t) = mengerCurvature p := by
  simpa using GEO_LEAN_TGT_009 (LinearIsometryEquiv.refl ℝ V) t p

/-- GEO-LEAN-TGT-011 — a nondegenerate triple on a Euclidean circle of
radius `r` has Menger curvature `1/r`.

The center is required to lie in the affine span of the triple, making this a
circle in the triple's affine plane rather than a higher-dimensional sphere
with an off-plane center.
-/
theorem GEO_LEAN_TGT_011 (p : Fin 3 → V) (h : AffineIndependent ℝ p)
    {c : V} {r : ℝ} (_hr : 0 < r)
    (hc : c ∈ affineSpan ℝ (Set.range p))
    (hon : ∀ i, dist (p i) c = r) :
    mengerCurvature p = 1 / r := by
  classical
  let s : Affine.Simplex ℝ V 2 := ⟨p, h⟩
  have hc' : c ∈ affineSpan ℝ (Set.range s.points) := by
    simpa [s] using hc
  have hon' : ∀ i, dist (s.points i) c = r := by
    intro i
    simpa [s] using hon i
  have hradius : r = s.circumradius :=
    s.eq_circumradius_of_dist_eq hc' hon'
  unfold mengerCurvature
  rw [dif_pos h]
  change 1 / s.circumradius = 1 / r
  rw [← hradius]

end

end GeoReason

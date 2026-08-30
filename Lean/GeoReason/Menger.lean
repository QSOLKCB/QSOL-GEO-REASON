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

/-- Multiplication by a nonzero real scalar as an affine equivalence. -/
def scaleAffineEquiv (s : ℝ) (hs : s ≠ 0) : V ≃ᵃ[ℝ] V where
  toEquiv := (LinearEquiv.smulOfNeZero ℝ V s hs).toEquiv
  linear := LinearEquiv.smulOfNeZero ℝ V s hs
  map_vadd' p v := by
    change s • (v + p) = s • v + s • p
    exact smul_add s v p

@[simp]
theorem scaleAffineEquiv_apply (s : ℝ) (hs : s ≠ 0) (x : V) :
    scaleAffineEquiv s hs x = s • x := by
  rfl

/-- Circumradius scales by the absolute value of a nonzero scalar. -/
theorem circumradius_scale (s : ℝ) (hs : s ≠ 0) (p : Fin 3 → V)
    (h : AffineIndependent ℝ p) :
    (⟨fun i => s • p i, by
        simpa [Function.comp_def] using
          h.map' (scaleAffineEquiv s hs).toAffineMap (scaleAffineEquiv s hs).injective⟩ :
      Affine.Simplex ℝ V 2).circumradius =
      |s| * (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumradius := by
  let e : V ≃ᵃ[ℝ] V := scaleAffineEquiv s hs
  have hsp : AffineIndependent ℝ (fun i => s • p i) := by
    simpa [e, Function.comp_def] using h.map' e.toAffineMap e.injective
  let base : Affine.Simplex ℝ V 2 := ⟨p, h⟩
  let scaled : Affine.Simplex ℝ V 2 := ⟨fun i => s • p i, hsp⟩
  have hcMap : e base.circumcenter ∈ affineSpan ℝ (Set.range (e ∘ p)) := by
    rw [Set.range_comp, ← AffineSubspace.map_span]
    exact AffineSubspace.mem_map_of_mem _ base.circumcenter_mem_affineSpan
  have hc : s • base.circumcenter ∈ affineSpan ℝ (Set.range fun i => s • p i) := by
    simpa [e, Function.comp_def] using hcMap
  have hd : ∀ i, dist ((fun j => s • p j) i) (s • base.circumcenter) =
      |s| * base.circumradius := by
    intro i
    calc
      dist (s • p i) (s • base.circumcenter) =
          ‖s • (p i - base.circumcenter)‖ := by
            simp [dist_eq_norm, smul_sub]
      _ = |s| * ‖p i - base.circumcenter‖ := by
            simp [norm_smul, Real.norm_eq_abs]
      _ = |s| * dist (p i) base.circumcenter := by
            rw [dist_eq_norm]
      _ = |s| * base.circumradius := by
            rw [base.dist_circumcenter_eq_circumradius]
  have hradius : |s| * base.circumradius = scaled.circumradius :=
    scaled.eq_circumradius_of_dist_eq hc hd
  change scaled.circumradius = |s| * base.circumradius
  exact hradius.symm

/-- GEO-LEAN-TGT-010 — Menger scaling law. -/
theorem GEO_LEAN_TGT_010 (s : ℝ) (hs : s ≠ 0) (p : Fin 3 → V) :
    mengerCurvature (fun i => s • p i) = (1 / |s|) * mengerCurvature p := by
  classical
  by_cases h : AffineIndependent ℝ p
  · have hsp : AffineIndependent ℝ (fun i => s • p i) := by
      simpa [Function.comp_def] using
        h.map' (scaleAffineEquiv s hs).toAffineMap (scaleAffineEquiv s hs).injective
    let base : Affine.Simplex ℝ V 2 := ⟨p, h⟩
    let scaled : Affine.Simplex ℝ V 2 := ⟨fun i => s • p i, hsp⟩
    have hradius : scaled.circumradius = |s| * base.circumradius := by
      simpa [base, scaled] using circumradius_scale s hs p h
    unfold mengerCurvature
    rw [dif_pos hsp, dif_pos h]
    change 1 / scaled.circumradius = (1 / |s|) * (1 / base.circumradius)
    rw [hradius]
    field_simp [abs_ne_zero.mpr hs, base.circumradius_pos.ne']
  · have hsp : ¬ AffineIndependent ℝ (fun i => s • p i) := by
      intro hbad
      apply h
      exact AffineIndependent.of_comp (scaleAffineEquiv s hs).toAffineMap (by
        simpa [Function.comp_def] using hbad)
    unfold mengerCurvature
    rw [dif_neg hsp, dif_neg h]
    simp

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

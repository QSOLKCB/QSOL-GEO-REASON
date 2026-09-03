import Mathlib.Geometry.Euclidean.Angle.Sphere
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Nlinarith

namespace GeoReason

noncomputable section

open Affine

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Exact unsigned area of the ordered triangle, using the standard
`A = ab sin(θ) / 2` formula at the middle point. -/
noncomputable def mengerTriangleArea (p : Fin 3 → V) : ℝ :=
  (dist (p 0) (p 1) * dist (p 1) (p 2) *
      Real.sin (EuclideanGeometry.angle (p 0) (p 1) (p 2))) / 2

/-- Product `abc` of the three side lengths in GEO-MATH-006. -/
def mengerSideProduct (p : Fin 3 → V) : ℝ :=
  dist (p 0) (p 1) * dist (p 1) (p 2) * dist (p 0) (p 2)

/-- GEO-MATH-006, defined from the frozen `4A/(abc)` formula.

For an affinely independent triple, `A` is `mengerTriangleArea` and `a b c`
is `mengerSideProduct`. Affinely dependent triples, including repeated or
collinear points, use the frozen project convention `κ = 0`.
-/
noncomputable def mengerCurvature (p : Fin 3 → V) : ℝ := by
  classical
  exact if AffineIndependent ℝ p then
    4 * mengerTriangleArea p / mengerSideProduct p
  else
    0

/-- The circumradius presentation used to derive transformation laws. This is
not the definition of project curvature; its equivalence to the frozen
`4A/(abc)` definition is proved below. -/
noncomputable def circumradiusMengerCurvature (p : Fin 3 → V) : ℝ := by
  classical
  exact if h : AffineIndependent ℝ p then
    1 / (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumradius
  else
    0

/-- For a nondegenerate triple, the frozen `4A/(abc)` definition equals the
reciprocal circumradius. This is the formal bridge asserted by GEO-MATH-006. -/
theorem mengerCurvature_of_affineIndependent (p : Fin 3 → V)
    (h : AffineIndependent ℝ p) :
    mengerCurvature p =
      1 / (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumradius := by
  classical
  let s : Affine.Simplex ℝ V 2 := ⟨p, h⟩
  have h01 : (0 : Fin 3) ≠ 1 := by decide
  have h02 : (0 : Fin 3) ≠ 2 := by decide
  have h12 : (1 : Fin 3) ≠ 2 := by decide
  have hp01 : p 0 ≠ p 1 := h.injective.ne h01
  have hp02 : p 0 ≠ p 2 := h.injective.ne h02
  have hp12 : p 1 ≠ p 2 := h.injective.ne h12
  have hd01 : dist (p 0) (p 1) ≠ 0 := dist_ne_zero.mpr hp01
  have hd02 : dist (p 0) (p 2) ≠ 0 := dist_ne_zero.mpr hp02
  have hd12 : dist (p 1) (p 2) ≠ 0 := dist_ne_zero.mpr hp12
  have hsineLaw :
      dist (p 0) (p 2) /
          Real.sin (EuclideanGeometry.angle (p 0) (p 1) (p 2)) =
        2 * s.circumradius := by
    simpa [s] using
      s.dist_div_sin_angle_eq_two_mul_circumradius h01 h02 h12
  have hsin :
      Real.sin (EuclideanGeometry.angle (p 0) (p 1) (p 2)) ≠ 0 := by
    intro hzero
    rw [hzero, div_zero] at hsineLaw
    nlinarith [s.circumradius_pos]
  have hchord :
      dist (p 0) (p 2) =
        (2 * s.circumradius) *
          Real.sin (EuclideanGeometry.angle (p 0) (p 1) (p 2)) :=
    (div_eq_iff hsin).mp hsineLaw
  unfold mengerCurvature
  rw [if_pos h]
  change
    4 *
          ((dist (p 0) (p 1) * dist (p 1) (p 2) *
              Real.sin (EuclideanGeometry.angle (p 0) (p 1) (p 2))) /
            2) /
        (dist (p 0) (p 1) * dist (p 1) (p 2) * dist (p 0) (p 2)) =
      1 / s.circumradius
  calc
    _ =
        2 * Real.sin (EuclideanGeometry.angle (p 0) (p 1) (p 2)) /
          dist (p 0) (p 2) := by
            field_simp [hd01, hd02, hd12]
            <;> ring
    _ = 1 / s.circumradius := by
      field_simp [hd02, s.circumradius_pos.ne']
      nlinarith [hchord]

/-- The area/side-length definition and the circumradius presentation agree on
both the nondegenerate branch and the frozen zero-convention branch. -/
theorem mengerCurvature_eq_circumradiusMengerCurvature (p : Fin 3 → V) :
    mengerCurvature p = circumradiusMengerCurvature p := by
  classical
  by_cases h : AffineIndependent ℝ p
  · rw [mengerCurvature_of_affineIndependent p h]
    simp [circumradiusMengerCurvature, h]
  · simp [mengerCurvature, circumradiusMengerCurvature, h]

/-- Affine Euclidean isometries preserve the exact project Menger curvature. -/
theorem mengerCurvature_affineIsometry (f : V →ᵃⁱ[ℝ] V) (p : Fin 3 → V) :
    mengerCurvature (fun i => f (p i)) = mengerCurvature p := by
  classical
  by_cases h : AffineIndependent ℝ p
  · have hfp : AffineIndependent ℝ (fun i => f (p i)) := by
      simpa [Function.comp_def] using h.map' f.toAffineMap f.injective
    rw [mengerCurvature_of_affineIndependent _ hfp,
      mengerCurvature_of_affineIndependent _ h]
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
    simp [mengerCurvature, h, hfp]

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
    have hmem : e.toAffineMap base.circumcenter ∈
        (affineSpan ℝ (Set.range p)).map e.toAffineMap :=
      AffineSubspace.mem_map_of_mem _ base.circumcenter_mem_affineSpan
    simpa [AffineSubspace.map_span, Set.range_comp] using hmem
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
    rw [mengerCurvature_of_affineIndependent _ hsp,
      mengerCurvature_of_affineIndependent _ h]
    change 1 / scaled.circumradius = (1 / |s|) * (1 / base.circumradius)
    rw [hradius]
    field_simp [abs_ne_zero.mpr hs, base.circumradius_pos.ne']
  · have hsp : ¬ AffineIndependent ℝ (fun i => s • p i) := by
      intro hbad
      apply h
      exact AffineIndependent.of_comp (scaleAffineEquiv s hs).toAffineMap (by
        simpa [Function.comp_def] using hbad)
    simp [mengerCurvature, h, hsp]

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
  rw [mengerCurvature_of_affineIndependent p h]
  change 1 / s.circumradius = 1 / r
  rw [← hradius]

end

end GeoReason

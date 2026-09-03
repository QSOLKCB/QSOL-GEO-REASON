import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Tactic.Abel
import Mathlib.Tactic.FieldSimp

namespace GeoReason

noncomputable section

open Affine
open scoped RealInnerProductSpace

variable {V : Type*} [NormedAddCommGroup V]

private def mengerEdge01 (p : Fin 3 → V) : V :=
  p 1 - p 0

private def mengerEdge02 (p : Fin 3 → V) : V :=
  p 2 - p 0

private def mengerEdge12 (p : Fin 3 → V) : V :=
  p 2 - p 1

private theorem mengerEdge02_sub_edge01 (p : Fin 3 → V) :
    mengerEdge02 p - mengerEdge01 p = mengerEdge12 p := by
  unfold mengerEdge02 mengerEdge01 mengerEdge12
  abel

variable [InnerProductSpace ℝ V]

/-- Gram determinant of the two edges based at the first point. -/
noncomputable def mengerGramDet (p : Fin 3 → V) : ℝ :=
  ‖mengerEdge01 p‖ ^ 2 * ‖mengerEdge02 p‖ ^ 2 -
    (⟪mengerEdge01 p, mengerEdge02 p⟫_ℝ) ^ 2

/-- Exact unsigned area of the ordered triangle.

This is the standard inner-product-space area formula
`A = sqrt (‖u‖² ‖v‖² - ⟪u,v⟫²) / 2`.
-/
noncomputable def mengerTriangleArea (p : Fin 3 → V) : ℝ :=
  Real.sqrt (mengerGramDet p) / 2

/-- Product `abc` of the three side lengths in GEO-MATH-006. -/
def mengerSideProduct (p : Fin 3 → V) : ℝ :=
  ‖mengerEdge01 p‖ * ‖mengerEdge12 p‖ * ‖mengerEdge02 p‖

/-- GEO-MATH-006, defined from the frozen `4A/(abc)` formula.

Affinely dependent triples, including repeated or collinear points, use
project convention `κ = 0`.
-/
noncomputable def mengerCurvature (p : Fin 3 → V) : ℝ := by
  classical
  exact if AffineIndependent ℝ p then
    4 * mengerTriangleArea p / mengerSideProduct p
  else
    0

private theorem circumcenter_sub_base_mem_span_edges (p : Fin 3 → V)
    (h : AffineIndependent ℝ p) :
    ∃ a b : ℝ,
      a • mengerEdge01 p + b • mengerEdge02 p =
        (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumcenter - p 0 := by
  classical
  let s : Affine.Simplex ℝ V 2 := ⟨p, h⟩
  have hcenter : s.circumcenter ∈ affineSpan ℝ (Set.range p) := by
    change s.circumcenter ∈ affineSpan ℝ (Set.range s.points)
    exact s.circumcenter_mem_affineSpan
  have hbase : p 0 ∈ affineSpan ℝ (Set.range p) :=
    subset_affineSpan ℝ (Set.range p) (Set.mem_range_self 0)
  have hw : s.circumcenter - p 0 ∈ vectorSpan ℝ (Set.range p) := by
    rw [← direction_affineSpan]
    exact AffineSubspace.vsub_mem_direction hcenter hbase
  have hle :
      vectorSpan ℝ (Set.range p) ≤
        Submodule.span ℝ ({mengerEdge01 p, mengerEdge02 p} : Set V) := by
    rw [vectorSpan_eq_span_vsub_set_right ℝ (Set.mem_range_self 0)]
    refine Submodule.span_le.mpr ?_
    rintro _ ⟨_, ⟨i, rfl⟩, rfl⟩
    refine Fin.cases ?_ (fun j => ?_) i
    · simp
    · refine Fin.cases ?_ (fun k => ?_) j
      · change mengerEdge01 p ∈
          Submodule.span ℝ ({mengerEdge01 p, mengerEdge02 p} : Set V)
        exact Submodule.subset_span (by simp)
      · refine Fin.cases ?_ (fun z => Fin.elim0 z) k
        change mengerEdge02 p ∈
          Submodule.span ℝ ({mengerEdge01 p, mengerEdge02 p} : Set V)
        exact Submodule.subset_span (by simp)
  exact Submodule.mem_span_pair.mp (hle hw)

/-- The algebraic circumradius identity for the Gram-area definition. -/
private theorem four_mul_gramDet_mul_circumradius_sq (p : Fin 3 → V)
    (h : AffineIndependent ℝ p) :
    4 * mengerGramDet p *
          (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumradius ^ 2 =
      ‖mengerEdge01 p‖ ^ 2 * ‖mengerEdge12 p‖ ^ 2 *
        ‖mengerEdge02 p‖ ^ 2 := by
  classical
  let s : Affine.Simplex ℝ V 2 := ⟨p, h⟩
  let u : V := mengerEdge01 p
  let v : V := mengerEdge02 p
  let w : V := s.circumcenter - p 0
  let R : ℝ := s.circumradius

  have hwR : ‖w‖ = R := by
    change ‖s.circumcenter - s.points 0‖ = s.circumradius
    rw [← dist_eq_norm]
    exact s.dist_circumcenter_eq_circumradius' 0

  have huSub : u - w = p 1 - s.circumcenter := by
    dsimp [u, w, mengerEdge01]
    abel
  have hvSub : v - w = p 2 - s.circumcenter := by
    dsimp [v, w, mengerEdge02]
    abel
  have huR : ‖u - w‖ = R := by
    rw [huSub]
    change ‖s.points 1 - s.circumcenter‖ = s.circumradius
    rw [← dist_eq_norm]
    exact s.dist_circumcenter_eq_circumradius 1
  have hvR : ‖v - w‖ = R := by
    rw [hvSub]
    change ‖s.points 2 - s.circumcenter‖ = s.circumradius
    rw [← dist_eq_norm]
    exact s.dist_circumcenter_eq_circumradius 2

  have huw : (⟪u, w⟫_ℝ) = ‖u‖ ^ 2 / 2 := by
    have hsquare : ‖u - w‖ ^ 2 = ‖w‖ ^ 2 := by
      rw [huR, hwR]
    rw [norm_sub_sq_real] at hsquare
    nlinarith
  have hvw : (⟪v, w⟫_ℝ) = ‖v‖ ^ 2 / 2 := by
    have hsquare : ‖v - w‖ ^ 2 = ‖w‖ ^ 2 := by
      rw [hvR, hwR]
    rw [norm_sub_sq_real] at hsquare
    nlinarith

  obtain ⟨a, b, hab⟩ := circumcenter_sub_base_mem_span_edges p h
  have hab' : a • u + b • v = w := by
    simpa [u, v, w, s] using hab

  have hgram :
      (‖u‖ ^ 2 * ‖v‖ ^ 2 - (⟪u, v⟫_ℝ) ^ 2) * ‖w‖ ^ 2 =
        ‖v‖ ^ 2 * (⟪u, w⟫_ℝ) ^ 2 -
          2 * (⟪u, v⟫_ℝ) * (⟪u, w⟫_ℝ) * (⟪v, w⟫_ℝ) +
            ‖u‖ ^ 2 * (⟪v, w⟫_ℝ) ^ 2 := by
    rw [← real_inner_self_eq_norm_sq w, ← hab']
    simp [inner_add_left, inner_add_right, real_inner_smul_left,
      real_inner_smul_right, real_inner_self_eq_norm_sq, real_inner_comm]
    ring

  have hedge :
      ‖mengerEdge12 p‖ ^ 2 =
        ‖v‖ ^ 2 + ‖u‖ ^ 2 - 2 * (⟪u, v⟫_ℝ) := by
    rw [← mengerEdge02_sub_edge01 p]
    change ‖v - u‖ ^ 2 = _
    rw [norm_sub_sq_real, real_inner_comm v u]
    ring

  change
    4 * (‖u‖ ^ 2 * ‖v‖ ^ 2 - (⟪u, v⟫_ℝ) ^ 2) * R ^ 2 =
      ‖u‖ ^ 2 * ‖mengerEdge12 p‖ ^ 2 * ‖v‖ ^ 2
  calc
    4 * (‖u‖ ^ 2 * ‖v‖ ^ 2 - (⟪u, v⟫_ℝ) ^ 2) * R ^ 2 =
        4 * ((‖u‖ ^ 2 * ‖v‖ ^ 2 - (⟪u, v⟫_ℝ) ^ 2) * ‖w‖ ^ 2) := by
          rw [hwR]
          ring
    _ = 4 *
        (‖v‖ ^ 2 * (⟪u, w⟫_ℝ) ^ 2 -
          2 * (⟪u, v⟫_ℝ) * (⟪u, w⟫_ℝ) * (⟪v, w⟫_ℝ) +
            ‖u‖ ^ 2 * (⟪v, w⟫_ℝ) ^ 2) := by rw [hgram]
    _ = ‖u‖ ^ 2 * ‖mengerEdge12 p‖ ^ 2 * ‖v‖ ^ 2 := by
      rw [huw, hvw, hedge]
      ring

/-- For a nondegenerate triple, the frozen `4A/(abc)` definition equals the
reciprocal circumradius. This proves the equality asserted by GEO-MATH-006
rather than taking it as the definition. -/
theorem mengerCurvature_of_affineIndependent (p : Fin 3 → V)
    (h : AffineIndependent ℝ p) :
    mengerCurvature p =
      1 / (⟨p, h⟩ : Affine.Simplex ℝ V 2).circumradius := by
  classical
  let s : Affine.Simplex ℝ V 2 := ⟨p, h⟩
  let D : ℝ := mengerGramDet p
  let P : ℝ := mengerSideProduct p
  let R : ℝ := s.circumradius

  have h01 : p 1 ≠ p 0 := h.injective.ne (by decide)
  have h02 : p 2 ≠ p 0 := h.injective.ne (by decide)
  have h12 : p 2 ≠ p 1 := h.injective.ne (by decide)
  have he01 : mengerEdge01 p ≠ 0 := by
    exact sub_ne_zero.mpr h01
  have he02 : mengerEdge02 p ≠ 0 := by
    exact sub_ne_zero.mpr h02
  have he12 : mengerEdge12 p ≠ 0 := by
    exact sub_ne_zero.mpr h12
  have hPpos : 0 < P := by
    dsimp [P, mengerSideProduct]
    positivity
  have hRpos : 0 < R := by
    change 0 < s.circumradius
    exact s.circumradius_pos

  have hidentity :
      4 * D * R ^ 2 =
        ‖mengerEdge01 p‖ ^ 2 * ‖mengerEdge12 p‖ ^ 2 *
          ‖mengerEdge02 p‖ ^ 2 := by
    simpa [D, R, s] using four_mul_gramDet_mul_circumradius_sq p h
  have hrhsPos :
      0 < ‖mengerEdge01 p‖ ^ 2 * ‖mengerEdge12 p‖ ^ 2 *
        ‖mengerEdge02 p‖ ^ 2 := by
    positivity
  have hDpos : 0 < D := by
    by_contra hnot
    have hDle : D ≤ 0 := le_of_not_gt hnot
    have hleft : 4 * D * R ^ 2 ≤ 0 := by
      exact mul_nonpos_of_nonpos_of_nonneg
        (mul_nonpos_of_nonneg_of_nonpos (by norm_num) hDle)
        (sq_nonneg R)
    rw [hidentity] at hleft
    exact (not_lt_of_ge hleft) hrhsPos

  have hroot : 2 * Real.sqrt D * R = P := by
    apply (mul_self_inj_of_nonneg (by positivity) (le_of_lt hPpos)).mp
    calc
      (2 * Real.sqrt D * R) * (2 * Real.sqrt D * R) =
          4 * (Real.sqrt D * Real.sqrt D) * R ^ 2 := by ring
      _ = 4 * D * R ^ 2 := by
        rw [Real.mul_self_sqrt (le_of_lt hDpos)]
      _ = ‖mengerEdge01 p‖ ^ 2 * ‖mengerEdge12 p‖ ^ 2 *
          ‖mengerEdge02 p‖ ^ 2 := hidentity
      _ = P * P := by
        dsimp [P, mengerSideProduct]
        ring

  unfold mengerCurvature
  rw [if_pos h]
  change 4 * (Real.sqrt D / 2) / P = 1 / R
  field_simp [hPpos.ne', hRpos.ne']
  nlinarith [hroot]

/-- Affine Euclidean isometries preserve exact project Menger curvature. -/
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

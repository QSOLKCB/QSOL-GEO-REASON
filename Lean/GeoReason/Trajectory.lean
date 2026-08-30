import Mathlib

namespace GeoReason

noncomputable section

section Trajectory

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- A finite ordered trajectory. -/
abbrev Trajectory (V : Type*) := List V

/-- One forward finite-difference step. -/
def forwardDiff : List V → List V
  | [] => []
  | [_] => []
  | x :: y :: xs => (y - x) :: forwardDiff (y :: xs)

/-- Repeated forward finite differences. -/
def iterDiff : ℕ → List V → List V
  | 0, xs => xs
  | k + 1, xs => iterDiff k (forwardDiff xs)

/-- Translation of every trajectory point by the same vector. -/
def translate (t : V) (xs : List V) : List V :=
  xs.map fun x => x + t

/-- Piecewise-linear path length of a finite trajectory. -/
def pathLength : List V → ℝ
  | [] => 0
  | [_] => 0
  | x :: y :: xs => ‖y - x‖ + pathLength (y :: xs)

/-- Recursive exact statement that all consecutive trajectory points are equal. -/
def AllEqual : List V → Prop
  | [] => True
  | [_] => True
  | x :: y :: xs => x = y ∧ AllEqual (y :: xs)

@[simp]
theorem forwardDiff_length : ∀ xs : List V, (forwardDiff xs).length = xs.length - 1
  | [] => by simp [forwardDiff]
  | [_] => by simp [forwardDiff]
  | x :: y :: xs => by
      simp [forwardDiff, forwardDiff_length (y :: xs)]

/-- Stronger length theorem underlying GEO-LEAN-TGT-001. -/
theorem iterDiff_length (k : ℕ) (xs : List V) :
    (iterDiff k xs).length = xs.length - k := by
  induction k generalizing xs with
  | zero => simp [iterDiff]
  | succ k ih =>
      rw [iterDiff, ih, forwardDiff_length]
      omega

/-- GEO-LEAN-TGT-001 — finite-difference length. -/
theorem GEO_LEAN_TGT_001 (k : ℕ) (xs : List V) (hk : k < xs.length) :
    (iterDiff k xs).length = xs.length - k :=
  iterDiff_length k xs

@[simp]
theorem forwardDiff_translate (t : V) : ∀ xs : List V,
    forwardDiff (translate t xs) = forwardDiff xs
  | [] => by simp [translate, forwardDiff]
  | [_] => by simp [translate, forwardDiff]
  | x :: y :: xs => by
      simp [translate, forwardDiff, forwardDiff_translate t (y :: xs), sub_eq_add_neg,
        add_assoc, add_left_comm, add_comm]

/-- GEO-LEAN-TGT-002 — every positive-order finite difference cancels translation. -/
theorem GEO_LEAN_TGT_002 (k : ℕ) (hk : 1 ≤ k) (t : V) (xs : List V) :
    iterDiff k (translate t xs) = iterDiff k xs := by
  cases k with
  | zero => omega
  | succ k =>
      simp [iterDiff, forwardDiff_translate]

/-- Path length is nonnegative. -/
theorem pathLength_nonneg : ∀ xs : List V, 0 ≤ pathLength xs
  | [] => by simp [pathLength]
  | [_] => by simp [pathLength]
  | x :: y :: xs =>
      add_nonneg (norm_nonneg _) (pathLength_nonneg (y :: xs))

/-- GEO-LEAN-TGT-003 — path-length translation invariance. -/
theorem GEO_LEAN_TGT_003 (t : V) : ∀ xs : List V,
    pathLength (translate t xs) = pathLength xs
  | [] => by simp [translate, pathLength]
  | [_] => by simp [translate, pathLength]
  | x :: y :: xs => by
      simp [translate, pathLength, GEO_LEAN_TGT_003 t (y :: xs), sub_eq_add_neg,
        add_assoc, add_left_comm, add_comm]

/-- GEO-LEAN-TGT-005 — zero path length exactly characterizes all-equal trajectories. -/
theorem GEO_LEAN_TGT_005 : ∀ xs : List V,
    pathLength xs = 0 ↔ AllEqual xs
  | [] => by simp [pathLength, AllEqual]
  | [_] => by simp [pathLength, AllEqual]
  | x :: y :: xs => by
      constructor
      · intro h
        have htail_nonneg := pathLength_nonneg (y :: xs)
        have hnorm_nonneg := norm_nonneg (y - x)
        have hnorm : ‖y - x‖ = 0 := by
          rw [pathLength] at h
          nlinarith
        have hxy : x = y := by
          have hyx : y - x = 0 := norm_eq_zero.mp hnorm
          exact (sub_eq_zero.mp hyx).symm
        have htail : pathLength (y :: xs) = 0 := by
          rw [pathLength, hnorm, zero_add] at h
          exact h
        exact ⟨hxy, (GEO_LEAN_TGT_005 (y :: xs)).mp htail⟩
      · rintro ⟨hxy, htail⟩
        have hz : pathLength (y :: xs) = 0 := (GEO_LEAN_TGT_005 (y :: xs)).mpr htail
        simp [pathLength, hxy, hz]

end Trajectory

section Isometry

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Pointwise affine Euclidean isometry `x ↦ Q x + t`. -/
def affineIsometryMap (Q : V ≃ₗᵢ[ℝ] V) (t : V) (xs : List V) : List V :=
  xs.map fun x => Q x + t

/-- GEO-LEAN-TGT-004 — path length is invariant under Euclidean isometries. -/
theorem GEO_LEAN_TGT_004 (Q : V ≃ₗᵢ[ℝ] V) (t : V) : ∀ xs : List V,
    pathLength (affineIsometryMap Q t xs) = pathLength xs
  | [] => by simp [affineIsometryMap, pathLength]
  | [_] => by simp [affineIsometryMap, pathLength]
  | x :: y :: xs => by
      simp [affineIsometryMap, pathLength, GEO_LEAN_TGT_004 Q t (y :: xs), ← Q.map_sub,
        sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

end Isometry

end

end GeoReason

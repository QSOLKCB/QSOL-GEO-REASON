import Lean.Elab.Tactic.Omega
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Normed.Operator.LinearIsometry
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith.Frontend

namespace GeoReason

noncomputable section

section Trajectory

variable {V : Type*} [NormedAddCommGroup V]

/-- GEO-MATH-001 — a finite ordered trajectory with at least one point.

The head/tail representation makes the frozen non-emptiness requirement
unrepresentable by construction. Finite-difference outputs may still become
empty at sufficiently high order, but a source trajectory cannot be empty.
-/
structure Trajectory (V : Type*) where
  head : V
  tail : List V

/-- Ordered point list of a nonempty trajectory. -/
def Trajectory.points (z : Trajectory V) : List V :=
  z.head :: z.tail

omit [NormedAddCommGroup V] in
@[simp]
theorem Trajectory.points_ne_nil (z : Trajectory V) : z.points ≠ [] := by
  simp [Trajectory.points]

omit [NormedAddCommGroup V] in
@[simp]
theorem Trajectory.points_length (z : Trajectory V) :
    z.points.length = z.tail.length + 1 := by
  simp [Trajectory.points]

/-- One forward finite-difference step. Its input may be empty because
higher-order differences are allowed to exhaust the source trajectory. -/
def forwardDiff : List V → List V
  | [] => []
  | [_] => []
  | x :: y :: xs => (y - x) :: forwardDiff (y :: xs)

/-- Repeated forward finite differences of an arbitrary finite sequence. -/
def iterDiff : ℕ → List V → List V
  | 0, xs => xs
  | k + 1, xs => iterDiff k (forwardDiff xs)

/-- Order-`k` forward finite differences of a nonempty source trajectory. -/
def trajectoryDiff (k : ℕ) (z : Trajectory V) : List V :=
  iterDiff k z.points

/-- Translation of every trajectory point by the same vector. -/
def translate (t : V) (z : Trajectory V) : Trajectory V where
  head := z.head + t
  tail := z.tail.map fun x => x + t

@[simp]
theorem points_translate (t : V) (z : Trajectory V) :
    (translate t z).points = z.points.map fun x => x + t :=
  rfl

private def listPathLength : List V → ℝ
  | [] => 0
  | [_] => 0
  | x :: y :: xs => ‖y - x‖ + listPathLength (y :: xs)

/-- GEO-MATH-003 — piecewise-linear path length of a nonempty trajectory. -/
def pathLength (z : Trajectory V) : ℝ :=
  listPathLength z.points

private def listAllEqual : List V → Prop
  | [] => True
  | [_] => True
  | x :: y :: xs => x = y ∧ listAllEqual (y :: xs)

/-- Exact statement that all consecutive points in a nonempty trajectory agree. -/
def AllEqual (z : Trajectory V) : Prop :=
  listAllEqual z.points

@[simp]
theorem forwardDiff_length : ∀ xs : List V, (forwardDiff xs).length = xs.length - 1
  | [] => by simp [forwardDiff]
  | [_] => by simp [forwardDiff]
  | x :: y :: xs => by
      simp [forwardDiff, forwardDiff_length (y :: xs)]

private theorem iterDiff_length (k : ℕ) (xs : List V) :
    (iterDiff k xs).length = xs.length - k := by
  induction k generalizing xs with
  | zero => simp [iterDiff]
  | succ k ih =>
      rw [iterDiff, ih, forwardDiff_length]
      omega

/-- GEO-LEAN-TGT-001 — finite-difference length. -/
theorem GEO_LEAN_TGT_001 (k : ℕ) (z : Trajectory V) (_hk : k < z.points.length) :
    (trajectoryDiff k z).length = z.points.length - k := by
  exact iterDiff_length k z.points

private theorem forwardDiff_map_add (t : V) : ∀ xs : List V,
    forwardDiff (xs.map fun x => x + t) = forwardDiff xs
  | [] => by simp [forwardDiff]
  | [_] => by simp [forwardDiff]
  | x :: y :: xs => by
      change ((y + t) - (x + t)) :: forwardDiff ((y :: xs).map fun a => a + t) =
        (y - x) :: forwardDiff (y :: xs)
      rw [forwardDiff_map_add t (y :: xs)]
      have hcancel : (y + t) - (x + t) = y - x := by abel
      rw [hcancel]

/-- GEO-LEAN-TGT-002 — every positive-order finite difference cancels translation. -/
theorem GEO_LEAN_TGT_002 (k : ℕ) (hk : 1 ≤ k) (t : V) (z : Trajectory V) :
    trajectoryDiff k (translate t z) = trajectoryDiff k z := by
  unfold trajectoryDiff
  rw [points_translate]
  cases k with
  | zero => omega
  | succ k =>
      simp [iterDiff, forwardDiff_map_add]

private theorem listPathLength_nonneg : ∀ xs : List V, 0 ≤ listPathLength xs
  | [] => by simp [listPathLength]
  | [_] => by simp [listPathLength]
  | x :: y :: xs =>
      add_nonneg (norm_nonneg _) (listPathLength_nonneg (y :: xs))

private theorem listPathLength_map_add (t : V) : ∀ xs : List V,
    listPathLength (xs.map fun x => x + t) = listPathLength xs
  | [] => by simp [listPathLength]
  | [_] => by simp [listPathLength]
  | x :: y :: xs => by
      change ‖(y + t) - (x + t)‖ +
          listPathLength ((y :: xs).map fun a => a + t) =
        ‖y - x‖ + listPathLength (y :: xs)
      rw [listPathLength_map_add t (y :: xs)]
      have hcancel : (y + t) - (x + t) = y - x := by abel
      rw [hcancel]

/-- GEO-LEAN-TGT-003 — path-length translation invariance. -/
theorem GEO_LEAN_TGT_003 (t : V) (z : Trajectory V) :
    pathLength (translate t z) = pathLength z := by
  unfold pathLength
  rw [points_translate, listPathLength_map_add]

private theorem listPathLength_eq_zero_iff_allEqual : ∀ xs : List V,
    listPathLength xs = 0 ↔ listAllEqual xs
  | [] => by simp [listPathLength, listAllEqual]
  | [_] => by simp [listPathLength, listAllEqual]
  | x :: y :: xs => by
      constructor
      · intro h
        have htail_nonneg := listPathLength_nonneg (y :: xs)
        have hnorm_nonneg := norm_nonneg (y - x)
        have hnorm : ‖y - x‖ = 0 := by
          rw [listPathLength] at h
          nlinarith
        have hxy : x = y := by
          have hyx : y - x = 0 := norm_eq_zero.mp hnorm
          exact (sub_eq_zero.mp hyx).symm
        have htail : listPathLength (y :: xs) = 0 := by
          rw [listPathLength, hnorm, zero_add] at h
          exact h
        exact ⟨hxy, (listPathLength_eq_zero_iff_allEqual (y :: xs)).mp htail⟩
      · rintro ⟨hxy, htail⟩
        have hz : listPathLength (y :: xs) = 0 :=
          (listPathLength_eq_zero_iff_allEqual (y :: xs)).mpr htail
        simp [listPathLength, hxy, hz]

/-- GEO-LEAN-TGT-005 — zero path length exactly characterizes all-equal
nonempty trajectories. -/
theorem GEO_LEAN_TGT_005 (z : Trajectory V) :
    pathLength z = 0 ↔ AllEqual z := by
  exact listPathLength_eq_zero_iff_allEqual z.points

end Trajectory

section Isometry

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Pointwise affine Euclidean isometry `x ↦ Q x + t`. -/
def affineIsometryMap (Q : V ≃ₗᵢ[ℝ] V) (t : V) (z : Trajectory V) : Trajectory V where
  head := Q z.head + t
  tail := z.tail.map fun x => Q x + t

@[simp]
theorem points_affineIsometryMap (Q : V ≃ₗᵢ[ℝ] V) (t : V) (z : Trajectory V) :
    (affineIsometryMap Q t z).points = z.points.map fun x => Q x + t :=
  rfl

private theorem listPathLength_map_affineIsometry (Q : V ≃ₗᵢ[ℝ] V) (t : V) :
    ∀ xs : List V,
      listPathLength (xs.map fun x => Q x + t) = listPathLength xs
  | [] => by simp [listPathLength]
  | [_] => by simp [listPathLength]
  | x :: y :: xs => by
      change ‖(Q y + t) - (Q x + t)‖ +
          listPathLength ((y :: xs).map fun a => Q a + t) =
        ‖y - x‖ + listPathLength (y :: xs)
      rw [listPathLength_map_affineIsometry Q t (y :: xs)]
      have hcancel : (Q y + t) - (Q x + t) = Q (y - x) := by
        rw [Q.map_sub]
        abel
      rw [hcancel, Q.norm_map]

/-- GEO-LEAN-TGT-004 — path length is invariant under Euclidean isometries. -/
theorem GEO_LEAN_TGT_004 (Q : V ≃ₗᵢ[ℝ] V) (t : V) (z : Trajectory V) :
    pathLength (affineIsometryMap Q t z) = pathLength z := by
  unfold pathLength
  rw [points_affineIsometryMap, listPathLength_map_affineIsometry]

end Isometry

end

end GeoReason

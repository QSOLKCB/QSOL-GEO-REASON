import Mathlib
import GeoReason.Trajectory

namespace GeoReason

noncomputable section

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- The final point of a nonempty path represented by a head plus a tail. -/
def lastPoint (x : V) : List V → V
  | [] => x
  | y :: ys => lastPoint y ys

/-- Linear interpolation in the ambient real vector space. -/
def lerp (x y : V) (u : ℝ) : V :=
  (1 - u) • x + u • y

/-- Evaluate a finite piecewise-linear path at an arc-length distance from its
current point. Zero-length segments are traversed without division. -/
def pointAtDistance (x : V) : List V → ℝ → V
  | [], _ => x
  | y :: ys, d =>
      let segment := ‖y - x‖
      if segment = 0 then
        pointAtDistance y ys d
      else if d ≤ segment then
        lerp x y (d / segment)
      else
        pointAtDistance y ys (d - segment)
termination_by ys => ys.length

/-- Normalized piecewise-linear arc-length parameterization of a nonempty
finite path. Values at or below `0` are clamped to the first point and values
at or above `1` are clamped to the final point. Interior values traverse the
path using arc-length distance. -/
def normalizedArcLengthPoint (x : V) (xs : List V) (u : ℝ) : V :=
  if u ≤ 0 then
    x
  else if 1 ≤ u then
    lastPoint x xs
  else
    pointAtDistance x xs (u * pathLength (x :: xs))

/-- GEO-LEAN-TGT-012 — a nonzero-length normalized piecewise-linear path
preserves its exact first and final endpoints at progress `0` and `1`. -/
theorem GEO_LEAN_TGT_012 (x : V) (xs : List V) (_h : 0 < pathLength (x :: xs)) :
    normalizedArcLengthPoint x xs 0 = x ∧
      normalizedArcLengthPoint x xs 1 = lastPoint x xs := by
  constructor <;> simp [normalizedArcLengthPoint]

end

end GeoReason

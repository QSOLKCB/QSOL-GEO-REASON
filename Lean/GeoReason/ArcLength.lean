import Mathlib
import GeoReason.Trajectory

namespace GeoReason

noncomputable section

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

private def lastPointFrom (x : V) : List V → V
  | [] => x
  | y :: ys => lastPointFrom y ys

/-- The exact final point of a nonempty trajectory. -/
def lastPoint (z : Trajectory V) : V :=
  lastPointFrom z.head z.tail

/-- Linear interpolation in the ambient real vector space. -/
def lerp (x y : V) (u : ℝ) : V :=
  (1 - u) • x + u • y

private def pointAtDistanceFrom (x : V) : List V → ℝ → V
  | [], _ => x
  | y :: ys, d =>
      let segment := ‖y - x‖
      if segment = 0 then
        pointAtDistanceFrom y ys d
      else if d ≤ segment then
        lerp x y (d / segment)
      else
        pointAtDistanceFrom y ys (d - segment)
termination_by ys => ys.length

/-- Evaluate a nonempty finite piecewise-linear trajectory at an arc-length
distance from its first point. Zero-length segments are traversed without
division. -/
def pointAtDistance (z : Trajectory V) (d : ℝ) : V :=
  pointAtDistanceFrom z.head z.tail d

/-- Normalized piecewise-linear arc-length parameterization of a nonempty
finite trajectory. Values at or below `0` are clamped to the first point and
values at or above `1` are clamped to the final point. Interior values traverse
the trajectory using arc-length distance. -/
def normalizedArcLengthPoint (z : Trajectory V) (u : ℝ) : V :=
  if u ≤ 0 then
    z.head
  else if 1 ≤ u then
    lastPoint z
  else
    pointAtDistance z (u * pathLength z)

/-- GEO-LEAN-TGT-012 — a nonzero-length normalized piecewise-linear
trajectory preserves its exact first and final endpoints at progress `0` and
`1`. The source trajectory is nonempty by type construction. -/
theorem GEO_LEAN_TGT_012 (z : Trajectory V) (_h : 0 < pathLength z) :
    normalizedArcLengthPoint z 0 = z.head ∧
      normalizedArcLengthPoint z 1 = lastPoint z := by
  constructor <;> simp [normalizedArcLengthPoint]

end

end GeoReason

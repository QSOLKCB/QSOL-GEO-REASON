"""Dependency-free Euclidean trajectory geometry for Phase 1."""

from __future__ import annotations

import math
from typing import Sequence

Point = list[float]
Trajectory = list[Point]

_EPS = 1e-15


def _validate_trajectory(points: Sequence[Sequence[float]], *, name: str = "trajectory") -> int:
    if not isinstance(points, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(points) == 0:
        raise ValueError(f"{name} must contain at least one point")
    dim = len(points[0])
    if dim == 0:
        raise ValueError(f"{name} points must have at least one dimension")
    for idx, point in enumerate(points):
        if len(point) != dim:
            raise ValueError(f"{name} dimension mismatch at point {idx}")
        for value in point:
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} contains non-finite value")
    return dim


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("point dimension mismatch")
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def finite_difference(points: Sequence[Sequence[float]], order: int = 1) -> Trajectory:
    """Return the repeated forward finite difference.

    order=0 returns a float-normalized copy of the input. If order is greater
    than or equal to the trajectory length, an empty sequence is returned.
    """

    _validate_trajectory(points)
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a non-negative integer")
    current = [[float(x) for x in point] for point in points]
    for _ in range(order):
        current = [
            [b - a for a, b in zip(current[i], current[i + 1])]
            for i in range(len(current) - 1)
        ]
        if not current:
            break
    return current


def path_length(points: Sequence[Sequence[float]]) -> float:
    _validate_trajectory(points)
    if len(points) == 1:
        return 0.0
    return math.fsum(_distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def resample_arclength(points: Sequence[Sequence[float]], count: int) -> Trajectory:
    """Linearly resample a trajectory at equally spaced arc-length targets."""

    dim = _validate_trajectory(points)
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    pts = [[float(x) for x in point] for point in points]
    if count == 1:
        return [pts[0][:]]
    if len(pts) == 1:
        return [pts[0][:] for _ in range(count)]

    cumulative = [0.0]
    for i in range(len(pts) - 1):
        cumulative.append(cumulative[-1] + _distance(pts[i], pts[i + 1]))
    total = cumulative[-1]
    if total <= _EPS:
        return [pts[0][:] for _ in range(count)]

    targets = [total * i / (count - 1) for i in range(count)]
    out: Trajectory = []
    segment = 0
    for target in targets:
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(pts):
            out.append(pts[-1][:])
            continue
        lo, hi = cumulative[segment], cumulative[segment + 1]
        if hi - lo <= _EPS:
            out.append(pts[segment + 1][:])
            continue
        weight = (target - lo) / (hi - lo)
        out.append([
            (1.0 - weight) * pts[segment][d] + weight * pts[segment + 1][d]
            for d in range(dim)
        ])
    return out


def align_pair(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    mode: str = "truncate",
) -> tuple[Trajectory, Trajectory]:
    left_dim = _validate_trajectory(left, name="left")
    right_dim = _validate_trajectory(right, name="right")
    if left_dim != right_dim:
        raise ValueError("trajectory dimension mismatch")

    if mode == "error":
        if len(left) != len(right):
            raise ValueError("trajectory length mismatch")
        return (
            [[float(x) for x in p] for p in left],
            [[float(x) for x in p] for p in right],
        )
    if mode == "truncate":
        size = min(len(left), len(right))
        return (
            [[float(x) for x in p] for p in left[:size]],
            [[float(x) for x in p] for p in right[:size]],
        )
    if mode == "arclength":
        size = max(len(left), len(right))
        return resample_arclength(left, size), resample_arclength(right, size)
    raise ValueError("mode must be one of: error, truncate, arclength")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch")
    aa = math.fsum(float(x) * float(x) for x in a)
    bb = math.fsum(float(x) * float(x) for x in b)
    if aa <= _EPS or bb <= _EPS:
        return 1.0 if aa <= _EPS and bb <= _EPS else 0.0
    dot = math.fsum(float(x) * float(y) for x, y in zip(a, b))
    value = dot / math.sqrt(aa * bb)
    return max(-1.0, min(1.0, value))


def cosine_alignment(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    order: int = 1,
    align: str = "truncate",
) -> float:
    """Mean pointwise cosine similarity after alignment and finite differencing."""

    a, b = align_pair(left, right, mode=align)
    a = finite_difference(a, order)
    b = finite_difference(b, order)
    if len(a) != len(b):
        size = min(len(a), len(b))
        a, b = a[:size], b[:size]
    if not a:
        return 0.0
    return math.fsum(_cosine(x, y) for x, y in zip(a, b)) / len(a)


def menger_curvature_sequence(points: Sequence[Sequence[float]]) -> list[float]:
    """Return Menger curvature for each consecutive triple.

    For side lengths a,b,c and triangle area A, kappa = 4A/(abc). The
    dimension-independent area is obtained from the Gram determinant.
    Repeated points and collinear triples are assigned curvature zero.
    """

    _validate_trajectory(points)
    if len(points) < 3:
        return []
    out: list[float] = []
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        u = [float(b) - float(a) for a, b in zip(p0, p1)]
        v = [float(c) - float(b) for b, c in zip(p1, p2)]
        a = math.sqrt(math.fsum(x * x for x in u))
        b = math.sqrt(math.fsum(x * x for x in v))
        chord = [float(c) - float(a0) for a0, c in zip(p0, p2)]
        c = math.sqrt(math.fsum(x * x for x in chord))
        if a <= _EPS or b <= _EPS or c <= _EPS:
            out.append(0.0)
            continue
        uu = math.fsum(x * x for x in u)
        vv = math.fsum(x * x for x in v)
        uv = math.fsum(x * y for x, y in zip(u, v))
        gram = max(0.0, uu * vv - uv * uv)
        area = 0.5 * math.sqrt(gram)
        kappa = (4.0 * area) / (a * b * c)
        out.append(0.0 if abs(kappa) <= _EPS else kappa)
    return out

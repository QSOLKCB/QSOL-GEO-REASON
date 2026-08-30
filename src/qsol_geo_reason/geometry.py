"""Dependency-free Euclidean trajectory geometry for Phase 1."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

Point = list[float]
Trajectory = list[Point]


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


def _vector_between(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("point dimension mismatch")
    delta = [float(y) - float(x) for x, y in zip(a, b)]
    if any(not math.isfinite(x) for x in delta):
        raise ValueError("point displacement is not finite")
    return delta


def _norm(vector: Sequence[float]) -> float:
    values = [float(x) for x in vector]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("vector contains non-finite value")
    return math.hypot(*values)


def _unit(vector: Sequence[float]) -> list[float] | None:
    values = [float(x) for x in vector]
    if not values:
        raise ValueError("vector must contain at least one component")
    scale = max(abs(x) for x in values)
    if scale == 0.0:
        return None
    scaled = [x / scale for x in values]
    norm = math.hypot(*scaled)
    return [x / norm for x in scaled]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _norm(_vector_between(a, b))


def finite_difference(points: Sequence[Sequence[float]], order: int = 1) -> Trajectory:
    """Return the repeated forward finite difference.

    order=0 returns a float-normalized copy of the input. If order is greater
    than or equal to the trajectory length, an empty sequence is returned.
    """
    _validate_trajectory(points)
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
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
    result = math.fsum(_distance(points[i], points[i + 1]) for i in range(len(points) - 1))
    if not math.isfinite(result):
        raise ValueError("path length is not finite")
    return result


def resample_arclength(points: Sequence[Sequence[float]], count: int) -> Trajectory:
    """Linearly resample a trajectory at equally spaced arc-length targets.

    Segment lengths are accumulated as exact rational representations of their
    binary64 values. This prevents a short late segment from disappearing when
    preceded by a much larger segment. The first and final samples are always
    the exact input endpoints.

    Only an exactly zero-length path is collapsed to repeated copies.
    """
    dim = _validate_trajectory(points)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    pts = [[float(x) for x in point] for point in points]
    if count == 1:
        return [pts[0][:]]
    if len(pts) == 1:
        return [pts[0][:] for _ in range(count)]

    segment_lengths = [_distance(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    exact_lengths = [Fraction.from_float(length) for length in segment_lengths]
    total = sum(exact_lengths, Fraction(0, 1))
    if total == 0:
        return [pts[0][:] for _ in range(count)]

    out: Trajectory = []
    segment = 0
    segment_start = Fraction(0, 1)

    for sample_index in range(count):
        if sample_index == 0:
            out.append(pts[0][:])
            continue
        if sample_index == count - 1:
            out.append(pts[-1][:])
            continue

        target = total * Fraction(sample_index, count - 1)
        while (
            segment < len(exact_lengths) - 1
            and segment_start + exact_lengths[segment] < target
        ):
            segment_start += exact_lengths[segment]
            segment += 1

        segment_length = exact_lengths[segment]
        if segment_length == 0:
            out.append(pts[segment + 1][:])
            continue

        weight = float((target - segment_start) / segment_length)
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
    ua = _unit(a)
    ub = _unit(b)
    if ua is None or ub is None:
        return 1.0 if ua is None and ub is None else 0.0
    value = math.fsum(x * y for x, y in zip(ua, ub))
    return max(-1.0, min(1.0, value))


def cosine_alignment(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    order: int = 1,
    align: str = "truncate",
) -> float:
    """Mean pointwise cosine similarity after alignment and finite differencing.

    Raises ValueError if the selected finite-difference order leaves no samples;
    an empty mean is undefined and is never encoded as an apparent zero score.
    """
    a, b = align_pair(left, right, mode=align)
    a = finite_difference(a, order)
    b = finite_difference(b, order)
    if len(a) != len(b):
        size = min(len(a), len(b))
        a, b = a[:size], b[:size]
    if not a:
        raise ValueError("finite-difference comparison has no samples")
    return math.fsum(_cosine(x, y) for x, y in zip(a, b)) / len(a)


def menger_curvature_sequence(points: Sequence[Sequence[float]]) -> list[float]:
    """Return Menger curvature for each consecutive triple.

    For consecutive displacement vectors u and v and endpoint chord c,
    kappa = 2 sin(theta) / ||c||. This is algebraically equivalent to
    4A/(abc) but uses scaled unit vectors to avoid overflow in Gram products.
    Exactly repeated points and collinear triples are assigned curvature zero.
    """
    _validate_trajectory(points)
    if len(points) < 3:
        return []

    out: list[float] = []
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        u = _vector_between(p0, p1)
        v = _vector_between(p1, p2)
        chord = _vector_between(p0, p2)
        a = _norm(u)
        b = _norm(v)
        c = _norm(chord)
        if a == 0.0 or b == 0.0 or c == 0.0:
            out.append(0.0)
            continue

        uu = _unit(u)
        vv = _unit(v)
        assert uu is not None and vv is not None
        dot = max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(uu, vv))))
        orthogonal_residual = [vv[j] - dot * uu[j] for j in range(len(uu))]
        sin_theta = _norm(orthogonal_residual)
        kappa = (2.0 * sin_theta) / c
        if not math.isfinite(kappa):
            raise ValueError("Menger curvature is not finite")
        out.append(0.0 if sin_theta == 0.0 else kappa)
    return out

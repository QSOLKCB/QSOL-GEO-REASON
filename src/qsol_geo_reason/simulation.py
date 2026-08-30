"""Deterministic synthetic trajectory simulation for GEO-SIM-001."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .canonical import sha256_json
from .geometry import cosine_alignment, finite_difference, menger_curvature_sequence, path_length

SCHEMA_VERSION = "1.0.0"
EVIDENCE_CLASS = "SIMULATION"
REPLICATION_STATUS = "not_attempted"


class RecipeError(ValueError):
    pass


def _point(value: Any, *, field: str, dim: int | None = None) -> list[float]:
    if not isinstance(value, list) or not value:
        raise RecipeError(f"{field} must be a non-empty numeric array")
    out = [float(x) for x in value]
    if any(not math.isfinite(x) for x in out):
        raise RecipeError(f"{field} contains a non-finite value")
    if dim is not None and len(out) != dim:
        raise RecipeError(f"{field} must have dimension {dim}")
    return out


def _positive_int(value: Any, *, field: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RecipeError(f"{field} must be an integer >= {minimum}")
    return value


def validate_recipe(recipe: dict[str, Any]) -> None:
    if not isinstance(recipe, dict):
        raise RecipeError("recipe must be an object")
    required = {"schema_version", "recipe_id", "seed", "trajectories", "comparisons"}
    missing = sorted(required - set(recipe))
    if missing:
        raise RecipeError(f"missing required keys: {', '.join(missing)}")
    unknown = sorted(set(recipe) - required - {"description"})
    if unknown:
        raise RecipeError(f"unknown top-level keys: {', '.join(unknown)}")
    if recipe["schema_version"] != SCHEMA_VERSION:
        raise RecipeError(f"unsupported schema_version: {recipe['schema_version']!r}")
    if not isinstance(recipe["recipe_id"], str) or not recipe["recipe_id"]:
        raise RecipeError("recipe_id must be a non-empty string")
    if not isinstance(recipe["seed"], int) or isinstance(recipe["seed"], bool):
        raise RecipeError("seed must be an integer")
    if not isinstance(recipe["trajectories"], list) or not recipe["trajectories"]:
        raise RecipeError("trajectories must be a non-empty array")
    if not isinstance(recipe["comparisons"], list):
        raise RecipeError("comparisons must be an array")

    ids: set[str] = set()
    for item in recipe["trajectories"]:
        if not isinstance(item, dict):
            raise RecipeError("each trajectory must be an object")
        tid = item.get("id")
        if not isinstance(tid, str) or not tid:
            raise RecipeError("trajectory id must be a non-empty string")
        if tid in ids:
            raise RecipeError(f"duplicate trajectory id: {tid}")
        ids.add(tid)
        kind = item.get("kind")
        params = item.get("parameters", {})
        if kind not in {"straight", "circle_arc", "branch", "noisy_straight", "null", "isometry", "suffix_shift", "translation"}:
            raise RecipeError(f"unsupported trajectory kind: {kind!r}")
        if not isinstance(params, dict):
            raise RecipeError(f"{tid}.parameters must be an object")
        if kind in {"isometry", "suffix_shift", "translation"}:
            source = params.get("source")
            if source not in ids:
                raise RecipeError(f"{tid} source must reference an earlier trajectory")
    for comp in recipe["comparisons"]:
        if not isinstance(comp, dict):
            raise RecipeError("each comparison must be an object")
        cid = comp.get("id")
        left, right = comp.get("left"), comp.get("right")
        if not isinstance(cid, str) or not cid:
            raise RecipeError("comparison id must be a non-empty string")
        if left not in ids or right not in ids:
            raise RecipeError(f"comparison {cid} references unknown trajectory")
        order = comp.get("order", 1)
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise RecipeError(f"comparison {cid} order must be a non-negative integer")
        if comp.get("align", "truncate") not in {"truncate", "arclength", "error"}:
            raise RecipeError(f"comparison {cid} has invalid align mode")


def _lcg_noise(seed: int, index: int, axis: int, amplitude: float) -> float:
    state = ((seed & 0xFFFFFFFF) ^ ((index + 1) * 0x9E3779B1) ^ ((axis + 1) * 0x85EBCA77)) & 0xFFFFFFFF
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    unit = state / 4294967295.0
    return amplitude * (2.0 * unit - 1.0)


def _straight(params: dict[str, Any]) -> list[list[float]]:
    start = _point(params.get("start"), field="start")
    direction = _point(params.get("direction"), field="direction", dim=len(start))
    steps = _positive_int(params.get("steps"), field="steps", minimum=1)
    step_size = float(params.get("step_size", 1.0))
    return [[start[d] + i * step_size * direction[d] for d in range(len(start))] for i in range(steps)]


def _circle_arc(params: dict[str, Any]) -> list[list[float]]:
    center = _point(params.get("center"), field="center")
    if len(center) < 2:
        raise RecipeError("circle_arc requires at least two dimensions")
    radius = float(params.get("radius"))
    if not math.isfinite(radius) or radius <= 0:
        raise RecipeError("radius must be finite and > 0")
    start_angle = float(params.get("start_angle"))
    end_angle = float(params.get("end_angle"))
    steps = _positive_int(params.get("steps"), field="steps", minimum=2)
    axes = params.get("plane_axes", [0, 1])
    if not isinstance(axes, list) or len(axes) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in axes) or axes[0] == axes[1] or min(axes) < 0 or max(axes) >= len(center):
        raise RecipeError("plane_axes must name two distinct valid dimensions")
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        angle = start_angle + t * (end_angle - start_angle)
        point = center[:]
        point[axes[0]] += radius * math.cos(angle)
        point[axes[1]] += radius * math.sin(angle)
        out.append(point)
    return out


def _branch(params: dict[str, Any]) -> list[list[float]]:
    start = _point(params.get("start"), field="start")
    trunk = _point(params.get("trunk_direction"), field="trunk_direction", dim=len(start))
    branch = _point(params.get("branch_direction"), field="branch_direction", dim=len(start))
    steps = _positive_int(params.get("steps"), field="steps", minimum=2)
    branch_index = params.get("branch_index")
    if not isinstance(branch_index, int) or isinstance(branch_index, bool) or not (1 <= branch_index < steps):
        raise RecipeError("branch_index must satisfy 1 <= branch_index < steps")
    step_size = float(params.get("step_size", 1.0))
    out = [start[:]]
    for i in range(1, steps):
        direction = trunk if i <= branch_index else branch
        out.append([out[-1][d] + step_size * direction[d] for d in range(len(start))])
    return out


def _noisy_straight(params: dict[str, Any], seed: int) -> list[list[float]]:
    base = _straight(params)
    amplitude = float(params.get("noise_amplitude", 0.0))
    if amplitude < 0 or not math.isfinite(amplitude):
        raise RecipeError("noise_amplitude must be finite and >= 0")
    return [[value + _lcg_noise(seed, i, d, amplitude) for d, value in enumerate(point)] for i, point in enumerate(base)]


def _null(params: dict[str, Any]) -> list[list[float]]:
    point = _point(params.get("point"), field="point")
    steps = _positive_int(params.get("steps"), field="steps", minimum=1)
    return [point[:] for _ in range(steps)]


def _translation(source: list[list[float]], params: dict[str, Any], *, suffix: bool) -> list[list[float]]:
    shift = _point(params.get("shift"), field="shift", dim=len(source[0]))
    start_index = 0
    if suffix:
        start_index = params.get("start_index")
        if not isinstance(start_index, int) or isinstance(start_index, bool) or not (0 <= start_index < len(source)):
            raise RecipeError("start_index must reference an existing trajectory point")
    out = deepcopy(source)
    for i in range(start_index, len(out)):
        out[i] = [out[i][d] + shift[d] for d in range(len(shift))]
    return out


def _isometry(source: list[list[float]], params: dict[str, Any]) -> list[list[float]]:
    dim = len(source[0])
    permutation = params.get("permutation", list(range(dim)))
    signs = params.get("signs", [1] * dim)
    translation = _point(params.get("translation", [0.0] * dim), field="translation", dim=dim)
    if sorted(permutation) != list(range(dim)):
        raise RecipeError("permutation must contain each dimension exactly once")
    if len(signs) != dim or any(sign not in {-1, 1} for sign in signs):
        raise RecipeError("signs must contain one -1 or 1 per dimension")
    return [[signs[d] * point[permutation[d]] + translation[d] for d in range(dim)] for point in source]


def _generate(item: dict[str, Any], trajectories: dict[str, list[list[float]]], seed: int) -> list[list[float]]:
    kind = item["kind"]
    params = item.get("parameters", {})
    if kind == "straight": return _straight(params)
    if kind == "circle_arc": return _circle_arc(params)
    if kind == "branch": return _branch(params)
    if kind == "noisy_straight": return _noisy_straight(params, int(params.get("seed", seed)))
    if kind == "null": return _null(params)
    source = trajectories[params["source"]]
    if kind == "isometry": return _isometry(source, params)
    if kind == "translation": return _translation(source, params, suffix=False)
    if kind == "suffix_shift": return _translation(source, params, suffix=True)
    raise AssertionError(kind)


def _round_float(value: float, digits: int = 15) -> float:
    value = round(float(value), digits)
    return 0.0 if value == -0.0 else value


def _round_nested(value: Any) -> Any:
    if isinstance(value, float): return _round_float(value)
    if isinstance(value, list): return [_round_nested(x) for x in value]
    if isinstance(value, dict): return {k: _round_nested(v) for k, v in value.items()}
    return value


def run_recipe(recipe: dict[str, Any], *, implementation_revision: str) -> dict[str, Any]:
    """Execute a validated recipe and return a hash-bound SIMULATION record."""

    validate_recipe(recipe)
    if not isinstance(implementation_revision, str) or not implementation_revision:
        raise RecipeError("implementation_revision must be a non-empty string")

    trajectories = {}
    records = []
    for item in recipe["trajectories"]:
        points = _generate(item, trajectories, recipe["seed"])
        trajectories[item["id"]] = points
        record = {
            "id": item["id"], "kind": item["kind"], "point_count": len(points), "dimension": len(points[0]),
            "points": points, "trajectory_sha256": sha256_json(_round_nested(points)),
            "metrics": {"path_length": path_length(points), "order_1": finite_difference(points, 1), "order_2": finite_difference(points, 2), "menger_curvature": menger_curvature_sequence(points)},
        }
        records.append(_round_nested(record))

    comparisons = []
    for comp in recipe["comparisons"]:
        comparisons.append(_round_nested({
            "id": comp["id"], "left": comp["left"], "right": comp["right"], "order": comp.get("order", 1), "align": comp.get("align", "truncate"),
            "mean_cosine_alignment": cosine_alignment(trajectories[comp["left"]], trajectories[comp["right"]], order=comp.get("order", 1), align=comp.get("align", "truncate")),
        }))

    payload = _round_nested({
        "schema_version": SCHEMA_VERSION, "evidence_class": EVIDENCE_CLASS, "replication_status": REPLICATION_STATUS,
        "recipe_id": recipe["recipe_id"], "recipe_sha256": sha256_json(recipe), "implementation_revision": implementation_revision,
        "trajectories": records, "comparisons": comparisons,
    })
    return {**payload, "artifact_sha256": sha256_json(payload)}

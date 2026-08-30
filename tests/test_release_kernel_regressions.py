import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.geometry import finite_difference, menger_curvature_sequence, path_length
from qsol_geo_reason.simulation import run_recipe

ROOT = Path(__file__).resolve().parents[1]


class ReleaseKernelRegressionTests(unittest.TestCase):
    def test_near_collinear_nonzero_curvature_survives(self):
        points = [
            [0.0, 0.0],
            [0.7921019866760063, 0.21315511101833268],
            [1.5842039733520128, 0.42631022203666535],
        ]
        curvature = menger_curvature_sequence(points)[0]
        self.assertGreater(curvature, 0.0)

    def test_serialized_points_and_metrics_share_one_trajectory(self):
        recipe = {
            "schema_version": "1.0.0",
            "recipe_id": "SERIALIZED-CONSISTENCY",
            "seed": 0,
            "trajectories": [
                {
                    "id": "large-offset",
                    "kind": "straight",
                    "parameters": {
                        "start": [1e16],
                        "direction": [1.0],
                        "steps": 3,
                        "step_size": 2.0,
                    },
                }
            ],
            "comparisons": [],
        }
        result = run_recipe(recipe, implementation_revision="test-revision")
        record = result["trajectories"][0]
        points = record["points"]

        self.assertNotEqual(points[0], points[1])
        self.assertNotEqual(points[1], points[2])
        self.assertEqual(finite_difference(points, 1), record["metrics"]["order_1"])
        self.assertEqual(path_length(points), record["metrics"]["path_length"])
        self.assertEqual(sha256_json(points), record["trajectory_sha256"])

    def test_math_spec_freezes_arclength_count_rule(self):
        text = (ROOT / "MATH-SPEC.md").read_text(encoding="utf-8")
        self.assertIn(r"m=\max(n_X,n_Y)", text)
        self.assertIn(r"\frac{j}{m-1}", text)


if __name__ == "__main__":
    unittest.main()

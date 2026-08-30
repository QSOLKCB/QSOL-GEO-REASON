import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.geometry import cosine_alignment, finite_difference, menger_curvature_sequence, path_length
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

    def test_ulp_sized_offset_survives_origin_canonicalization(self):
        recipe = {
            "schema_version": "1.0.0",
            "recipe_id": "ULP-OFFSET",
            "seed": 0,
            "trajectories": [
                {
                    "id": "ulp-step",
                    "kind": "straight",
                    "parameters": {
                        "start": [9007199254740991.0],
                        "direction": [1.0],
                        "steps": 2,
                        "step_size": 1.0,
                    },
                }
            ],
            "comparisons": [],
        }
        result = run_recipe(recipe, implementation_revision="test-revision")
        record = result["trajectories"][0]
        self.assertEqual(
            record["points"],
            [[9007199254740991.0], [9007199254740992.0]],
        )
        self.assertEqual(record["metrics"]["order_1"], [[1.0]])
        self.assertEqual(record["metrics"]["path_length"], 1.0)

    def test_overflowing_finite_difference_is_rejected(self):
        points = [[-1e308], [1e308]]
        with self.assertRaisesRegex(ValueError, "displacement is not finite"):
            finite_difference(points, 1)
        with self.assertRaisesRegex(ValueError, "displacement is not finite"):
            cosine_alignment(points, points, order=1, align="error")

    def test_math_spec_freezes_arclength_count_rule(self):
        text = (ROOT / "MATH-SPEC.md").read_text(encoding="utf-8")
        self.assertIn(r"m=\max(n_X,n_Y)", text)
        self.assertIn(r"\frac{j}{m-1}", text)


if __name__ == "__main__":
    unittest.main()

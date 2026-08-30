import json
import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.simulation import RecipeError, run_recipe, validate_recipe

ROOT = Path(__file__).resolve().parents[1]


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.recipe = json.loads((ROOT / "recipes/reference-suite.json").read_text(encoding="utf-8"))

    def test_reference_recipe_validates(self): validate_recipe(self.recipe)

    def test_unknown_top_level_key_rejected(self):
        bad = dict(self.recipe); bad["surprise"] = True
        with self.assertRaises(RecipeError): validate_recipe(bad)

    def test_forward_reference_rejected(self):
        bad = json.loads(json.dumps(self.recipe))
        bad["trajectories"][0] = {"id": "bad", "kind": "translation", "parameters": {"source": "carrier_a", "shift": [1, 1]}}
        with self.assertRaises(RecipeError): validate_recipe(bad)

    def test_reference_invariants(self):
        result = run_recipe(self.recipe, implementation_revision="test-revision")
        by_id = {x["id"]: x for x in result["trajectories"]}
        comparisons = {x["id"]: x for x in result["comparisons"]}
        self.assertEqual(result["evidence_class"], "SIMULATION")
        self.assertEqual(result["replication_status"], "not_attempted")
        self.assertEqual(result["recipe_sha256"], sha256_json(self.recipe))
        self.assertEqual(by_id["straight"]["metrics"]["menger_curvature"], [0.0] * 4)
        self.assertTrue(all(abs(k - 0.5) < 1e-12 for k in by_id["curved"]["metrics"]["menger_curvature"]))
        self.assertEqual(by_id["null"]["metrics"]["path_length"], 0.0)
        self.assertNotEqual(by_id["carrier_a"]["points"], by_id["carrier_b"]["points"])
        self.assertEqual(by_id["carrier_a"]["metrics"]["order_1"], by_id["carrier_b"]["metrics"]["order_1"])
        self.assertEqual(by_id["carrier_a"]["metrics"]["menger_curvature"], by_id["carrier_b"]["metrics"]["menger_curvature"])
        self.assertEqual(comparisons["carrier_velocity_alignment"]["mean_cosine_alignment"], 1.0)
        self.assertEqual(comparisons["control_velocity_alignment"]["mean_cosine_alignment"], 1.0)
        self.assertLess(comparisons["causal_velocity_alignment"]["mean_cosine_alignment"], 1.0)

    def test_deterministic_replay(self):
        first = run_recipe(self.recipe, implementation_revision="same-revision")
        second = run_recipe(self.recipe, implementation_revision="same-revision")
        self.assertEqual(first, second)
        self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])

    def test_revision_is_hash_bound(self):
        self.assertNotEqual(run_recipe(self.recipe, implementation_revision="rev-a")["artifact_sha256"], run_recipe(self.recipe, implementation_revision="rev-b")["artifact_sha256"])

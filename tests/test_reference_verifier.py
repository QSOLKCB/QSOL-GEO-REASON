import importlib.util
import json
import unittest
from pathlib import Path

from qsol_geo_reason.simulation import run_recipe

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_reference", ROOT / "tools" / "verify_reference.py")
verify_reference = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_reference)


class ReferenceVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((ROOT / "recipes/reference-suite.json").read_text(encoding="utf-8"))
        cls.result = run_recipe(cls.recipe, implementation_revision="implementation-test")
        cls.metadata = {
            "schema_version": cls.result["schema_version"],
            "protocol_id": cls.result["protocol_id"],
            "recipe_id": cls.result["recipe_id"],
            "implementation_revision": cls.result["implementation_revision"],
            "recipe_sha256": cls.result["recipe_sha256"],
            "artifact_sha256": cls.result["artifact_sha256"],
            "evidence_class": cls.result["evidence_class"],
            "replication_status": cls.result["replication_status"],
        }

    def test_consistent_identities_pass(self):
        self.assertEqual(
            verify_reference._identity_errors(self.recipe, self.metadata, self.result, self.result),
            [],
        )

    def test_every_metadata_identity_is_enforced(self):
        replacements = {
            "schema_version": "9.9.9",
            "protocol_id": "WRONG-PROTOCOL",
            "recipe_id": "WRONG-RECIPE",
            "implementation_revision": "wrong-revision",
            "recipe_sha256": "0" * 64,
            "artifact_sha256": "1" * 64,
            "evidence_class": "OBSERVATION",
            "replication_status": "replicated",
        }
        for key, replacement in replacements.items():
            with self.subTest(key=key):
                tampered = dict(self.metadata)
                tampered[key] = replacement
                errors = verify_reference._identity_errors(self.recipe, tampered, self.result, self.result)
                self.assertTrue(any(key in error for error in errors), errors)

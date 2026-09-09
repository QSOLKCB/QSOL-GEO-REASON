"""Round 38 regressions for live-vault hashing, templates, MPS identity, and vectors."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import (
    CaptureContractError,
    HuggingFacePyTorchBackend as CanonicalBackend,
    verify_capture_bundle,
)
from qsol_geo_reason.capture_backend_final import _remember_final_construction_request
from qsol_geo_reason.capture_execute import _assert_observation_backend_execution_methods
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from qsol_geo_reason.capture_validation import validate_capture_request
from test_capture import execute, fixture_request as simulation_fixture_request, rehash_outer_bundle
from test_capture_codex_round6 import (
    fixture_request as production_fixture_request,
    valid_production_shape,
)

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound38RegressionTests(unittest.TestCase):
    def test_live_final_construction_vault_entries_are_opaque_to_dependency_hashing(self):
        # Use the current concrete OBSERVATION boundary. It still inherits the final
        # construction vault whose mutable WeakKeyDictionary contents this regression
        # protects, while the import-time adapter receipt correctly tracks the newest
        # production subclass.
        backend = object.__new__(CanonicalBackend)
        _assert_observation_backend_execution_methods(backend)

        # A real final backend adds itself to a closure-owned WeakKeyDictionary in
        # __new__. That mutable trust-vault content is state, not executable code, and
        # must not change the import-time execution dependency receipt.
        _remember_final_construction_request(
            backend,
            ("1" * 64, "2" * 64),
        )
        _assert_observation_backend_execution_methods(backend)

    def test_production_template_includes_and_explains_frozen_tree_receipts(self):
        template = json.loads(
            (ROOT / "examples" / "GEO-CAP-001.example.json").read_text(
                encoding="utf-8"
            )
        )
        validated = validate_capture_request(template)
        model = validated["model"]
        self.assertEqual(model["revision_tree_sha256"], "0" * 64)
        self.assertEqual(model["tokenizer_revision_tree_sha256"], "0" * 64)
        notes = template["notes"]
        self.assertIn("trees/<commit>.json", notes)
        self.assertIn("SHA-256", notes)
        self.assertIn("freeze", notes)

    def test_production_validator_rejects_mps_without_concrete_hardware_identity(self):
        request = production_fixture_request()
        request["backend"]["device"] = "mps"
        observed = valid_production_shape(request)
        observed["mps_mac_model"] = None
        observed["mps_cpu_brand"] = None
        with self.assertRaisesRegex(CaptureContractError, "MPS provenance requires"):
            _validate_production_metadata_shape(observed, request)

        observed["mps_mac_model"] = "Mac15,7"
        _validate_production_metadata_shape(observed, request)

    def test_oversized_vector_integer_is_rejected_as_contract_error(self):
        request = simulation_fixture_request()
        manifest, trajectory = execute(request)
        record = trajectory["steps"][0]["layers"][0]
        # 10**1000 is JSON-serializable under Python's default integer-string
        # limit but still far outside binary64, so float(value) raises OverflowError.
        record["vector"][0] = 10**1000
        record["vector_sha256"] = sha256_json(record["vector"])
        rehash_outer_bundle(manifest, trajectory)

        with self.assertRaisesRegex(CaptureContractError, "vector content/dimension mismatch"):
            verify_capture_bundle(request, manifest, trajectory)


if __name__ == "__main__":
    unittest.main()

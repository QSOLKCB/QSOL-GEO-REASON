from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_provenance import _validate_backend_identity
from qsol_geo_reason.capture_validation import validate_capture_request

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


class CaptureRound7RegressionTests(unittest.TestCase):
    def test_reused_backend_is_bound_to_model_and_tokenizer_repository_ids(self):
        request = fixture_request()
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._applied_seed = request["determinism"]["seed"]
        backend._model_identifier = request["model"]["identifier"]
        backend._tokenizer_identifier = request["model"]["tokenizer_identifier"]
        backend.assert_execution_request(request)

        mirrored_model = copy.deepcopy(request)
        mirrored_model["model"]["identifier"] = "mirror/model"
        with self.assertRaisesRegex(CaptureContractError, "model repository identity"):
            backend.assert_execution_request(mirrored_model)

        mirrored_tokenizer = copy.deepcopy(request)
        mirrored_tokenizer["model"]["tokenizer_identifier"] = "mirror/tokenizer"
        with self.assertRaisesRegex(CaptureContractError, "tokenizer repository identity"):
            backend.assert_execution_request(mirrored_tokenizer)

    def test_hub_repo_ids_reject_repeated_forbidden_separators(self):
        for field in ("identifier", "tokenizer_identifier"):
            for bad in ("org/model--copy", "org/model..copy"):
                request = fixture_request()
                request["model"][field] = bad
                with self.subTest(field=field, bad=bad):
                    with self.assertRaisesRegex(CaptureContractError, "namespace/repository"):
                        validate_capture_request(request)

    def test_hub_repo_id_schema_patterns_match_runtime_separator_rule(self):
        request_schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(encoding="utf-8")
        )
        manifest_schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        patterns = (
            request_schema["$defs"]["hubRepoId"]["pattern"],
            manifest_schema["$defs"]["hubRepoId"]["pattern"],
        )
        for pattern in patterns:
            self.assertIsNotNone(re.fullmatch(pattern, "org/model-copy"))
            self.assertIsNotNone(re.fullmatch(pattern, "org/model.copy"))
            self.assertIsNone(re.fullmatch(pattern, "org/model--copy"))
            self.assertIsNone(re.fullmatch(pattern, "org/model..copy"))

    def test_simulation_and_observation_backend_names_are_distinct(self):
        request = fixture_request()
        observed = {
            "name": "software-simulation",
            "observed_model_commit": request["model"]["revision"],
            "observed_tokenizer_commit": request["model"]["tokenizer_revision"],
        }
        _validate_backend_identity(observed, request, "SIMULATION")

        observed["name"] = "huggingface-pytorch"
        with self.assertRaisesRegex(CaptureContractError, "software-simulation"):
            _validate_backend_identity(observed, request, "SIMULATION")
        _validate_backend_identity(observed, request, "OBSERVATION")

        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            schema["$defs"]["backendObservedSimulation"]["properties"]["name"]["const"],
            "software-simulation",
        )
        self.assertEqual(
            schema["$defs"]["backendObservedProduction"]["properties"]["name"]["const"],
            "huggingface-pytorch",
        )


if __name__ == "__main__":
    unittest.main()

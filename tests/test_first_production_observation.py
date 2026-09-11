from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_first_production_observation.py"
SPEC = importlib.util.spec_from_file_location("qsol_first_observation_tool", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class FirstProductionObservationTests(unittest.TestCase):
    def test_template_freezes_small_cpu_reference_model_and_capture_definition(self) -> None:
        request = TOOL._load_template()
        self.assertEqual(request["model"]["identifier"], "openai-community/gpt2")
        self.assertEqual(
            request["model"]["revision"],
            "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        )
        self.assertEqual(request["model"]["tokenizer_identifier"], "openai-community/gpt2")
        self.assertEqual(
            request["model"]["tokenizer_revision"],
            "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        )
        self.assertEqual(request["backend"]["device"], "cpu")
        self.assertEqual(request["backend"]["dtype"], "float32")
        self.assertEqual(request["capture"]["layers"], [0, 3, 6, 9, 12])
        self.assertEqual(request["capture"]["pooling"], {"mode": "step_mean"})
        self.assertEqual(request["determinism"], {"mode": "required", "seed": 20260912})
        self.assertNotIn("revision_tree_sha256", request["model"])
        self.assertNotIn("tokenizer_revision_tree_sha256", request["model"])

    def test_only_authenticated_tree_receipts_may_materialize_after_preregistration(self) -> None:
        request = TOOL._load_template()
        final = json.loads(json.dumps(request))
        final["model"]["revision_tree_sha256"] = "1" * 64
        final["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        validated = TOOL._assert_exact_experiment_request(final, require_receipts=True)
        self.assertEqual(validated["model"]["revision_tree_sha256"], "1" * 64)
        self.assertEqual(validated["model"]["tokenizer_revision_tree_sha256"], "2" * 64)

        final["capture"]["layers"] = [0, 6, 12]
        with self.assertRaises(CaptureContractError):
            TOOL._assert_exact_experiment_request(final, require_receipts=True)

    def test_prepare_writes_one_final_request_and_refuses_overwrite(self) -> None:
        receipts = {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "final-request.json"
            with mock.patch.object(TOOL, "prepare_tree_receipts", return_value=receipts):
                request_sha256 = TOOL.prepare(output)
                written = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(written["model"]["revision_tree_sha256"], "a" * 64)
                self.assertEqual(written["model"]["tokenizer_revision_tree_sha256"], "b" * 64)
                self.assertEqual(request_sha256, TOOL.sha256_json(written))
                with self.assertRaises(CaptureContractError):
                    TOOL.prepare(output)


if __name__ == "__main__":
    unittest.main()

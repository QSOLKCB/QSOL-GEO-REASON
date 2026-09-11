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
    def _materialized_request(self) -> dict:
        request = TOOL._load_template()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return TOOL._assert_exact_experiment_request(request, require_receipts=True)

    def test_template_freezes_small_cpu_reference_model_and_capture_definition(self) -> None:
        request = TOOL._load_template()
        self.assertEqual(request["model"]["identifier"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(
            request["model"]["revision"],
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
        )
        self.assertEqual(request["model"]["tokenizer_identifier"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(
            request["model"]["tokenizer_revision"],
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
        )
        self.assertEqual(request["backend"]["device"], "cpu")
        self.assertEqual(request["backend"]["dtype"], "float32")
        self.assertEqual(request["capture"]["layers"], [0, 6, 12, 18, 24])
        self.assertEqual(request["capture"]["pooling"], {"mode": "step_mean"})
        self.assertEqual(request["determinism"], {"mode": "required", "seed": 20260912})
        self.assertNotIn("revision_tree_sha256", request["model"])
        self.assertNotIn("tokenizer_revision_tree_sha256", request["model"])

    def test_only_authenticated_tree_receipts_may_materialize_after_preregistration(self) -> None:
        final = self._materialized_request()
        self.assertEqual(final["model"]["revision_tree_sha256"], "1" * 64)
        self.assertEqual(final["model"]["tokenizer_revision_tree_sha256"], "2" * 64)

        final["capture"]["layers"] = [0, 12, 24]
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

    def test_failed_payload_sync_does_not_publish_partial_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output = directory / "request.json"
            with mock.patch.object(TOOL.os, "fsync", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(CaptureContractError, "unable to persist"):
                    TOOL._exclusive_write_json(output, {"value": 1})
            self.assertFalse(output.exists())
            self.assertEqual(list(directory.glob(".request.json.tmp.*")), [])

    def test_failed_observation_is_preserved_without_blocking_retry_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request_path = directory / "request.json"
            request_path.write_text(
                json.dumps(self._materialized_request()), encoding="utf-8"
            )
            output_root = directory / "observation"

            with mock.patch.object(
                TOOL,
                "_run_capture",
                side_effect=CaptureContractError("transient backend failure"),
            ):
                with self.assertRaisesRegex(
                    CaptureContractError, "incomplete evidence preserved"
                ):
                    TOOL.observe(request_path, output_root, None)

            self.assertFalse(output_root.exists())
            failed = list(directory.glob("observation.failed-*"))
            self.assertEqual(len(failed), 1)
            marker = json.loads(
                (failed[0] / "execution-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["attempt_status"], "failed_before_replay_verdict")
            self.assertEqual(marker["completed_run_directories"], [])


if __name__ == "__main__":
    unittest.main()

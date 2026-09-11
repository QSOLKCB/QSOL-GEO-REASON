from __future__ import annotations

import inspect
import json
import subprocess
import sys
import types
import unittest
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason import capture_hub_prepare_worker


class Phase2AProductionBoundaryHardeningTests(unittest.TestCase):
    def test_isolated_hub_preparation_uses_no_site_child_sanitized_environment_and_returns_package_receipt(self) -> None:
        seen: dict[str, object] = {}
        evidence = {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
            "huggingface_hub_package_file_count": 137,
            "huggingface_hub_package_receipt_sha256": "c" * 64,
        }

        def fake_run(command, **kwargs):
            seen["command"] = list(command)
            seen["environment"] = dict(kwargs["env"])
            seen["input"] = kwargs["input"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(evidence, sort_keys=True) + "\n",
                stderr="",
            )

        hostile = {
            "PYTHONPATH": "/tmp/evil-python",
            "PYTHONHOME": "/tmp/evil-home",
            "LD_PRELOAD": "/tmp/evil.so",
            "HTTPS_PROXY": "http://127.0.0.1:8080",
            "REQUESTS_CA_BUNDLE": "/tmp/mitm.pem",
            "HF_ENDPOINT": "https://mirror.invalid",
            "HF_HUB_OFFLINE": "1",
        }
        with (
            mock.patch.object(TOOL.subprocess, "run", side_effect=fake_run),
            mock.patch.dict(TOOL.os.environ, hostile, clear=False),
        ):
            observed = TOOL.prepare_hub_evidence({"frozen": "request"})

        self.assertEqual(observed, evidence)
        command = seen["command"]
        self.assertEqual(command[:4], [sys.executable, "-I", "-S", "-B"])
        self.assertIn("capture_hub_prepare_worker", command[5])
        environment = seen["environment"]
        for key in hostile:
            self.assertNotIn(key, environment)

    def test_hub_tree_compatibility_surface_does_not_leak_package_fields_into_request(self) -> None:
        evidence = {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
            "huggingface_hub_package_file_count": 137,
            "huggingface_hub_package_receipt_sha256": "c" * 64,
        }
        with mock.patch.object(
            TOOL,
            "prepare_hub_evidence",
            return_value=evidence,
        ):
            receipts = TOOL.prepare_tree_receipts({"frozen": "request"})
        self.assertEqual(
            receipts,
            {
                "revision_tree_sha256": "a" * 64,
                "tokenizer_revision_tree_sha256": "b" * 64,
            },
        )

    def test_hub_worker_rejects_preloaded_huggingface_hub(self) -> None:
        fake_hub = types.ModuleType("huggingface_hub")
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with self.assertRaisesRegex(CaptureContractError, "absent before"):
                capture_hub_prepare_worker._preimport_hub_package_provenance()

    def test_hub_worker_source_binds_package_before_after_and_into_output(self) -> None:
        source = inspect.getsource(capture_hub_prepare_worker)
        self.assertIn("sys.flags.no_site", source)
        self.assertIn('"sitecustomize" in sys.modules', source)
        self.assertIn("_preimport_hub_package_provenance()", source)
        self.assertIn("after_import != before", source)
        self.assertIn("after_work != before", source)
        self.assertIn('"huggingface_hub_package_file_count"', source)
        self.assertIn('"huggingface_hub_package_receipt_sha256"', source)
        self.assertLess(
            source.index("before = _preimport_hub_package_provenance()"),
            source.index("import huggingface_hub"),
        )

    def test_orchestrator_authenticates_launcher_and_reference_lock(self) -> None:
        source = inspect.getsource(TOOL.prepare)
        helper_source = inspect.getsource(TOOL.prepare.__globals__["_authenticate_external_inputs"])
        self.assertIn("_authenticate_external_inputs", source)
        self.assertIn("authenticate_tracked_tool_against_revision", helper_source)
        self.assertIn("REFERENCE_LOCK", helper_source)


if __name__ == "__main__":
    unittest.main()

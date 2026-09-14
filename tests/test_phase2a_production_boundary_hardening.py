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
from qsol_geo_reason import capture_hub_prepare_worker, no_site_subprocess
from reference_environment_fixture import (
    hub_transport_package_provenance,
    reference_environment_receipt,
)


class Phase2AProductionBoundaryHardeningTests(unittest.TestCase):
    def _worker_evidence(self) -> dict[str, object]:
        return {
            "revision_tree_sha256": "a" * 64,
            "tokenizer_revision_tree_sha256": "b" * 64,
            "huggingface_hub_package_file_count": 137,
            "huggingface_hub_package_receipt_sha256": "7" * 64,
            "hub_transport_package_provenance": hub_transport_package_provenance(),
        }

    def _authenticated_source_patches(self):
        return (
            mock.patch.object(
                no_site_subprocess,
                "_AUTHENTICATED_SOURCE_REVISION",
                "a" * 40,
            ),
            mock.patch.object(
                no_site_subprocess,
                "_AUTHENTICATED_SOURCE_MANIFEST",
                {"__init__.py": "b" * 64},
            ),
        )

    def test_isolated_hub_preparation_uses_authenticated_no_site_child_sanitized_environment_and_returns_package_receipts(self) -> None:
        seen: dict[str, object] = {}
        evidence = self._worker_evidence()

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
        revision_patch, manifest_patch = self._authenticated_source_patches()
        with (
            mock.patch.object(TOOL.subprocess, "run", side_effect=fake_run),
            mock.patch.dict(TOOL.os.environ, hostile, clear=False),
            revision_patch,
            manifest_patch,
        ):
            observed = TOOL.prepare_hub_evidence({"frozen": "request"})

        self.assertEqual(observed, evidence)
        command = seen["command"]
        self.assertEqual(command[:4], [sys.executable, "-I", "-S", "-B"])
        bootstrap = command[5]
        self.assertIn("capture_hub_prepare_worker", bootstrap)
        self.assertIn("source_manifest=json.loads(sys.argv[3])", bootstrap)
        self.assertIn("sourcebad=sorted", bootstrap)
        self.assertLess(
            bootstrap.index("sourcebad=sorted"),
            bootstrap.index("sys.path.insert(0,src)"),
        )
        self.assertLess(
            bootstrap.index("sys.path.insert(0,src)"),
            bootstrap.index(
                "from qsol_geo_reason.capture_hub_prepare_worker import main"
            ),
        )
        self.assertIn("a" * 40, command)
        environment = seen["environment"]
        for key in hostile:
            self.assertNotIn(key, environment)

    def test_hub_tree_compatibility_surface_binds_package_and_transport_then_returns_only_request_fields(self) -> None:
        evidence = self._worker_evidence()

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(evidence, sort_keys=True) + "\n",
                stderr="",
            )

        revision_patch, manifest_patch = self._authenticated_source_patches()
        with (
            mock.patch.object(TOOL.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                TOOL,
                "verify_current_reference_environment",
                return_value=reference_environment_receipt(),
            ),
            revision_patch,
            manifest_patch,
        ):
            receipts = TOOL.prepare_tree_receipts({"frozen": "request"})
        self.assertEqual(
            receipts,
            {
                "revision_tree_sha256": "a" * 64,
                "tokenizer_revision_tree_sha256": "b" * 64,
            },
        )

    def test_hub_tree_compatibility_surface_rejects_worker_transport_digest_drift(self) -> None:
        evidence = self._worker_evidence()
        transport = json.loads(json.dumps(evidence["hub_transport_package_provenance"]))
        transport["requests"]["receipt_sha256"] = "d" * 64
        evidence["hub_transport_package_provenance"] = transport

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(evidence, sort_keys=True) + "\n",
                stderr="",
            )

        revision_patch, manifest_patch = self._authenticated_source_patches()
        with (
            mock.patch.object(TOOL.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                TOOL,
                "verify_current_reference_environment",
                return_value=reference_environment_receipt(),
            ),
            revision_patch,
            manifest_patch,
        ):
            with self.assertRaisesRegex(
                CaptureContractError,
                "runtime provenance does not match",
            ):
                TOOL.prepare_tree_receipts({"frozen": "request"})

    def test_hub_worker_rejects_preloaded_huggingface_hub(self) -> None:
        fake_hub = types.ModuleType("huggingface_hub")
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with self.assertRaisesRegex(CaptureContractError, "absent before"):
                capture_hub_prepare_worker._preimport_hub_package_provenance()

    def test_hub_worker_source_binds_hub_and_transport_before_after_and_into_output(self) -> None:
        source = inspect.getsource(capture_hub_prepare_worker)
        self.assertIn("sys.flags.no_site", source)
        self.assertIn('"sitecustomize" in sys.modules', source)
        self.assertIn("_preimport_hub_package_provenance()", source)
        self.assertIn("_preimport_transport_package_provenance()", source)
        self.assertIn("transport_after_import != transport_before", source)
        self.assertIn("transport_after_work != transport_before", source)
        self.assertIn('"huggingface_hub_package_file_count"', source)
        self.assertIn('"huggingface_hub_package_receipt_sha256"', source)
        self.assertIn('"hub_transport_package_provenance"', source)
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

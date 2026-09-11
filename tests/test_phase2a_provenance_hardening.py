from __future__ import annotations

import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import provenance, tracked_artifact
from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_preparation import (
    build_preparation_receipt,
    verify_preparation_receipt,
)
from test_capture_codex_round6 import fixture_request


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "run_first_production_observation.py"
EXPERIMENT_DOC = ROOT / "experiments" / "GEO-CAP-001-EXP-001.md"


class Phase2AProvenanceHardeningTests(unittest.TestCase):
    def _load_launcher(self):
        spec = importlib.util.spec_from_file_location(
            "qsol_test_phase2a_hardened_launcher",
            LAUNCHER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_launcher_strips_proxy_ca_git_and_path_overrides(self) -> None:
        launcher = self._load_launcher()
        hostile = {
            "PATH": "/tmp/fake-bin",
            "HTTPS_PROXY": "http://127.0.0.1:8080",
            "http_proxy": "http://127.0.0.1:8081",
            "ALL_PROXY": "socks5://127.0.0.1:1080",
            "NO_PROXY": "huggingface.co",
            "REQUESTS_CA_BUNDLE": "/tmp/mitm.pem",
            "CURL_CA_BUNDLE": "/tmp/curl.pem",
            "SSL_CERT_FILE": "/tmp/ssl.pem",
            "SSL_CERT_DIR": "/tmp/ssl-dir",
            "HF_HUB_DISABLE_SSL_VERIFICATION": "1",
            "GIT_DIR": "/tmp/fake-git-dir",
            "GIT_OBJECT_DIRECTORY": "/tmp/fake-objects",
            "QSOL_SAFE_SENTINEL": "keep-me",
        }
        with mock.patch.dict(launcher.os.environ, hostile, clear=True):
            environment = launcher._sanitized_orchestrator_environment()

        self.assertEqual(environment["QSOL_SAFE_SENTINEL"], "keep-me")
        self.assertEqual(environment["PATH"], launcher._trusted_system_path())
        for key in hostile:
            if key not in {"QSOL_SAFE_SENTINEL", "PATH"}:
                self.assertNotIn(key, environment)
        self.assertEqual(environment["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)

    @unittest.skipUnless(os.name == "posix", "system-Git trust regression is POSIX-specific")
    def test_path_git_shim_cannot_control_provenance_or_template_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            marker = directory / "shim-invoked"
            shim = directory / "git"
            shim.write_text(
                "#!/bin/sh\nprintf invoked > \"$QSOL_GIT_SHIM_MARKER\"\n"
                "printf '%s\\n' ffffffffffffffffffffffffffffffffffffffff\n",
                encoding="utf-8",
            )
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR)

            environment = {
                "PATH": str(directory),
                "QSOL_GIT_SHIM_MARKER": str(marker),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                revision = provenance.git_source_revision(
                    require_clean=False,
                    reject_importable_bytecode=False,
                )
                self.assertRegex(revision or "", r"^[0-9a-f]{40}$")
                tracked = tracked_artifact.authenticate_tracked_file_against_revision(
                    ROOT / "experiments" / "GEO-CAP-001-EXP-001.request.template.json",
                    revision,
                )

            self.assertEqual(
                tracked,
                "experiments/GEO-CAP-001-EXP-001.request.template.json",
            )
            self.assertFalse(marker.exists())
            self.assertTrue(provenance._trusted_git_executable().is_absolute())

    def test_uppercase_preparation_commit_is_rejected_even_after_rehash(self) -> None:
        request = fixture_request()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        receipt = build_preparation_receipt(
            request=request,
            repository_commit="a" * 40,
            experiment_id="EXP-TEST-001",
        )
        receipt["preparation_repository_commit"] = "A" * 40
        receipt["preparation_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in receipt.items()
                if key != "preparation_receipt_sha256"
            }
        )
        with self.assertRaisesRegex(
            CaptureContractError,
            "canonical lowercase 40-hex",
        ):
            verify_preparation_receipt(
                receipt,
                request=request,
                experiment_id="EXP-TEST-001",
            )

    def test_experiment_uses_python311_frozen_reference_installation(self) -> None:
        document = EXPERIMENT_DOC.read_text(encoding="utf-8")
        self.assertIn(
            "python3.11 -m venv /tmp/qsol-geo-reason-capture-py311",
            document,
        )
        self.assertNotIn("python3.11 -m venv .venv-capture-py311", document)
        self.assertIn("outside the repository checkout", document)
        self.assertIn("constraints/capture-reference-py311.txt", document)
        self.assertIn("torch==2.2.2+cpu", document)
        self.assertIn("verify_capture_reference_environment.py", document)
        self.assertIn("complete resolved runtime lock", document)
        self.assertIn("download.pytorch.org/whl/cpu", document)
        self.assertIn("must report Python 3.11.x", document)


if __name__ == "__main__":
    unittest.main()

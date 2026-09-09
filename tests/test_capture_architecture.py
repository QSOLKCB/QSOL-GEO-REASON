from __future__ import annotations

import json
import unittest
from pathlib import Path

import qsol_geo_reason


ROOT = Path(__file__).resolve().parents[1]


class CaptureArchitectureRemediationTests(unittest.TestCase):
    def test_public_capture_facade_has_one_backend_composition_import(self):
        source = (ROOT / "src" / "qsol_geo_reason" / "capture.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from .capture_backend_canonical import HuggingFacePyTorchBackend", source)
        self.assertNotIn("capture_backend_round", source)

        composition = (
            ROOT / "src" / "qsol_geo_reason" / "capture_backend_canonical.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Single composition boundary", composition)
        self.assertIn("Historical ``capture_backend_round*`` modules", composition)

    def test_canonical_cli_delegates_observation_to_isolated_worker(self):
        cli = (ROOT / "src" / "qsol_geo_reason" / "capture_cli.py").read_text(
            encoding="utf-8"
        )
        worker = (ROOT / "src" / "qsol_geo_reason" / "capture_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"-I"', cli)
        self.assertIn('"-B"', cli)
        self.assertIn('"qsol_geo_reason.capture_worker"', cli)
        self.assertIn('"HF_HUB_OFFLINE": "1"', cli)
        self.assertIn("sys.flags.isolated", worker)
        self.assertIn("preloaded production dependencies", worker)

    def test_capture_reference_environment_is_frozen_and_real_backend_ci_exists(self):
        constraints = (
            ROOT / "constraints" / "capture-reference-py311.txt"
        ).read_text(encoding="utf-8")
        for requirement in (
            "torch==2.2.2",
            "transformers==4.40.2",
            "huggingface-hub==0.23.5",
            "tokenizers==0.19.1",
            "safetensors==0.4.3",
        ):
            self.assertIn(requirement, constraints)

        workflow = (
            ROOT / ".github" / "workflows" / "capture-production-integration.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("run_capture_production_integration.py", workflow)
        self.assertIn("pip check", workflow)

        integration = (
            ROOT / "tools" / "run_capture_production_integration.py"
        ).read_text(encoding="utf-8")
        self.assertIn("GPT2LMHeadModel", integration)
        self.assertIn("PreTrainedTokenizerFast", integration)
        self.assertIn("verify_capture_bundle", integration)
        self.assertIn("Draft202012Validator", integration)
        self.assertIn("deterministic replay produced different", integration)

    def test_package_identity_advances_for_phase2a(self):
        self.assertEqual(qsol_geo_reason.__version__, "0.3.0.dev0")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('version = "0.3.0.dev0"', pyproject)
        self.assertIn('"torch>=2.2.2,<2.3"', pyproject)
        self.assertIn('"transformers>=4.40.2,<4.41"', pyproject)

    def test_threat_model_draws_trusted_cli_boundary(self):
        threat_model = (
            ROOT / "docs" / "GEO-CAP-001-THREAT-MODEL.md"
        ).read_text(encoding="utf-8")
        self.assertIn("trusted local CLI that launches a fresh isolated Python worker", threat_model)
        self.assertIn("arbitrary hostile mutation inside an already-running embedding process", threat_model)
        self.assertIn("not a sandbox", threat_model)
        self.assertIn("capture-reference-py311.txt", threat_model)


if __name__ == "__main__":
    unittest.main()

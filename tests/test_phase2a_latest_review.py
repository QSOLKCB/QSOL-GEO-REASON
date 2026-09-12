from __future__ import annotations

import secrets
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DOC = ROOT / "experiments" / "GEO-CAP-001-EXP-001.md"


class Phase2ALatestReviewTests(unittest.TestCase):
    def test_prepare_rejects_checkout_output_before_revision_or_hub_work(self) -> None:
        output = ROOT / f".qsol-test-prepare-{secrets.token_hex(8)}.json"
        sidecar = TOOL._default_preparation_receipt_path(output)
        self.assertFalse(output.exists())
        self.assertFalse(sidecar.exists())
        with (
            mock.patch.object(TOOL, "resolve_implementation_revision") as resolver,
            mock.patch.object(TOOL, "prepare_tree_receipts") as warmup,
        ):
            with self.assertRaisesRegex(
                CaptureContractError,
                "preparation output must be outside the source checkout",
            ):
                TOOL.prepare(output)
        resolver.assert_not_called()
        warmup.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(sidecar.exists())

    def test_external_preparation_path_boundary_accepts_external_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            TOOL._assert_preparation_output_outside_checkout(Path(tmp) / "request.json")

    def test_experiment_uses_no_site_initial_launcher_invocation(self) -> None:
        document = EXPERIMENT_DOC.read_text(encoding="utf-8")
        self.assertIn(
            "python -I -S -B tools/run_first_production_observation.py prepare",
            document,
        )
        self.assertIn(
            "python -I -S -B tools/run_first_production_observation.py observe",
            document,
        )
        self.assertNotIn(
            "\npython tools/run_first_production_observation.py prepare",
            document,
        )


if __name__ == "__main__":
    unittest.main()

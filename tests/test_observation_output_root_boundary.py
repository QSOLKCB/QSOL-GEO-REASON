from __future__ import annotations

import json
import secrets
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]


class ObservationOutputRootBoundaryTests(unittest.TestCase):
    def _materialized_request(self) -> dict:
        request = TOOL._load_template()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return TOOL._assert_exact_experiment_request(
            request,
            require_receipts=True,
        )

    def test_observe_rejects_checkout_contained_output_before_capture_or_failure_archive(self) -> None:
        request = self._materialized_request()
        repository_commit = "f" * 40
        preparation = TOOL.build_preparation_receipt(
            request=request,
            repository_commit=repository_commit,
            experiment_id=TOOL.EXPERIMENT_ID,
        )

        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp)
            request_path = external / "request.json"
            receipt_path = external / "request.preparation-receipt.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            receipt_path.write_text(json.dumps(preparation), encoding="utf-8")

            output_root = ROOT / f".qsol-test-observation-{secrets.token_hex(8)}"
            self.assertFalse(output_root.exists())
            failed_pattern = f"{output_root.name}.failed-*"
            self.assertEqual(list(ROOT.glob(failed_pattern)), [])

            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=repository_commit,
                ),
                mock.patch.object(
                    TOOL,
                    "authenticate_tracked_file_against_revision",
                ),
                mock.patch.object(TOOL, "_run_capture") as run_capture,
            ):
                with self.assertRaisesRegex(
                    CaptureContractError,
                    "outside the source checkout",
                ):
                    TOOL.observe(
                        request_path,
                        output_root,
                        None,
                        receipt_path,
                    )

            run_capture.assert_not_called()
            self.assertFalse(output_root.exists())
            self.assertEqual(list(ROOT.glob(failed_pattern)), [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import first_production_observation as TOOL
from qsol_geo_reason.capture_common import CaptureContractError


class PreparationCrashRecoveryTests(unittest.TestCase):
    def _prepared_request_and_receipt(
        self,
        *,
        preparation_commit: str = "c" * 40,
    ) -> tuple[dict, dict]:
        request = TOOL._load_template()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        request = TOOL._assert_exact_experiment_request(
            request,
            require_receipts=True,
        )
        receipt = TOOL.build_preparation_receipt(
            request=request,
            repository_commit=preparation_commit,
            experiment_id=TOOL.EXPERIMENT_ID,
        )
        return request, receipt

    def test_receipt_only_crash_state_recovers_without_hub_warmup(self) -> None:
        recovery_commit = "f" * 40
        preparation_commit = "c" * 40
        request, receipt = self._prepared_request_and_receipt(
            preparation_commit=preparation_commit
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "final-request.json"
            sidecar = TOOL._default_preparation_receipt_path(output)
            sidecar.write_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=recovery_commit,
                ) as resolver,
                mock.patch.object(
                    TOOL,
                    "authenticate_tracked_file_against_revision",
                    return_value="experiments/GEO-CAP-001-EXP-001.request.template.json",
                ) as authenticate,
                mock.patch.object(
                    TOOL,
                    "prepare_tree_receipts",
                    side_effect=AssertionError("recovery must not contact the Hub"),
                ) as warmup,
            ):
                request_sha256 = TOOL.prepare(output)

            self.assertFalse(warmup.called)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                request,
            )
            self.assertEqual(request_sha256, TOOL.sha256_json(request))
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8")),
                receipt,
            )
            resolver.assert_has_calls(
                [
                    mock.call(None, require_checkout=True),
                    mock.call(recovery_commit, require_checkout=True),
                ]
            )
            authenticated_revisions = [call.args[1] for call in authenticate.call_args_list]
            self.assertIn(recovery_commit, authenticated_revisions)
            self.assertIn(preparation_commit, authenticated_revisions)

    def test_malformed_receipt_only_state_fails_closed_without_hub_warmup(self) -> None:
        recovery_commit = "f" * 40
        request, receipt = self._prepared_request_and_receipt()
        receipt["request_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "final-request.json"
            sidecar = TOOL._default_preparation_receipt_path(output)
            sidecar.write_text(json.dumps(receipt), encoding="utf-8")

            with (
                mock.patch.object(
                    TOOL,
                    "resolve_implementation_revision",
                    return_value=recovery_commit,
                ),
                mock.patch.object(
                    TOOL,
                    "authenticate_tracked_file_against_revision",
                    return_value="experiments/GEO-CAP-001-EXP-001.request.template.json",
                ),
                mock.patch.object(
                    TOOL,
                    "prepare_tree_receipts",
                    side_effect=AssertionError("malformed recovery must not contact the Hub"),
                ) as warmup,
            ):
                with self.assertRaisesRegex(
                    CaptureContractError,
                    "request_sha256 does not match",
                ):
                    TOOL.prepare(output)

            self.assertFalse(warmup.called)
            self.assertFalse(output.exists())
            self.assertTrue(sidecar.exists())
            self.assertEqual(request["model"]["revision_tree_sha256"], "1" * 64)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_execution
from qsol_geo_reason.capture_common import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]


class CaptureExecutionReceiptTests(unittest.TestCase):
    def _artifacts(self):
        request = {"run_id": "REQUEST-001", "frozen": True}
        manifest = {
            "repository_commit": "a" * 40,
            "run_manifest_id": "b" * 64,
            "manifest_sha256": "c" * 64,
        }
        trajectory = {"trajectory_sha256": "d" * 64}
        return request, manifest, trajectory

    def test_schema_is_closed_and_binds_bundle_file_hashes(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "capture-execution-receipt.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        for field in (
            "execution_id",
            "capture_request_file_sha256",
            "run_manifest_file_sha256",
            "captured_trajectory_file_sha256",
            "execution_receipt_sha256",
        ):
            self.assertIn(field, schema["required"])

    def test_receipt_is_self_hashed_and_bundle_bound(self) -> None:
        request, manifest, trajectory = self._artifacts()
        with mock.patch.object(
            capture_execution,
            "verify_capture_bundle",
            return_value=request,
        ):
            receipt = capture_execution.build_execution_receipt(
                execution_id="EXEC-001",
                request=request,
                manifest=manifest,
                trajectory=trajectory,
            )
            verified = capture_execution.verify_execution_receipt(
                receipt,
                request=request,
                manifest=manifest,
                trajectory=trajectory,
            )
            self.assertEqual(verified, receipt)

            changed_manifest = dict(manifest)
            changed_manifest["manifest_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                CaptureContractError, "does not match the verified capture bundle"
            ):
                capture_execution.verify_execution_receipt(
                    receipt,
                    request=request,
                    manifest=changed_manifest,
                    trajectory=trajectory,
                )

    def test_execution_receipt_publication_is_no_replace(self) -> None:
        request, manifest, trajectory = self._artifacts()
        with mock.patch.object(
            capture_execution,
            "verify_capture_bundle",
            return_value=request,
        ):
            receipt = capture_execution.build_execution_receipt(
                execution_id="EXEC-001",
                request=request,
                manifest=manifest,
                trajectory=trajectory,
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "execution-receipt.json"
            capture_execution.write_execution_receipt(path, receipt)
            self.assertTrue(path.is_file())
            with self.assertRaisesRegex(CaptureContractError, "refusing to overwrite"):
                capture_execution.write_execution_receipt(path, receipt)


if __name__ == "__main__":
    unittest.main()

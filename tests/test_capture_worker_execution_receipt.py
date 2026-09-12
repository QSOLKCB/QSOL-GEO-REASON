from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture_worker


class CaptureWorkerExecutionReceiptBoundaryTests(unittest.TestCase):
    def test_execution_receipt_must_stay_outside_bundle_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "run-a"
            nested = bundle / "execution-receipt.json"
            sibling = root / "run-a-execution-receipt.json"

            with self.assertRaisesRegex(RuntimeError, "outside --output-dir"):
                capture_worker._assert_execution_receipt_outside_bundle(bundle, nested)

            capture_worker._assert_execution_receipt_outside_bundle(bundle, sibling)
            capture_worker._assert_execution_receipt_outside_bundle(bundle, None)

    def test_bundle_directory_itself_is_not_a_valid_receipt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "run-a"
            with self.assertRaisesRegex(RuntimeError, "outside --output-dir"):
                capture_worker._assert_execution_receipt_outside_bundle(bundle, bundle)

    def test_blank_execution_identity_is_rejected_before_reservation_or_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text("{}\n", encoding="utf-8")
            output = root / "run-a"
            receipt = root / "run-a-execution-receipt.json"
            with mock.patch.object(capture_worker, "_assert_fresh_worker_boundary"):
                status = capture_worker.main(
                    [
                        str(request),
                        "--output-dir",
                        str(output),
                        "--execution-id",
                        "   ",
                        "--execution-receipt",
                        str(receipt),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(receipt.exists())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

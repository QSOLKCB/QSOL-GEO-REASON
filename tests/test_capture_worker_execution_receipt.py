from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

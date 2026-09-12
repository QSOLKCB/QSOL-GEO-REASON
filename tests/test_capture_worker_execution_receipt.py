from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qsol_geo_reason import capture, capture_execute, capture_worker
from qsol_geo_reason import execution_receipt_reservation
from qsol_geo_reason.capture_common import CaptureContractError


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

    def test_post_publish_fsync_failure_preserves_reserved_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text("{}\n", encoding="utf-8")
            output = root / "run-a"
            receipt = root / "run-a-execution-receipt.json"
            reservation = object()

            def reserve(path: Path):
                Path(path).write_text("reserved\n", encoding="utf-8")
                return reservation

            def publish_then_fail(output_dir, _request, _manifest, _trajectory):
                directory = Path(output_dir)
                directory.mkdir()
                for name in capture_worker._CANONICAL_BUNDLE_FILES:
                    (directory / name).write_text("{}\n", encoding="utf-8")
                raise OSError("parent directory fsync failed after rename")

            manifest = {"manifest_sha256": "a" * 64}
            with (
                mock.patch.object(capture_worker, "_assert_fresh_worker_boundary"),
                mock.patch.object(
                    execution_receipt_reservation,
                    "reserve_execution_receipt_destination",
                    side_effect=reserve,
                ),
                mock.patch.object(
                    execution_receipt_reservation,
                    "release_execution_receipt_reservation",
                ) as release,
                mock.patch.object(capture, "validate_capture_request", return_value={}),
                mock.patch.object(
                    capture_execute,
                    "resolve_implementation_revision",
                    return_value="f" * 40,
                ),
                mock.patch.object(
                    capture,
                    "HuggingFacePyTorchBackend",
                    return_value=object(),
                ),
                mock.patch.object(
                    capture,
                    "execute_capture",
                    return_value=(manifest, {}),
                ),
                mock.patch.object(
                    capture,
                    "write_capture_bundle",
                    side_effect=publish_then_fail,
                ),
            ):
                status = capture_worker.main(
                    [
                        str(request),
                        "--output-dir",
                        str(output),
                        "--execution-id",
                        "EXEC-A",
                        "--execution-receipt",
                        str(receipt),
                    ]
                )

            self.assertEqual(status, 2)
            self.assertTrue(capture_worker._complete_bundle_present(output))
            self.assertTrue(receipt.is_file())
            self.assertEqual(receipt.read_text(encoding="utf-8"), "reserved\n")
            release.assert_not_called()

    def test_preexisting_bundle_still_releases_new_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text("{}\n", encoding="utf-8")
            output = root / "run-a"
            output.mkdir()
            for name in capture_worker._CANONICAL_BUNDLE_FILES:
                (output / name).write_text("{}\n", encoding="utf-8")
            receipt = root / "run-a-execution-receipt.json"
            reservation = object()

            manifest = {"manifest_sha256": "a" * 64}
            with (
                mock.patch.object(capture_worker, "_assert_fresh_worker_boundary"),
                mock.patch.object(
                    execution_receipt_reservation,
                    "reserve_execution_receipt_destination",
                    return_value=reservation,
                ),
                mock.patch.object(
                    execution_receipt_reservation,
                    "release_execution_receipt_reservation",
                ) as release,
                mock.patch.object(capture, "validate_capture_request", return_value={}),
                mock.patch.object(
                    capture_execute,
                    "resolve_implementation_revision",
                    return_value="f" * 40,
                ),
                mock.patch.object(
                    capture,
                    "HuggingFacePyTorchBackend",
                    return_value=object(),
                ),
                mock.patch.object(
                    capture,
                    "execute_capture",
                    return_value=(manifest, {}),
                ),
                mock.patch.object(
                    capture,
                    "write_capture_bundle",
                    side_effect=CaptureContractError(
                        "output_dir already exists; canonical capture bundles are immutable publications"
                    ),
                ),
            ):
                status = capture_worker.main(
                    [
                        str(request),
                        "--output-dir",
                        str(output),
                        "--execution-id",
                        "EXEC-A",
                        "--execution-receipt",
                        str(receipt),
                    ]
                )

            self.assertEqual(status, 2)
            release.assert_called_once_with(reservation)


if __name__ == "__main__":
    unittest.main()

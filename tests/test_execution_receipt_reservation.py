from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qsol_geo_reason.execution_receipt_reservation import (
    commit_execution_receipt_reservation,
    release_execution_receipt_reservation,
    reserve_execution_receipt_destination,
)


class ExecutionReceiptReservationTests(unittest.TestCase):
    def test_existing_destination_is_rejected_before_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution-receipt.json"
            path.write_text("already here\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                reserve_execution_receipt_destination(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "already here\n")

    def test_unavailable_parent_is_rejected_before_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocked_parent = Path(tmp) / "not-a-directory"
            blocked_parent.write_text("blocked\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "parent is unavailable"):
                reserve_execution_receipt_destination(
                    blocked_parent / "execution-receipt.json"
                )

    def test_reservation_is_atomically_replaced_by_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution-receipt.json"
            reservation = reserve_execution_receipt_destination(path)
            self.assertEqual(path.read_bytes(), reservation.marker)

            receipt = {
                "schema_version": "1.0.0",
                "execution_id": "EXEC-001",
            }
            commit_execution_receipt_reservation(reservation, receipt)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                receipt,
            )

    def test_prebundle_release_removes_only_intact_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution-receipt.json"
            reservation = reserve_execution_receipt_destination(path)
            release_execution_receipt_reservation(reservation)
            self.assertFalse(path.exists())

            reservation = reserve_execution_receipt_destination(path)
            path.write_text("changed\n", encoding="utf-8")
            release_execution_receipt_reservation(reservation)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "changed\n")


if __name__ == "__main__":
    unittest.main()

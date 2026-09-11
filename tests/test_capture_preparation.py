from __future__ import annotations

import json
import unittest
from pathlib import Path

from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_hub_tree import CANONICAL_HF_ENDPOINT
from qsol_geo_reason.capture_preparation import (
    build_preparation_receipt,
    verify_preparation_receipt,
)
from test_capture_codex_round6 import fixture_request


ROOT = Path(__file__).resolve().parents[1]


class CapturePreparationReceiptTests(unittest.TestCase):
    def _request(self) -> dict:
        request = fixture_request()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return request

    def test_schema_is_closed_and_pins_endpoint_and_revision(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "capture-preparation-receipt.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["hub_endpoint"]["const"],
            CANONICAL_HF_ENDPOINT,
        )
        self.assertIn("preparation_repository_commit", schema["required"])
        self.assertIn("preparation_receipt_sha256", schema["required"])

    def test_receipt_binds_request_endpoint_and_repository_revision(self) -> None:
        request = self._request()
        receipt = build_preparation_receipt(
            request=request,
            repository_commit="a" * 40,
            experiment_id="EXP-TEST-001",
        )
        verified = verify_preparation_receipt(
            receipt,
            request=request,
            experiment_id="EXP-TEST-001",
        )
        self.assertEqual(verified, receipt)
        self.assertEqual(receipt["preparation_repository_commit"], "a" * 40)
        self.assertEqual(receipt["hub_endpoint"], CANONICAL_HF_ENDPOINT)
        self.assertRegex(receipt["preparation_receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_receipt_rejects_request_or_endpoint_tampering(self) -> None:
        request = self._request()
        receipt = build_preparation_receipt(
            request=request,
            repository_commit="a" * 40,
            experiment_id="EXP-TEST-001",
        )

        tampered_request = json.loads(json.dumps(receipt))
        tampered_request["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(CaptureContractError, "request_sha256 does not match"):
            verify_preparation_receipt(
                tampered_request,
                request=request,
                experiment_id="EXP-TEST-001",
            )

        tampered_endpoint = json.loads(json.dumps(receipt))
        tampered_endpoint["hub_endpoint"] = "https://mirror.invalid"
        with self.assertRaisesRegex(CaptureContractError, "canonical Hugging Face endpoint"):
            verify_preparation_receipt(
                tampered_endpoint,
                request=request,
                experiment_id="EXP-TEST-001",
            )


if __name__ == "__main__":
    unittest.main()

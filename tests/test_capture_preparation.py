from __future__ import annotations

import json
import unittest
from pathlib import Path

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_hub_tree import CANONICAL_HF_ENDPOINT
from qsol_geo_reason.capture_preparation import (
    PREPARATION_RECEIPT_SCHEMA_VERSION,
    build_preparation_receipt,
    verify_preparation_receipt,
)
from reference_environment_fixture import reference_environment_receipt
from test_capture_codex_round6 import fixture_request


ROOT = Path(__file__).resolve().parents[1]


class CapturePreparationReceiptTests(unittest.TestCase):
    def _request(self) -> dict:
        request = fixture_request()
        request["model"]["revision_tree_sha256"] = "1" * 64
        request["model"]["tokenizer_revision_tree_sha256"] = "2" * 64
        return request

    def _receipt(self, request: dict) -> dict:
        return build_preparation_receipt(
            request=request,
            repository_commit="a" * 40,
            experiment_id="EXP-TEST-001",
            reference_environment_receipt=reference_environment_receipt(),
        )

    def test_schema_is_closed_and_pins_endpoint_revision_reference_environment_and_hub_content(self) -> None:
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
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            PREPARATION_RECEIPT_SCHEMA_VERSION,
        )
        self.assertEqual(PREPARATION_RECEIPT_SCHEMA_VERSION, "1.3.0")
        self.assertIn("preparation_repository_commit", schema["required"])
        self.assertIn("reference_environment", schema["required"])
        self.assertIn("preparation_receipt_sha256", schema["required"])
        reference_schema = schema["$defs"]["referenceEnvironment"]
        self.assertEqual(reference_schema["properties"]["schema_version"]["const"], "1.2.0")
        self.assertEqual(
            reference_schema["properties"]["python_implementation"]["const"],
            "CPython",
        )
        self.assertEqual(
            reference_schema["properties"]["platform_system"]["const"],
            "Linux",
        )
        self.assertIn(
            "huggingface_hub_package_file_count",
            reference_schema["required"],
        )
        self.assertIn(
            "huggingface_hub_package_receipt_sha256",
            reference_schema["required"],
        )
        self.assertIn(
            "hub_transport_package_provenance",
            reference_schema["required"],
        )
        transport_schema = schema["$defs"]["hubTransportPackageProvenance"]
        self.assertEqual(
            set(transport_schema["required"]),
            {"requests", "urllib3", "certifi", "charset-normalizer", "idna"},
        )

    def test_receipt_binds_request_endpoint_repository_revision_complete_environment_and_hub_content(self) -> None:
        request = self._request()
        receipt = self._receipt(request)
        verified = verify_preparation_receipt(
            receipt,
            request=request,
            experiment_id="EXP-TEST-001",
        )
        self.assertEqual(verified, receipt)
        self.assertEqual(receipt["preparation_repository_commit"], "a" * 40)
        self.assertEqual(receipt["hub_endpoint"], CANONICAL_HF_ENDPOINT)
        self.assertEqual(receipt["reference_environment"]["distribution_count"], 28)
        self.assertEqual(receipt["reference_environment"]["python_version"], "3.11.16")
        self.assertEqual(
            receipt["reference_environment"]["huggingface_hub_package_file_count"],
            137,
        )
        self.assertEqual(
            receipt["reference_environment"]["huggingface_hub_package_receipt_sha256"],
            "7" * 64,
        )
        self.assertEqual(
            set(receipt["reference_environment"]["hub_transport_package_provenance"]),
            {"requests", "urllib3", "certifi", "charset-normalizer", "idna"},
        )
        self.assertRegex(
            receipt["reference_environment"]["environment_receipt_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(receipt["preparation_receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_receipt_rejects_request_endpoint_environment_or_hub_content_tampering(self) -> None:
        request = self._request()
        receipt = self._receipt(request)

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

        tampered_environment = json.loads(json.dumps(receipt))
        tampered_environment["reference_environment"]["python_version"] = "3.12.9"
        tampered_environment["reference_environment"]["environment_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in tampered_environment["reference_environment"].items()
                if key != "environment_receipt_sha256"
            }
        )
        tampered_environment["preparation_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in tampered_environment.items()
                if key != "preparation_receipt_sha256"
            }
        )
        with self.assertRaisesRegex(CaptureContractError, "Python 3.11"):
            verify_preparation_receipt(
                tampered_environment,
                request=request,
                experiment_id="EXP-TEST-001",
            )

        tampered_hub = json.loads(json.dumps(receipt))
        tampered_hub["reference_environment"][
            "huggingface_hub_package_receipt_sha256"
        ] = "not-a-digest"
        tampered_hub["reference_environment"]["environment_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in tampered_hub["reference_environment"].items()
                if key != "environment_receipt_sha256"
            }
        )
        tampered_hub["preparation_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in tampered_hub.items()
                if key != "preparation_receipt_sha256"
            }
        )
        with self.assertRaisesRegex(CaptureContractError, "package provenance receipt_sha256"):
            verify_preparation_receipt(
                tampered_hub,
                request=request,
                experiment_id="EXP-TEST-001",
            )

        tampered_transport = json.loads(json.dumps(receipt))
        tampered_transport["reference_environment"]["hub_transport_package_provenance"][
            "requests"
        ]["receipt_sha256"] = "not-a-digest"
        tampered_transport["reference_environment"]["environment_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in tampered_transport["reference_environment"].items()
                if key != "environment_receipt_sha256"
            }
        )
        tampered_transport["preparation_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in tampered_transport.items()
                if key != "preparation_receipt_sha256"
            }
        )
        with self.assertRaisesRegex(CaptureContractError, "transport dependency requests"):
            verify_preparation_receipt(
                tampered_transport,
                request=request,
                experiment_id="EXP-TEST-001",
            )


if __name__ == "__main__":
    unittest.main()

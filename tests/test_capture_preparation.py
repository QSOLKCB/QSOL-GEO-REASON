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

    @staticmethod
    def _rehash(receipt: dict) -> None:
        receipt["reference_environment"]["environment_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in receipt["reference_environment"].items()
                if key != "environment_receipt_sha256"
            }
        )
        receipt["preparation_receipt_sha256"] = sha256_json(
            {
                key: value
                for key, value in receipt.items()
                if key != "preparation_receipt_sha256"
            }
        )

    def test_schema_is_closed_and_pins_endpoint_revision_reference_environment_and_package_content(self) -> None:
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
        self.assertEqual(PREPARATION_RECEIPT_SCHEMA_VERSION, "1.4.0")
        self.assertIn("preparation_repository_commit", schema["required"])
        self.assertIn("reference_environment", schema["required"])
        self.assertIn("preparation_receipt_sha256", schema["required"])
        reference_schema = schema["$defs"]["referenceEnvironment"]
        self.assertEqual(reference_schema["properties"]["schema_version"]["const"], "1.4.0")
        for field in (
            "huggingface_hub_package_file_count",
            "huggingface_hub_package_receipt_sha256",
            "hub_transport_package_provenance",
            "capture_direct_package_provenance",
            "capture_transitive_package_provenance",
        ):
            self.assertIn(field, reference_schema["required"])
        transport_schema = schema["$defs"]["hubTransportPackageProvenance"]
        for package in (
            "requests", "urllib3", "certifi", "charset-normalizer", "idna",
            "filelock", "fsspec", "packaging", "pyyaml", "tqdm", "typing-extensions",
        ):
            self.assertIn(package, transport_schema["required"])
        direct_schema = schema["$defs"]["captureDirectPackageProvenance"]
        self.assertEqual(
            set(direct_schema["required"]),
            {"torch", "transformers", "tokenizers", "safetensors"},
        )
        transitive_schema = schema["$defs"]["captureTransitivePackageProvenance"]
        for package in ("numpy", "regex", "pyyaml", "sympy", "typing-extensions"):
            self.assertIn(package, transitive_schema["required"])

    def test_receipt_binds_request_endpoint_repository_revision_complete_environment_and_package_content(self) -> None:
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
        environment = receipt["reference_environment"]
        self.assertEqual(environment["distribution_count"], 28)
        self.assertEqual(environment["python_version"], "3.11.16")
        self.assertEqual(environment["huggingface_hub_package_file_count"], 137)
        for package in ("filelock", "fsspec", "packaging", "tqdm"):
            self.assertIn(package, environment["hub_transport_package_provenance"])
        self.assertEqual(
            set(environment["capture_direct_package_provenance"]),
            {"torch", "transformers", "tokenizers", "safetensors"},
        )
        for package in ("numpy", "regex", "pyyaml"):
            self.assertIn(package, environment["capture_transitive_package_provenance"])
        self.assertRegex(environment["environment_receipt_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(receipt["preparation_receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_receipt_rejects_request_endpoint_environment_or_package_content_tampering(self) -> None:
        request = self._request()
        receipt = self._receipt(request)

        tampered_request = json.loads(json.dumps(receipt))
        tampered_request["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(CaptureContractError, "request_sha256 does not match"):
            verify_preparation_receipt(tampered_request, request=request, experiment_id="EXP-TEST-001")

        tampered_endpoint = json.loads(json.dumps(receipt))
        tampered_endpoint["hub_endpoint"] = "https://mirror.invalid"
        with self.assertRaisesRegex(CaptureContractError, "canonical Hugging Face endpoint"):
            verify_preparation_receipt(tampered_endpoint, request=request, experiment_id="EXP-TEST-001")

        tampered_environment = json.loads(json.dumps(receipt))
        tampered_environment["reference_environment"]["python_version"] = "3.12.9"
        self._rehash(tampered_environment)
        with self.assertRaisesRegex(CaptureContractError, "Python 3.11"):
            verify_preparation_receipt(tampered_environment, request=request, experiment_id="EXP-TEST-001")

        tampered_hub = json.loads(json.dumps(receipt))
        tampered_hub["reference_environment"]["huggingface_hub_package_receipt_sha256"] = "not-a-digest"
        self._rehash(tampered_hub)
        with self.assertRaisesRegex(CaptureContractError, "package provenance receipt_sha256"):
            verify_preparation_receipt(tampered_hub, request=request, experiment_id="EXP-TEST-001")

        tampered_hub_dependency = json.loads(json.dumps(receipt))
        tampered_hub_dependency["reference_environment"]["hub_transport_package_provenance"]["filelock"]["receipt_sha256"] = "not-a-digest"
        self._rehash(tampered_hub_dependency)
        with self.assertRaisesRegex(CaptureContractError, "execution dependency filelock"):
            verify_preparation_receipt(tampered_hub_dependency, request=request, experiment_id="EXP-TEST-001")

        tampered_direct = json.loads(json.dumps(receipt))
        tampered_direct["reference_environment"]["capture_direct_package_provenance"]["torch"]["receipt_sha256"] = "not-a-digest"
        self._rehash(tampered_direct)
        with self.assertRaisesRegex(CaptureContractError, "direct dependency torch"):
            verify_preparation_receipt(tampered_direct, request=request, experiment_id="EXP-TEST-001")

        tampered_transitive = json.loads(json.dumps(receipt))
        tampered_transitive["reference_environment"]["capture_transitive_package_provenance"]["numpy"]["receipt_sha256"] = "not-a-digest"
        self._rehash(tampered_transitive)
        with self.assertRaisesRegex(CaptureContractError, "transitive dependency numpy"):
            verify_preparation_receipt(tampered_transitive, request=request, experiment_id="EXP-TEST-001")


if __name__ == "__main__":
    unittest.main()

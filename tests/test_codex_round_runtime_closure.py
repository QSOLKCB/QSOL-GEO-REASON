from __future__ import annotations

import inspect
import json
import sys
import types
import unittest
from unittest import mock

from qsol_geo_reason import capture_hub_prepare_worker
from qsol_geo_reason import capture_reference_environment as reference
from qsol_geo_reason.capture_common import CaptureContractError
from reference_environment_fixture import reference_environment_receipt


class RuntimeClosureReviewRegressions(unittest.TestCase):
    def test_preparation_environment_binds_all_direct_capture_packages(self) -> None:
        receipt = reference_environment_receipt()
        self.assertEqual(reference.REFERENCE_ENVIRONMENT_SCHEMA_VERSION, "1.4.0")
        self.assertEqual(
            set(reference.CAPTURE_DIRECT_PACKAGE_IMPORTS),
            {"torch", "transformers", "tokenizers", "safetensors"},
        )
        self.assertEqual(
            set(receipt["capture_direct_package_provenance"]),
            set(reference.CAPTURE_DIRECT_PACKAGE_IMPORTS),
        )

        changed = json.loads(json.dumps(receipt))
        changed["capture_direct_package_provenance"]["torch"]["receipt_sha256"] = "f" * 64
        changed["environment_receipt_sha256"] = reference.sha256_json(
            {
                key: value
                for key, value in changed.items()
                if key != "environment_receipt_sha256"
            }
        )
        with mock.patch.object(
            reference,
            "build_reference_environment_receipt",
            return_value=changed,
        ):
            with self.assertRaisesRegex(
                CaptureContractError,
                "does not match the preparation reference environment receipt",
            ):
                reference.verify_current_reference_environment(receipt)

    def test_hub_execution_closure_includes_non_requests_dependencies(self) -> None:
        required = {
            "filelock",
            "fsspec",
            "packaging",
            "pyyaml",
            "tqdm",
            "typing-extensions",
            "requests",
            "urllib3",
            "certifi",
            "charset-normalizer",
            "idna",
        }
        self.assertEqual(set(reference.HUB_TRANSPORT_PACKAGE_IMPORTS), required)
        receipt = reference_environment_receipt()
        self.assertEqual(set(receipt["hub_transport_package_provenance"]), required)

    def test_hub_worker_measures_complete_closure_before_import(self) -> None:
        source = inspect.getsource(capture_hub_prepare_worker.main)
        self.assertLess(
            source.index("transport_before = _preimport_transport_package_provenance()"),
            source.index("import huggingface_hub"),
        )
        self.assertIn("transport_after_import != transport_before", source)
        self.assertIn("transport_after_work != transport_before", source)

        fake_filelock = types.ModuleType("filelock")
        with mock.patch.dict(sys.modules, {"filelock": fake_filelock}):
            with self.assertRaisesRegex(CaptureContractError, "absent before"):
                capture_hub_prepare_worker._preimport_package_provenance(
                    "filelock",
                    "filelock",
                    "Hugging Face Hub execution dependency filelock",
                )


if __name__ == "__main__":
    unittest.main()

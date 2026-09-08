"""Round 40 regression for validated embedded CUDA runtime provenance."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_runtime import _validate_torch_build_metadata

ROOT = Path(__file__).resolve().parents[1]
_PREFIX = "QSOL_GEO_CUDA_RUNTIME="


def _config(*, count: int = 2, receipt: str = "a" * 64) -> str:
    payload = json.dumps(
        {
            "loaded_cuda_runtime_libraries": {
                "cuda_runtime_library_file_count": count,
                "cuda_runtime_library_receipt_sha256": receipt,
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "SYNTHETIC\n" + _PREFIX + payload + "\n"


def _observed(config: str, device: str) -> dict[str, str]:
    return {
        "device": device,
        "torch_build_config": config,
        "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
    }


class CaptureRound40RegressionTests(unittest.TestCase):
    def test_cuda_build_config_requires_runtime_library_receipt(self):
        config = "SYNTHETIC\n"
        with self.assertRaisesRegex(
            CaptureContractError,
            "missing the authenticated CUDA runtime library receipt",
        ):
            _validate_torch_build_metadata(_observed(config, "cuda:0"))

    def test_valid_cuda_runtime_library_receipt_is_accepted(self):
        _validate_torch_build_metadata(_observed(_config(), "cuda:0"))

    def test_malformed_cuda_runtime_library_receipt_is_rejected(self):
        for count, receipt in (
            (0, "a" * 64),
            (2, "A" * 64),
            (2, "not-a-digest"),
        ):
            with self.subTest(count=count, receipt=receipt):
                with self.assertRaises(CaptureContractError):
                    _validate_torch_build_metadata(
                        _observed(_config(count=count, receipt=receipt), "cuda:0")
                    )

    def test_duplicate_or_nonfinal_cuda_runtime_records_are_rejected(self):
        valid = _config()
        duplicate = valid + valid.split("\n", 1)[1]
        with self.assertRaisesRegex(CaptureContractError, "exactly one CUDA runtime"):
            _validate_torch_build_metadata(_observed(duplicate, "cuda:0"))

        nonfinal = _config().rstrip("\n") + "\nAFTER=1\n"
        with self.assertRaisesRegex(CaptureContractError, "final torch_build_config line"):
            _validate_torch_build_metadata(_observed(nonfinal, "cuda:0"))

    def test_cuda_runtime_provenance_is_forbidden_outside_cuda(self):
        with self.assertRaisesRegex(CaptureContractError, "absent outside CUDA"):
            _validate_torch_build_metadata(_observed(_config(), "cpu"))

    def test_schema_requires_canonical_cuda_runtime_record(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        rules = schema["$defs"]["backendObservedProduction"]["allOf"]
        matches = []
        for rule in rules:
            device = rule.get("if", {}).get("properties", {}).get("device", {})
            if device.get("pattern") != "^cuda:[0-9]+$":
                continue
            then_config = rule.get("then", {}).get("properties", {}).get(
                "torch_build_config"
            )
            else_config = rule.get("else", {}).get("properties", {}).get(
                "torch_build_config"
            )
            if then_config is not None:
                matches.append((then_config, else_config))
        self.assertEqual(len(matches), 1)
        then_config, else_config = matches[0]
        pattern = then_config["pattern"]
        self.assertIn("QSOL_GEO_CUDA_RUNTIME=", pattern)
        self.assertIn("cuda_runtime_library_file_count", pattern)
        self.assertIn("cuda_runtime_library_receipt_sha256", pattern)
        self.assertEqual(
            else_config,
            {"not": {"pattern": "(?:^|\\n)QSOL_GEO_CUDA_RUNTIME="}},
        )


if __name__ == "__main__":
    unittest.main()

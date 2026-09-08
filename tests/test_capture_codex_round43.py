"""Round 43 regressions for source-less callables, backend overflow, and CUDA indices."""
from __future__ import annotations

import json
import re
import types
import unittest
from pathlib import Path

import qsol_geo_reason.capture_backend_core as backend_core
from qsol_geo_reason.capture_common import CaptureContractError, _validate_backend_layer
from qsol_geo_reason.capture_validation import validate_capture_request
from qsol_geo_reason.provenance import (
    SourceIdentityError,
    _assert_loaded_importable_callables_match_source,
    source_repo_root,
)
from test_capture_codex_round6 import fixture_request

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound43RegressionTests(unittest.TestCase):
    def test_source_less_loaded_qsol_callable_is_rejected(self):
        root = source_repo_root()
        namespace = {"__name__": "qsol_geo_reason.capture_backend_core"}
        exec(
            compile(
                "def _extract_hidden_tensor(value):\n    return None\n",
                "<string>",
                "exec",
            ),
            namespace,
        )
        replacement = namespace["_extract_hidden_tensor"]
        self.assertIsInstance(replacement, types.FunctionType)
        self.assertEqual(replacement.__code__.co_filename, "<string>")

        original = backend_core._extract_hidden_tensor
        try:
            backend_core._extract_hidden_tensor = replacement
            with self.assertRaisesRegex(
                SourceIdentityError,
                "no checkout-backed source filename",
            ):
                _assert_loaded_importable_callables_match_source(root)
        finally:
            backend_core._extract_hidden_tensor = original

        _assert_loaded_importable_callables_match_source(root)

    def test_oversized_backend_vector_integer_is_a_contract_error(self):
        with self.assertRaisesRegex(CaptureContractError, "binary64"):
            _validate_backend_layer(
                {
                    "vector": [10**10000],
                    "vector_dimension": 1,
                    "observed_dtype": "float32",
                },
                layer_index=0,
                expected_dimension=None,
                where="backend layer 0",
            )

    def test_cuda_device_index_is_bounded_before_production_parsing(self):
        request = fixture_request()
        request["backend"]["device"] = "cuda:" + ("9" * 4301)
        with self.assertRaisesRegex(CaptureContractError, "1-10 digit"):
            validate_capture_request(request)

        schema = json.loads(
            (ROOT / "schemas" / "capture-request.schema.json").read_text(
                encoding="utf-8"
            )
        )
        pattern = schema["$defs"]["backend"]["properties"]["device"]["oneOf"][1]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, "cuda:0"))
        self.assertIsNotNone(re.fullmatch(pattern, "cuda:" + ("9" * 10)))
        self.assertIsNone(re.fullmatch(pattern, "cuda:" + ("9" * 11)))
        self.assertIsNone(re.fullmatch(pattern, request["backend"]["device"]))


if __name__ == "__main__":
    unittest.main()

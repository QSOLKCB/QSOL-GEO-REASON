"""Round 42 regressions for UTF-8 requests, CPU pooling runtimes, and source binding."""
from __future__ import annotations

import hashlib
import inspect
import json
import types
import unittest
from pathlib import Path

import qsol_geo_reason.capture_backend_core as backend_core
from qsol_geo_reason.capture_backend_round39 import HuggingFacePyTorchBackend as Round39Backend
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_runtime import _validate_torch_build_metadata
from qsol_geo_reason.capture_validation import validate_capture_request
from qsol_geo_reason.provenance import (
    SourceIdentityError,
    _assert_loaded_importable_callables_match_source,
    source_repo_root,
)
from test_capture_codex_round6 import fixture_request

ROOT = Path(__file__).resolve().parents[1]
_CPU_PREFIX = "QSOL_GEO_CPU_RUNTIME="
_CUDA_PREFIX = "QSOL_GEO_CUDA_RUNTIME="


def _runtime_line(prefix: str, key: str, payload: dict[str, object]) -> str:
    return prefix + json.dumps({key: payload}, sort_keys=True, separators=(",", ":"))


def _cpu_line() -> str:
    return _runtime_line(
        _CPU_PREFIX,
        "loaded_cpu_runtime_libraries",
        {
            "cpu_runtime_library_file_count": 0,
            "cpu_runtime_library_receipt_sha256": "a" * 64,
        },
    )


def _cuda_line() -> str:
    return _runtime_line(
        _CUDA_PREFIX,
        "loaded_cuda_runtime_libraries",
        {
            "cuda_runtime_library_file_count": 2,
            "cuda_runtime_library_receipt_sha256": "b" * 64,
        },
    )


def _observed(device: str, lines: list[str], *, cpu_pooling: bool = True) -> dict[str, object]:
    config = "SYNTHETIC\n" + "\n".join(lines) + "\n"
    observed: dict[str, object] = {
        "device": device,
        "torch_build_config": config,
        "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
    }
    if cpu_pooling:
        observed["pool_accumulation_device"] = "cpu"
    return observed


class CaptureRound42RegressionTests(unittest.TestCase):
    def test_request_validation_rejects_lone_surrogates_before_hashing(self):
        mutators = {
            "run_id": lambda request: request.__setitem__("run_id", "\ud800"),
            "notes": lambda request: request.__setitem__("notes", "\ud800"),
            "prefix_text": lambda request: request["capture"].__setitem__("prefix_text", "\ud800"),
            "step_joiner": lambda request: request["capture"].__setitem__("step_joiner", "\ud800"),
            "step_id": lambda request: request["steps"][0].__setitem__("step_id", "\ud800"),
            "step_text": lambda request: request["steps"][0].__setitem__("text", "\ud800"),
            "generation_key": lambda request: request["generation_parameters"].__setitem__("\ud800", None),
        }
        for field, mutate in mutators.items():
            request = fixture_request()
            mutate(request)
            with self.subTest(field=field):
                with self.assertRaisesRegex(CaptureContractError, "UTF-8"):
                    validate_capture_request(request)

    def test_cpu_pooling_runtime_receipt_is_required_for_accelerator_lanes(self):
        _validate_torch_build_metadata(_observed("mps", [_cpu_line()]))
        _validate_torch_build_metadata(
            _observed("cuda:0", [_cpu_line(), _cuda_line()])
        )

        with self.assertRaisesRegex(
            CaptureContractError,
            "missing the authenticated CPU runtime library receipt",
        ):
            _validate_torch_build_metadata(_observed("mps", []))
        with self.assertRaisesRegex(
            CaptureContractError,
            "missing the authenticated CPU runtime library receipt",
        ):
            _validate_torch_build_metadata(_observed("cuda:0", [_cuda_line()]))

    def test_cpu_runtime_record_precedes_final_cuda_record(self):
        wrong = _observed("cuda:0", [_cuda_line(), _cpu_line()])
        with self.assertRaisesRegex(CaptureContractError, "final torch_build_config line"):
            _validate_torch_build_metadata(wrong)

    def test_round39_emits_cpu_runtime_for_every_production_device(self):
        source = inspect.getsource(Round39Backend.metadata)
        self.assertIn("production_device", source)
        self.assertIn("loaded_cpu_runtime_library_provenance()", source)
        self.assertIn("if production_device:", source)
        self.assertIn("if cuda_active:", source)
        self.assertLess(source.index("QSOL_GEO_CPU_RUNTIME="), source.index("QSOL_GEO_CUDA_RUNTIME="))

    def test_manifest_schema_requires_terminal_cpu_pooling_runtime_record(self):
        schema = json.loads(
            (ROOT / "schemas" / "capture-run-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        config = schema["$defs"]["backendObservedProduction"]["properties"]["torch_build_config"]
        pattern = config["pattern"]
        self.assertIn("QSOL_GEO_CPU_RUNTIME=", pattern)
        self.assertIn("cpu_runtime_library_file_count", pattern)
        self.assertIn("cpu_runtime_library_receipt_sha256", pattern)
        self.assertIn("QSOL_GEO_CUDA_RUNTIME=", pattern)

    def test_loaded_qsol_callable_must_match_clean_tracked_source(self):
        root = source_repo_root()
        source_path = root / "src" / "qsol_geo_reason" / "capture_backend_core.py"
        namespace = {"__name__": "qsol_geo_reason.capture_backend_core"}
        exec(
            compile(
                "def _extract_hidden_tensor(value):\n    return None\n",
                str(source_path),
                "exec",
            ),
            namespace,
        )
        replacement = namespace["_extract_hidden_tensor"]
        self.assertIsInstance(replacement, types.FunctionType)
        original = backend_core._extract_hidden_tensor
        try:
            backend_core._extract_hidden_tensor = replacement
            with self.assertRaisesRegex(
                SourceIdentityError,
                "does not match the clean tracked source",
            ):
                _assert_loaded_importable_callables_match_source(root)
        finally:
            backend_core._extract_hidden_tensor = original

        _assert_loaded_importable_callables_match_source(root)


if __name__ == "__main__":
    unittest.main()

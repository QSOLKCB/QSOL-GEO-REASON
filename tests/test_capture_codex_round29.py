"""Round 29 regressions for adapter dependencies, semantic schemas, exclusion, and MPS identity."""
from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import qsol_geo_reason.capture_backend_core as backend_core
from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_common import (
    CaptureContractError,
    _LAYER_INDEX_SEMANTICS,
    _STEP_SPAN_SEMANTICS,
)
from qsol_geo_reason.capture_dispatch import _validate_dispatch_metadata
from qsol_geo_reason.capture_execute import (
    _assert_observation_backend_execution_methods,
    execute_capture,
)
from test_capture_codex_round6 import fixture_request, valid_production_shape

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound29RegressionTests(unittest.TestCase):
    def test_backend_global_helper_replacement_changes_trusted_dependency_receipt(self):
        backend = object.__new__(Backend)
        _assert_observation_backend_execution_methods(backend)
        original = backend_core._extract_hidden_tensor
        try:
            backend_core._extract_hidden_tensor = lambda value: None
            with self.assertRaisesRegex(CaptureContractError, "execution dependencies"):
                _assert_observation_backend_execution_methods(backend)
        finally:
            backend_core._extract_hidden_tensor = original
        _assert_observation_backend_execution_methods(backend)

    def test_trajectory_schema_binds_canonical_representation_semantics(self):
        schema = json.loads((ROOT / "schemas/captured-trajectory.schema.json").read_text())
        properties = schema["properties"]["representation_definition"]["properties"]
        self.assertEqual(properties["layer_index_semantics"], {"const": _LAYER_INDEX_SEMANTICS})
        self.assertEqual(properties["step_span_semantics"], {"const": _STEP_SPAN_SEMANTICS})

    def test_adapter_and_routing_are_rechecked_after_exclusive_boundary_starts(self):
        source = inspect.getsource(execute_capture)
        begin = source.index("backend.begin_observation()")
        protected_try = source.rfind("\n    try:", 0, begin)
        self.assertNotEqual(protected_try, -1)
        self.assertLess(protected_try, begin)

        tail = source[begin:]
        method_recheck = tail.index("_assert_observation_backend_execution_methods(backend)")
        routing_recheck = tail.index("_assert_observation_backend_routing_state(backend)")
        capture = tail.index("steps, prefix_ids = _capture_steps(")
        active_guard = tail.index('getattr(backend, "_observation_active", False)')
        cleanup = tail.index("backend.end_observation()")
        self.assertLess(method_recheck, capture)
        self.assertLess(routing_recheck, capture)
        self.assertGreater(active_guard, capture)
        self.assertLess(active_guard, cleanup)
        self.assertGreater(cleanup, method_recheck)
        self.assertGreater(cleanup, routing_recheck)

    def test_mps_requires_concrete_hardware_identity_in_runtime_and_schema(self):
        request = fixture_request()
        request["backend"]["device"] = "mps"
        observed = valid_production_shape(request)
        observed["mps_mac_model"] = None
        observed["mps_cpu_brand"] = None
        with self.assertRaisesRegex(CaptureContractError, "MPS provenance requires"):
            _validate_dispatch_metadata(observed, "mps")

        for field, value in (("mps_mac_model", "Mac15,7"), ("mps_cpu_brand", "Apple M3 Pro")):
            candidate = dict(observed)
            candidate[field] = value
            with self.subTest(field=field):
                _validate_dispatch_metadata(candidate, "mps")

        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        mps_rule = schema["$defs"]["backendObservedProduction"]["allOf"][0]
        self.assertEqual(mps_rule["if"]["properties"]["device"]["const"], "mps")
        alternatives = mps_rule["then"]["anyOf"]
        fields = {
            next(iter(alternative["properties"])): alternative
            for alternative in alternatives
        }
        for field in ("mps_mac_model", "mps_cpu_brand"):
            constraint = fields[field]["properties"][field]
            self.assertEqual(constraint["type"], "string")
            self.assertEqual(constraint["pattern"], r"\S")
            self.assertIn(field, fields[field]["required"])


if __name__ == "__main__":
    unittest.main()

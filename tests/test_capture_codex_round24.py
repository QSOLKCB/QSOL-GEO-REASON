"""Behavioral regressions for execution ownership and numerical provenance.

All model/runtime stand-ins and forged bundles below are software fixtures, not
empirical observations. The Python thread tests use actual interpreter workers.
"""
from __future__ import annotations

import _thread
import copy
import functools
import inspect
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_backend_isolated import HuggingFacePyTorchBackend as IsolatedBackend
from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend as CoreBackend
from qsol_geo_reason.capture_common import (
    _ALLOWED_OBSERVED_DTYPES, _PRODUCTION_BACKEND_KEYS,
    _require_observed_dtype, _validate_backend_layer,
)
from qsol_geo_reason.capture_execution_state import (
    _assert_exclusive_interpreter_thread, _callable_execution_identity,
    _cudnn_algorithm_policy_state,
)
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from qsol_geo_reason.capture_verify import verify_capture_bundle
from test_capture_codex_round6 import fixture_request, valid_production_shape
from test_capture_codex_round11 import execute_simulation, production_observed, rebind_bundle
from test_capture_codex_round20 import FakePolicyTorch

ROOT = Path(__file__).resolve().parents[1]


class HelperAttention:
    def _attn(self, value):
        return value

    def forward(self, value=None):
        return self._attn(value)


class HelperModel:
    def __init__(self):
        self.attention = HelperAttention()

    def forward(self, value=None):
        return self.attention.forward(value)

    def named_modules(self):
        return [("", self), ("attention", self.attention)]


def policy_torch():
    torch = FakePolicyTorch()
    torch.backends.cpu = SimpleNamespace(get_cpu_capability=lambda: "DEFAULT")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    return torch


class CaptureRound24RegressionTests(unittest.TestCase):
    def assert_starts_blocked(self):
        with self.assertRaisesRegex(CaptureContractError, "starting another Python thread"):
            threading.Thread(target=lambda: None).start()

    def test_constructor_rejects_concurrent_thread_before_snapshot_or_loading(self):
        started, release = threading.Event(), threading.Event()

        def worker():
            started.set()
            release.wait(timeout=5)

        worker_thread = threading.Thread(target=worker)
        original = threading.Thread.start
        worker_thread.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            with (
                patch.object(HuggingFacePyTorchBackend, "_snapshot_torch_process_state") as snapshot,
                patch.object(IsolatedBackend, "__init__", return_value=None) as load,
            ):
                with self.assertRaisesRegex(CaptureContractError, "exclusive Python-thread execution"):
                    HuggingFacePyTorchBackend(fixture_request())
                snapshot.assert_not_called()
                load.assert_not_called()
            self.assertIs(threading.Thread.start, original)
        finally:
            release.set()
            worker_thread.join(timeout=2)
        self.assertFalse(worker_thread.is_alive())

    def test_constructor_keeps_exclusion_through_snapshot_load_and_restoration(self):
        for failure in (None, RuntimeError("load failure"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                torch = policy_torch()
                snapshot = HuggingFacePyTorchBackend._snapshot_torch_process_state
                restore = HuggingFacePyTorchBackend._restore_torch_process_state
                before = snapshot(torch, "cpu")
                calls = []
                original_start = threading.Thread.start

                def take_snapshot(module, device):
                    self.assert_starts_blocked()
                    calls.append("snapshot")
                    return snapshot(module, device)

                def load(backend, request):
                    self.assert_starts_blocked()
                    calls.append("load")
                    backend._torch = torch
                    backend._attention_implementation = "eager"
                    torch.cpu_rng = b"changed by constructor"
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    if failure is not None:
                        raise failure

                def restore_state(module, state):
                    self.assert_starts_blocked()
                    calls.append("restore")
                    restore(module, state)

                with (
                    patch.dict(sys.modules, {"torch": torch}),
                    patch.object(IsolatedBackend, "__init__", new=load),
                    patch.object(HuggingFacePyTorchBackend, "_snapshot_torch_process_state", side_effect=take_snapshot),
                    patch.object(HuggingFacePyTorchBackend, "_restore_torch_process_state", side_effect=restore_state),
                    patch.object(HuggingFacePyTorchBackend, "_model_runtime_attributes_seal", return_value="fixture"),
                ):
                    if failure is None:
                        HuggingFacePyTorchBackend(fixture_request())
                    else:
                        with self.assertRaises(type(failure)):
                            HuggingFacePyTorchBackend(fixture_request())
                self.assertEqual(calls, ["snapshot", "load", "restore"])
                self.assertEqual(snapshot(torch, "cpu"), before)
                self.assertIs(threading.Thread.start, original_start)

    def test_constructor_releases_boundary_when_snapshot_or_restore_fails(self):
        for method in ("_snapshot_torch_process_state", "_restore_torch_process_state"):
            with self.subTest(method=method):
                original = threading.Thread.start
                with (
                    patch.dict(sys.modules, {"torch": policy_torch()}),
                    patch.object(IsolatedBackend, "__init__", return_value=None),
                    patch.object(HuggingFacePyTorchBackend, method, side_effect=RuntimeError("fixture failure")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                        HuggingFacePyTorchBackend(fixture_request())
                self.assertIs(threading.Thread.start, original)

    def test_raw_thread_absent_from_threading_registry_blocks_both_boundaries(self):
        started, release = threading.Event(), threading.Event()
        original_starts = (_thread.start_new_thread, _thread.start_new, threading.Thread.start)
        baseline_count = _thread._count()

        def raw_worker():
            # Do not call threading.current_thread(): this worker stays unregistered.
            started.set()
            release.wait(timeout=5)

        ident = _thread.start_new_thread(raw_worker, ())
        self.assertTrue(started.wait(timeout=2))
        try:
            self.assertNotIn(ident, threading._active)
            self.assertIn(ident, sys._current_frames())
            backend = object.__new__(HuggingFacePyTorchBackend)
            with patch.object(IsolatedBackend, "begin_observation") as begin:
                with self.assertRaisesRegex(CaptureContractError, "exclusive Python-thread execution"):
                    backend.begin_observation()
                begin.assert_not_called()
            with patch.object(HuggingFacePyTorchBackend, "_snapshot_torch_process_state") as snapshot:
                with self.assertRaisesRegex(CaptureContractError, "exclusive Python-thread execution"):
                    HuggingFacePyTorchBackend(fixture_request())
                snapshot.assert_not_called()
            self.assertEqual((_thread.start_new_thread, _thread.start_new, threading.Thread.start), original_starts)
        finally:
            release.set()
            deadline = time.monotonic() + 2
            while _thread._count() != baseline_count and time.monotonic() < deadline:
                time.sleep(0.001)
        self.assertEqual(_thread._count(), baseline_count)
        _assert_exclusive_interpreter_thread()

    def test_thread_census_fails_closed_and_restores_start_entry_points(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        original = _thread.start_new_thread
        with patch.object(sys, "_current_frames", side_effect=RuntimeError("census failed")):
            with self.assertRaisesRegex(CaptureContractError, "inspect active CPython"):
                backend._enter_exclusive_python_thread_boundary()
        self.assertIs(_thread.start_new_thread, original)
        with patch.object(_thread, "_count", return_value=1):
            with self.assertRaisesRegex(CaptureContractError, "exclusive Python-thread execution"):
                backend._enter_exclusive_python_thread_boundary()
        self.assertIs(_thread.start_new_thread, original)

    def test_callable_helper_override_changes_runtime_seal_without_changing_forward(self):
        for existing_override in (False, True):
            with self.subTest(existing_override=existing_override):
                backend = object.__new__(HuggingFacePyTorchBackend)
                backend._model = HelperModel()
                backend._torch = SimpleNamespace(is_tensor=lambda value: False)
                backend._canonical_torch_module = backend._torch
                if existing_override:
                    backend._model.attention._attn = lambda value: value
                forward_seal = backend._model_executable_state_seal()
                backend._canonical_model_runtime_attributes = backend._model_runtime_attributes_seal()
                backend._assert_model_runtime_attributes()
                backend._model.attention._attn = lambda value: ("changed", value)
                self.assertNotEqual(backend._model_executable_state_seal(), forward_seal)
                with self.assertRaisesRegex(CaptureContractError, "runtime execution attributes changed"):
                    backend._assert_model_runtime_attributes()
                with patch.object(IsolatedBackend, "begin_observation") as begin:
                    with self.assertRaisesRegex(CaptureContractError, "runtime execution attributes changed"):
                        backend.begin_observation()
                    begin.assert_not_called()

    def test_callable_identity_covers_code_constants_bound_methods_and_partials(self):
        def helper(value=1):
            return value + 1

        def changed(value=1):
            return value + 2

        receipt = _callable_execution_identity(helper)
        helper.__code__ = changed.__code__
        self.assertNotEqual(_callable_execution_identity(helper), receipt)
        instance = HelperAttention()
        self.assertEqual(_callable_execution_identity(instance._attn), _callable_execution_identity(instance._attn))
        partial = functools.partial(helper, value=2)
        receipt = _callable_execution_identity(partial)
        partial.keywords["value"] = 3
        self.assertNotEqual(_callable_execution_identity(partial), receipt)

    def test_cudnn_flags_are_restored_with_complete_process_state(self):
        torch = policy_torch()
        before = HuggingFacePyTorchBackend._snapshot_torch_process_state(torch, "cpu")
        self.assertIs(before["execution_policies"]["cudnn_benchmark"], True)
        self.assertIs(before["execution_policies"]["cudnn_deterministic"], False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        HuggingFacePyTorchBackend._restore_torch_process_state(torch, before)
        self.assertEqual(HuggingFacePyTorchBackend._snapshot_torch_process_state(torch, "cpu"), before)

    def test_best_effort_cudnn_policy_is_forced_verified_and_recorded(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = policy_torch()
        backend._determinism_mode = "best_effort"
        backend._canonical_cuda_float32_policy = backend._cuda_float32_policy_state()
        backend._canonical_cudnn_algorithm_policy = _cudnn_algorithm_policy_state(backend._torch)
        backend._torch.backends.cudnn.benchmark = False
        backend._torch.backends.cudnn.deterministic = True
        backend._force_cuda_float32_policy()
        self.assertEqual(_cudnn_algorithm_policy_state(backend._torch), backend._canonical_cudnn_algorithm_policy)
        with patch.object(IsolatedBackend, "metadata", return_value={"inherited": "preserved"}):
            observed = backend.metadata()
            self.assertEqual(observed, {"inherited": "preserved", "cudnn_benchmark": True, "cudnn_deterministic": False,
                                        "cuda_matmul_allow_fp16_accumulation": None, "cpu_aten_capability": None,
                                        "aten_cpu_capability_env": None, "aten_cpu_capability_env_known": None})
            backend._torch.backends.cudnn.deterministic = True
            with self.assertRaisesRegex(CaptureContractError, "cuDNN.*drifted"):
                backend.metadata()
        with self.assertRaisesRegex(CaptureContractError, "cuDNN.*drifted"):
            backend._assert_cuda_float32_policy()
        source = inspect.getsource(CoreBackend.hidden_states)
        forward = source.index("self._base_model(")
        self.assertLess(source.index("self._force_cuda_float32_policy()"), forward)
        self.assertGreater(source.index("self._assert_cuda_float32_policy()"), forward)

    def test_cudnn_provenance_requires_boolean_flags_on_cuda_and_null_elsewhere(self):
        request = fixture_request()
        request["backend"]["device"] = "cuda:0"
        request["determinism"]["mode"] = "best_effort"
        observed = valid_production_shape(request)
        observed.update(cuda_device_name="SYNTHETIC GPU", cuda_device_capability="8.0", cuda_resolved_device_index=0,
                        float32_matmul_precision="highest", cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
                        cuda_matmul_allow_fp16_reduced_precision_reduction=False,
                        cuda_matmul_allow_bf16_reduced_precision_reduction=False)
        _validate_production_metadata_shape(observed, request)
        for field in ("cudnn_benchmark", "cudnn_deterministic"):
            for value in (None, 0, 1, "false"):
                with self.subTest(field=field, value=value):
                    mutated = dict(observed, **{field: value})
                    with self.assertRaisesRegex(CaptureContractError, field):
                        _validate_production_metadata_shape(mutated, request)
            cpu_request = fixture_request()
            cpu_observed = valid_production_shape(cpu_request)
            cpu_observed[field] = False
            with self.assertRaisesRegex(CaptureContractError, "null outside CUDA"):
                _validate_production_metadata_shape(cpu_observed, cpu_request)

    def test_floating_dtype_domain_applies_to_producer_records(self):
        for dtype in _ALLOWED_OBSERVED_DTYPES:
            self.assertEqual(_require_observed_dtype(dtype, "fixture dtype"), dtype)
        for invalid in ("not-a-torch-dtype", "torch.float32", "double", "int64", "complex64", "float8_e4m3fn", " ", None, True, []):
            with self.subTest(invalid=invalid):
                with self.assertRaises(CaptureContractError):
                    _validate_backend_layer({"vector": [1.0], "vector_dimension": 1, "observed_dtype": invalid},
                                            layer_index=0, expected_dimension=None, where="fixture layer")

    def test_consistently_rehashed_fabricated_dtype_bundle_is_rejected(self):
        request, original_manifest, original_trajectory = execute_simulation()
        original_manifest["backend_observed"] = production_observed(request)
        original_trajectory["evidence_class"] = "OBSERVATION"
        rebind_bundle(original_manifest, original_trajectory)
        verify_capture_bundle(request, original_manifest, original_trajectory)
        for dtype in (*sorted(_ALLOWED_OBSERVED_DTYPES), "not-a-torch-dtype", "int64"):
            manifest, trajectory = copy.deepcopy(original_manifest), copy.deepcopy(original_trajectory)
            for step in trajectory["steps"]:
                for layer in step["layers"]:
                    layer["observed_dtype"] = dtype
            manifest["backend_observed"]["observed_hidden_state_dtypes"] = {
                str(layer): [dtype] for layer in request["capture"]["layers"]
            }
            rebind_bundle(manifest, trajectory)
            with self.subTest(dtype=dtype):
                if dtype in _ALLOWED_OBSERVED_DTYPES:
                    verify_capture_bundle(request, manifest, trajectory)
                else:
                    with self.assertRaisesRegex(CaptureContractError, "supported floating dtype"):
                        verify_capture_bundle(request, manifest, trajectory)

    def test_dtype_map_rejects_unhashable_values_as_contract_errors(self):
        request = fixture_request()
        for bad in ([], {}, ["float32"]):
            observed = valid_production_shape(request)
            observed["observed_hidden_state_dtypes"][str(request["capture"]["layers"][0])] = [bad]
            with self.subTest(bad=bad):
                with self.assertRaises(CaptureContractError):
                    _validate_production_metadata_shape(observed, request)

    def test_schemas_match_dtype_and_cudnn_domains(self):
        manifest_schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        trajectory_schema = json.loads((ROOT / "schemas/captured-trajectory.schema.json").read_text())
        production = manifest_schema["$defs"]["backendObservedProduction"]
        dtype_item = production["properties"]["observed_hidden_state_dtypes"]["additionalProperties"]["items"]
        self.assertEqual(set(dtype_item["enum"]), _ALLOWED_OBSERVED_DTYPES)
        self.assertEqual(set(trajectory_schema["$defs"]["layer"]["properties"]["observed_dtype"]["enum"]), _ALLOWED_OBSERVED_DTYPES)
        for field in ("cudnn_benchmark", "cudnn_deterministic"):
            self.assertIn(field, production["required"])
            self.assertIn(field, _PRODUCTION_BACKEND_KEYS)
            conditional = next(rule for rule in production["allOf"] if field in rule["then"].get("properties", {}))
            self.assertEqual(conditional["then"]["properties"][field], {"type": "boolean"})
            self.assertEqual(conditional["else"]["properties"][field], {"type": "null"})


if __name__ == "__main__":
    unittest.main()
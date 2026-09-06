"""Round 25 dispatch and callable-dependency regressions, with software fixtures only."""
from __future__ import annotations

import builtins
import copy
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_backend_isolated import HuggingFacePyTorchBackend as IsolatedBackend
from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend as CoreBackend
from qsol_geo_reason.capture_common import _PRODUCTION_BACKEND_KEYS
from qsol_geo_reason.capture_dispatch import (
    _MathSDPABaseModel, _effective_cpu_capability, _execution_dependencies_sha256,
    _fp16_accumulation_state, _model_execution_dependency_roots,
    _validate_dispatch_metadata,
)
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape
from qsol_geo_reason.capture_verify import verify_capture_bundle
from test_capture_codex_round6 import fixture_request, valid_production_shape
from test_capture_codex_round11 import execute_simulation, production_observed, rebind_bundle
from test_capture_codex_round24 import policy_torch

ROOT = Path(__file__).resolve().parents[1]


def graph_model(namespace):
    class Model:
        forward = namespace["forward"]

        def named_modules(self):
            return [("", self)]
    return Model()


class CaptureRound25RegressionTests(unittest.TestCase):
    def test_fp16_accumulation_is_forced_verified_and_restored(self):
        torch = policy_torch()
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        backend = object.__new__(Backend)
        backend._torch = torch
        backend._canonical_fp16_accumulation_supported = True
        before = Backend._snapshot_torch_process_state(torch, "cpu")
        self.assertIs(before["execution_policies"]["cuda_matmul_allow_fp16_accumulation"], True)
        backend._force_cuda_reduced_precision_policy()
        self.assertIs(torch.backends.cuda.matmul.allow_fp16_accumulation, False)
        self.assertIs(backend._last_fp16_accumulation, False)
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        with self.assertRaisesRegex(CaptureContractError, "FP16 accumulation"):
            backend._assert_cuda_reduced_precision_policy()
        Backend._restore_torch_process_state(torch, before)
        self.assertEqual(Backend._snapshot_torch_process_state(torch, "cpu"), before)
        source = inspect.getsource(CoreBackend.hidden_states)
        forward = source.index("self._base_model(")
        self.assertLess(source.index("self._force_cuda_reduced_precision_policy()"), forward)
        self.assertGreater(source.index("self._assert_cuda_reduced_precision_policy()"), forward)

    def test_fp16_accumulation_optional_api_is_not_confused_with_invalid_values(self):
        torch = policy_torch()
        self.assertIsNone(_fp16_accumulation_state(torch))
        backend = object.__new__(Backend)
        backend._torch = torch
        backend._canonical_fp16_accumulation_supported = False
        backend._force_cuda_reduced_precision_policy()
        for value in (None, 0, 1, "false"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = value
            with self.subTest(value=value):
                with self.assertRaises(CaptureContractError):
                    _fp16_accumulation_state(torch)
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        backend._canonical_fp16_accumulation_supported = True
        del torch.backends.cuda.matmul.allow_fp16_accumulation
        with self.assertRaisesRegex(CaptureContractError, "availability changed"):
            backend._assert_fp16_accumulation_policy()

    def test_fp16_accumulation_metadata_records_the_enforced_value(self):
        backend = object.__new__(Backend)
        backend._torch = policy_torch()
        backend._torch.backends.cuda.matmul.allow_fp16_accumulation = True
        backend._device_type = "cuda"
        backend._canonical_fp16_accumulation_supported = True
        backend._force_cuda_reduced_precision_policy()
        with patch.object(IsolatedBackend, "metadata", return_value={}):
            self.assertIs(backend.metadata()["cuda_matmul_allow_fp16_accumulation"], False)
            backend._torch.backends.cuda.matmul.allow_fp16_accumulation = True
            with self.assertRaises(CaptureContractError):
                backend.metadata()

    def test_module_global_helper_replacement_invalidates_unchanged_forward(self):
        namespace = {}
        exec("def helper(x): return x + 1\ndef forward(self, x): return helper(x)", namespace)
        backend = object.__new__(Backend)
        backend._model = graph_model(namespace)
        old_forward = backend._model.forward.__func__
        old_code = old_forward.__code__
        old_graph = IsolatedBackend._model_executable_state_seal(backend)
        backend._canonical_model_executable_state = backend._model_executable_state_seal()
        backend._assert_live_state_authentication()
        namespace["helper"] = lambda x: x + 2
        self.assertIs(backend._model.forward.__func__, old_forward)
        self.assertIs(old_forward.__code__, old_code)
        self.assertEqual(IsolatedBackend._model_executable_state_seal(backend), old_graph)
        with self.assertRaisesRegex(CaptureContractError, "executable graph changed"):
            backend._assert_live_state_authentication()

    def test_qualified_and_transitive_global_dependencies_are_authenticated(self):
        api = ModuleType("synthetic_activation_api")
        api.activation = lambda x: x
        namespace = {"api": api}
        exec("def helper(x): return api.activation(x)\ndef forward(self, x): return helper(x)", namespace)
        model = graph_model(namespace)
        roots = _model_execution_dependency_roots(model)
        receipt = _execution_dependencies_sha256(roots)
        self.assertEqual(_execution_dependencies_sha256(roots), receipt)
        api.activation = lambda x: -x
        self.assertNotEqual(_execution_dependencies_sha256(roots), receipt)

    def test_globals_of_self_resolved_helpers_are_authenticated(self):
        namespace = {}
        exec("def activation(x): return x\ndef helper(self, x): return activation(x)\ndef forward(self, x): return self.helper(x)", namespace)
        model = graph_model(namespace)
        type(model).helper = namespace["helper"]
        roots = _model_execution_dependency_roots(model)
        before = _execution_dependencies_sha256(roots)
        namespace["activation"] = lambda x: -x
        self.assertNotEqual(_execution_dependencies_sha256(roots), before)

    def test_dependency_cycles_terminate_and_unreferenced_globals_do_not_change_receipts(self):
        namespace = {}
        exec("def left(x): return right(x)\ndef right(x): return left(x)", namespace)
        before = _execution_dependencies_sha256([namespace["left"]])
        namespace["unrelated"] = lambda x: None
        self.assertEqual(_execution_dependencies_sha256([namespace["left"]]), before)
        namespace["right"] = lambda x: x
        self.assertNotEqual(_execution_dependencies_sha256([namespace["left"]]), before)

    def test_effective_cpu_isa_is_mandatory_and_not_inferred_from_environment(self):
        torch = policy_torch()
        with patch.dict(os.environ, {"ATEN_CPU_CAPABILITY": "avx512"}):
            self.assertEqual(_effective_cpu_capability(torch), "DEFAULT")
        for value in (None, "", "   ", "fabricated-isa", 3, True):
            torch.backends.cpu.get_cpu_capability = lambda value=value: value
            with self.subTest(value=value):
                with self.assertRaises(CaptureContractError):
                    _effective_cpu_capability(torch)
        with self.assertRaises(CaptureContractError):
            _effective_cpu_capability(SimpleNamespace())
        with patch.object(torch.backends.cpu, "get_cpu_capability", side_effect=RuntimeError("probe")):
            with self.assertRaisesRegex(CaptureContractError, "establish effective"):
                _effective_cpu_capability(torch)

    def test_cpu_dispatch_and_override_drift_are_rejected(self):
        backend = object.__new__(Backend)
        backend._torch = policy_torch()
        backend._canonical_cpu_dispatch = {"cpu_aten_capability": "DEFAULT"}
        backend._canonical_cpu_environment = None
        with patch.dict(os.environ, {}, clear=True):
            backend._assert_cpu_dispatch_policy()
            with patch.object(backend._torch.backends.cpu, "get_cpu_capability", return_value="AVX512"):
                with self.assertRaisesRegex(CaptureContractError, "dispatch policy drifted"):
                    backend._assert_cpu_dispatch_policy()
            os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
            with self.assertRaisesRegex(CaptureContractError, "override drifted"):
                backend._assert_cpu_dispatch_policy()

    def test_cpu_constructor_distinguishes_known_and_unknown_initialization_override(self):
        real_import = builtins.__import__
        for preimported in (False, True):
            torch = policy_torch()

            def import_fixture(name, *args, **kwargs):
                return torch if name == "torch" else real_import(name, *args, **kwargs)

            def load(backend, request):
                backend._torch = torch
                backend._device_type = "cpu"
                backend._attention_implementation = "eager"

            with self.subTest(preimported=preimported):
                with (
                    patch.dict(os.environ, {"ATEN_CPU_CAPABILITY": "default"}),
                    patch.dict(sys.modules),
                    patch("builtins.__import__", side_effect=import_fixture),
                    patch.object(IsolatedBackend, "__init__", new=load),
                    patch.object(Backend, "_model_runtime_attributes_seal", return_value="fixture"),
                ):
                    if preimported:
                        sys.modules["torch"] = torch
                    else:
                        sys.modules.pop("torch", None)
                    backend = Backend(fixture_request())
                    expected = {
                        "cpu_aten_capability": "DEFAULT",
                        "aten_cpu_capability_env": None if preimported else "default",
                        "aten_cpu_capability_env_known": not preimported,
                    }
                    self.assertEqual(backend._canonical_cpu_dispatch, expected)
                    with patch.object(IsolatedBackend, "metadata", return_value={}):
                        observed = backend.metadata()
                    for key, value in expected.items():
                        self.assertEqual(observed[key], value)

    def test_cpu_and_mps_sdpa_are_math_only_at_the_actual_model_call(self):
        for device in ("cpu", "mps"):
            with self.subTest(device=device):
                backend = object.__new__(Backend)
                backend._torch = policy_torch()
                backend._device_type = device
                before = Backend._snapshot_torch_process_state(backend._torch, "cpu")
                calls = []

                def forward(**kwargs):
                    state = backend._assert_sdpa_math_policy()
                    calls.append(state)
                    return "captured"

                proxy = _MathSDPABaseModel(forward, backend)
                try:
                    self.assertEqual(proxy(input_ids=[1]), "captured")
                    self.assertEqual(len(calls), 1)
                    self.assertIs(backend._last_sdpa_policy["math"], True)
                    self.assertIs(backend._last_sdpa_policy["flash"], False)
                    self.assertIs(backend._last_sdpa_policy["fp16_bf16_math_reduction"], False)
                finally:
                    Backend._restore_torch_process_state(backend._torch, before)
                self.assertEqual(Backend._snapshot_torch_process_state(backend._torch, "cpu"), before)

    def test_non_cuda_sdpa_drift_and_exception_do_not_escape_restoration(self):
        for error in (False, True):
            backend = object.__new__(Backend)
            backend._torch = policy_torch()
            before = Backend._snapshot_torch_process_state(backend._torch, "cpu")

            def forward():
                if error:
                    raise RuntimeError("synthetic forward failure")
                backend._torch.backends.cuda.enable_flash_sdp(True)

            try:
                with self.assertRaises(RuntimeError if error else CaptureContractError):
                    _MathSDPABaseModel(forward, backend)()
            finally:
                Backend._restore_torch_process_state(backend._torch, before)
            self.assertEqual(Backend._snapshot_torch_process_state(backend._torch, "cpu"), before)
        source = inspect.getsource(Backend.__init__)
        self.assertIn('_MathSDPABaseModel(self._base_model, self)', source)

    def test_non_cuda_sdpa_metadata_requires_the_enforced_math_flags(self):
        for device in ("cpu", "mps"):
            request = fixture_request()
            request["backend"]["device"] = device
            observed = valid_production_shape(request)
            observed.update(attention_implementation="sdpa", sdpa_flash_enabled=False,
                            sdpa_mem_efficient_enabled=False, sdpa_math_enabled=True,
                            sdpa_cudnn_enabled=False)
            _validate_production_metadata_shape(observed, request)
            for field in ("sdpa_flash_enabled", "sdpa_mem_efficient_enabled", "sdpa_math_enabled"):
                with self.subTest(device=device, field=field):
                    modified = dict(observed, **{field: None})
                    with self.assertRaises(CaptureContractError):
                        _validate_production_metadata_shape(modified, request)

    def test_rehashed_bundles_cannot_erase_cpu_dispatch_or_enable_fp16_accumulation(self):
        request, manifest, trajectory = execute_simulation()
        manifest["backend_observed"] = production_observed(request)
        trajectory["evidence_class"] = "OBSERVATION"
        rebind_bundle(manifest, trajectory)
        verify_capture_bundle(request, manifest, trajectory)
        for field, value in (("cpu_aten_capability", None), ("cpu_aten_capability", " "),
                             ("aten_cpu_capability_env_known", 0),
                             ("aten_cpu_capability_env", "avx2"),
                             ("cuda_matmul_allow_fp16_accumulation", True)):
            changed_m, changed_t = copy.deepcopy(manifest), copy.deepcopy(trajectory)
            changed_m["backend_observed"][field] = value
            rebind_bundle(changed_m, changed_t)
            with self.subTest(field=field, value=value):
                with self.assertRaises(CaptureContractError):
                    verify_capture_bundle(request, changed_m, changed_t)
        before = sha256_json(manifest["backend_observed"])
        manifest["backend_observed"]["cpu_aten_capability"] = "AVX512"
        self.assertNotEqual(sha256_json(manifest["backend_observed"]), before)

    def test_dispatch_schema_and_runtime_have_matching_fields_and_domains(self):
        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        production = schema["$defs"]["backendObservedProduction"]
        self.assertEqual(set(production["required"]), _PRODUCTION_BACKEND_KEYS)
        for field in ("cpu_aten_capability", "aten_cpu_capability_env", "aten_cpu_capability_env_known", "cuda_matmul_allow_fp16_accumulation"):
            self.assertIn(field, production["properties"])
        self.assertEqual(production["properties"]["cuda_matmul_allow_fp16_accumulation"]["enum"], [False, None])
        math_rule = next(rule for rule in production["allOf"]
                         if rule["if"].get("properties", {}).get("attention_implementation", {}).get("const") == "sdpa")
        self.assertNotIn("device", math_rule["if"]["properties"])
        self.assertIs(math_rule["then"]["properties"]["sdpa_math_enabled"]["const"], True)
        for value in (True, 0, 1, "false"):
            with self.assertRaises(CaptureContractError):
                _validate_dispatch_metadata({"cuda_matmul_allow_fp16_accumulation": value}, "cuda:0")


if __name__ == "__main__":
    unittest.main()

"""Round 31 software regressions; no empirical model capture is claimed."""
from __future__ import annotations

import importlib.util
import inspect
import marshal
import py_compile
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsol_geo_reason import capture_package
from qsol_geo_reason.canonical import sha256_json
from qsol_geo_reason.capture import execute_capture, verify_capture_bundle, write_capture_bundle
from qsol_geo_reason.capture_backend_isolated import HuggingFacePyTorchBackend as IsolatedBackend
from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_package import _python_package_provenance
from test_capture_codex_round11 import (
    REV, FakeBackend, execute_simulation, fixture_request, rebind_bundle,
)
from test_capture_codex_round24 import policy_torch


class TokenRewritingProxy:
    def __init__(self, module):
        self.module = module
        self.tensor_calls = 0
        self.delegations = 0

    def __getattr__(self, name):
        self.delegations += 1
        return getattr(self.module, name)

    def tensor(self, rows, **kwargs):
        self.tensor_calls += 1
        return self.module.tensor([[token + 1 for token in row] for row in rows], **kwargs)


def package_fixture(directory):
    package = Path(directory) / "transformers"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text('__version__ = "fixture-same-version"\n', encoding="utf-8")
    model = package / "modeling_fixture.py"
    model.write_text(
        'def forward(x):\n    """Synthetic forward."""\n    assert x is not None\n    return x + 1\n',
        encoding="utf-8",
    )
    return SimpleNamespace(__file__=str(init)), model


def bound_runtime_fixture():
    backend = object.__new__(Backend)
    runtime = SimpleNamespace(tensor=lambda rows, **kwargs: rows)
    backend._torch = runtime
    backend._canonical_torch_module = runtime
    return backend, runtime


class CaptureRound31RegressionTests(unittest.TestCase):
    def test_constructor_binds_the_runtime_it_imported(self):
        runtime = policy_torch()

        def load(backend, request):
            backend._torch = runtime
            backend._attention_implementation = "eager"

        with (
            patch.dict(sys.modules, {"torch": runtime, "transformers": None}),
            patch.object(IsolatedBackend, "__init__", new=load),
            patch.object(Backend, "_model_runtime_attributes_seal", return_value="fixture"),
        ):
            backend = Backend(fixture_request())
        self.assertIs(backend._canonical_torch_module, runtime)
        backend._assert_torch_runtime_identity()

    def test_replacement_runtime_fails_before_request_authentication(self):
        backend, runtime = bound_runtime_fixture()
        proxy = TokenRewritingProxy(runtime)
        self.assertEqual(proxy.tensor([[1, 2]]), [[2, 3]])
        proxy.tensor_calls = 0
        backend._torch = proxy
        with patch.object(IsolatedBackend, "assert_execution_request") as inherited:
            with self.assertRaisesRegex(CaptureContractError, "PyTorch runtime object"):
                backend.assert_execution_request(fixture_request())
            inherited.assert_not_called()
        self.assertEqual(proxy.tensor_calls, 0)
        self.assertEqual(proxy.delegations, 0)
        backend._torch = runtime
        backend._assert_torch_runtime_identity()

    def test_runtime_replacement_in_boundary_gap_is_rejected_before_startup(self):
        backend, runtime = bound_runtime_fixture()
        backend._assert_torch_runtime_identity()
        original_enter = Backend._enter_exclusive_python_thread_boundary
        original_start = threading.Thread.start
        proxy = TokenRewritingProxy(runtime)

        def enter_and_replace(instance):
            original_enter(instance)
            instance._torch = proxy

        with (
            patch.object(Backend, "_enter_exclusive_python_thread_boundary", new=enter_and_replace),
            patch.object(IsolatedBackend, "begin_observation") as inherited,
            patch.object(Backend, "_assert_no_active_torch_override_modes") as modes,
        ):
            with self.assertRaisesRegex(CaptureContractError, "PyTorch runtime object"):
                backend.begin_observation()
            inherited.assert_not_called()
            modes.assert_not_called()
        self.assertIs(threading.Thread.start, original_start)
        self.assertIsNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(proxy.tensor_calls, 0)
        self.assertEqual(proxy.delegations, 0)

    def test_missing_runtime_binding_fails_closed(self):
        backend = object.__new__(Backend)
        for runtime in (None, SimpleNamespace()):
            backend._torch = runtime
            with self.subTest(runtime=runtime):
                with self.assertRaisesRegex(CaptureContractError, "construction binding"):
                    backend._assert_torch_runtime_identity()
        source = inspect.getsource(Backend.begin_observation)
        self.assertLess(source.index("self._enter_exclusive_python_thread_boundary()"),
                        source.index("self._assert_torch_runtime_identity()"))
        self.assertLess(source.index("self._assert_torch_runtime_identity()"),
                        source.index("super().begin_observation()"))

    def test_valid_caches_preserve_receipt_in_all_invalidation_and_optimization_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            module, model = package_fixture(directory)
            expected = _python_package_provenance(module, "Transformers")
            for mode in py_compile.PycInvalidationMode:
                for optimization in (0, 1, 2):
                    with self.subTest(mode=mode.name, optimization=optimization):
                        py_compile.compile(
                            str(model), doraise=True, optimize=optimization,
                            invalidation_mode=mode,
                            dfile="/previous/wheel/location/transformers/modeling_fixture.py",
                        )
                        self.assertEqual(_python_package_provenance(module, "Transformers"), expected)
            # A legacy adjacent cache is executable too and must match source.
            py_compile.compile(str(model), cfile=str(model.with_suffix(".pyc")), doraise=True, optimize=0)
            self.assertEqual(_python_package_provenance(module, "Transformers"), expected)

    def test_valid_header_caches_with_different_executable_code_are_rejected(self):
        for mode in py_compile.PycInvalidationMode:
            with self.subTest(mode=mode.name), tempfile.TemporaryDirectory() as directory:
                module, model = package_fixture(directory)
                source = model.read_bytes()
                baseline = _python_package_provenance(module, "Transformers")
                cache = Path(py_compile.compile(str(model), doraise=True, optimize=0, invalidation_mode=mode))
                original_header = cache.read_bytes()[:16]
                changed = compile("def forward(x): return x + 2\n", str(model), "exec", dont_inherit=True)
                cache.write_bytes(original_header + marshal.dumps(changed))
                # Demonstrate actual CPython execution of the synthetic substituted
                # cache while the source's timestamp/hash validity is unchanged.
                spec = importlib.util.spec_from_file_location("qsol_synthetic_cache_probe", model)
                imported = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imported)
                self.assertEqual(imported.forward(5), 7)
                self.assertEqual(model.read_bytes(), source)
                with self.assertRaisesRegex(CaptureContractError, "package bytecode"):
                    _python_package_provenance(module, "Transformers")
                cache.unlink()
                self.assertEqual(_python_package_provenance(module, "Transformers"), baseline)

    def test_malformed_foreign_and_noncode_package_caches_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            module, model = package_fixture(directory)
            cache = Path(py_compile.compile(str(model), doraise=True, optimize=0))
            valid = cache.read_bytes()
            for raw in (
                b"invalid", b"BAD!" + valid[4:], valid[:4] + b"\x04\x00\x00\x00" + valid[8:],
                valid[:16] + marshal.dumps([1, 2]), valid + b"trailing bytes",
            ):
                with self.subTest(raw=raw[:20]):
                    cache.write_bytes(raw)
                    with self.assertRaisesRegex(CaptureContractError, "package bytecode"):
                        _python_package_provenance(module, "Transformers")
            cache.write_bytes(valid)
            foreign = cache.with_name("modeling_fixture.foreign-interpreter.pyc")
            cache.rename(foreign)
            with self.assertRaisesRegex(CaptureContractError, "package bytecode"):
                _python_package_provenance(module, "Transformers")

    def test_sourceless_package_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            module, model = package_fixture(directory)
            py_compile.compile(str(model), doraise=True, optimize=0)
            model.unlink()
            with self.assertRaisesRegex(CaptureContractError, "package bytecode"):
                _python_package_provenance(module, "Transformers")

    def test_bytecode_verification_uses_the_source_bytes_bound_by_the_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            module, model = package_fixture(directory)
            py_compile.compile(str(model), doraise=True, optimize=0)
            real_hash = capture_package._hash_regular_file

            def hash_then_change(path, where):
                digest = real_hash(path, where)
                if path == model:
                    model.write_text("def forward(x): return x + 2\n", encoding="utf-8")
                return digest

            with patch.object(capture_package, "_hash_regular_file", side_effect=hash_then_change):
                with self.assertRaisesRegex(CaptureContractError, "package bytecode"):
                    _python_package_provenance(module, "Transformers")

    def test_nonempty_prefix_cannot_be_erased_by_rehashing_and_rewriting_spans(self):
        for context_mode in ("cumulative", "isolated"):
            for prefix in ("prefix", "   "):
                with self.subTest(context_mode=context_mode, prefix=prefix):
                    request = fixture_request()
                    request["capture"].update(
                        prefix_text=prefix, context_mode=context_mode, pooling={"mode": "step_mean"}
                    )
                    request, manifest, trajectory = execute_simulation(request)
                    verify_capture_bundle(request, manifest, trajectory)
                    representation = trajectory["representation_definition"]
                    self.assertTrue(representation["prefix_input_ids"])
                    representation["prefix_input_ids"] = []
                    representation["prefix_input_ids_sha256"] = sha256_json([])
                    changed_steps = trajectory["steps"] if context_mode == "isolated" else trajectory["steps"][:1]
                    for step in changed_steps:
                        span = [0, step["token_count"]]
                        step["changed_token_span"] = span
                        for record in step["layers"]:
                            record["pool_span"] = list(span)
                    rebind_bundle(manifest, trajectory)
                    with self.assertRaisesRegex(CaptureContractError, "prefix_input_ids"):
                        verify_capture_bundle(request, manifest, trajectory)
                    with tempfile.TemporaryDirectory() as directory:
                        destination = Path(directory) / "forged"
                        with self.assertRaisesRegex(CaptureContractError, "prefix_input_ids"):
                            write_capture_bundle(destination, request, manifest, trajectory)
                        self.assertFalse(destination.exists())

    def test_empty_prefix_preserves_both_empty_and_special_token_encodings(self):
        class SpecialTokenBackend(FakeBackend):
            def tokenize(self, text):
                return [0] + super().tokenize(text)

        for context_mode in ("cumulative", "isolated"):
            for backend_type in (FakeBackend, SpecialTokenBackend):
                with self.subTest(context_mode=context_mode, backend=backend_type.__name__):
                    request = fixture_request()
                    request["capture"]["prefix_text"] = ""
                    request["capture"]["context_mode"] = context_mode
                    manifest, trajectory = execute_capture(
                        request, implementation_revision=REV, backend=backend_type(request)
                    )
                    expected = [] if backend_type is FakeBackend else [0]
                    self.assertEqual(trajectory["representation_definition"]["prefix_input_ids"], expected)
                    verify_capture_bundle(request, manifest, trajectory)


if __name__ == "__main__":
    unittest.main()

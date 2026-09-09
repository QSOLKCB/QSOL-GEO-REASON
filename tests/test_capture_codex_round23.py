"""Behavioral regressions for device-scoped state and runtime provenance.

These use explicit software doubles, not empirical CUDA/MPS/model observations.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qsol_geo_reason import provenance
from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend as CoreBackend
from qsol_geo_reason.capture_backend_isolated import HuggingFacePyTorchBackend as IsolatedBackend
from qsol_geo_reason.capture_common import _PRODUCTION_BACKEND_KEYS
from qsol_geo_reason.capture_provenance import _validate_backend_metadata, _validate_production_metadata_shape
from qsol_geo_reason.capture_runtime import (
    _cuda_device_identity,
    _restore_accelerator_rng,
    _seed_capture_generators,
    _snapshot_accelerator_rng,
    _torch_build_metadata,
    _validate_torch_build_metadata,
)
from qsol_geo_reason.capture_verify import verify_capture_bundle
import test_capture_codex_round11 as round11
import test_capture_codex_round18 as round18
import test_capture_codex_round19 as round19
from test_capture_codex_round6 import fixture_request, valid_production_shape

ROOT = Path(__file__).resolve().parents[1]


class UnusedAccelerator:
    def is_available(self):
        return True

    def is_initialized(self):
        return False

    def __getattr__(self, name):
        raise AssertionError(f"CPU-only capture touched unused accelerator: {name}")


class SelectedCuda:
    def __init__(self):
        self.current = 0
        self.states = {0: b"ambient-zero", 1: b"ambient-one"}
        self.reads = []
        self.writes = []

    @contextlib.contextmanager
    def device(self, index):
        previous = self.current
        self.current = index
        try:
            yield
        finally:
            self.current = previous

    def get_rng_state(self, index):
        self.reads.append(index)
        return self.states[index]

    def set_rng_state(self, state, index):
        self.writes.append(index)
        self.states[index] = state

    def manual_seed(self, seed):
        self.states[self.current] = f"seed:{seed}".encode("ascii")


class CaptureRound23RegressionTests(unittest.TestCase):
    def test_cpu_capture_snapshot_seed_and_restore_never_initialize_accelerators(self):
        torch = round19.FakeProcessTorch()
        torch.cuda = UnusedAccelerator()
        torch.mps = UnusedAccelerator()
        torch.xpu = UnusedAccelerator()
        torch.version = SimpleNamespace(hip=None)
        ambient = HuggingFacePyTorchBackend._snapshot_torch_process_state(torch, "cpu")
        self.assertIsNone(ambient["accelerator_rng"])
        # The default_generator retains its CPU-only method; the global seeder
        # is forbidden because real torch.manual_seed also seeds accelerators.
        with patch.object(torch, "manual_seed", side_effect=AssertionError("global seed used")):
            _seed_capture_generators(torch, "cpu", 17)
        self.assertEqual(torch.cpu_rng, b"seed:17")
        HuggingFacePyTorchBackend._restore_torch_process_state(torch, ambient)
        self.assertEqual(torch.cpu_rng, ambient["cpu_rng"])
        self.assertFalse(torch.cuda.is_initialized())
        HuggingFacePyTorchBackend._assert_pristine_cuda_runtime(torch, "cuda:0")

    def test_accelerator_rng_is_scoped_to_the_requested_cuda_index(self):
        torch = round19.FakeProcessTorch()
        torch.cuda = SelectedCuda()
        torch.mps = UnusedAccelerator()
        ambient = HuggingFacePyTorchBackend._snapshot_torch_process_state(torch, "cuda:1")
        _seed_capture_generators(torch, "cuda:1", 23)
        self.assertEqual(torch.cuda.current, 0)
        self.assertEqual(torch.cuda.states, {0: b"ambient-zero", 1: b"seed:23"})
        HuggingFacePyTorchBackend._restore_torch_process_state(torch, ambient)
        self.assertEqual(torch.cuda.states, {0: b"ambient-zero", 1: b"ambient-one"})
        self.assertEqual(torch.cuda.reads, [1])
        self.assertEqual(torch.cuda.writes, [1])
        self.assertEqual(torch.cpu_rng, ambient["cpu_rng"])

    def test_mps_rng_does_not_access_cuda(self):
        torch = round19.FakeProcessTorch()
        torch.cuda = UnusedAccelerator()
        state = {"rng": b"mps-ambient"}
        torch.mps = SimpleNamespace(
            get_rng_state=lambda: state["rng"],
            set_rng_state=lambda value: state.update(rng=value),
            manual_seed=lambda seed: state.update(rng=f"mps:{seed}".encode("ascii")),
        )
        receipt = _snapshot_accelerator_rng(torch, "mps")
        _seed_capture_generators(torch, "mps", 31)
        self.assertEqual(state["rng"], b"mps:31")
        _restore_accelerator_rng(torch, receipt)
        self.assertEqual(state["rng"], b"mps-ambient")

    def test_construction_and_observation_use_device_scoped_rng_helpers(self):
        core = inspect.getsource(CoreBackend.__init__)
        session = inspect.getsource(IsolatedBackend.begin_observation)
        for source in (core, session):
            self.assertIn("_seed_capture_generators(", source)
            self.assertNotIn("manual_seed_all", source)
            self.assertNotIn("torch.manual_seed(", source)
        for cls in (IsolatedBackend, HuggingFacePyTorchBackend):
            self.assertIn(
                "self._snapshot_torch_process_state(process_torch, device)",
                inspect.getsource(cls.__init__),
            )

    def test_cuda_identity_queries_fail_closed_but_uuid_is_optional(self):
        cuda = SimpleNamespace(
            get_device_name=Mock(return_value="SYNTHETIC GPU"),
            get_device_capability=Mock(return_value=(8, 0)),
            get_device_properties=Mock(side_effect=RuntimeError("UUID API unavailable")),
        )
        torch = SimpleNamespace(cuda=cuda)
        self.assertEqual(_cuda_device_identity(torch, 2), {
            "cuda_device_name": "SYNTHETIC GPU",
            "cuda_device_capability": "8.0",
            "cuda_device_uuid": None,
        })
        for name in ("get_device_name", "get_device_capability"):
            with patch.object(cuda, name, side_effect=RuntimeError("identity unavailable")):
                with self.assertRaisesRegex(CaptureContractError, "required CUDA hardware identity"):
                    _cuda_device_identity(torch, 2)
        source = inspect.getsource(CoreBackend.__init__)
        self.assertLess(source.index("_cuda_device_identity("), source.index("Path(snapshot_download("))

    def test_cuda_provenance_requires_nonblank_name_and_capability(self):
        request = fixture_request()
        request["backend"]["device"] = "cuda:0"
        observed = valid_production_shape(request)
        observed.update({
            "cuda_device_name": "SYNTHETIC GPU",
            "cuda_device_capability": "8.0",
            "cuda_resolved_device_index": 0,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
            "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
            "cublas_workspace_config": ":4096:8",
        })
        _validate_production_metadata_shape(observed, request)
        for field in ("cuda_device_name", "cuda_device_capability"):
            for bad in (None, "", " \t\n", True, []):
                mutated = dict(observed, **{field: bad})
                with self.subTest(field=field, bad=bad):
                    with self.assertRaisesRegex(CaptureContractError, field):
                        _validate_production_metadata_shape(mutated, request)
        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        rule = schema["$defs"]["backendObservedProduction"]["allOf"][2]
        self.assertIsNotNone(re.fullmatch(rule["if"]["properties"]["device"]["pattern"], "cuda:0"))
        for field in ("cuda_device_name", "cuda_device_capability"):
            constraint = rule["then"]["properties"][field]
            self.assertEqual(constraint["type"], "string")
            expected_pattern = r"\S" if field == "cuda_device_name" else r"^[0-9]+\.[0-9]+$"
            self.assertEqual(constraint["pattern"], expected_pattern)

    def test_compiled_call_substitution_is_rejected_at_each_module(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = round19.FakeGraphModel()
        original_forward = IsolatedBackend._model_executable_state_seal(backend)
        original = backend._model_executable_state_seal()
        for module in (backend._model, backend._model.child):
            module._compiled_call_impl = lambda *args, **kwargs: "compiled"
            # The forward-only seal remains unchanged, but the stronger dependency
            # seal now also binds added callable attributes. Keep both assertions
            # and the independent explicit compiled-call rejection regression.
            self.assertEqual(IsolatedBackend._model_executable_state_seal(backend), original_forward)
            self.assertNotEqual(backend._model_executable_state_seal(), original)
            with self.assertRaisesRegex(CaptureContractError, "compiled module call"):
                backend._assert_live_state_authentication()
            module._compiled_call_impl = None
            self.assertEqual(backend._model_executable_state_seal(), original)
        backend._assert_live_state_authentication()

    def _sealed_backend(self):
        backend = round18.CaptureRound18RegressionTests()._sealed_backend()
        request = fixture_request()
        for field, value in {
            "_applied_seed": request["determinism"]["seed"],
            "_determinism_mode": request["determinism"]["mode"],
            "_model_identifier": request["model"]["identifier"],
            "_model_revision": request["model"]["revision"],
            "_tokenizer_identifier": request["model"]["tokenizer_identifier"],
            "_tokenizer_revision": request["model"]["tokenizer_revision"],
            "_device": request["backend"]["device"],
            "_dtype_name": request["backend"]["dtype"],
        }.items():
            setattr(backend, field, value)
        return backend, request

    def test_one_content_hash_pass_per_guard_and_per_request(self):
        backend, request = self._sealed_backend()
        with patch.object(backend, "_model_content_state_seal", wraps=backend._model_content_state_seal) as hashes:
            backend._assert_live_state_authentication()
            self.assertEqual(hashes.call_count, 1)
            hashes.reset_mock()
            backend.assert_execution_request(request)
            self.assertEqual(hashes.call_count, 1)

    def test_forward_keeps_one_pre_and_one_post_content_check(self):
        backend, _ = self._sealed_backend()
        with (
            patch.object(backend, "_model_content_state_seal", wraps=backend._model_content_state_seal) as hashes,
            patch.object(CoreBackend, "hidden_states", return_value={}),
        ):
            backend.hidden_states([0], [0], pool_span=(0, 1))
            self.assertEqual(hashes.call_count, 2)

        backend._model.weight.content[0] ^= 1
        with patch.object(CoreBackend, "hidden_states", return_value={}) as forward:
            with self.assertRaisesRegex(CaptureContractError, "tensor contents changed"):
                backend.hidden_states([0], [0], pool_span=(0, 1))
            forward.assert_not_called()
        backend._model.weight.content[0] ^= 1

        def mutate_during_forward(*args, **kwargs):
            backend._model.cache.content[0] ^= 1
            return {}

        with patch.object(CoreBackend, "hidden_states", side_effect=mutate_during_forward):
            with self.assertRaisesRegex(CaptureContractError, "tensor contents changed"):
                backend.hidden_states([0], [0], pool_span=(0, 1))

    def test_tokenizer_guards_do_not_rehash_model_weights(self):
        backend, _ = self._sealed_backend()
        backend._tokenizer = round19.FakeSlowTokenizer()
        backend._canonical_tokenizer_live_state = backend._tokenizer_live_state_seal()
        with (
            patch.object(backend, "_model_content_state_seal", side_effect=AssertionError("model hashing during tokenization")),
            patch.object(backend, "_tokenizer_live_state_seal", wraps=backend._tokenizer_live_state_seal) as seals,
            patch.object(CoreBackend, "tokenize", return_value=[0]),
        ):
            self.assertEqual(backend.tokenize("a"), [0])
            self.assertEqual(seals.call_count, 2)

        def mutate_tokenizer(text):
            backend._tokenizer.add_prefix_space = True
            return [0]

        with patch.object(CoreBackend, "tokenize", side_effect=mutate_tokenizer):
            with self.assertRaisesRegex(CaptureContractError, "live tokenizer state changed"):
                backend.tokenize("a")

    def test_untracked_pyd_invalidates_real_git_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/qsol_geo_reason"
            source.mkdir(parents=True)
            (source / "capture.py").write_text("# tracked source\n", encoding="utf-8")
            (root / ".gitignore").write_text((ROOT / ".gitignore").read_text(), encoding="utf-8")

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True, capture_output=True, text=True,
                )

            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=Capture Test", "-c", "user.email=capture-test@example.invalid", "commit", "-qm", "fixture")
            with patch.object(provenance, "source_repo_root", return_value=root):
                expected = provenance.resolve_implementation_revision(require_checkout=True)
                for filename in ("capture.pyd", "capture.cp313-win_amd64.pyd"):
                    extension = source / filename
                    extension.write_bytes(b"synthetic extension placeholder")
                    with self.subTest(filename=filename):
                        self.assertFalse(provenance._is_generated_untracked(f"src/qsol_geo_reason/{filename}"))
                        with self.assertRaisesRegex(provenance.SourceIdentityError, "dirty"):
                            provenance.resolve_implementation_revision(require_checkout=True)
                    extension.unlink()
                build = root / "build"
                build.mkdir()
                (build / "capture.pyd").write_bytes(b"non-imported build output")
                self.assertEqual(provenance.resolve_implementation_revision(require_checkout=True), expected)

    def test_build_receipt_distinguishes_same_version_different_builds(self):
        first = SimpleNamespace(__version__="same-version", __config__=SimpleNamespace(show=lambda: "BLAS=MKL; compiler=A\n"))
        second = SimpleNamespace(__version__="same-version", __config__=SimpleNamespace(show=lambda: "BLAS=OpenBLAS; compiler=B\n"))
        a, b = _torch_build_metadata(first), _torch_build_metadata(second)
        self.assertEqual(first.__version__, second.__version__)
        self.assertNotEqual(a["torch_build_config_sha256"], b["torch_build_config_sha256"])
        self.assertEqual(a["torch_build_config"], first.__config__.show())
        _validate_torch_build_metadata(a)
        with self.assertRaisesRegex(CaptureContractError, "does not authenticate"):
            _validate_torch_build_metadata(dict(a, torch_build_config=b["torch_build_config"]))
        for bad in (None, "", " \n\t"):
            with self.assertRaisesRegex(CaptureContractError, "non-whitespace"):
                _torch_build_metadata(SimpleNamespace(__config__=SimpleNamespace(show=lambda: bad)))
        with self.assertRaisesRegex(CaptureContractError, "__config__"):
            _torch_build_metadata(SimpleNamespace())

    def test_build_receipt_is_required_and_reverified_after_outer_rehash(self):
        request, manifest, trajectory = round11.execute_simulation()
        manifest["backend_observed"] = round11.production_observed(request)
        trajectory["evidence_class"] = "OBSERVATION"
        round11.rebind_bundle(manifest, trajectory)
        verify_capture_bundle(request, manifest, trajectory)
        for field in ("torch_build_config", "torch_build_config_sha256"):
            observed = dict(manifest["backend_observed"])
            del observed[field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(CaptureContractError, "missing required"):
                    _validate_backend_metadata(observed, request, "OBSERVATION")

        manifest["backend_observed"]["torch_build_config"] += "altered build"
        round11.rebind_bundle(manifest, trajectory)
        with self.assertRaisesRegex(CaptureContractError, "does not authenticate"):
            verify_capture_bundle(request, manifest, trajectory)
        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        production = schema["$defs"]["backendObservedProduction"]
        for field in ("torch_build_config", "torch_build_config_sha256"):
            self.assertIn(field, production["required"])
            self.assertIn(field, _PRODUCTION_BACKEND_KEYS)
        self.assertIn("_torch_build_metadata(torch)", inspect.getsource(CoreBackend.__init__))
        self.assertIn("**self._torch_build_provenance", inspect.getsource(CoreBackend.metadata))


if __name__ == "__main__":
    unittest.main()

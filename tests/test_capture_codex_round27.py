"""Round 27 regressions for executable provenance and fail-closed validation."""
from __future__ import annotations

import copy
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.capture_backend import HuggingFacePyTorchBackend as PolicyBackend
from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_common import CaptureContractError, _LOADING_INFO_KEYS
from qsol_geo_reason.capture_dispatch import (
    _execution_dependencies_sha256,
    _model_execution_dependency_roots,
)
from qsol_geo_reason.capture_execute import _assert_observation_backend_execution_methods
from qsol_geo_reason.capture_validation import _validate_loading_info, validate_capture_request
from qsol_geo_reason.provenance import SourceIdentityError, git_source_revision

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "fixtures" / "capture-contract-request.json"


def fixture_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


class DispatchModel:
    def __call__(self, *args, **kwargs):
        return self._call_impl(*args, **kwargs)

    def _wrapped_call_impl(self, *args, **kwargs):
        return self._call_impl(*args, **kwargs)

    def _call_impl(self, value=None, **_kwargs):
        return self.forward(value)

    def forward(self, value=None):
        return value

    def named_modules(self):
        return [("", self)]


class SlowTokenizer:
    def __init__(self):
        self.vocab = {"a": 0, "b": 1}
        self.added_tokens_encoder = {}
        self.special_tokens_map = {}
        self.all_special_tokens = []
        self.all_special_ids = []
        self.init_kwargs = {}

    def get_vocab(self):
        return dict(self.vocab)

    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text):
        return self._tokenize(text)

    def _tokenize(self, text):
        return list(text)


class CaptureRound27RegressionTests(unittest.TestCase):
    def test_observation_rejects_private_adapter_method_shadowing(self):
        backend = object.__new__(Backend)
        _assert_observation_backend_execution_methods(backend)
        for name in (
            "_assert_live_state_authentication",
            "_force_canonical_determinism_policy",
            "_assert_tokenizer_state_authentication",
        ):
            with self.subTest(name=name):
                setattr(backend, name, lambda *args, **kwargs: None)
                with self.assertRaisesRegex(CaptureContractError, "cannot be overridden"):
                    _assert_observation_backend_execution_methods(backend)
                delattr(backend, name)
                _assert_observation_backend_execution_methods(backend)

    def test_model_call_dispatch_changes_executable_dependency_receipt(self):
        model = DispatchModel()
        before = _execution_dependencies_sha256(_model_execution_dependency_roots(model))
        original = DispatchModel._call_impl
        try:
            DispatchModel._call_impl = lambda self, value=None, **_kwargs: ("patched", value)
            after = _execution_dependencies_sha256(_model_execution_dependency_roots(model))
        finally:
            DispatchModel._call_impl = original
        self.assertNotEqual(before, after)

    def test_tokenizer_helper_and_class_call_changes_live_state_receipt(self):
        backend = object.__new__(PolicyBackend)
        backend._tokenizer = SlowTokenizer()
        before = backend._tokenizer_live_state_seal()

        original_tokenize = SlowTokenizer._tokenize
        try:
            SlowTokenizer._tokenize = lambda self, text: ["patched", text]
            self.assertNotEqual(backend._tokenizer_live_state_seal(), before)
        finally:
            SlowTokenizer._tokenize = original_tokenize

        before_call = backend._tokenizer_live_state_seal()
        original_call = SlowTokenizer.__call__
        try:
            SlowTokenizer.__call__ = lambda self, text: ["call-patched", text]
            self.assertNotEqual(backend._tokenizer_live_state_seal(), before_call)
        finally:
            SlowTokenizer.__call__ = original_call

    def test_malformed_enum_values_raise_contract_errors(self):
        cases = (
            (("backend", "dtype"), "backend.dtype"),
            (("capture", "context_mode"), "capture.context_mode"),
            (("capture", "pooling", "mode"), "capture.pooling.mode"),
            (("determinism", "mode"), "determinism.mode"),
        )
        for path, label in cases:
            for malformed in ([], {}, ["float32"], {"value": "required"}):
                request = fixture_request()
                target = request
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = copy.deepcopy(malformed)
                with self.subTest(path=path, malformed=malformed):
                    with self.assertRaisesRegex(CaptureContractError, re.escape(label)):
                        validate_capture_request(request)

    def test_loading_info_requires_every_diagnostic_key(self):
        clean = {key: [] for key in _LOADING_INFO_KEYS}
        _validate_loading_info(clean)
        for key in _LOADING_INFO_KEYS:
            malformed = dict(clean)
            del malformed[key]
            with self.subTest(key=key):
                with self.assertRaisesRegex(CaptureContractError, "missing required diagnostic"):
                    _validate_loading_info(malformed)

    def test_mps_schema_rejects_enabled_fallback_and_fast_math(self):
        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        mps_rule = schema["$defs"]["backendObservedProduction"]["allOf"][0]
        constraints = mps_rule["then"]["properties"]
        for field in ("mps_fallback_env", "mps_fast_math_env"):
            rule = constraints[field]
            string_rule = next(item for item in rule["anyOf"] if item.get("type") == "string")
            pattern = re.compile(string_rule["pattern"])
            for accepted in ("", "0", "false", "FALSE", "no", "OFF", "  off  "):
                self.assertIsNotNone(pattern.fullmatch(accepted), (field, accepted))
            for rejected in ("1", "true", "yes", "on", "enabled"):
                self.assertIsNone(pattern.fullmatch(rejected), (field, rejected))
            self.assertTrue(any(item.get("type") == "null" for item in rule["anyOf"]))

    def test_checkout_bound_observation_rejects_ignored_package_bytecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "src" / "qsol_geo_reason"
            package.mkdir(parents=True)
            (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
            (package / "capture.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

            cache = package / "__pycache__"
            cache.mkdir()
            (cache / "capture.cpython-313.pyc").write_bytes(b"synthetic-bytecode")

            with patch("qsol_geo_reason.provenance.source_repo_root", return_value=root):
                # Ordinary source-revision lookup still tolerates disposable caches.
                self.assertRegex(git_source_revision(require_clean=True), r"^[0-9a-f]{40}$")
                with self.assertRaisesRegex(SourceIdentityError, "importable bytecode caches"):
                    git_source_revision(
                        require_clean=True,
                        reject_importable_bytecode=True,
                    )


if __name__ == "__main__":
    unittest.main()

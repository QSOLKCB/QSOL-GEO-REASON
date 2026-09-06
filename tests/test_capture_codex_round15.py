from __future__ import annotations

import inspect
import os
import unittest
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_backend import (
    _CoreHuggingFacePyTorchBackend,
    _ExplicitNoAttentionBaseModel,
)
from qsol_geo_reason.capture_common import _LOADING_INFO_KEYS
from qsol_geo_reason.capture_validation import _validate_loading_info


class Recorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return "ok"


class FakePolicyTorch:
    def __init__(self, deterministic=False, intra=8, inter=2):
        self.deterministic = deterministic
        self.intra = intra
        self.inter = inter

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def use_deterministic_algorithms(self, value):
        self.deterministic = bool(value)

    def get_num_threads(self):
        return self.intra

    def get_num_interop_threads(self):
        return self.inter

    def set_num_threads(self, value):
        self.intra = value

    def set_num_interop_threads(self, value):
        self.inter = value


class CaptureRound15RegressionTests(unittest.TestCase):
    def test_attention_outputs_are_explicitly_disabled(self):
        recorder = Recorder()
        proxy = _ExplicitNoAttentionBaseModel(recorder)
        self.assertEqual(proxy(output_attentions=True, input_ids="ids"), "ok")
        self.assertIs(recorder.kwargs["output_attentions"], False)
        self.assertEqual(recorder.kwargs["input_ids"], "ids")

    def test_cpu_thread_policy_wraps_pooling_for_all_devices(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakePolicyTorch(intra=8, inter=2)
        backend._canonical_cpu_thread_policy = {
            "torch_num_threads": 8,
            "torch_num_interop_threads": 2,
        }
        backend._last_cpu_thread_policy = None
        backend._canonical_deterministic_algorithms_enabled = None
        backend._canonical_cuda_environment = None
        backend._torch.intra = 1
        backend._torch.inter = 1

        with patch.object(
            _CoreHuggingFacePyTorchBackend,
            "_pool_tensor_record",
            return_value={"vector": [1.0], "vector_dimension": 1, "observed_dtype": "float32"},
        ):
            record = backend._pool_tensor_record(None, layer_index=0, token_count=1, pool_span=(0, 1))
        self.assertEqual(record["vector"], [1.0])
        self.assertEqual(backend._torch.intra, 8)
        self.assertEqual(backend._torch.inter, 2)
        source = inspect.getsource(HuggingFacePyTorchBackend._pool_tensor_record)
        self.assertIn("_force_cpu_thread_policy", source)
        self.assertIn("_assert_cpu_thread_policy", source)
        self.assertNotIn('self._device_type == "cpu"', source)

    def test_best_effort_determinism_is_frozen_not_late_sampled(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakePolicyTorch(deterministic=True)
        backend._determinism_mode = "best_effort"
        backend._canonical_deterministic_algorithms_enabled = False
        backend._last_deterministic_algorithms_enabled = None
        backend._force_canonical_determinism_policy()
        self.assertIs(backend._torch.deterministic, False)
        backend._torch.deterministic = True
        with self.assertRaisesRegex(CaptureContractError, "deterministic-algorithm policy drifted"):
            backend._assert_canonical_determinism_policy()

    def test_best_effort_and_cpu_threads_are_rechecked_after_entire_forward(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        forward = source.index("result = super().hidden_states")
        self.assertLess(source.index("self._force_canonical_determinism_policy()"), forward)
        self.assertLess(source.index("self._force_cpu_thread_policy()"), forward)
        self.assertGreater(source.index("self._assert_canonical_determinism_policy()", forward), forward)
        self.assertGreater(source.index("self._assert_cpu_thread_policy()", forward), forward)

    def test_cuda_environment_controls_are_snapshotted_and_drift_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._canonical_cuda_environment = {
            "cuda_visible_devices": None,
            "nvidia_tf32_override": None,
            "torch_allow_tf32_cublas_override": None,
            "cublas_workspace_config": None,
        }
        with patch.dict(os.environ, {}, clear=True):
            backend._assert_cuda_environment_policy()
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            with self.assertRaisesRegex(CaptureContractError, "CUDA environment policy drifted"):
                backend._assert_cuda_environment_policy()
        init_source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        self.assertLess(
            init_source.index("self._canonical_cuda_environment = self._cuda_environment_state()"),
            init_source.index("super().__init__(validated)"),
        )

    def test_loading_info_requires_all_four_diagnostics(self):
        clean = {key: [] for key in _LOADING_INFO_KEYS}
        _validate_loading_info(clean)
        for key in _LOADING_INFO_KEYS:
            partial = dict(clean)
            partial.pop(key)
            with self.subTest(missing=key):
                with self.assertRaisesRegex(CaptureContractError, "incomplete"):
                    _validate_loading_info(partial)
        for key in _LOADING_INFO_KEYS:
            malformed = dict(clean)
            malformed[key] = None
            with self.subTest(malformed=key):
                with self.assertRaisesRegex(CaptureContractError, "invalid shape"):
                    _validate_loading_info(malformed)

    def test_existing_core_and_facade_capture_guards_remain_visible(self):
        core_hidden = inspect.getsource(_CoreHuggingFacePyTorchBackend.hidden_states)
        self.assertIn("self._base_model(", core_hidden)
        self.assertIn("output_hidden_states=False", core_hidden)
        self.assertIn("use_cache=False", core_hidden)
        core_metadata = inspect.getsource(_CoreHuggingFacePyTorchBackend.metadata)
        self.assertIn("self._last_cuda_float32_policy", core_metadata)
        self.assertNotIn("get_float32_matmul_precision", core_metadata)
        self.assertIn('"cpu_mkldnn_enabled"', core_metadata)
        facade_metadata = "\n".join(
            inspect.getsource(cls.__dict__["metadata"])
            for cls in HuggingFacePyTorchBackend.__mro__
            if "metadata" in cls.__dict__
        )
        self.assertIn('data["torch_num_threads"]', facade_metadata)
        self.assertIn("self._canonical_cuda_environment", facade_metadata)


if __name__ == "__main__":
    unittest.main()

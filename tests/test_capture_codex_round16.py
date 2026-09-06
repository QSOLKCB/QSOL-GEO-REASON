from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend as CoreBackend


class FakeEvalModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self


class FakeDeterminismTorch:
    def __init__(self, *, enabled: bool = True, warn_only: bool = True):
        self.enabled = enabled
        self.warn_only = warn_only

    def are_deterministic_algorithms_enabled(self):
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def use_deterministic_algorithms(self, enabled, *, warn_only=False):
        self.enabled = bool(enabled)
        self.warn_only = bool(warn_only)


class FakeCudaSDPA:
    def __init__(self):
        self.flash = True
        self.mem_efficient = True
        self.math = False
        self.cudnn = True
        self.reduced_math = True

    def flash_sdp_enabled(self):
        return self.flash

    def mem_efficient_sdp_enabled(self):
        return self.mem_efficient

    def math_sdp_enabled(self):
        return self.math

    def cudnn_sdp_enabled(self):
        return self.cudnn

    def fp16_bf16_reduction_math_sdp_allowed(self):
        return self.reduced_math

    def enable_flash_sdp(self, enabled):
        self.flash = bool(enabled)

    def enable_mem_efficient_sdp(self, enabled):
        self.mem_efficient = bool(enabled)

    def enable_math_sdp(self, enabled):
        self.math = bool(enabled)

    def enable_cudnn_sdp(self, enabled):
        self.cudnn = bool(enabled)

    def allow_fp16_bf16_reduction_math_sdp(self, enabled):
        self.reduced_math = bool(enabled)


class CaptureRound16RegressionTests(unittest.TestCase):
    def test_evaluation_mode_is_forced_and_post_forward_drift_is_rejected(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = FakeEvalModel()
        backend._canonical_evaluation_mode = True

        backend._force_model_eval_policy()
        self.assertIs(backend._model.training, False)

        backend._model.training = True
        with self.assertRaisesRegex(CaptureContractError, "evaluation mode"):
            backend._assert_model_eval_policy()

        pre_source = inspect.getsource(HuggingFacePyTorchBackend._assert_attention_implementation)
        forward_source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        core_source = inspect.getsource(CoreBackend.hidden_states)
        self.assertIn("self._force_model_eval_policy()", pre_source)
        self.assertIn("self._assert_model_eval_policy()", forward_source)
        self.assertLess(
            core_source.index("self._assert_attention_implementation()"),
            core_source.index("self._base_model("),
        )

    def test_required_determinism_forces_warn_only_false_and_rejects_drift(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeDeterminismTorch(enabled=True, warn_only=True)
        backend._determinism_mode = "required"
        backend._canonical_deterministic_algorithms_enabled = True
        backend._canonical_deterministic_warn_only_enabled = False
        backend._last_deterministic_algorithms_enabled = None

        backend._force_canonical_determinism_policy()
        self.assertIs(backend._torch.enabled, True)
        self.assertIs(backend._torch.warn_only, False)

        backend._torch.warn_only = True
        with self.assertRaisesRegex(CaptureContractError, "warn-only"):
            backend._assert_canonical_determinism_policy()

    def test_math_sdpa_reduced_precision_reductions_are_disabled_and_verified(self):
        cuda = FakeCudaSDPA()
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = SimpleNamespace(backends=SimpleNamespace(cuda=cuda))
        backend._last_sdpa_policy = None

        backend._force_sdpa_math_policy()
        self.assertIs(cuda.flash, False)
        self.assertIs(cuda.mem_efficient, False)
        self.assertIs(cuda.math, True)
        self.assertIs(cuda.cudnn, False)
        self.assertIs(cuda.reduced_math, False)
        self.assertIs(
            backend._assert_sdpa_math_policy()["fp16_bf16_math_reduction"],
            False,
        )

        cuda.reduced_math = True
        with self.assertRaisesRegex(CaptureContractError, "reduced-precision"):
            backend._assert_sdpa_math_policy()


if __name__ == "__main__":
    unittest.main()

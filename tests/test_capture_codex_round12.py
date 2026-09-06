from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_provenance import _validate_production_metadata_shape


class FakeCudaTorch:
    def __init__(self):
        self.precision = "high"
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(
                    allow_tf32=True,
                    allow_fp16_reduced_precision_reduction=False,
                    allow_bf16_reduced_precision_reduction=False,
                )
            ),
            cudnn=SimpleNamespace(allow_tf32=False),
        )

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, value):
        self.precision = value


class CaptureRound12RegressionTests(unittest.TestCase):
    def test_cuda_float32_policy_is_restored_to_construction_snapshot(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeCudaTorch()
        backend._canonical_cuda_float32_policy = backend._cuda_float32_policy_state()
        backend._last_cuda_float32_policy = None

        backend._torch.precision = "medium"
        backend._torch.backends.cuda.matmul.allow_tf32 = False
        backend._torch.backends.cudnn.allow_tf32 = True

        backend._force_cuda_float32_policy()
        self.assertEqual(
            backend._last_cuda_float32_policy,
            {
                "float32_matmul_precision": "high",
                "cuda_matmul_allow_tf32": True,
                "cudnn_allow_tf32": False,
            },
        )
        self.assertEqual(backend._assert_cuda_float32_policy(), backend._canonical_cuda_float32_policy)

    def test_cuda_float32_policy_rejects_post_forward_drift(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeCudaTorch()
        backend._canonical_cuda_float32_policy = backend._cuda_float32_policy_state()
        backend._torch.backends.cuda.matmul.allow_tf32 = False
        with self.assertRaisesRegex(CaptureContractError, "float32/TF32 policy drifted"):
            backend._assert_cuda_float32_policy()

    def test_each_cuda_forward_reasserts_then_rechecks_float32_policy(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        forward = source.index("self._base_model(")
        self.assertLess(source.index("self._force_cuda_float32_policy()"), forward)
        self.assertGreater(source.index("self._assert_cuda_float32_policy()"), forward)

    def test_metadata_uses_last_enforced_policy_not_ambient_resampling(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.metadata)
        self.assertIn("self._last_cuda_float32_policy", source)
        self.assertNotIn("get_float32_matmul_precision", source)
        self.assertNotIn("torch.backends.cuda.matmul.allow_tf32", source)
        self.assertNotIn("torch.backends.cudnn.allow_tf32", source)

    def test_verifier_requires_cuda_float32_policy_and_nulls_it_elsewhere(self):
        source = inspect.getsource(_validate_production_metadata_shape)
        self.assertIn("float32_policy_fields", source)
        self.assertIn('{"highest", "high", "medium"}', source)
        self.assertIn("canonical CUDA capture requires boolean", source)
        self.assertIn("CUDA float32/TF32 policy fields must be null outside CUDA", source)


if __name__ == "__main__":
    unittest.main()

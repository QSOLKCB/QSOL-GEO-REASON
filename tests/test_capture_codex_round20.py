from __future__ import annotations

import unittest
from types import SimpleNamespace

from qsol_geo_reason.capture import HuggingFacePyTorchBackend


class FakeCudaBackend:
    def __init__(self):
        self.matmul = SimpleNamespace(
            allow_tf32=True,
            allow_fp16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction=True,
        )
        self._flash = True
        self._mem_efficient = True
        self._math = False
        self._cudnn = True
        self._reduced_math = True

    def flash_sdp_enabled(self):
        return self._flash

    def mem_efficient_sdp_enabled(self):
        return self._mem_efficient

    def math_sdp_enabled(self):
        return self._math

    def cudnn_sdp_enabled(self):
        return self._cudnn

    def fp16_bf16_reduction_math_sdp_allowed(self):
        return self._reduced_math

    def enable_flash_sdp(self, enabled):
        self._flash = bool(enabled)

    def enable_mem_efficient_sdp(self, enabled):
        self._mem_efficient = bool(enabled)

    def enable_math_sdp(self, enabled):
        self._math = bool(enabled)

    def enable_cudnn_sdp(self, enabled):
        self._cudnn = bool(enabled)

    def allow_fp16_bf16_reduction_math_sdp(self, enabled):
        self._reduced_math = bool(enabled)


class FakePolicyTorch:
    def __init__(self):
        self.cpu_rng = b"ambient-rng"
        self.deterministic = False
        self.warn_only = True
        self.num_threads = 7
        self.num_interop_threads = 3
        self.float32_precision = "high"
        self.backends = SimpleNamespace(
            mkldnn=SimpleNamespace(
                enabled=True,
                matmul=SimpleNamespace(fp32_precision="ieee"),
            ),
            cuda=FakeCudaBackend(),
            cudnn=SimpleNamespace(allow_tf32=True),
        )
        self.cuda = SimpleNamespace(is_available=lambda: False)
        self.mps = SimpleNamespace(is_available=lambda: False)
        self.xpu = SimpleNamespace(is_available=lambda: False)

    def get_rng_state(self):
        return self.cpu_rng

    def set_rng_state(self, state):
        self.cpu_rng = state

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def use_deterministic_algorithms(self, enabled, *, warn_only=False):
        self.deterministic = bool(enabled)
        self.warn_only = bool(warn_only)

    def get_num_threads(self):
        return self.num_threads

    def set_num_threads(self, value):
        self.num_threads = int(value)

    def get_num_interop_threads(self):
        return self.num_interop_threads

    def set_num_interop_threads(self, value):
        self.num_interop_threads = int(value)

    def get_float32_matmul_precision(self):
        return self.float32_precision

    def set_float32_matmul_precision(self, value):
        self.float32_precision = value


class CaptureRound20RegressionTests(unittest.TestCase):
    def test_process_state_restoration_includes_all_mutated_execution_policies(self):
        torch = FakePolicyTorch()
        ambient = HuggingFacePyTorchBackend._snapshot_torch_process_state(torch)

        torch.cpu_rng = b"capture-rng"
        torch.deterministic = True
        torch.warn_only = False
        torch.num_threads = 1
        torch.num_interop_threads = 1
        torch.backends.mkldnn.enabled = False
        torch.backends.mkldnn.matmul.fp32_precision = "tf32"
        torch.float32_precision = "medium"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

        HuggingFacePyTorchBackend._restore_torch_process_state(torch, ambient)
        restored = HuggingFacePyTorchBackend._snapshot_torch_process_state(torch)

        self.assertEqual(restored["cpu_rng"], ambient["cpu_rng"])
        self.assertIs(restored["deterministic_algorithms_enabled"], False)
        self.assertIs(restored["deterministic_warn_only_enabled"], True)
        self.assertEqual(restored["execution_policies"], ambient["execution_policies"])
        self.assertEqual(
            set(restored["execution_policies"]),
            {
                "torch_num_threads",
                "torch_num_interop_threads",
                "cpu_mkldnn_enabled",
                "cpu_mkldnn_matmul_fp32_precision",
                "float32_matmul_precision",
                "cuda_matmul_allow_tf32",
                "cuda_allow_fp16_reduced_precision_reduction",
                "cuda_allow_bf16_reduced_precision_reduction",
                "cudnn_allow_tf32",
                "cuda_flash_sdp_enabled",
                "cuda_mem_efficient_sdp_enabled",
                "cuda_math_sdp_enabled",
                "cuda_cudnn_sdp_enabled",
                "cuda_fp16_bf16_reduction_math_sdp_allowed",
            },
        )


if __name__ == "__main__":
    unittest.main()

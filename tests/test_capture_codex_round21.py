from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend
from qsol_geo_reason.capture_execute import execute_capture


class FakeHookModule:
    def __init__(self):
        self._forward_pre_hooks = {}
        self._forward_hooks = {}
        self._backward_pre_hooks = {}
        self._backward_hooks = {}
        self._forward_pre_hooks_with_kwargs = {}
        self._forward_hooks_with_kwargs = {}
        self._forward_hooks_always_called = {}


class FakeHookModel(FakeHookModule):
    def __init__(self):
        super().__init__()
        self.child = FakeHookModule()

    def named_modules(self):
        return [("", self), ("child", self.child)]


class FakeCudaRuntime:
    def __init__(self, initialized: bool):
        self._initialized = initialized

    def is_initialized(self):
        return self._initialized


class FakeTorchRuntime:
    def __init__(self, *, initialized: bool = False, hip: str | None = None):
        self.version = SimpleNamespace(hip=hip)
        self.cuda = FakeCudaRuntime(initialized)


class CaptureRound21RegressionTests(unittest.TestCase):
    def test_registered_forward_hooks_are_rejected_at_live_state_boundary(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._model = FakeHookModel()
        backend._assert_no_registered_module_hooks()

        backend._model.child._forward_hooks[7] = lambda module, args, output: output
        with self.assertRaisesRegex(CaptureContractError, "registered model execution hooks"):
            backend._assert_no_registered_module_hooks()

        source = inspect.getsource(HuggingFacePyTorchBackend._assert_live_state_authentication)
        self.assertIn("self._assert_no_registered_module_hooks()", source)

    def test_cuda_observation_requires_runtime_uninitialized_before_snapshot(self):
        clean = FakeTorchRuntime(initialized=False, hip=None)
        HuggingFacePyTorchBackend._assert_pristine_cuda_runtime(clean, "cuda:0")

        initialized = FakeTorchRuntime(initialized=True, hip=None)
        with self.assertRaisesRegex(CaptureContractError, "uninitialized CUDA runtime"):
            HuggingFacePyTorchBackend._assert_pristine_cuda_runtime(initialized, "cuda:0")

        source = inspect.getsource(HuggingFacePyTorchBackend.__init__)
        self.assertLess(
            source.index("self._assert_pristine_cuda_runtime(process_torch, device)"),
            source.index("self._snapshot_torch_process_state(process_torch)"),
        )

    def test_rocm_build_cannot_enter_cuda_observation_lane(self):
        rocm = FakeTorchRuntime(initialized=False, hip="6.4.0")
        with self.assertRaisesRegex(CaptureContractError, "ROCm/HIP"):
            HuggingFacePyTorchBackend._assert_pristine_cuda_runtime(rocm, "cuda:0")

        # A CPU request is not mislabeled as CUDA and remains outside this guard.
        HuggingFacePyTorchBackend._assert_pristine_cuda_runtime(rocm, "cpu")

    def test_execute_capture_requires_the_final_concrete_boundary(self):
        source = inspect.getsource(execute_capture)
        self.assertIn("type(backend) is not HuggingFacePyTorchBackend", source)
        self.assertEqual(
            HuggingFacePyTorchBackend.__module__,
            "qsol_geo_reason.capture_backend_production",
        )


if __name__ == "__main__":
    unittest.main()

"""Round 54 regressions for restoration ownership and timestamp-safe runtime stability."""
from __future__ import annotations

import time
import unittest

import qsol_geo_reason.capture_backend_production as production
import qsol_geo_reason.capture_backend_round39 as round39
import qsol_geo_reason.capture_cuda_runtime as cuda_runtime
from qsol_geo_reason.capture_common import CaptureContractError


class CaptureRound54RegressionTests(unittest.TestCase):
    def _bare_production_backend(self):
        backend = object.__new__(production.HuggingFacePyTorchBackend)
        backend._observation_active = True
        backend._observation_ambient_process_state = {"rng": "ambient"}
        backend._torch = object()
        backend._exclusive_thread_boundary_state = {"patches": ("synthetic",)}
        backend.restore_should_fail = True
        backend.leave_calls = 0

        def restore(_torch, _ambient):
            if backend.restore_should_fail:
                raise CaptureContractError("synthetic ambient restore failure")

        def leave():
            backend.leave_calls += 1
            backend._exclusive_thread_boundary_state = None

        backend._restore_torch_process_state = restore
        backend._leave_exclusive_python_thread_boundary = leave
        return backend

    def test_end_observation_retains_thread_exclusion_until_restore_succeeds(self):
        backend = self._bare_production_backend()
        with self.assertRaisesRegex(CaptureContractError, "synthetic ambient restore failure"):
            production.HuggingFacePyTorchBackend.end_observation(backend)

        self.assertTrue(backend._observation_active)
        self.assertEqual(backend._observation_ambient_process_state, {"rng": "ambient"})
        self.assertIsNotNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(backend.leave_calls, 0)

        backend.restore_should_fail = False
        production.HuggingFacePyTorchBackend.end_observation(backend)
        self.assertFalse(backend._observation_active)
        self.assertIsNone(backend._observation_ambient_process_state)
        self.assertIsNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(backend.leave_calls, 1)

    def test_runtime_stability_allows_unchanged_baseline_with_future_mtime(self):
        started = time.time_ns()
        entry = (
            "/tmp/libgomp.so.1",
            "libgomp.so.1",
            "a" * 64,
            1,
            2,
            4096,
            started + 10_000_000_000,
            started - 1_000_000,
        )
        cuda_runtime.assert_runtime_library_state_stable(
            (entry,), (entry,), observation_started_ns=started, label="CPU"
        )

    def test_runtime_stability_allows_lazy_library_with_future_mtime_when_ctime_is_old(self):
        started = time.time_ns()
        lazy = (
            "/tmp/libcudnn.so.9",
            "libcudnn.so.9",
            "b" * 64,
            1,
            3,
            8192,
            started + 10_000_000_000,
            started - 1_000_000,
        )
        cuda_runtime.assert_runtime_library_state_stable(
            (), (lazy,), observation_started_ns=started, label="CUDA"
        )

    def test_runtime_stability_rejects_lazy_library_with_post_start_ctime(self):
        started = time.time_ns()
        lazy = (
            "/tmp/libcudnn.so.9",
            "libcudnn.so.9",
            "b" * 64,
            1,
            3,
            8192,
            started - 1_000_000,
            started + 1,
        )
        with self.assertRaisesRegex(CaptureContractError, "changed after canonical observation began"):
            cuda_runtime.assert_runtime_library_state_stable(
                (), (lazy,), observation_started_ns=started, label="CUDA"
            )

    def test_round39_baseline_is_not_misclassified_as_lazy_load(self):
        source = __import__("inspect").getsource(
            round39.HuggingFacePyTorchBackend._begin_runtime_library_stability_window
        )
        self.assertIn("loaded_cpu_runtime_library_snapshot()", source)
        self.assertNotIn("assert_runtime_library_state_stable", source)


if __name__ == "__main__":
    unittest.main()

"""Round 53 regressions for startup cleanup and runtime-library stability."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import qsol_geo_reason.capture_backend_isolated as isolated
import qsol_geo_reason.capture_backend_round39 as round39
import qsol_geo_reason.capture_cuda_runtime as cuda_runtime
from qsol_geo_reason.capture_common import CaptureContractError


class CaptureRound53RegressionTests(unittest.TestCase):
    def test_begin_failure_retains_receipt_when_restore_fails(self):
        class FakeBackend:
            def __init__(self):
                self._observation_consumed = False
                self._observation_active = False
                self._observation_ambient_process_state = None
                self._torch = object()
                self._device = "cpu"
                self._applied_seed = 7
                self.fail_restore = True

            def _snapshot_torch_process_state(self, _torch, _device):
                return {"rng": "ambient"}

            def _restore_torch_process_state(self, _torch, _ambient):
                if self.fail_restore:
                    raise CaptureContractError("synthetic restore failure")

            def _force_canonical_determinism_policy(self):
                return None

            def _assert_live_state_authentication(self):
                return None

        backend = FakeBackend()
        with mock.patch.object(
            isolated,
            "_seed_capture_generators",
            side_effect=CaptureContractError("synthetic startup failure"),
        ):
            with self.assertRaisesRegex(CaptureContractError, "synthetic restore failure"):
                isolated.HuggingFacePyTorchBackend.begin_observation(backend)

        self.assertTrue(backend._observation_active)
        self.assertEqual(backend._observation_ambient_process_state, {"rng": "ambient"})
        backend.fail_restore = False
        isolated.HuggingFacePyTorchBackend.end_observation(backend)
        self.assertFalse(backend._observation_active)
        self.assertIsNone(backend._observation_ambient_process_state)

    def test_begin_failure_retries_interrupting_restore_before_dropping_receipt(self):
        class FakeBackend:
            def __init__(self):
                self._observation_consumed = False
                self._observation_active = False
                self._observation_ambient_process_state = None
                self._torch = object()
                self._device = "cpu"
                self._applied_seed = 7
                self.restore_calls = 0

            def _snapshot_torch_process_state(self, _torch, _device):
                return {"threads": 4}

            def _restore_torch_process_state(self, _torch, _ambient):
                self.restore_calls += 1
                if self.restore_calls == 1:
                    raise KeyboardInterrupt

            def _force_canonical_determinism_policy(self):
                return None

            def _assert_live_state_authentication(self):
                return None

        backend = FakeBackend()
        with mock.patch.object(
            isolated,
            "_seed_capture_generators",
            side_effect=CaptureContractError("synthetic startup failure"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                isolated.HuggingFacePyTorchBackend.begin_observation(backend)

        self.assertEqual(backend.restore_calls, 2)
        self.assertFalse(backend._observation_active)
        self.assertIsNone(backend._observation_ambient_process_state)

    def test_mapped_digest_rejects_in_place_mutation_during_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libcuda.so.1"
            library.write_bytes(b"runtime-old")
            identity = library.stat()
            mapped = cuda_runtime._MappedLibrary(
                path=library,
                content_path=library,
                device=identity.st_dev,
                inode=identity.st_ino,
            )
            original_hash = cuda_runtime._sha256_fd
            calls = 0

            def mutate_after_hash(fd, where):
                nonlocal calls
                digest = original_hash(fd, where)
                calls += 1
                if calls == 1:
                    with library.open("r+b") as handle:
                        handle.seek(0)
                        handle.write(b"runtime-new")
                        handle.flush()
                        os.fsync(handle.fileno())
                return digest

            with mock.patch.object(cuda_runtime, "_sha256_fd", side_effect=mutate_after_hash):
                with self.assertRaisesRegex(CaptureContractError, "changed while hashing"):
                    cuda_runtime._cuda_runtime_library_receipt([mapped])

    def test_runtime_stability_rejects_write_restore_aba_by_ctime(self):
        started = time.time_ns()
        before = ((
            "/tmp/libcuda.so.1",
            "libcuda.so.1",
            "a" * 64,
            1,
            2,
            4096,
            started - 10_000,
            started - 10_000,
        ),)
        after = ((
            "/tmp/libcuda.so.1",
            "libcuda.so.1",
            "a" * 64,
            1,
            2,
            4096,
            started - 10_000,
            started + 1,
        ),)
        with self.assertRaisesRegex(CaptureContractError, "changed after canonical observation began"):
            cuda_runtime.assert_runtime_library_state_stable(
                before, after, observation_started_ns=started, label="CUDA"
            )

    def test_runtime_stability_allows_unchanged_lazy_load(self):
        started = time.time_ns()
        after = ((
            "/tmp/libcudnn.so.9",
            "libcudnn.so.9",
            "b" * 64,
            1,
            3,
            8192,
            started - 20_000,
            started - 20_000,
        ),)
        cuda_runtime.assert_runtime_library_state_stable(
            (), after, observation_started_ns=started, label="CUDA"
        )

    def test_round39_binds_runtime_state_before_capture_and_rechecks_metadata(self):
        hook_source = __import__("inspect").getsource(
            round39.HuggingFacePyTorchBackend._begin_runtime_library_stability_window
        )
        begin_source = __import__("inspect").getsource(
            round39.HuggingFacePyTorchBackend.begin_observation
        )
        metadata_source = __import__("inspect").getsource(round39.HuggingFacePyTorchBackend.metadata)
        self.assertIn("loaded_cpu_runtime_library_snapshot()", hook_source)
        self.assertIn("_remember_runtime_library_baseline", hook_source)
        self.assertIn("self._enter_exclusive_python_thread_boundary()", begin_source)
        self.assertIn("self._begin_runtime_library_stability_window()", begin_source)
        self.assertIn("assert_runtime_library_state_stable", metadata_source)
        self.assertIn("_recall_runtime_library_baseline", metadata_source)


if __name__ == "__main__":
    unittest.main()

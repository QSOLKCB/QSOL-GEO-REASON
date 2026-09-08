"""Round 52 regressions for unprivileged mapped receipts and retryable cleanup."""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path

import qsol_geo_reason.capture_backend_round46 as round46
import qsol_geo_reason.capture_cuda_runtime as cuda_runtime
from qsol_geo_reason.capture_backend_isolated import (
    HuggingFacePyTorchBackend as IsolatedBackend,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_cuda_runtime import (
    _MappedLibrary,
    _cuda_runtime_library_receipt,
)


class CaptureRound52RegressionTests(unittest.TestCase):
    def test_linux_mapping_enumerator_does_not_require_proc_map_files(self):
        source = inspect.getsource(cuda_runtime._linux_loaded_library_mappings)
        self.assertNotIn('Path("/proc/self/map_files")', source)
        self.assertIn("content_path=Path(candidate)", source)

    def test_mapped_receipt_authenticates_path_descriptor_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libcuda.so.1"
            library.write_bytes(b"mapped-runtime")
            identity = library.stat()
            mapped = _MappedLibrary(
                path=library,
                content_path=library,
                device=identity.st_dev,
                inode=identity.st_ino,
            )
            count, receipt = _cuda_runtime_library_receipt([mapped])
            self.assertEqual(count, 1)
            self.assertEqual(len(receipt), 64)

            replacement = root / "replacement"
            replacement.write_bytes(b"different-runtime")
            os.replace(replacement, library)
            with self.assertRaisesRegex(CaptureContractError, "identity changed"):
                _cuda_runtime_library_receipt([mapped])

    def test_round46_finish_retains_vault_ownership_during_cleanup(self):
        class Instance:
            pass

        instance = Instance()
        observed = {"owned": False}

        class Stage:
            def cleanup(self):
                try:
                    round46._remember_round46_stage(instance, self)
                except CaptureContractError as exc:
                    observed["owned"] = "already installed" in str(exc)
                return None

        round46._remember_round46_stage(instance, Stage())
        round46._finish_round46_stage(instance)
        self.assertTrue(observed["owned"])

    def test_round46_cleanup_defers_interrupt_until_redirect_restored(self):
        class InterruptingPatch:
            def __init__(self):
                self.calls = 0

            def restore(self):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt

        class Tempdir:
            def __init__(self):
                self.calls = 0

            def cleanup(self):
                self.calls += 1

        class Instance:
            pass

        patch = InterruptingPatch()
        tempdir = Tempdir()
        stage = round46._AuthenticatedLoadStage(tempdir, [patch])
        instance = Instance()
        round46._remember_round46_stage(instance, stage)
        with self.assertRaises(KeyboardInterrupt):
            round46._finish_round46_stage(instance)
        self.assertEqual(patch.calls, 2)
        self.assertEqual(tempdir.calls, 1)
        self.assertEqual(stage.patches, [])
        round46._finish_round46_stage(instance)

    def test_end_observation_retries_interrupt_before_dropping_receipt(self):
        class FakeBackend:
            def __init__(self):
                self._observation_active = True
                self._observation_ambient_process_state = {"rng": "ambient"}
                self._torch = object()
                self.calls = 0

            def _restore_torch_process_state(self, _torch, ambient):
                self.calls += 1
                self.seen = ambient
                if self.calls == 1:
                    raise KeyboardInterrupt

        backend = FakeBackend()
        with self.assertRaises(KeyboardInterrupt):
            IsolatedBackend.end_observation(backend)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(backend.seen, {"rng": "ambient"})
        self.assertFalse(backend._observation_active)
        self.assertIsNone(backend._observation_ambient_process_state)

    def test_end_observation_keeps_receipt_after_restore_failure(self):
        class FakeBackend:
            def __init__(self):
                self._observation_active = True
                self._observation_ambient_process_state = {"threads": 8}
                self._torch = object()
                self.fail = True

            def _restore_torch_process_state(self, _torch, _ambient):
                if self.fail:
                    raise CaptureContractError("synthetic restore failure")

        backend = FakeBackend()
        with self.assertRaisesRegex(CaptureContractError, "synthetic restore failure"):
            IsolatedBackend.end_observation(backend)
        self.assertTrue(backend._observation_active)
        self.assertEqual(backend._observation_ambient_process_state, {"threads": 8})

        backend.fail = False
        IsolatedBackend.end_observation(backend)
        self.assertFalse(backend._observation_active)
        self.assertIsNone(backend._observation_ambient_process_state)


if __name__ == "__main__":
    unittest.main()

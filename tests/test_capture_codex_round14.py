from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend, _snapshot_file_hashes


class FakeThreadTorch:
    def __init__(self, intra: int = 8, inter: int = 2):
        self.intra = intra
        self.inter = inter
        self.backends = SimpleNamespace(mkldnn=None)

    def get_num_threads(self):
        return self.intra

    def get_num_interop_threads(self):
        return self.inter

    def set_num_threads(self, value):
        self.intra = value

    def set_num_interop_threads(self, value):
        self.inter = value


class CaptureRound14RegressionTests(unittest.TestCase):
    def test_standard_hf_snapshot_symlink_is_dereferenced_and_hashed(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob_dir = root / "blobs"
            blob_dir.mkdir()
            blob = blob_dir / "deadbeef"
            payload = b"canonical model bytes"
            blob.write_bytes(payload)

            snapshot = root / revision
            snapshot.mkdir()
            (snapshot / "config.json").symlink_to(blob)

            hashes = _snapshot_file_hashes(snapshot, revision, "model")
            self.assertEqual(
                hashes,
                {"config.json": hashlib.sha256(payload).hexdigest()},
            )

    def test_broken_snapshot_symlink_is_rejected(self):
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / revision
            snapshot.mkdir()
            (snapshot / "config.json").symlink_to(Path(tmp) / "missing-blob")
            with self.assertRaisesRegex(CaptureContractError, "unreadable or broken artifact"):
                _snapshot_file_hashes(snapshot, revision, "model")

    def test_cpu_thread_policy_is_restored_to_construction_snapshot(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeThreadTorch(8, 2)
        backend._canonical_cpu_thread_policy = backend._cpu_thread_policy_state()
        backend._last_cpu_thread_policy = None

        backend._torch.intra = 1
        backend._torch.inter = 1
        backend._force_cpu_thread_policy()
        self.assertEqual(
            backend._last_cpu_thread_policy,
            {"torch_num_threads": 8, "torch_num_interop_threads": 2},
        )

    def test_cpu_thread_policy_rejects_post_forward_drift(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = FakeThreadTorch(8, 2)
        backend._canonical_cpu_thread_policy = backend._cpu_thread_policy_state()
        backend._torch.intra = 4
        with self.assertRaisesRegex(CaptureContractError, "CPU thread policy drifted"):
            backend._assert_cpu_thread_policy()

    def test_each_cpu_forward_reasserts_then_rechecks_thread_policy(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)
        forward = source.index("self._base_model(")
        self.assertLess(source.index("self._force_cpu_thread_policy()"), forward)
        self.assertGreater(source.index("self._assert_cpu_thread_policy()"), forward)

    def test_metadata_uses_last_verified_thread_policy(self):
        source = inspect.getsource(HuggingFacePyTorchBackend.metadata)
        self.assertIn("self._last_cpu_thread_policy", source)
        self.assertIn('cpu_hardware["torch_num_threads"]', source)
        self.assertIn('cpu_hardware["torch_num_interop_threads"]', source)


if __name__ == "__main__":
    unittest.main()

"""Round 51 regressions for interrupt ownership and mapped runtime receipts."""
from __future__ import annotations

import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_backend_round46 as round46
from qsol_geo_reason.capture_backend_production import (
    HuggingFacePyTorchBackend as ProductionBackend,
)
from qsol_geo_reason.capture_cpu_runtime import _cpu_runtime_library_receipt
from qsol_geo_reason.capture_cuda_runtime import (
    _MappedLibrary,
    _cuda_runtime_library_receipt,
)
from qsol_geo_reason.capture_common import CaptureContractError


class CaptureRound51RegressionTests(unittest.TestCase):
    def test_loader_stage_restores_if_interrupt_lands_during_vault_handoff(self):
        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class Instance:
            pass

        module = types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )
        tokenizer_descriptor = vars(AutoTokenizer)["from_pretrained"]
        model_descriptor = vars(AutoModelForCausalLM)["from_pretrained"]
        instance = Instance()
        real_remember = round46._remember_round46_stage

        def remember_then_interrupt(inst, stage):
            real_remember(inst, stage)
            raise KeyboardInterrupt

        tempdir = tempfile.TemporaryDirectory(prefix="round51-stage-")
        root = Path(tempdir.name)
        for name in ("model-source", "tokenizer-source", "model-stage", "tokenizer-stage"):
            (root / name).mkdir()
        try:
            with patch.object(
                round46,
                "_remember_round46_stage",
                side_effect=remember_then_interrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    round46._install_round46_stage(
                        instance,
                        module,
                        tempdir=tempdir,
                        model_source=root / "model-source",
                        tokenizer_source=root / "tokenizer-source",
                        model_stage=root / "model-stage",
                        tokenizer_stage=root / "tokenizer-stage",
                    )
            self.assertIs(vars(AutoTokenizer)["from_pretrained"], tokenizer_descriptor)
            self.assertIs(vars(AutoModelForCausalLM)["from_pretrained"], model_descriptor)
        finally:
            # The constructor's outer finally performs this second, idempotent vault
            # cleanup in production if the handoff stored the stage before interrupting.
            round46._finish_round46_stage(instance)
            tempdir.cleanup()

    def test_thread_restore_retries_interrupt_before_clearing_state(self):
        class Owner:
            def __init__(self):
                object.__setattr__(self, "interrupt_next_restore", False)
                object.__setattr__(self, "start", "blocked")

            def __setattr__(self, name, value):
                if name == "start" and self.interrupt_next_restore:
                    object.__setattr__(self, "interrupt_next_restore", False)
                    raise KeyboardInterrupt
                object.__setattr__(self, name, value)

        backend = object.__new__(ProductionBackend)
        owner = Owner()
        original = object()
        owner.interrupt_next_restore = True
        backend._exclusive_thread_boundary_state = {
            "lock": threading.RLock(),
            "patches": [(owner, "start", original)],
        }

        with self.assertRaises(KeyboardInterrupt):
            backend._leave_exclusive_python_thread_boundary()

        self.assertIs(owner.start, original)
        self.assertIsNone(backend._exclusive_thread_boundary_state)

    def test_linux_mapped_cuda_receipt_hashes_mapped_inode_not_replaced_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_path = root / "libcuda.so.1"
            mapped_handle = root / "mapped-handle"
            control_dir = root / "control"
            control_dir.mkdir()
            control = control_dir / "libcuda.so.1"

            live_path.write_bytes(b"mapped-old-bytes")
            os.link(live_path, mapped_handle)
            mapped_stat = mapped_handle.stat()
            control.write_bytes(b"mapped-old-bytes")

            replacement = root / "replacement"
            replacement.write_bytes(b"new-path-bytes")
            os.replace(replacement, live_path)

            mapped = _MappedLibrary(
                path=live_path,
                content_path=mapped_handle,
                device=mapped_stat.st_dev,
                inode=mapped_stat.st_ino,
            )
            count, mapped_receipt = _cuda_runtime_library_receipt([mapped])
            _, control_receipt = _cuda_runtime_library_receipt([control])
            _, pathname_receipt = _cuda_runtime_library_receipt([live_path])

            self.assertEqual(count, 1)
            self.assertEqual(mapped_receipt, control_receipt)
            self.assertNotEqual(mapped_receipt, pathname_receipt)

            wrong_identity = _MappedLibrary(
                path=live_path,
                content_path=live_path,
                device=mapped_stat.st_dev,
                inode=mapped_stat.st_ino,
            )
            with self.assertRaisesRegex(CaptureContractError, "identity changed"):
                _cuda_runtime_library_receipt([wrong_identity])

    def test_cpu_runtime_receipt_uses_same_mapped_identity_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_path = root / "libgomp.so.1"
            mapped_handle = root / "mapped-cpu-handle"
            control_dir = root / "control"
            control_dir.mkdir()
            control = control_dir / "libgomp.so.1"

            live_path.write_bytes(b"old-openmp")
            os.link(live_path, mapped_handle)
            mapped_stat = mapped_handle.stat()
            control.write_bytes(b"old-openmp")
            replacement = root / "replacement"
            replacement.write_bytes(b"new-openmp")
            os.replace(replacement, live_path)

            mapped = _MappedLibrary(
                path=live_path,
                content_path=mapped_handle,
                device=mapped_stat.st_dev,
                inode=mapped_stat.st_ino,
            )
            _, mapped_receipt = _cpu_runtime_library_receipt([mapped])
            _, control_receipt = _cpu_runtime_library_receipt([control])
            _, pathname_receipt = _cpu_runtime_library_receipt([live_path])
            self.assertEqual(mapped_receipt, control_receipt)
            self.assertNotEqual(mapped_receipt, pathname_receipt)


if __name__ == "__main__":
    unittest.main()

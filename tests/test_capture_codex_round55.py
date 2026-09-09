"""Round 55 regressions for constructor cleanup ownership and CLI decoding."""
from __future__ import annotations

import contextlib
import inspect
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qsol_geo_reason.capture_backend_production as production
import qsol_geo_reason.capture_cli as capture_cli
from qsol_geo_reason.capture_common import CaptureContractError


class CaptureRound55RegressionTests(unittest.TestCase):
    def tearDown(self):
        # Synthetic recovery tests always clean the class-level owner. Keep a failed
        # assertion from leaking its fake owner into later regression modules.
        production.HuggingFacePyTorchBackend._construction_recovery_owner = None

    def _bare_construction_backend(self):
        backend = object.__new__(production.HuggingFacePyTorchBackend)
        backend._construction_ambient_process_state = {"rng": "ambient"}
        backend._construction_process_torch = object()
        backend._exclusive_thread_boundary_state = {"patches": ("synthetic",)}
        backend.restore_should_fail = True
        backend.restore_calls = 0
        backend.leave_calls = 0

        def restore(_torch, _ambient):
            backend.restore_calls += 1
            if backend.restore_should_fail:
                raise CaptureContractError("synthetic constructor restore failure")

        def leave():
            backend.leave_calls += 1
            backend._exclusive_thread_boundary_state = None

        backend._restore_torch_process_state = restore
        backend._leave_exclusive_python_thread_boundary = leave
        return backend

    def test_constructor_restore_failure_retains_receipts_until_retry_succeeds(self):
        backend = self._bare_construction_backend()
        production.HuggingFacePyTorchBackend._construction_recovery_owner = backend

        with self.assertRaisesRegex(
            CaptureContractError, "synthetic constructor restore failure"
        ):
            production.HuggingFacePyTorchBackend._retry_failed_construction_restoration()

        self.assertIs(
            production.HuggingFacePyTorchBackend._construction_recovery_owner, backend
        )
        self.assertEqual(backend._construction_ambient_process_state, {"rng": "ambient"})
        self.assertIsNotNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(backend.leave_calls, 0)

        backend.restore_should_fail = False
        production.HuggingFacePyTorchBackend._retry_failed_construction_restoration()

        self.assertIsNone(production.HuggingFacePyTorchBackend._construction_recovery_owner)
        self.assertIsNone(backend._construction_ambient_process_state)
        self.assertIsNone(backend._construction_process_torch)
        self.assertIsNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(backend.leave_calls, 1)

    def test_constructor_restore_interrupt_completes_cleanup_before_propagating(self):
        backend = self._bare_construction_backend()
        backend.restore_should_fail = False

        def interrupt_once(_torch, _ambient):
            backend.restore_calls += 1
            if backend.restore_calls == 1:
                raise KeyboardInterrupt

        backend._restore_torch_process_state = interrupt_once
        production.HuggingFacePyTorchBackend._construction_recovery_owner = backend

        with self.assertRaises(KeyboardInterrupt):
            production.HuggingFacePyTorchBackend._retry_failed_construction_restoration()

        self.assertEqual(backend.restore_calls, 2)
        self.assertEqual(backend.leave_calls, 1)
        self.assertIsNone(backend._construction_ambient_process_state)
        self.assertIsNone(backend._exclusive_thread_boundary_state)
        self.assertIsNone(production.HuggingFacePyTorchBackend._construction_recovery_owner)

    def test_constructor_source_does_not_release_boundary_with_live_ambient_receipt(self):
        source = inspect.getsource(production.HuggingFacePyTorchBackend.__init__)
        self.assertIn("_construction_recovery_owner = self", source)
        self.assertIn(
            'if getattr(self, "_construction_ambient_process_state", None) is None:',
            source,
        )

    def test_invalid_utf8_is_an_argparse_error_in_every_cli_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_bytes(b'{"invalid":"\xff"}')
            modes = (
                [],
                ["--validate-only"],
                ["--prepare-tree-receipts"],
                ["--output-dir", str(root / "capture")],
            )
            for mode in modes:
                with self.subTest(mode=mode):
                    stderr = io.StringIO()
                    argv = ["qsol-geo-capture", str(request), *mode]
                    with (
                        mock.patch.object(sys, "argv", argv),
                        contextlib.redirect_stderr(stderr),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        capture_cli.main()
                    self.assertEqual(raised.exception.code, 2)
                    diagnostic = stderr.getvalue()
                    self.assertIn("error:", diagnostic)
                    self.assertIn("utf-8", diagnostic.lower())
                    self.assertNotIn("Traceback", diagnostic)


if __name__ == "__main__":
    unittest.main()

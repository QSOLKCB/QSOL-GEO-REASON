"""Round 50 regressions for interrupt-safe boundaries and durable publication."""
from __future__ import annotations

import _thread
import threading
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_backend_production as production
import qsol_geo_reason.capture_backend_round46 as round46
import qsol_geo_reason.capture_publish as capture_publish
from qsol_geo_reason.capture_backend_round45 import (
    HuggingFacePyTorchBackend as Round45Backend,
)
from qsol_geo_reason.capture_backend_production import (
    HuggingFacePyTorchBackend as ProductionBackend,
)
from qsol_geo_reason.capture_common import CaptureContractError


class _InterruptOnExitLock:
    def __init__(self, backend):
        self.backend = backend
        self.state_was_recorded = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.state_was_recorded = (
            getattr(self.backend, "_exclusive_thread_boundary_state", None) is not None
        )
        raise KeyboardInterrupt


class CaptureRound50RegressionTests(unittest.TestCase):
    def test_thread_patch_state_is_recorded_before_lock_exit(self):
        backend = object.__new__(ProductionBackend)
        backend._exclusive_thread_boundary_state = None
        lock = _InterruptOnExitLock(backend)
        original_thread_start = threading.Thread.start
        original_raw_start = _thread.start_new_thread
        current_ident = threading.get_ident()

        with (
            patch.object(production.threading, "_active_limbo_lock", lock),
            patch.object(production.threading, "_active", {current_ident: object()}),
            patch.object(production.threading, "_limbo", {}),
            patch.object(production, "_assert_exclusive_interpreter_thread", return_value=None),
        ):
            with self.assertRaises(KeyboardInterrupt):
                backend._enter_exclusive_python_thread_boundary()

        self.assertTrue(lock.state_was_recorded)
        self.assertIsNone(backend._exclusive_thread_boundary_state)
        self.assertIs(threading.Thread.start, original_thread_start)
        self.assertIs(_thread.start_new_thread, original_raw_start)

    def test_loader_redirect_installation_restores_on_keyboard_interrupt(self):
        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        module = types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )
        tokenizer_descriptor = vars(AutoTokenizer)["from_pretrained"]
        model_descriptor = vars(AutoModelForCausalLM)["from_pretrained"]
        real_patch = round46._LoaderPatch
        calls = 0

        def interrupt_second_patch(owner, replacement):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return real_patch(owner, replacement)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model-source", "tokenizer-source", "model-stage", "tokenizer-stage"):
                (root / name).mkdir()
            with patch.object(round46, "_LoaderPatch", side_effect=interrupt_second_patch):
                with self.assertRaises(KeyboardInterrupt):
                    round46._install_authenticated_loader_redirects(
                        module,
                        model_source=root / "model-source",
                        tokenizer_source=root / "tokenizer-source",
                        model_stage=root / "model-stage",
                        tokenizer_stage=root / "tokenizer-stage",
                    )

        self.assertIs(vars(AutoTokenizer)["from_pretrained"], tokenizer_descriptor)
        self.assertIs(vars(AutoModelForCausalLM)["from_pretrained"], model_descriptor)

    def test_fresh_import_boundary_includes_native_loading_packages(self):
        for root, loaded_name in (
            ("safetensors", "safetensors.torch"),
            ("tokenizers", "tokenizers.tokenizers"),
            ("huggingface_hub", "huggingface_hub.file_download"),
        ):
            module = types.ModuleType(loaded_name)
            with self.subTest(root=root):
                with self.assertRaisesRegex(CaptureContractError, root.split("_")[0]):
                    Round45Backend._assert_pristine_mps_import_state(
                        "cpu", modules={loaded_name: module}
                    )

        Round45Backend._assert_pristine_mps_import_state("cpu", modules={})

    def test_new_publication_ancestors_are_fsynced_as_they_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "level-1" / "level-2" / "level-3"
            synced: list[Path] = []

            with patch.object(
                capture_publish,
                "_fsync_directory",
                side_effect=lambda path: synced.append(Path(path)),
            ):
                capture_publish._ensure_parent_directory_durable(parent)

            self.assertTrue(parent.is_dir())
            self.assertEqual(
                synced,
                [root, root / "level-1", root / "level-1" / "level-2"],
            )


if __name__ == "__main__":
    unittest.main()

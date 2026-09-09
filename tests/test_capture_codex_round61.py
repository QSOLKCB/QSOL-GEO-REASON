from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from qsol_geo_reason import capture
from qsol_geo_reason import capture_backend_round56 as round56
from qsol_geo_reason import capture_backend_round61 as round61
from qsol_geo_reason import capture_backend_round65 as round65


class Round61PreloadStabilityTests(unittest.TestCase):
    def test_public_backend_retains_round61_preload_constructor_under_later_layers(self):
        self.assertIs(capture.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(capture.HuggingFacePyTorchBackend, round61.HuggingFacePyTorchBackend)
        self.assertIs(capture.HuggingFacePyTorchBackend, round65.HuggingFacePyTorchBackend)
        public_init = round56.HuggingFacePyTorchBackend.__init__
        self.assertIs(public_init, round65.HuggingFacePyTorchBackend.__init__)
        closure_values = tuple(cell.cell_contents for cell in public_init.__closure__ or ())
        self.assertTrue(
            any(
                callable(value) and getattr(value, "__module__", None) == round61.__name__
                for value in closure_values
            ),
            "Round-65 constructor must closure-bind the earlier Round-61 preload constructor chain",
        )

    def test_preload_stability_receipt_detects_write_restore_aba(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "torch"
            root.mkdir()
            init = root / "__init__.py"
            lazy = root / "lazy.py"
            init.write_text("from . import lazy\n", encoding="utf-8")
            original = b"VALUE = 'trusted'\n"
            lazy.write_bytes(original)

            before = round61._torch_package_stability_from_root_round61(root)

            lazy.write_bytes(b"VALUE = 'hostile'\n")
            restored = root / ".lazy.py.restored"
            restored.write_bytes(original)
            os.replace(restored, lazy)

            self.assertEqual(lazy.read_bytes(), original)
            after = round61._torch_package_stability_from_root_round61(root)
            self.assertNotEqual(after, before)

    def test_constructor_binds_original_init_in_closure(self):
        init = round61.HuggingFacePyTorchBackend.__init__
        self.assertNotIn("_ORIGINAL_INIT", init.__globals__)
        self.assertIsNotNone(init.__closure__)
        closure_values = tuple(cell.cell_contents for cell in init.__closure__ or ())
        self.assertTrue(any(callable(value) for value in closure_values))


if __name__ == "__main__":
    unittest.main()

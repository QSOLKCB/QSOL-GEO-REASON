"""Round 45 regressions for fresh runtimes and CPU pooling provenance."""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_backend_round45 as round45
from qsol_geo_reason.capture_backend_round45 import (
    HuggingFacePyTorchBackend as Round45Backend,
    _assert_transformers_instance_loader_source_bound,
    _is_canonical_snapshot_path_round45,
    _validate_backend_metadata_round45,
)
from qsol_geo_reason.capture_common import CaptureContractError, _PRODUCTION_BACKEND_KEYS


class CaptureRound45RegressionTests(unittest.TestCase):
    def test_every_production_lane_requires_fresh_real_torch_and_transformers_imports(self):
        for device in ("cpu", "cuda:0"):
            Round45Backend._assert_pristine_mps_import_state(device, modules={})
            with self.assertRaisesRegex(CaptureContractError, "fresh PyTorch/Transformers"):
                Round45Backend._assert_pristine_mps_import_state(
                    device,
                    modules={"torch.library": types.ModuleType("torch.library")},
                )
            with self.assertRaisesRegex(CaptureContractError, "fresh PyTorch/Transformers"):
                Round45Backend._assert_pristine_mps_import_state(
                    device,
                    modules={
                        "transformers.models.synthetic": types.ModuleType(
                            "transformers.models.synthetic"
                        )
                    },
                )

        # Preserve MPS's older, stricter pre-import contract and diagnostic.
        with self.assertRaisesRegex(CaptureContractError, "fresh PyTorch import boundary"):
            Round45Backend._assert_pristine_mps_import_state(
                "mps", modules={"torch": object()}
            )

        # Plain test doubles are not executable preloaded runtimes on CPU/CUDA.
        Round45Backend._assert_pristine_mps_import_state(
            "cpu", modules={"torch": object(), "transformers": object()}
        )

    def test_concrete_transformers_loader_is_bound_to_package_source(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "transformers"
            package.mkdir()
            init_file = package / "__init__.py"
            init_file.write_text("# synthetic package\n", encoding="utf-8")
            source_file = package / "synthetic.py"
            source = (
                "class ConcreteModel:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *args, **kwargs):\n"
                "        return cls()\n"
            )
            source_file.write_text(source, encoding="utf-8")
            namespace = {"__name__": "transformers.synthetic"}
            exec(
                compile(source, str(source_file), "exec", dont_inherit=True),
                namespace,
            )
            fake_transformers = types.SimpleNamespace(__file__=str(init_file))
            instance = namespace["ConcreteModel"]()
            _assert_transformers_instance_loader_source_bound(
                fake_transformers, instance, "model"
            )

            replacement_namespace = {"__name__": "transformers.synthetic"}
            exec(
                compile(
                    "def replacement(cls, *args, **kwargs):\n    return cls()\n",
                    "<instrumentation>",
                    "exec",
                    dont_inherit=True,
                ),
                replacement_namespace,
            )
            namespace["ConcreteModel"].from_pretrained = classmethod(
                replacement_namespace["replacement"]
            )
            with self.assertRaisesRegex(CaptureContractError, "receipt-backed"):
                _assert_transformers_instance_loader_source_bound(
                    fake_transformers, instance, "model"
                )

    def test_cpu_processor_identity_is_required_for_accelerator_pooling_too(self):
        observed = {
            "cpu_processor": "AMD Ryzen 9 5950X 16-Core Processor",
            "cpu_flush_denormal": False,
        }
        with patch.object(round45, "_ORIGINAL_VALIDATE_BACKEND_METADATA", return_value=None):
            _validate_backend_metadata_round45(observed, {}, "OBSERVATION")
            for missing in (None, "x86_64", "unknown"):
                malformed = dict(observed, cpu_processor=missing)
                with self.assertRaisesRegex(CaptureContractError, "concrete processor"):
                    _validate_backend_metadata_round45(malformed, {}, "OBSERVATION")

    def test_lone_surrogate_snapshot_path_fails_as_contract_error(self):
        bad_path = "weights/\ud800.safetensors"
        self.assertFalse(_is_canonical_snapshot_path_round45(bad_path))
        observed = {
            "model_snapshot_file_sha256": {bad_path: "a" * 64},
            "model_snapshot_file_count": 1,
            "model_snapshot_receipt_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(CaptureContractError, "artifact hashes are malformed"):
            round45._capture_provenance._validate_snapshot_receipt(observed, "model")

    def test_cpu_flush_denormal_is_forced_to_gradual_underflow(self):
        calls: list[bool] = []

        def set_flush_denormal(value: bool) -> bool:
            calls.append(value)
            return True

        backend = object.__new__(Round45Backend)
        backend._torch = types.SimpleNamespace(set_flush_denormal=set_flush_denormal)
        backend._force_round45_cpu_flush_denormal_policy()
        self.assertEqual(calls, [False])
        self.assertIs(backend._canonical_cpu_flush_denormal, False)

        backend._torch = types.SimpleNamespace(set_flush_denormal=lambda _value: False)
        with self.assertRaisesRegex(CaptureContractError, "gradual-underflow"):
            backend._force_round45_cpu_flush_denormal_policy()

    def test_cpu_flush_denormal_policy_is_required_and_content_bound(self):
        self.assertIn("cpu_flush_denormal", _PRODUCTION_BACKEND_KEYS)
        observed = {
            "cpu_processor": "AMD Ryzen 9 5950X 16-Core Processor",
            "cpu_flush_denormal": False,
        }
        with patch.object(round45, "_ORIGINAL_VALIDATE_BACKEND_METADATA", return_value=None):
            _validate_backend_metadata_round45(observed, {}, "OBSERVATION")
            with self.assertRaisesRegex(CaptureContractError, "cpu_flush_denormal=false"):
                _validate_backend_metadata_round45(
                    dict(observed, cpu_flush_denormal=True), {}, "OBSERVATION"
                )
            with self.assertRaisesRegex(CaptureContractError, "cpu_flush_denormal=false"):
                _validate_backend_metadata_round45(
                    {"cpu_processor": observed["cpu_processor"]}, {}, "OBSERVATION"
                )


if __name__ == "__main__":
    unittest.main()

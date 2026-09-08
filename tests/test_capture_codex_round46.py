"""Round 46 regressions for immutable loads and trusted runtime identity."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_hardware as hardware
from qsol_geo_reason.capture_backend_round46 import (
    HuggingFacePyTorchBackend as Round46Backend,
    _copy_authenticated_snapshot,
    _cpu_flush_denormal_state,
    _install_authenticated_loader_redirects,
    _trusted_linux_nvidia_driver_version,
)
from qsol_geo_reason.capture_common import CaptureContractError


class _Scalar:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _NormalProbe:
    def __init__(self, torch: "_DenormalTorch", tiny: float):
        self._torch = torch
        self._tiny = tiny

    def __mul__(self, value: float) -> _Scalar:
        result = self._tiny * value
        return _Scalar(0.0 if self._torch.flush_denormal else result)


class _DenormalTorch:
    float32 = object()

    def __init__(self, enabled: bool):
        self.flush_denormal = enabled
        self.num_threads = 4
        self.num_interop_threads = 2
        self.backends = types.SimpleNamespace(
            mkldnn=None,
            cuda=None,
            cudnn=None,
        )

    def finfo(self, _dtype: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(tiny=1.1754943508222875e-38)

    def tensor(self, values, *, dtype, device):
        self.assertEqualLike(values, [1.1754943508222875e-38])
        if dtype is not self.float32 or device != "cpu":
            raise AssertionError("unexpected probe tensor")
        return _NormalProbe(self, values[0])

    @staticmethod
    def assertEqualLike(actual, expected) -> None:
        if actual != expected:
            raise AssertionError((actual, expected))

    def set_flush_denormal(self, enabled: bool) -> bool:
        self.flush_denormal = enabled
        return True

    def get_num_threads(self) -> int:
        return self.num_threads

    def set_num_threads(self, value: int) -> None:
        self.num_threads = value

    def get_num_interop_threads(self) -> int:
        return self.num_interop_threads

    def set_num_interop_threads(self, value: int) -> None:
        self.num_interop_threads = value


class _AutoTokenizer:
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        del cls, args, kwargs
        return (Path(path) / "tokenizer.json").read_text(encoding="utf-8")


class _AutoModel:
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        del cls, args, kwargs
        return (Path(path) / "model.safetensors").read_bytes()


class CaptureRound46RegressionTests(unittest.TestCase):
    def test_denormal_state_is_snapshot_and_restored(self):
        torch = _DenormalTorch(enabled=True)
        self.assertIs(_cpu_flush_denormal_state(torch), True)
        ambient = Round46Backend._snapshot_torch_execution_policy_state(torch)
        self.assertIs(ambient["cpu_flush_denormal"], True)

        torch.set_flush_denormal(False)
        self.assertIs(_cpu_flush_denormal_state(torch), False)
        Round46Backend._restore_torch_execution_policy_state(torch, ambient)
        self.assertIs(torch.flush_denormal, True)
        self.assertIs(_cpu_flush_denormal_state(torch), True)

    def test_authenticated_private_stage_defeats_shared_cache_aba(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_source = root / "shared-model"
            tokenizer_source = root / "shared-tokenizer"
            model_source.mkdir()
            tokenizer_source.mkdir()
            model_good = b"authenticated-model-bytes"
            tokenizer_good = '{"authenticated":true}'
            (model_source / "model.safetensors").write_bytes(model_good)
            (tokenizer_source / "tokenizer.json").write_text(
                tokenizer_good, encoding="utf-8"
            )
            model_hashes = {
                "model.safetensors": hashlib.sha256(model_good).hexdigest()
            }
            tokenizer_hashes = {
                "tokenizer.json": hashlib.sha256(
                    tokenizer_good.encode("utf-8")
                ).hexdigest()
            }

            model_stage = root / "private" / "model" / ("a" * 40)
            tokenizer_stage = root / "private" / "tokenizer" / ("b" * 40)
            self.assertEqual(
                _copy_authenticated_snapshot(
                    model_source, model_stage, model_hashes, "model"
                ),
                model_hashes,
            )
            self.assertEqual(
                _copy_authenticated_snapshot(
                    tokenizer_source,
                    tokenizer_stage,
                    tokenizer_hashes,
                    "tokenizer",
                ),
                tokenizer_hashes,
            )

            fake_transformers = types.SimpleNamespace(
                AutoTokenizer=_AutoTokenizer,
                AutoModelForCausalLM=_AutoModel,
            )
            patches = _install_authenticated_loader_redirects(
                fake_transformers,
                model_source=model_source,
                tokenizer_source=tokenizer_source,
                model_stage=model_stage,
                tokenizer_stage=tokenizer_stage,
            )
            try:
                # Simulate another process substituting cache bytes after the trusted
                # copy is authenticated and before Transformers deserializes them.
                (model_source / "model.safetensors").write_bytes(b"substituted")
                (tokenizer_source / "tokenizer.json").write_text(
                    '{"substituted":true}', encoding="utf-8"
                )
                tokenizer = _AutoTokenizer.from_pretrained(
                    str(tokenizer_source),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = _AutoModel.from_pretrained(
                    str(model_source),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                self.assertEqual(tokenizer, tokenizer_good)
                self.assertEqual(model, model_good)
            finally:
                for loader_patch in reversed(patches):
                    loader_patch.restore()

    def test_private_stage_rejects_bytes_changed_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            path = source / "weights.safetensors"
            path.write_bytes(b"original")
            expected = {
                "weights.safetensors": hashlib.sha256(b"original").hexdigest()
            }
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(CaptureContractError, "changed while creating"):
                _copy_authenticated_snapshot(
                    source, root / "private" / "weights", expected, "model"
                )

    def test_nvidia_driver_probe_uses_loaded_kernel_module_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = root / "sys-version"
            procfs = root / "proc-version"
            sysfs.write_text("580.173.02\n", encoding="ascii")
            procfs.write_text("ignored", encoding="ascii")
            with patch("qsol_geo_reason.capture_backend_round46.platform.system", return_value="Linux"):
                self.assertEqual(
                    _trusted_linux_nvidia_driver_version(
                        sysfs_path=sysfs, proc_path=procfs
                    ),
                    "580.173.02",
                )

            sysfs.write_text("not-a-version", encoding="ascii")
            procfs.write_text(
                "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.173.02  test\n",
                encoding="ascii",
            )
            with patch("qsol_geo_reason.capture_backend_round46.platform.system", return_value="Linux"):
                self.assertEqual(
                    _trusted_linux_nvidia_driver_version(
                        sysfs_path=sysfs, proc_path=procfs
                    ),
                    "580.173.02",
                )

    def test_windows_cpu_identity_never_falls_back_to_environment_derived_platform_values(self):
        with (
            patch.object(hardware.platform, "system", return_value="Windows"),
            patch.object(hardware, "_windows_cpu_model", return_value=None),
            patch.object(
                hardware.platform,
                "processor",
                side_effect=AssertionError("platform.processor must not be consulted"),
            ),
            patch.object(
                hardware.platform,
                "uname",
                side_effect=AssertionError("platform.uname must not be consulted"),
            ),
        ):
            with self.assertRaisesRegex(CaptureContractError, "concrete processor"):
                hardware._concrete_cpu_identity()

    def test_manifest_schema_requires_denormal_receipt_and_unconditional_cpu_identity(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "capture-run-manifest.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        production = schema["$defs"]["backendObservedProduction"]
        torch_pattern = production["properties"]["torch_build_config"]["pattern"]
        cpu_runtime = (
            'QSOL_GEO_CPU_RUNTIME={"loaded_cpu_runtime_libraries":'
            '{"cpu_runtime_library_file_count":0,'
            '"cpu_runtime_library_receipt_sha256":"' + ("a" * 64) + '"}}\n'
        )
        self.assertIsNone(re.search(torch_pattern, "build\n" + cpu_runtime))
        self.assertIsNotNone(
            re.search(
                torch_pattern,
                "build\nQSOL_GEO_CPU_FLUSH_DENORMAL=false\n" + cpu_runtime,
            )
        )

        cpu_processor = production["properties"]["cpu_processor"]
        self.assertEqual(cpu_processor["type"], "string")
        self.assertNotIn("null", cpu_processor.get("type", []))
        self.assertIn("not", cpu_processor)


if __name__ == "__main__":
    unittest.main()

"""Round 41 regressions for ABI, CPU runtime, publication, and evidence boundaries."""
from __future__ import annotations

import ctypes
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from qsol_geo_reason.capture import (
    CaptureContractError,
    HuggingFacePyTorchBackend,
    execute_capture,
    write_capture_bundle,
)
from qsol_geo_reason.capture_cpu_runtime import _cpu_runtime_library_receipt
from qsol_geo_reason.capture_cuda_runtime import _configure_windows_module_api
from qsol_geo_reason.capture_runtime import _validate_torch_build_metadata
from test_capture import execute, fixture_request


_CPU_PREFIX = "QSOL_GEO_CPU_RUNTIME="


def _cpu_config(*, count: int = 0, receipt: str = "a" * 64) -> str:
    payload = json.dumps(
        {
            "loaded_cpu_runtime_libraries": {
                "cpu_runtime_library_file_count": count,
                "cpu_runtime_library_receipt_sha256": receipt,
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "SYNTHETIC\n" + _CPU_PREFIX + payload + "\n"


def _observed(config: str, device: str) -> dict[str, str]:
    return {
        "device": device,
        "torch_build_config": config,
        "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
    }


class _FunctionStub:
    pass


class CaptureRound41RegressionTests(unittest.TestCase):
    def test_windows_module_api_uses_pointer_sized_handles(self):
        kernel32 = SimpleNamespace(GetCurrentProcess=_FunctionStub())
        psapi = SimpleNamespace(
            EnumProcessModules=_FunctionStub(),
            GetModuleFileNameExW=_FunctionStub(),
        )
        _configure_windows_module_api(kernel32, psapi)

        self.assertIs(kernel32.GetCurrentProcess.restype, ctypes.c_void_p)
        self.assertIs(psapi.EnumProcessModules.argtypes[0], ctypes.c_void_p)
        self.assertIs(psapi.GetModuleFileNameExW.argtypes[0], ctypes.c_void_p)
        self.assertIs(psapi.GetModuleFileNameExW.argtypes[1], ctypes.c_void_p)
        self.assertEqual(
            ctypes.sizeof(psapi.GetModuleFileNameExW.argtypes[1]),
            ctypes.sizeof(ctypes.c_void_p),
        )
        self.assertIs(psapi.GetModuleFileNameExW.restype, ctypes.c_uint32)

    def test_cpu_runtime_receipt_content_binds_external_math_libraries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mkl = root / "libmkl_rt.so"
            omp = root / "libgomp.so.1"
            mkl.write_bytes(b"mkl-a")
            omp.write_bytes(b"omp-a")
            count_a, receipt_a = _cpu_runtime_library_receipt([mkl, omp])
            self.assertEqual(count_a, 2)
            self.assertRegex(receipt_a, r"^[0-9a-f]{64}$")

            mkl.write_bytes(b"mkl-b")
            count_b, receipt_b = _cpu_runtime_library_receipt([mkl, omp])
            self.assertEqual(count_b, 2)
            self.assertNotEqual(receipt_a, receipt_b)

    def test_cpu_build_config_requires_validated_runtime_receipt(self):
        with self.assertRaisesRegex(
            CaptureContractError,
            "missing the authenticated CPU runtime library receipt",
        ):
            _validate_torch_build_metadata(_observed("SYNTHETIC\n", "cpu"))

        _validate_torch_build_metadata(_observed(_cpu_config(), "cpu"))

        with self.assertRaisesRegex(CaptureContractError, "absent outside CPU"):
            _validate_torch_build_metadata(_observed(_cpu_config(), "mps"))

        with self.assertRaises(CaptureContractError):
            _validate_torch_build_metadata(
                _observed(_cpu_config(count=-1), "cpu")
            )
        with self.assertRaises(CaptureContractError):
            _validate_torch_build_metadata(
                _observed(_cpu_config(receipt="A" * 64), "cpu")
            )

    def test_parent_directory_fsync_failure_occurs_before_publish_rename(self):
        request = fixture_request()
        manifest, trajectory = execute(request)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bundle"
            parent = output_dir.parent

            def fail_parent(path: Path) -> None:
                if Path(path) == parent:
                    raise OSError("directory fsync unsupported")

            with mock.patch(
                "qsol_geo_reason.capture_publish._fsync_directory",
                side_effect=fail_parent,
            ), mock.patch(
                "qsol_geo_reason.capture_publish._rename_directory_noreplace"
            ) as rename:
                with self.assertRaisesRegex(OSError, "directory fsync unsupported"):
                    write_capture_bundle(output_dir, request, manifest, trajectory)
                rename.assert_not_called()
                self.assertFalse(output_dir.exists())

    def test_production_backend_is_rejected_before_simulation_execution(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        with self.assertRaisesRegex(
            CaptureContractError,
            "may execute only as OBSERVATION",
        ):
            execute_capture(
                fixture_request(),
                implementation_revision="d" * 40,
                backend=backend,
            )


if __name__ == "__main__":
    unittest.main()

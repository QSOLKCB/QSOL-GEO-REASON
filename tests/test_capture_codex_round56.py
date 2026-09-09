from __future__ import annotations

import ctypes
import inspect
import os
import stat
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason import capture
from qsol_geo_reason import capture_backend_round56 as round56
from qsol_geo_reason import capture_common
from qsol_geo_reason import capture_execute


class _FakeSysctlByName:
    def __init__(self, payload: bytes):
        self.payload = payload + b"\x00"
        self.argtypes = None
        self.restype = None

    def __call__(self, _name, output, size_pointer, _new_value, _new_size):
        size = ctypes.cast(size_pointer, ctypes.POINTER(ctypes.c_size_t))
        if output is None:
            size[0] = len(self.payload)
            return 0
        ctypes.memmove(output, self.payload, len(self.payload))
        size[0] = len(self.payload)
        return 0


class Round56TrustBoundaryTests(unittest.TestCase):
    def test_public_backend_is_round56(self):
        self.assertIs(capture.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)

    def test_evidence_allowlist_is_immutable_and_exact(self):
        self.assertIsInstance(capture_common._ALLOWED_EVIDENCE, frozenset)
        self.assertEqual(
            capture_common._ALLOWED_EVIDENCE,
            frozenset({"SIMULATION", "OBSERVATION"}),
        )
        self.assertIs(capture_execute._ALLOWED_EVIDENCE, capture_common._ALLOWED_EVIDENCE)
        with self.assertRaises(AttributeError):
            capture_common._ALLOWED_EVIDENCE.add("MECHANISM")
        with self.assertRaisesRegex(
            capture_common.CaptureContractError, "unsupported capture evidence class"
        ):
            round56._validate_backend_metadata_round56({}, {}, "MECHANISM")

    def test_hook_registry_names_are_bound_outside_mutable_class_attribute(self):
        backend_type = round56.HuggingFacePyTorchBackend
        attribute = "_GLOBAL_EXECUTION_HOOK_REGISTRIES"
        inherited = getattr(backend_type, attribute)
        had_local = attribute in backend_type.__dict__
        local_value = backend_type.__dict__.get(attribute)
        try:
            setattr(
                backend_type,
                attribute,
                tuple(name for name in inherited if name != "_global_forward_hooks"),
            )
            backend = object.__new__(backend_type)
            with self.assertRaisesRegex(
                capture_common.CaptureContractError, "hook-registry allowlist changed"
            ):
                backend._assert_round56_hook_registry_binding()
        finally:
            if had_local:
                setattr(backend_type, attribute, local_value)
            else:
                delattr(backend_type, attribute)

    def test_torch_package_receipt_is_reauthenticated_against_external_baseline(self):
        backend_type = round56.HuggingFacePyTorchBackend
        backend = object.__new__(backend_type)
        torch_module = types.ModuleType("torch")
        torch_module.__file__ = __file__
        backend._torch = torch_module
        backend._torch_build_provenance = {
            "torch_package_file_count": 7,
            "torch_package_receipt_sha256": "a" * 64,
        }
        with patch.object(
            round56,
            "_python_package_provenance",
            return_value={"file_count": 7, "receipt_sha256": "a" * 64},
        ):
            backend._initialize_round56_torch_package_baseline()
            backend._assert_round56_torch_package_provenance()

        with patch.object(
            round56,
            "_python_package_provenance",
            return_value={"file_count": 7, "receipt_sha256": "b" * 64},
        ):
            with self.assertRaisesRegex(
                capture_common.CaptureContractError,
                "PyTorch package changed after authenticated backend construction",
            ):
                backend._assert_round56_torch_package_provenance()

        begin_source = inspect.getsource(
            backend_type._begin_runtime_library_stability_window
        )
        metadata_source = inspect.getsource(backend_type.metadata)
        self.assertIn("_assert_round56_torch_package_provenance", begin_source)
        self.assertIn("_assert_round56_torch_package_provenance", metadata_source)

    def test_native_darwin_sysctl_uses_process_libc(self):
        fake = types.SimpleNamespace(
            sysctlbyname=_FakeSysctlByName(b"Apple M4 Max")
        )
        with (
            patch.object(round56.sys, "platform", "darwin"),
            patch.object(round56.ctypes, "CDLL", return_value=fake),
        ):
            self.assertEqual(
                round56._native_darwin_sysctl_text("machdep.cpu.brand_string"),
                "Apple M4 Max",
            )
        source = inspect.getsource(round56._native_darwin_sysctl_text)
        self.assertIn("sysctlbyname", source)
        self.assertNotIn("subprocess.run", source)

    def test_darwin_runtime_measurement_binds_path_to_mapped_macho_uuid(self):
        mapped_uuid = bytes(range(16))
        header = struct.pack(
            "<IiiIIIII",
            0xFEEDFACF,
            0x01000007,
            3,
            6,
            1,
            24,
            0,
            0,
        )
        command = struct.pack("<II16s", 0x1B, 24, mapped_uuid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "libopenblas.dylib"
            path.write_bytes(header + command + b"payload")
            entry = round56._DarwinMappedLibrary(path, mapped_uuid.hex())
            measured = round56._runtime_library_measurement_round56(
                entry,
                predicate=lambda _value: True,
                label="CPU",
            )
            self.assertIsNotNone(measured)
            fd = os.open(path, os.O_RDONLY)
            try:
                self.assertEqual(
                    round56._macho_uuids_from_fd(fd),
                    frozenset({mapped_uuid.hex()}),
                )
            finally:
                os.close(fd)

            replaced_uuid = bytes(reversed(range(16)))
            path.write_bytes(
                header
                + struct.pack("<II16s", 0x1B, 24, replaced_uuid)
                + b"replacement"
            )
            with self.assertRaisesRegex(
                capture_common.CaptureContractError,
                "no longer matches its pathname",
            ):
                round56._runtime_library_measurement_round56(
                    entry,
                    predicate=lambda _value: True,
                    label="CPU",
                )

    @unittest.skipIf(os.name == "nt", "system Git location assertion is POSIX-specific")
    def test_git_source_identity_ignores_path_wrappers(self):
        trusted = round56._trusted_git_executable()
        self.assertTrue(trusted.is_absolute())
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "git"
            marker = Path(directory) / "wrapper-used"
            wrapper.write_text(
                "#!/bin/sh\n"
                f"printf used > {marker!s}\n"
                "exit 1\n",
                encoding="utf-8",
            )
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
            with patch.dict(os.environ, {"PATH": directory}, clear=False):
                completed = round56._git_run_round56(
                    round56._provenance.source_repo_root(),
                    "--version",
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason import capture
from qsol_geo_reason import capture_backend_round56 as round56
from qsol_geo_reason import capture_backend_round57 as round57
from qsol_geo_reason import capture_common
from qsol_geo_reason import provenance


class Round57TrustBoundaryTests(unittest.TestCase):
    def test_public_backend_identity_is_preserved_while_round57_hardens_it(self):
        self.assertIs(capture.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(round57.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(
            round56.HuggingFacePyTorchBackend._assert_round56_torch_package_provenance,
            round57._assert_round57_torch_package_provenance,
        )

    def test_source_identity_uses_private_git_runner_after_module_slot_rebind(self):
        sealed_resolve = provenance.resolve_implementation_revision
        sealed_git_source = provenance.git_source_revision
        original = provenance._git_run

        def forged_git(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout="f" * 40 + "\n", stderr="")

        try:
            provenance._git_run = forged_git
            self.assertIs(
                sealed_resolve.__globals__["_git_run"],
                round57._git_run_round57,
            )
            self.assertIs(
                sealed_git_source.__globals__["_git_run"],
                round57._git_run_round57,
            )
            self.assertIsNot(sealed_resolve.__globals__, provenance.__dict__)
            self.assertIsNot(sealed_git_source.__globals__, provenance.__dict__)
        finally:
            provenance._git_run = original

    def test_git_child_environment_strips_dynamic_loader_injection(self):
        hostile = {
            "LD_PRELOAD": "/tmp/evil.so",
            "LD_AUDIT": "/tmp/audit.so",
            "LD_LIBRARY_PATH": "/tmp/lib",
            "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
            "_RLD_LIST": "/tmp/evil.so",
            "LDR_PRELOAD": "/tmp/evil.a",
            "GLIBC_TUNABLES": "glibc.malloc.check=3",
            "LIBPATH": "/tmp/aix",
            "SHLIB_PATH": "/tmp/hpux",
            "GIT_EXEC_PATH": "/tmp/fake-git-core",
            "ROUND57_SAFE_SENTINEL": "kept",
        }
        with patch.dict(os.environ, hostile, clear=False):
            environment = round57._git_child_environment_round57()

        for key in hostile:
            if key != "ROUND57_SAFE_SENTINEL":
                self.assertNotIn(key, environment)
        self.assertEqual(environment["ROUND57_SAFE_SENTINEL"], "kept")
        self.assertEqual(environment["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)

    @staticmethod
    def _macho64(uuid_bytes: bytes, payload: bytes) -> bytes:
        header_size = 32
        segment_size = 72
        uuid_size = 24
        file_size = header_size + segment_size + uuid_size + len(payload)
        header = struct.pack(
            "<IiiIIIII",
            0xFEEDFACF,
            0x01000007,
            3,
            6,
            2,
            segment_size + uuid_size,
            0,
            0,
        )
        segment = struct.pack(
            "<II16sQQQQiiII",
            round57._LC_SEGMENT_64,
            segment_size,
            b"__TEXT" + b"\x00" * 10,
            0x100000000,
            file_size,
            0,
            file_size,
            5,
            5,
            0,
            0,
        )
        uuid_command = struct.pack("<II16s", round56._LC_UUID, uuid_size, uuid_bytes)
        return header + segment + uuid_command + payload

    def test_darwin_same_uuid_replacement_fails_mapped_code_digest(self):
        mapped_uuid = bytes(range(16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "libtorch_cpu.dylib"
            path.write_bytes(self._macho64(mapped_uuid, b"A" * 64))
            fd = os.open(path, os.O_RDONLY)
            try:
                identities = round57._macho_identities_from_fd_round57(fd)
            finally:
                os.close(fd)
            self.assertEqual(len(identities), 1)
            mapped_identity = next(iter(identities))
            self.assertEqual(mapped_identity[0], mapped_uuid.hex())
            entry = round57._DarwinMappedLibraryRound57(
                path, mapped_identity[0], mapped_identity[1]
            )

            path.write_bytes(self._macho64(mapped_uuid, b"B" * 64))
            fd = os.open(path, os.O_RDONLY)
            try:
                replaced = round57._macho_identities_from_fd_round57(fd)
            finally:
                os.close(fd)
            self.assertEqual({item[0] for item in replaced}, {mapped_uuid.hex()})
            self.assertNotIn(mapped_identity, replaced)

            with self.assertRaisesRegex(
                capture_common.CaptureContractError,
                "executable bytes no longer match their pathname",
            ):
                round57._runtime_library_measurement_round57(
                    entry,
                    predicate=lambda _value: True,
                    label="CPU",
                )

    def test_torch_stability_receipt_detects_digest_preserving_aba_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "torch"
            root.mkdir()
            init = root / "__init__.py"
            source = root / "lazy.py"
            init.write_text("from . import lazy\n", encoding="utf-8")
            original = b"VALUE = 'trusted'\n"
            source.write_bytes(original)
            module = types.ModuleType("torch")
            module.__file__ = str(init)

            baseline = round57._torch_package_stability_receipt(module)

            source.write_bytes(b"VALUE = 'hostile'\n")
            restored = root / ".lazy.py.restored"
            restored.write_bytes(original)
            os.replace(restored, source)

            self.assertEqual(source.read_bytes(), original)
            observed = round57._torch_package_stability_receipt(module)
            self.assertNotEqual(observed, baseline)


if __name__ == "__main__":
    unittest.main()

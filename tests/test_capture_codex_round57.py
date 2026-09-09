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
from qsol_geo_reason import capture_backend_round58 as round58
from qsol_geo_reason import capture_backend_round60 as round60
from qsol_geo_reason import capture_backend_round66 as round66
from qsol_geo_reason import capture_common
from qsol_geo_reason import capture_execute
from qsol_geo_reason import provenance


class Round58TrustBoundaryTests(unittest.TestCase):
    def test_public_backend_identity_is_preserved_while_round58_hardens_it(self):
        self.assertIs(capture.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(round58.HuggingFacePyTorchBackend, round56.HuggingFacePyTorchBackend)
        self.assertIs(
            round56.HuggingFacePyTorchBackend._assert_round56_torch_package_provenance,
            round58._assert_round58_torch_package_provenance,
        )

    def test_canonical_source_identity_uses_private_git_runner_after_module_rebind(self):
        sealed_resolve = capture_execute.resolve_implementation_revision
        original = provenance._git_run

        def forged_git(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout="f" * 40 + "\n", stderr="")

        try:
            provenance._git_run = forged_git
            self.assertIs(
                sealed_resolve,
                round66._resolve_implementation_revision_round66,
            )
            closure_values = tuple(
                cell.cell_contents for cell in sealed_resolve.__closure__ or ()
            )
            self.assertIn(round60._resolve_implementation_revision_round60, closure_values)

            # Round 60 intentionally stopped exposing the private provenance graph
            # through its resolver's writable module globals. Round 66 only adds the
            # ignored-Python-source guard and closure-binds this same Round-60 resolver.
            canonical = round60._resolve_implementation_revision_round60
            self.assertNotIn("_git_run", canonical.__globals__)
            self.assertNotIn("git_source_revision", canonical.__globals__)
            self.assertIsNot(canonical.__globals__, provenance.__dict__)

            # Even if an embedded caller plants the historical names in the public
            # Round-60 module globals, they are not execution dependencies of the
            # canonical source resolver held by Round 66.
            globals_dict = canonical.__globals__
            missing = object()
            old_runner = globals_dict.get("_git_run", missing)
            old_source = globals_dict.get("git_source_revision", missing)
            globals_dict["_git_run"] = forged_git
            globals_dict["git_source_revision"] = lambda **_kwargs: "f" * 40
            try:
                observed = round60._git_source_revision_round60(
                    require_clean=False,
                    reject_importable_bytecode=False,
                )
                self.assertRegex(observed or "", r"^[0-9a-f]{40}$")
                self.assertNotEqual(observed, "f" * 40)
            finally:
                if old_runner is missing:
                    globals_dict.pop("_git_run", None)
                else:
                    globals_dict["_git_run"] = old_runner
                if old_source is missing:
                    globals_dict.pop("git_source_revision", None)
                else:
                    globals_dict["git_source_revision"] = old_source
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
            "ROUND58_SAFE_SENTINEL": "kept",
        }
        with patch.dict(os.environ, hostile, clear=False):
            environment = round58._git_child_environment_round58()

        for key in hostile:
            if key != "ROUND58_SAFE_SENTINEL":
                self.assertNotIn(key, environment)
        self.assertEqual(environment["ROUND58_SAFE_SENTINEL"], "kept")
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
            round58._LC_SEGMENT_64,
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
                identities = round58._macho_identities_from_fd_round58(fd)
            finally:
                os.close(fd)
            self.assertEqual(len(identities), 1)
            mapped_identity = next(iter(identities))
            self.assertEqual(mapped_identity[0], mapped_uuid.hex())
            entry = round58._DarwinMappedLibraryRound58(
                path, mapped_identity[0], mapped_identity[1]
            )

            path.write_bytes(self._macho64(mapped_uuid, b"B" * 64))
            fd = os.open(path, os.O_RDONLY)
            try:
                replaced = round58._macho_identities_from_fd_round58(fd)
            finally:
                os.close(fd)
            self.assertEqual({item[0] for item in replaced}, {mapped_uuid.hex()})
            self.assertNotIn(mapped_identity, replaced)

            with self.assertRaisesRegex(
                capture_common.CaptureContractError,
                "executable bytes no longer match their pathname",
            ):
                round58._runtime_library_measurement_round58(
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

            baseline = round58._torch_package_stability_receipt_round58(module)

            source.write_bytes(b"VALUE = 'hostile'\n")
            restored = root / ".lazy.py.restored"
            restored.write_bytes(original)
            os.replace(restored, source)

            self.assertEqual(source.read_bytes(), original)
            observed = round58._torch_package_stability_receipt_round58(module)
            self.assertNotEqual(observed, baseline)


if __name__ == "__main__":
    unittest.main()

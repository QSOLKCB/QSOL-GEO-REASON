from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "tools" / "run_first_production_observation.py"


class _ExecIntercept(RuntimeError):
    pass


class ProductionObservationLauncherTests(unittest.TestCase):
    def _load_launcher(self):
        spec = importlib.util.spec_from_file_location(
            "qsol_test_first_production_launcher",
            LAUNCHER_PATH,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_launcher_execve_strips_python_and_native_loader_injection(self) -> None:
        launcher = self._load_launcher()
        seen: dict[str, object] = {}

        def fake_execve(executable, argv, environment):
            seen["executable"] = executable
            seen["argv"] = list(argv)
            seen["environment"] = dict(environment)
            raise _ExecIntercept()

        hostile = {
            "LD_PRELOAD": "/tmp/inject.so",
            "LD_LIBRARY_PATH": "/tmp/lib",
            "DYLD_INSERT_LIBRARIES": "/tmp/dylib",
            "_RLD_LIST": "evil",
            "LDR_PRELOAD": "evil",
            "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=999",
            "LIBPATH": "/tmp/aix",
            "SHLIB_PATH": "/tmp/hpux",
            "PYTHONPATH": "/tmp/python",
            "PYTHONHOME": "/tmp/home",
            "QSOL_SAFE_SENTINEL": "preserve-me",
        }
        with (
            mock.patch.dict(launcher.os.environ, hostile, clear=True),
            mock.patch.object(launcher.os, "execve", side_effect=fake_execve),
        ):
            with self.assertRaises(_ExecIntercept):
                launcher.main()

        self.assertEqual(seen["executable"], sys.executable)
        argv = seen["argv"]
        self.assertEqual(
            argv[:5],
            [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "qsol_geo_reason.first_production_observation",
            ],
        )
        environment = seen["environment"]
        self.assertEqual(environment["QSOL_SAFE_SENTINEL"], "preserve-me")
        forbidden = {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "_RLD_LIST",
            "LDR_PRELOAD",
            "GLIBC_TUNABLES",
            "LIBPATH",
            "SHLIB_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
        }
        self.assertTrue(forbidden.isdisjoint(environment))


if __name__ == "__main__":
    unittest.main()

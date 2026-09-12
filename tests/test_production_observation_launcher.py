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

    def test_launcher_requires_initial_isolated_no_site_interpreter(self) -> None:
        launcher = self._load_launcher()
        with mock.patch.object(
            launcher.sys,
            "flags",
            type(
                "Flags",
                (),
                {
                    "isolated": 0,
                    "no_site": 0,
                    "no_user_site": 0,
                    "ignore_environment": 0,
                },
            )(),
        ):
            with self.assertRaisesRegex(RuntimeError, "python -I -S -B"):
                launcher._assert_initial_launcher_boundary()

    def test_launcher_execve_strips_injection_and_keeps_second_interpreter_no_site(self) -> None:
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
            mock.patch.object(launcher, "_assert_initial_launcher_boundary"),
            mock.patch.object(
                launcher,
                "_literal_site_package_paths",
                return_value=["/trusted/site-packages"],
            ),
            mock.patch.object(launcher.os, "execve", side_effect=fake_execve),
        ):
            with self.assertRaises(_ExecIntercept):
                launcher.main()

        self.assertEqual(seen["executable"], sys.executable)
        argv = seen["argv"]
        self.assertEqual(
            argv[:6],
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                launcher._ORCHESTRATOR_BOOTSTRAP,
            ],
        )
        self.assertIn(str((ROOT / "src").resolve()), argv)
        self.assertIn("/trusted/site-packages", argv)
        self.assertIn("--", argv)
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

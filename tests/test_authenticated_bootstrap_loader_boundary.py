from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "GEO-CAP-001-EXP-001.md"


def _documented_bootstrap_function() -> str:
    text = EXPERIMENT.read_text(encoding="utf-8")
    return text.split("qsol_first_observation() {", 1)[1].split("\n}\n```", 1)[0]


class AuthenticatedBootstrapLoaderBoundaryTests(unittest.TestCase):
    def test_documented_bootstrap_scrubs_loader_injection_before_python(self) -> None:
        text = EXPERIMENT.read_text(encoding="utf-8")
        function = _documented_bootstrap_function()

        for prefix_expansion in (
            "${!LD_@}",
            "${!DYLD_@}",
            "${!_RLD_@}",
            "${!LDR_@}",
        ):
            self.assertIn(prefix_expansion, function)
        self.assertIn("GLIBC_TUNABLES LIBPATH SHLIB_PATH; do", function)
        self.assertIn('! \\unset "$qsol_loader_var" 2>/dev/null', function)
        self.assertIn("\\exit 126", function)
        self.assertIn("loader variable survived scrub", function)
        self.assertNotIn("env -u", function)

        self.assertIn("\\local qsol_loader_var qsol_python", function)
        self.assertIn(
            'qsol_python="$(\\command type -P python 2>/dev/null)" || qsol_python=',
            function,
        )
        self.assertIn('"$qsol_python" != /*', function)
        self.assertIn('! -f "$qsol_python"', function)
        self.assertIn('! -x "$qsol_python"', function)
        self.assertIn("cannot resolve an absolute executable", function)
        self.assertNotIn(
            '\n    python -I -S -B -c "$QSOL_FIRST_OBSERVATION_BOOTSTRAP"',
            function,
        )
        self.assertNotIn(
            '\n    command "$qsol_python" -I -S -B -c "$QSOL_FIRST_OBSERVATION_BOOTSTRAP"',
            function,
        )

        python_start = function.index(
            '\\command "$qsol_python" -I -S -B -c "$QSOL_FIRST_OBSERVATION_BOOTSTRAP"'
        )
        self.assertLess(function.index("${!LD_@}"), python_start)
        self.assertLess(function.index("! \\unset"), python_start)
        self.assertLess(function.index("loader variable survived scrub"), python_start)
        self.assertLess(function.index("\\command type -P python"), python_start)

        self.assertIn(
            "Before starting any Python process",
            text,
        )
        self.assertIn(
            "uses only Bash built-ins",
            text,
        )
        self.assertIn(
            "Every `unset` is checked",
            text,
        )
        self.assertIn(
            "path-only lookup, which ignores shell functions",
            text,
        )
        self.assertIn(
            "backslash-escaped at every command position",
            text,
        )

    def test_readonly_loader_variable_aborts_before_python(self) -> None:
        function_body = _documented_bootstrap_function()
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "python-called"
            marker_shell = shlex.quote(str(marker))
            script = (
                function_definition
                + "\npython() { printf 'called\\n' > "
                + marker_shell
                + "; }\n"
                + "export LD_PRELOAD=/tmp/qsol-readonly-loader-test.so\n"
                + "readonly LD_PRELOAD\n"
                + "qsol_first_observation\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 126, completed.stderr)
            self.assertFalse(marker.exists(), "Python tripwire was reached after scrub failure")
            self.assertIn("cannot clear loader variable LD_PRELOAD", completed.stderr)

    def test_python_shell_function_cannot_intercept_bootstrap(self) -> None:
        function_body = _documented_bootstrap_function()
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            python_path = tmp_path / "python"
            python_path.symlink_to(Path(sys.executable).resolve())
            marker = tmp_path / "python-function-called"
            marker_shell = shlex.quote(str(marker))
            path_shell = shlex.quote(str(tmp_path))
            script = (
                function_definition
                + "\nPATH="
                + path_shell
                + "\n"
                + "python() { printf 'intercepted\\n' > "
                + marker_shell
                + "; return 0; }\n"
                + "QSOL_FIRST_OBSERVATION_BOOTSTRAP='import sys; sys.exit(23)'\n"
                + "qsol_first_observation\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 23, completed.stderr)
            self.assertFalse(
                marker.exists(),
                "python shell function intercepted the authenticated bootstrap",
            )

    def test_interactive_aliases_cannot_rewrite_bootstrap_function_definition(self) -> None:
        function_body = _documented_bootstrap_function()
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            python_path = tmp_path / "python"
            python_path.symlink_to(Path(sys.executable).resolve())
            marker = tmp_path / "alias-intercepted"
            marker_shell = shlex.quote(str(marker))
            path_shell = shlex.quote(str(tmp_path))
            alias_payload = "printf intercepted > " + marker_shell + "; false"
            script = (
                "shopt -s expand_aliases\n"
                + "alias command="
                + shlex.quote(alias_payload)
                + "\n"
                + "alias local="
                + shlex.quote(alias_payload)
                + "\n"
                + "alias unset="
                + shlex.quote(alias_payload)
                + "\n"
                + "alias printf="
                + shlex.quote(alias_payload)
                + "\n"
                + "alias exit="
                + shlex.quote(alias_payload)
                + "\n"
                + "PATH="
                + path_shell
                + "\n"
                + function_definition
                + "QSOL_FIRST_OBSERVATION_BOOTSTRAP='import sys; sys.exit(29)'\n"
                + "qsol_first_observation\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc"],
                input=script,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 29, completed.stderr)
            self.assertFalse(
                marker.exists(),
                "interactive alias expansion rewrote the authenticated bootstrap wrapper",
            )


if __name__ == "__main__":
    unittest.main()

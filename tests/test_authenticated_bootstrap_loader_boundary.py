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


def _documented_bootstrap_literal() -> str:
    function = _documented_bootstrap_function()
    marker = "' qsol_first_observation \"${VIRTUAL_ENV-}\" '"
    return function.split(marker, 1)[1].split("' \"$@\"", 1)[0]


def _documented_bootstrap_function_with(code: str) -> str:
    function = _documented_bootstrap_function()
    literal = _documented_bootstrap_literal()
    return function.replace(f"'{literal}'", shlex.quote(code), 1)


def _make_reference_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    return venv


class AuthenticatedBootstrapLoaderBoundaryTests(unittest.TestCase):
    def test_documented_bootstrap_scrubs_loader_then_execs_function_free_shell(self) -> None:
        text = EXPERIMENT.read_text(encoding="utf-8")
        function = _documented_bootstrap_function()
        bootstrap = _documented_bootstrap_literal()

        for prefix_expansion in (
            "${!LD_@}",
            "${!DYLD_@}",
            "${!_RLD_@}",
            "${!LDR_@}",
        ):
            self.assertIn(prefix_expansion, function)
        self.assertIn("POSIXLY_CORRECT=1", function)
        self.assertIn("GLIBC_TUNABLES LIBPATH SHLIB_PATH; do", function)
        self.assertIn('! \\unset "$qsol_loader_var" 2>/dev/null', function)
        self.assertIn("\\exit 126", function)
        self.assertNotIn("env -u", function)
        self.assertNotIn("\\local ", function)
        self.assertNotIn("\\printf ", function)

        clean_shell = "\\exec /bin/bash --noprofile --norc -p -c"
        self.assertIn(clean_shell, function)
        self.assertIn("qsol_venv=$1", function)
        self.assertIn("qsol_bootstrap=$2", function)
        self.assertIn('qsol_python="$qsol_venv/bin/python"', function)
        self.assertIn('"$qsol_python" != /*', function)
        self.assertIn('! -f "$qsol_python"', function)
        self.assertIn('! -x "$qsol_python"', function)
        self.assertIn("cannot resolve the activated virtualenv Python executable", function)
        self.assertIn(
            '"$qsol_python" -I -S -B -c "$qsol_bootstrap" "$@"',
            function,
        )
        self.assertTrue(
            bootstrap.startswith("import hashlib,os,pathlib,stat,subprocess,sys;"),
            bootstrap,
        )
        self.assertIn("exec(compile(src", bootstrap)
        self.assertIn(f"'{bootstrap}'", function)
        self.assertNotIn("QSOL_FIRST_OBSERVATION_BOOTSTRAP=", text)
        self.assertNotIn("$QSOL_FIRST_OBSERVATION_BOOTSTRAP", function)
        self.assertNotIn("command type -P python", function)
        self.assertNotIn('\\command "$qsol_python"', function)
        self.assertNotIn("\n    /bin/bash --noprofile --norc -p -c", function)

        posix_start = function.index("POSIXLY_CORRECT=1")
        loader_unset = function.index("! \\unset")
        clean_shell_start = function.index(clean_shell)
        python_start = function.index(
            '"$qsol_python" -I -S -B -c "$qsol_bootstrap" "$@"'
        )
        bootstrap_arg = function.index(f"'{bootstrap}'")
        self.assertLess(posix_start, loader_unset)
        self.assertLess(loader_unset, clean_shell_start)
        self.assertLess(clean_shell_start, bootstrap_arg)
        self.assertLess(clean_shell_start, python_start)

        self.assertIn("Before starting any Python process", text)
        self.assertIn("POSIX mode", text)
        self.assertIn("special builtins", text)
        self.assertIn("exec", text)
        self.assertIn("does not import exported shell functions", text)
        self.assertIn("bootstrap program is passed as a literal positional argument", text)
        self.assertIn("activated virtual environment", text)

    def test_readonly_loader_variable_aborts_before_python(self) -> None:
        function_body = _documented_bootstrap_function_with("import sys; sys.exit(19)")
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = _make_reference_venv(tmp_path)
            marker = tmp_path / "python-called"
            marker_shell = shlex.quote(str(marker))
            venv_shell = shlex.quote(str(venv))
            script = (
                function_definition
                + "\npython() { printf 'called\\n' > "
                + marker_shell
                + "; }\n"
                + "VIRTUAL_ENV="
                + venv_shell
                + "\n"
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

    def test_readonly_prebound_bootstrap_variable_cannot_replace_literal(self) -> None:
        function_body = _documented_bootstrap_function_with("import sys; sys.exit(41)")
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = _make_reference_venv(tmp_path)
            marker = tmp_path / "readonly-bootstrap-executed"
            venv_shell = shlex.quote(str(venv))
            malicious = (
                "import pathlib;"
                f"pathlib.Path({str(marker)!r}).write_text('intercepted', encoding='utf-8');"
                "raise SystemExit(0)"
            )
            script = (
                "QSOL_FIRST_OBSERVATION_BOOTSTRAP="
                + shlex.quote(malicious)
                + "\nreadonly QSOL_FIRST_OBSERVATION_BOOTSTRAP\n"
                + "VIRTUAL_ENV="
                + venv_shell
                + "\n"
                + function_definition
                + "qsol_first_observation\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 41, completed.stderr)
            self.assertFalse(
                marker.exists(),
                "readonly ambient bootstrap variable replaced the documented literal",
            )

    def test_python_shell_function_cannot_intercept_bootstrap(self) -> None:
        function_body = _documented_bootstrap_function_with("import sys; sys.exit(23)")
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = _make_reference_venv(tmp_path)
            marker = tmp_path / "python-function-called"
            marker_shell = shlex.quote(str(marker))
            venv_shell = shlex.quote(str(venv))
            script = (
                function_definition
                + "\nVIRTUAL_ENV="
                + venv_shell
                + "\n"
                + "python() { printf 'intercepted\\n' > "
                + marker_shell
                + "; return 0; }\n"
                + "export -f python\n"
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

    def test_command_shell_function_cannot_intercept_clean_child(self) -> None:
        function_body = _documented_bootstrap_function_with("import sys; sys.exit(31)")
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = _make_reference_venv(tmp_path)
            marker = tmp_path / "command-function-called"
            marker_shell = shlex.quote(str(marker))
            venv_shell = shlex.quote(str(venv))
            script = (
                function_definition
                + "\nVIRTUAL_ENV="
                + venv_shell
                + "\n"
                + "command() { printf 'intercepted\\n' > "
                + marker_shell
                + "; return 0; }\n"
                + "export -f command\n"
                + "qsol_first_observation\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 31, completed.stderr)
            self.assertFalse(
                marker.exists(),
                "ambient command shell function intercepted the authenticated bootstrap",
            )

    def test_path_named_bash_and_special_builtin_functions_cannot_intercept_boundary(self) -> None:
        function_body = _documented_bootstrap_function_with("import sys; sys.exit(37)")
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = _make_reference_venv(tmp_path)
            marker = tmp_path / "ambient-function-called"
            marker_shell = shlex.quote(str(marker))
            venv_shell = shlex.quote(str(venv))
            tripwire = "printf intercepted > " + marker_shell + "; return 0"
            script = (
                function_definition
                + "\nVIRTUAL_ENV="
                + venv_shell
                + "\n"
                + "function /bin/bash { "
                + tripwire
                + "; }\n"
                + "exec() { "
                + tripwire
                + "; }\n"
                + "unset() { "
                + tripwire
                + "; }\n"
                + "exit() { "
                + tripwire
                + "; }\n"
                + "qsol_first_observation\n"
            )
            completed = subprocess.run(
                ["/bin/bash", "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 37, completed.stderr)
            self.assertFalse(
                marker.exists(),
                "ambient path-named or special-builtin shell function intercepted the boundary",
            )

    def test_interactive_aliases_cannot_rewrite_bootstrap_function_definition(self) -> None:
        function_body = _documented_bootstrap_function_with("import sys; sys.exit(29)")
        function_definition = "qsol_first_observation() {" + function_body + "\n}\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = _make_reference_venv(tmp_path)
            marker = tmp_path / "alias-intercepted"
            marker_shell = shlex.quote(str(marker))
            venv_shell = shlex.quote(str(venv))
            alias_payload = "printf intercepted > " + marker_shell + "; false"
            script = (
                "shopt -s expand_aliases\n"
                + "alias unset="
                + shlex.quote(alias_payload)
                + "\n"
                + "alias exit="
                + shlex.quote(alias_payload)
                + "\n"
                + "alias exec="
                + shlex.quote(alias_payload)
                + "\n"
                + "VIRTUAL_ENV="
                + venv_shell
                + "\n"
                + function_definition
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

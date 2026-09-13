from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "GEO-CAP-001-EXP-001.md"


class AuthenticatedBootstrapLoaderBoundaryTests(unittest.TestCase):
    def test_documented_bootstrap_scrubs_loader_injection_before_python(self) -> None:
        text = EXPERIMENT.read_text(encoding="utf-8")
        function = text.split("qsol_first_observation() {", 1)[1].split("\n}\n```", 1)[0]

        for prefix_expansion in (
            "${!LD_@}",
            "${!DYLD_@}",
            "${!_RLD_@}",
            "${!LDR_@}",
        ):
            self.assertIn(prefix_expansion, function)
        self.assertIn("unset GLIBC_TUNABLES LIBPATH SHLIB_PATH", function)
        self.assertIn('unset "$qsol_loader_var"', function)
        self.assertNotIn("env -u", function)

        python_start = function.index('python -I -S -B -c "$QSOL_FIRST_OBSERVATION_BOOTSTRAP"')
        self.assertLess(function.index("${!LD_@}"), python_start)
        self.assertLess(function.index("unset GLIBC_TUNABLES LIBPATH SHLIB_PATH"), python_start)

        self.assertIn(
            "Before starting any Python process",
            text,
        )
        self.assertIn(
            "uses only Bash built-ins",
            text,
        )


if __name__ == "__main__":
    unittest.main()

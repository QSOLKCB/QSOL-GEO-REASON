from __future__ import annotations

import hashlib
import inspect
import json
import unittest

from qsol_geo_reason import capture_backend_round39 as round39
from qsol_geo_reason import capture_backend_round62 as round62
from qsol_geo_reason import capture_mps_runtime as mps_runtime
from qsol_geo_reason.capture_common import CaptureContractError


class Round62MPSRuntimeTests(unittest.TestCase):
    @staticmethod
    def _config(*, include_mps: bool = True) -> str:
        lines = ["PyTorch built with synthetic test configuration"]
        if include_mps:
            lines.append(
                "QSOL_GEO_MPS_RUNTIME="
                + json.dumps(
                    {
                        "loaded_mps_runtime_libraries": {
                            "mps_runtime_library_file_count": 3,
                            "mps_runtime_library_receipt_sha256": "a" * 64,
                        }
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        lines.append("QSOL_GEO_CPU_FLUSH_DENORMAL=false")
        lines.append(
            "QSOL_GEO_CPU_RUNTIME="
            + json.dumps(
                {
                    "loaded_cpu_runtime_libraries": {
                        "cpu_runtime_library_file_count": 0,
                        "cpu_runtime_library_receipt_sha256": "b" * 64,
                    }
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return "\n".join(lines) + "\n"

    def test_mps_runtime_predicate_covers_metal_and_mps_frameworks(self):
        for path in (
            "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
            "/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/MetalPerformanceShaders",
            "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/MetalPerformanceShadersGraph",
            "/System/Library/Extensions/AGXMetalG15X.bundle/Contents/MacOS/AGXMetalG15X",
        ):
            with self.subTest(path=path):
                self.assertTrue(mps_runtime._is_mps_runtime_library(path))
        self.assertFalse(mps_runtime._is_mps_runtime_library("/usr/lib/libSystem.B.dylib"))

    def test_mps_build_receipt_is_required_and_position_bound(self):
        config = self._config()
        observed = {
            "device": "mps",
            "pool_accumulation_device": "cpu",
            "torch_build_config": config,
            "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        }
        round62._validate_torch_build_metadata_round62(observed)

        missing = self._config(include_mps=False)
        observed["torch_build_config"] = missing
        observed["torch_build_config_sha256"] = hashlib.sha256(
            missing.encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(CaptureContractError, "MPS torch_build_config"):
            round62._validate_torch_build_metadata_round62(observed)

    def test_mps_receipt_is_forbidden_outside_mps(self):
        config = self._config()
        observed = {
            "device": "cpu",
            "pool_accumulation_device": "cpu",
            "torch_build_config": config,
            "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        }
        with self.assertRaisesRegex(CaptureContractError, "absent outside MPS"):
            round62._validate_torch_build_metadata_round62(observed)

    def test_round39_binds_mps_baseline_and_final_runtime_state(self):
        source = inspect.getsource(round39)
        self.assertIn("loaded_mps_runtime_library_snapshot", source)
        self.assertIn("mps_before", source)
        self.assertIn('label="MPS"', source)
        self.assertIn("QSOL_GEO_MPS_RUNTIME=", source)


if __name__ == "__main__":
    unittest.main()

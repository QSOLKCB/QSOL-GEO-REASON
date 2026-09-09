"""Round 48 regressions for production dispatch, startup cleanup, and docs."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import qsol_geo_reason.capture_execute as capture_execute
from qsol_geo_reason.capture import CaptureContractError


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REQUEST = ROOT / "fixtures" / "capture-contract-request.json"
IMPLEMENTATION_REVISION = "d" * 40
Backend = capture_execute.HuggingFacePyTorchBackend


class _ProductionBackendSubclass(Backend):
    pass


def fixture_request() -> dict:
    return json.loads(FIXTURE_REQUEST.read_text(encoding="utf-8"))


class CaptureRound48RegressionTests(unittest.TestCase):
    def test_production_subclass_is_rejected_before_simulation_execution(self):
        backend = object.__new__(_ProductionBackendSubclass)
        with self.assertRaisesRegex(
            CaptureContractError,
            "including subclasses.*only as OBSERVATION",
        ):
            capture_execute.execute_capture(
                fixture_request(),
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=backend,
            )

    def test_production_subclass_cannot_gain_observation_authority(self):
        backend = object.__new__(_ProductionBackendSubclass)
        with self.assertRaisesRegex(
            CaptureContractError,
            "requires the concrete HuggingFacePyTorchBackend",
        ):
            capture_execute.execute_capture(
                fixture_request(),
                implementation_revision=IMPLEMENTATION_REVISION,
                backend=backend,
                evidence_class="OBSERVATION",
            )

    def test_interrupt_after_activation_runs_observation_cleanup(self):
        backend = object.__new__(Backend)
        backend._observed_hidden_state_dtypes = {}
        backend._observation_active = False
        events: list[str] = []

        def begin(instance) -> None:
            events.append("begin")
            instance._observation_active = True
            # Models the asynchronous window after startup has made the session
            # active but before normal capture execution can proceed.
            raise KeyboardInterrupt()

        def end(instance) -> None:
            events.append("end")
            instance._observation_active = False

        with (
            patch.object(
                capture_execute,
                "_assert_observation_backend_execution_methods",
            ),
            patch.object(
                capture_execute,
                "_assert_observation_backend_routing_state",
            ),
            patch.object(
                capture_execute,
                "resolve_implementation_revision",
                return_value=IMPLEMENTATION_REVISION,
            ),
            patch.object(Backend, "assert_execution_request", autospec=True),
            patch.object(Backend, "begin_observation", begin),
            patch.object(Backend, "end_observation", end),
        ):
            with self.assertRaises(KeyboardInterrupt):
                capture_execute.execute_capture(
                    fixture_request(),
                    implementation_revision=IMPLEMENTATION_REVISION,
                    backend=backend,
                    evidence_class="OBSERVATION",
                )

        self.assertEqual(events, ["begin", "end"])
        self.assertFalse(backend._observation_active)

    def test_protocol_describes_trusted_linux_driver_probe(self):
        protocol = (ROOT / "protocols" / "GEO-CAP-001.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("/sys/module/nvidia/version", protocol)
        self.assertIn("/proc/driver/nvidia/version", protocol)
        self.assertIn("canonical probe does not invoke `nvidia-smi`", protocol)
        self.assertIn("unsupported platforms", protocol)
        self.assertNotIn(
            "NVIDIA driver version reported by `nvidia-smi`",
            protocol,
        )


if __name__ == "__main__":
    unittest.main()

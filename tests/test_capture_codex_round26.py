"""Round 26 regressions for observation cleanup, adapter identity, and schema parity."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.capture import CaptureContractError
from qsol_geo_reason.capture_backend_isolated import HuggingFacePyTorchBackend as IsolatedBackend
from qsol_geo_reason.capture_backend_production import HuggingFacePyTorchBackend as Backend
from qsol_geo_reason.capture_execute import (
    _OBSERVATION_BACKEND_EXECUTION_METHODS,
    _assert_observation_backend_execution_methods,
)

ROOT = Path(__file__).resolve().parents[1]


class CaptureRound26RegressionTests(unittest.TestCase):
    def test_observation_startup_baseexceptions_restore_ambient_process_state(self):
        for interrupt in (KeyboardInterrupt, SystemExit):
            with self.subTest(interrupt=interrupt.__name__):
                backend = object.__new__(IsolatedBackend)
                backend._observation_consumed = False
                backend._observation_active = False
                backend._observation_ambient_process_state = None
                backend._torch = object()
                backend._device = "cpu"
                backend._applied_seed = 7
                ambient = {"marker": "ambient"}

                with (
                    patch.object(IsolatedBackend, "_snapshot_torch_process_state", return_value=ambient),
                    patch("qsol_geo_reason.capture_backend_isolated._seed_capture_generators"),
                    patch.object(
                        IsolatedBackend,
                        "_force_canonical_determinism_policy",
                        side_effect=interrupt(),
                    ),
                    patch.object(IsolatedBackend, "_restore_torch_process_state") as restore,
                ):
                    with self.assertRaises(interrupt):
                        backend.begin_observation()

                restore.assert_called_once_with(backend._torch, ambient)
                self.assertFalse(backend._observation_active)
                self.assertIsNone(backend._observation_ambient_process_state)

    def test_observation_adapter_rejects_instance_method_substitution(self):
        backend = object.__new__(Backend)
        _assert_observation_backend_execution_methods(backend)

        for name in _OBSERVATION_BACKEND_EXECUTION_METHODS:
            with self.subTest(name=name):
                setattr(backend, name, lambda *args, **kwargs: None)
                with self.assertRaisesRegex(CaptureContractError, "cannot be overridden"):
                    _assert_observation_backend_execution_methods(backend)
                delattr(backend, name)
                _assert_observation_backend_execution_methods(backend)

    def test_required_determinism_schema_requires_enabled_algorithms(self):
        schema = json.loads((ROOT / "schemas/capture-run-manifest.schema.json").read_text())
        production = schema["$defs"]["backendObservedProduction"]
        self.assertEqual(
            production["if"]["properties"]["determinism_mode"]["const"],
            "required",
        )
        self.assertIn("determinism_mode", production["if"]["required"])
        self.assertIs(
            production["then"]["properties"]["deterministic_algorithms_enabled"]["const"],
            True,
        )


if __name__ == "__main__":
    unittest.main()

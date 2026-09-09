"""Round 49 regressions for partial observation cleanup and position limits."""
from __future__ import annotations

import inspect
import types
import unittest
from unittest.mock import patch

from qsol_geo_reason.capture_backend_core import HuggingFacePyTorchBackend as CoreBackend
from qsol_geo_reason.capture_backend_production import (
    HuggingFacePyTorchBackend as ProductionBackend,
)
from qsol_geo_reason.capture_common import CaptureContractError
from qsol_geo_reason.capture_execute import execute_capture


class CaptureRound49RegressionTests(unittest.TestCase):
    def test_partial_thread_boundary_interrupt_is_cleaned_by_begin_observation(self):
        backend = object.__new__(ProductionBackend)

        def partial_enter(instance):
            instance._exclusive_thread_boundary_state = {"partial": True}
            raise KeyboardInterrupt

        def leave(instance):
            instance._exclusive_thread_boundary_state = None

        with (
            patch.object(ProductionBackend, "_enter_exclusive_python_thread_boundary", partial_enter),
            patch.object(ProductionBackend, "_leave_exclusive_python_thread_boundary", leave),
        ):
            with self.assertRaises(KeyboardInterrupt):
                backend.begin_observation()

        self.assertIsNone(backend._exclusive_thread_boundary_state)

    def test_execute_cleanup_recognizes_partial_thread_boundary_state(self):
        source = inspect.getsource(execute_capture)
        finally_tail = source[source.index("    finally:"):]
        self.assertIn('getattr(backend, "_observation_active", False)', finally_tail)
        self.assertIn(
            'getattr(backend, "_exclusive_thread_boundary_state", None) is not None',
            finally_tail,
        )
        self.assertIn("backend.end_observation()", finally_tail)

    def test_context_over_position_limit_fails_before_execution_policy_or_hooks(self):
        backend = object.__new__(CoreBackend)
        backend._hidden_state_count = 2
        backend._model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_position_embeddings=4)
        )

        with self.assertRaisesRegex(
            CaptureContractError,
            "tokenized context length 5 exceeds model position limit 4",
        ):
            backend.hidden_states(
                [1, 2, 3, 4, 5],
                [0],
                pool_span=(0, 5),
            )

    def test_position_limit_accepts_exact_limit_and_rejects_malformed_config(self):
        backend = object.__new__(CoreBackend)
        backend._model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_position_embeddings=1024)
        )
        self.assertEqual(backend._model_position_limit(), 1024)

        for bad in (True, 0, -1, 1.5, "1024"):
            backend._model.config.max_position_embeddings = bad
            with self.subTest(value=bad):
                with self.assertRaisesRegex(
                    CaptureContractError,
                    "max_position_embeddings must be a positive integer",
                ):
                    backend._model_position_limit()

        backend._model.config = types.SimpleNamespace()
        self.assertIsNone(backend._model_position_limit())
        del backend._model
        self.assertIsNone(backend._model_position_limit())


if __name__ == "__main__":
    unittest.main()

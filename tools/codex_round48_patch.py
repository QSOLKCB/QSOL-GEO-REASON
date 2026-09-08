"""Temporary guarded patcher for Codex review round 48; removed by CI."""
from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one guarded match, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# A production-derived subclass must never enter the unbracketed SIMULATION lane,
# while OBSERVATION authority remains restricted to the exact trusted concrete type.
replace_once(
    "src/qsol_geo_reason/capture_execute.py",
    '''    production_backend = type(backend) is HuggingFacePyTorchBackend
    if production_backend and evidence_class != "OBSERVATION":
        raise CaptureContractError(
            "the concrete HuggingFacePyTorchBackend may execute only as OBSERVATION; "
            "use a software simulation backend for SIMULATION"
        )

    observation_started = False
''',
    '''    production_backend_instance = isinstance(backend, HuggingFacePyTorchBackend)
    production_backend = type(backend) is HuggingFacePyTorchBackend
    if production_backend_instance and evidence_class != "OBSERVATION":
        raise CaptureContractError(
            "HuggingFacePyTorchBackend instances, including subclasses, may execute "
            "only as OBSERVATION; use a software simulation backend for SIMULATION"
        )

''',
)

# Establish the cleanup-protected region before begin_observation(). Cleanup reads
# the backend's actual active state, covering interrupts both inside and immediately
# after startup without calling end_observation() for a session that never activated.
replace_once(
    "src/qsol_geo_reason/capture_execute.py",
    '''        backend.assert_execution_request(validated)
        backend._observed_hidden_state_dtypes.clear()
        backend.begin_observation()
        observation_started = True

    try:
        if evidence_class == "OBSERVATION":
            # begin_observation() owns the exclusive Python-thread boundary. Repeat
''',
    '''        backend.assert_execution_request(validated)
        backend._observed_hidden_state_dtypes.clear()

    try:
        if evidence_class == "OBSERVATION":
            backend.begin_observation()
            # begin_observation() owns the exclusive Python-thread boundary. Repeat
''',
)
replace_once(
    "src/qsol_geo_reason/capture_execute.py",
    '''    finally:
        if observation_started:
            backend.end_observation()
''',
    '''    finally:
        if production_backend and getattr(backend, "_observation_active", False):
            backend.end_observation()
''',
)

# Reconcile the normative protocol with the trusted implementation actually used.
replace_once(
    "protocols/GEO-CAP-001.md",
    '''- NVIDIA driver version reported by `nvidia-smi` for CUDA captures;
''',
    '''- NVIDIA driver version read, on Linux only, from the loaded kernel module via `/sys/module/nvidia/version`, falling back to `/proc/driver/nvidia/version`; the field is null when those trusted interfaces are unavailable and on unsupported platforms, and the canonical probe does not invoke `nvidia-smi`;
''',
)

# Update the earlier ordering regression to assert the stronger active-state cleanup
# boundary rather than the removed bookkeeping boolean.
replace_once(
    "tests/test_capture_codex_round29.py",
    '''    def test_adapter_and_routing_are_rechecked_after_exclusive_boundary_starts(self):
        source = inspect.getsource(execute_capture)
        begin = source.index("backend.begin_observation()")
        tail = source[begin:]
        active = tail.index("observation_started = True")
        method_recheck = tail.index("_assert_observation_backend_execution_methods(backend)")
        routing_recheck = tail.index("_assert_observation_backend_routing_state(backend)")
        capture = tail.index("steps, prefix_ids = _capture_steps(")
        cleanup = tail.index("backend.end_observation()")
        self.assertLess(active, method_recheck)
        self.assertLess(active, routing_recheck)
        self.assertLess(method_recheck, capture)
        self.assertLess(routing_recheck, capture)
        self.assertGreater(cleanup, method_recheck)
        self.assertGreater(cleanup, routing_recheck)
''',
    '''    def test_adapter_and_routing_are_rechecked_after_exclusive_boundary_starts(self):
        source = inspect.getsource(execute_capture)
        begin = source.index("backend.begin_observation()")
        protected_try = source.rfind("\\n    try:", 0, begin)
        self.assertNotEqual(protected_try, -1)
        self.assertLess(protected_try, begin)

        tail = source[begin:]
        method_recheck = tail.index("_assert_observation_backend_execution_methods(backend)")
        routing_recheck = tail.index("_assert_observation_backend_routing_state(backend)")
        capture = tail.index("steps, prefix_ids = _capture_steps(")
        active_guard = tail.index('getattr(backend, "_observation_active", False)')
        cleanup = tail.index("backend.end_observation()")
        self.assertLess(method_recheck, capture)
        self.assertLess(routing_recheck, capture)
        self.assertGreater(active_guard, capture)
        self.assertLess(active_guard, cleanup)
        self.assertGreater(cleanup, method_recheck)
        self.assertGreater(cleanup, routing_recheck)
''',
)

Path("tests/test_capture_codex_round48.py").write_text(
    '''"""Round 48 regressions for production dispatch, startup cleanup, and docs."""
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
''',
    encoding="utf-8",
)

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one replacement target, found {count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "src/qsol_geo_reason/capture_backend_production.py",
    '''        except BaseException:\n            self._leave_exclusive_python_thread_boundary()\n            raise\n\n    def end_observation(self) -> None:\n        try:\n            super().end_observation()\n        finally:\n            self._leave_exclusive_python_thread_boundary()\n''',
    '''        except BaseException:\n            # Do not release thread exclusion while inherited startup still owns\n            # an ambient-state receipt.  A failed restoration must be retried while\n            # caller threads remain excluded from the temporary process-global state.\n            if (\n                not getattr(self, "_observation_active", False)\n                and getattr(self, "_observation_ambient_process_state", None) is None\n            ):\n                self._leave_exclusive_python_thread_boundary()\n            raise\n\n    def end_observation(self) -> None:\n        try:\n            super().end_observation()\n        finally:\n            # Release the Python-thread boundary only after the inherited session has\n            # fully restored and relinquished its ambient-state receipt. Ordinary\n            # restore failures keep both ownership records live for a safe retry.\n            if (\n                not getattr(self, "_observation_active", False)\n                and getattr(self, "_observation_ambient_process_state", None) is None\n            ):\n                self._leave_exclusive_python_thread_boundary()\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_cuda_runtime.py",
    '''    """Reject runtime files modified after the canonical observation began.\n\n    Existing mappings must retain the same descriptor identity/content/stat receipt.\n    A runtime library loaded lazily during the forward is allowed only when its file\n    metadata proves it had not been modified since observation start.  Linux ctime\n    makes an in-place write/restore ABA visible even when final bytes match again.\n    """\n''',
    '''    """Reject runtime-file drift across the canonical observation window.\n\n    Existing mappings are authenticated by exact baseline-to-final descriptor, content,\n    and stat equality.  Their absolute mtimes are deliberately not compared with wall\n    clock start time because package extraction may preserve a legitimate future build\n    timestamp.  A library first observed after the baseline is accepted only when its\n    change-time predates the observation; on POSIX, ctime also exposes in-place\n    write/restore ABA even when final bytes and preserved mtime match again.\n    """\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_cuda_runtime.py",
    '''            if mtime_ns > observation_started_ns or ctime_ns > observation_started_ns:\n                raise CaptureContractError(\n                    f"{label} runtime library changed after canonical observation began: {display_path}"\n                )\n            if display_path in normalized:\n''',
    '''            if display_path in normalized:\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_cuda_runtime.py",
    '''    expected = normalize(before)\n    observed = normalize(after)\n    for display_path, expected_state in expected.items():\n        if observed.get(display_path) != expected_state:\n            raise CaptureContractError(\n                f"{label} runtime library identity/content changed during canonical observation: "\n                f"{display_path}"\n            )\n''',
    '''    expected = normalize(before)\n    observed = normalize(after)\n    for display_path, expected_state in expected.items():\n        if observed.get(display_path) != expected_state:\n            raise CaptureContractError(\n                f"{label} runtime library identity/content changed during canonical observation: "\n                f"{display_path}"\n            )\n\n    # Only entries absent from the pre-execution baseline need a wall-clock\n    # freshness test.  Do not use mtime for that test: preserved package/build\n    # timestamps may legitimately lie in the future.  ctime/change-time is retained\n    # in the exact receipt and is the post-start mutation discriminator here.\n    for display_path, observed_state in observed.items():\n        if display_path in expected:\n            continue\n        ctime_ns = observed_state[6]\n        if ctime_ns > observation_started_ns:\n            raise CaptureContractError(\n                f"{label} runtime library changed after canonical observation began: {display_path}"\n            )\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_backend_round39.py",
    '''        _cpu_provenance, cpu_state = loaded_cpu_runtime_library_snapshot()\n        assert_runtime_library_state_stable(\n            (),\n            cpu_state,\n            observation_started_ns=observation_started_ns,\n            label="CPU",\n        )\n        cuda_state = None\n        if cuda_active:\n            _cuda_provenance, cuda_state = loaded_cuda_runtime_library_snapshot()\n            assert_runtime_library_state_stable(\n                (),\n                cuda_state,\n                observation_started_ns=observation_started_ns,\n                label="CUDA",\n            )\n''',
    '''        _cpu_provenance, cpu_state = loaded_cpu_runtime_library_snapshot()\n        # This is the pre-execution baseline, not a set of libraries discovered\n        # after start.  Absolute file timestamps are therefore not freshness tests;\n        # exact baseline-to-final receipts enforce stability at metadata time.\n        cuda_state = None\n        if cuda_active:\n            _cuda_provenance, cuda_state = loaded_cuda_runtime_library_snapshot()\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_execute.py",
    '''            # Also cover a partially entered production boundary whose thread-start\n            # patches are active even though inherited observation activation has not\n            # completed yet. end_observation() is idempotent for the inactive parent\n            # session and always releases the production thread boundary in finally.\n            backend.end_observation()\n''',
    '''            # Also cover a partially entered production boundary whose thread-start\n            # patches are active even though inherited observation activation has not\n            # completed yet.  end_observation() releases exclusion only after ambient\n            # restoration fully succeeds; a failed restore keeps ownership retryable.\n            backend.end_observation()\n''',
)

replace_once(
    "tests/test_capture_codex_round53.py",
    '''        with self.assertRaisesRegex(CaptureContractError, "changed after canonical observation began"):\n            cuda_runtime.assert_runtime_library_state_stable(\n                before, after, observation_started_ns=started, label="CUDA"\n            )\n''',
    '''        with self.assertRaisesRegex(CaptureContractError, "identity/content changed during canonical observation"):\n            cuda_runtime.assert_runtime_library_state_stable(\n                before, after, observation_started_ns=started, label="CUDA"\n            )\n''',
)

round54 = r'''"""Round 54 regressions for restoration ownership and timestamp-safe runtime stability."""
from __future__ import annotations

import time
import unittest

import qsol_geo_reason.capture_backend_production as production
import qsol_geo_reason.capture_backend_round39 as round39
import qsol_geo_reason.capture_cuda_runtime as cuda_runtime
from qsol_geo_reason.capture_common import CaptureContractError


class CaptureRound54RegressionTests(unittest.TestCase):
    def _bare_production_backend(self):
        backend = object.__new__(production.HuggingFacePyTorchBackend)
        backend._observation_active = True
        backend._observation_ambient_process_state = {"rng": "ambient"}
        backend._torch = object()
        backend._exclusive_thread_boundary_state = {"patches": ("synthetic",)}
        backend.restore_should_fail = True
        backend.leave_calls = 0

        def restore(_torch, _ambient):
            if backend.restore_should_fail:
                raise CaptureContractError("synthetic ambient restore failure")

        def leave():
            backend.leave_calls += 1
            backend._exclusive_thread_boundary_state = None

        backend._restore_torch_process_state = restore
        backend._leave_exclusive_python_thread_boundary = leave
        return backend

    def test_end_observation_retains_thread_exclusion_until_restore_succeeds(self):
        backend = self._bare_production_backend()
        with self.assertRaisesRegex(CaptureContractError, "synthetic ambient restore failure"):
            production.HuggingFacePyTorchBackend.end_observation(backend)

        self.assertTrue(backend._observation_active)
        self.assertEqual(backend._observation_ambient_process_state, {"rng": "ambient"})
        self.assertIsNotNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(backend.leave_calls, 0)

        backend.restore_should_fail = False
        production.HuggingFacePyTorchBackend.end_observation(backend)
        self.assertFalse(backend._observation_active)
        self.assertIsNone(backend._observation_ambient_process_state)
        self.assertIsNone(backend._exclusive_thread_boundary_state)
        self.assertEqual(backend.leave_calls, 1)

    def test_runtime_stability_allows_unchanged_baseline_with_future_mtime(self):
        started = time.time_ns()
        entry = (
            "/tmp/libgomp.so.1",
            "libgomp.so.1",
            "a" * 64,
            1,
            2,
            4096,
            started + 10_000_000_000,
            started - 1_000_000,
        )
        cuda_runtime.assert_runtime_library_state_stable(
            (entry,), (entry,), observation_started_ns=started, label="CPU"
        )

    def test_runtime_stability_allows_lazy_library_with_future_mtime_when_ctime_is_old(self):
        started = time.time_ns()
        lazy = (
            "/tmp/libcudnn.so.9",
            "libcudnn.so.9",
            "b" * 64,
            1,
            3,
            8192,
            started + 10_000_000_000,
            started - 1_000_000,
        )
        cuda_runtime.assert_runtime_library_state_stable(
            (), (lazy,), observation_started_ns=started, label="CUDA"
        )

    def test_runtime_stability_rejects_lazy_library_with_post_start_ctime(self):
        started = time.time_ns()
        lazy = (
            "/tmp/libcudnn.so.9",
            "libcudnn.so.9",
            "b" * 64,
            1,
            3,
            8192,
            started - 1_000_000,
            started + 1,
        )
        with self.assertRaisesRegex(CaptureContractError, "changed after canonical observation began"):
            cuda_runtime.assert_runtime_library_state_stable(
                (), (lazy,), observation_started_ns=started, label="CUDA"
            )

    def test_round39_baseline_is_not_misclassified_as_lazy_load(self):
        source = __import__("inspect").getsource(
            round39.HuggingFacePyTorchBackend._begin_runtime_library_stability_window
        )
        self.assertIn("loaded_cpu_runtime_library_snapshot()", source)
        self.assertNotIn("assert_runtime_library_state_stable", source)


if __name__ == "__main__":
    unittest.main()
'''
Path("tests/test_capture_codex_round54.py").write_text(round54, encoding="utf-8")

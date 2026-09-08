from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one follow-up target, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/qsol_geo_reason/capture_backend_round39.py",
    '''    def begin_observation(self) -> None:\n        super().begin_observation()\n        observation_started_ns = time.time_ns()\n        try:\n            device = getattr(self, "_device", None)\n            cuda_active = isinstance(device, str) and device.startswith("cuda:")\n            cpu_provenance, cpu_state = loaded_cpu_runtime_library_snapshot()\n            # A library that was changed between the observation timestamp and this\n            # first receipt is already outside the immutable execution boundary.\n            assert_runtime_library_state_stable(\n                (),\n                cpu_state,\n                observation_started_ns=observation_started_ns,\n                label="CPU",\n            )\n            cuda_state = None\n            if cuda_active:\n                _cuda_provenance, cuda_state = loaded_cuda_runtime_library_snapshot()\n                assert_runtime_library_state_stable(\n                    (),\n                    cuda_state,\n                    observation_started_ns=observation_started_ns,\n                    label="CUDA",\n                )\n            _remember_runtime_library_baseline(\n                self,\n                (observation_started_ns, cpu_state, cuda_state),\n            )\n        except BaseException:\n            # super().begin_observation() has already entered the restoration-owned\n            # session.  Do not expose a failed receipt acquisition with capture\n            # policies still active.\n            self.end_observation()\n            raise\n\n    def end_observation(self) -> None:\n        try:\n            super().end_observation()\n        finally:\n            if not getattr(self, "_observation_active", False):\n                _forget_runtime_library_baseline(self)\n\n''',
    '''    def _begin_runtime_library_stability_window(self) -> None:\n        # The production boundary calls this only after exclusive-thread ownership\n        # and the retryable observation session are established, but before control\n        # returns to execute_capture() and before the first tokenization/forward.\n        super()._begin_runtime_library_stability_window()\n        observation_started_ns = time.time_ns()\n        device = getattr(self, "_device", None)\n        cuda_active = isinstance(device, str) and device.startswith("cuda:")\n        _cpu_provenance, cpu_state = loaded_cpu_runtime_library_snapshot()\n        assert_runtime_library_state_stable(\n            (),\n            cpu_state,\n            observation_started_ns=observation_started_ns,\n            label="CPU",\n        )\n        cuda_state = None\n        if cuda_active:\n            _cuda_provenance, cuda_state = loaded_cuda_runtime_library_snapshot()\n            assert_runtime_library_state_stable(\n                (),\n                cuda_state,\n                observation_started_ns=observation_started_ns,\n                label="CUDA",\n            )\n        _remember_runtime_library_baseline(\n            self,\n            (observation_started_ns, cpu_state, cuda_state),\n        )\n\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_backend_production.py",
    '''    def begin_observation(self) -> None:\n        try:\n''',
    '''    def _begin_runtime_library_stability_window(self) -> None:\n        """Extension hook owned by the existing exclusive observation boundary."""\n        return None\n\n    def begin_observation(self) -> None:\n        try:\n''',
)

replace_once(
    "src/qsol_geo_reason/capture_backend_production.py",
    '''            self._assert_model_runtime_attributes()\n            super().begin_observation()\n        except BaseException:\n            self._leave_exclusive_python_thread_boundary()\n            raise\n''',
    '''            self._assert_model_runtime_attributes()\n            super().begin_observation()\n            try:\n                # Runtime-library stability starts only after inherited startup has\n                # acquired its ambient-state receipt. A hook failure therefore uses\n                # the same retryable restoration path before this boundary releases\n                # thread exclusion.\n                self._begin_runtime_library_stability_window()\n            except BaseException:\n                super().end_observation()\n                raise\n        except BaseException:\n            self._leave_exclusive_python_thread_boundary()\n            raise\n''',
)

replace_once(
    "tests/test_capture_codex_round53.py",
    '''        begin_source = __import__("inspect").getsource(round39.HuggingFacePyTorchBackend.begin_observation)\n        metadata_source = __import__("inspect").getsource(round39.HuggingFacePyTorchBackend.metadata)\n        self.assertIn("loaded_cpu_runtime_library_snapshot()", begin_source)\n        self.assertIn("_remember_runtime_library_baseline", begin_source)\n        self.assertIn("assert_runtime_library_state_stable", metadata_source)\n        self.assertIn("_recall_runtime_library_baseline", metadata_source)\n''',
    '''        hook_source = __import__("inspect").getsource(\n            round39.HuggingFacePyTorchBackend._begin_runtime_library_stability_window\n        )\n        begin_source = __import__("inspect").getsource(\n            round39.HuggingFacePyTorchBackend.begin_observation\n        )\n        metadata_source = __import__("inspect").getsource(round39.HuggingFacePyTorchBackend.metadata)\n        self.assertIn("loaded_cpu_runtime_library_snapshot()", hook_source)\n        self.assertIn("_remember_runtime_library_baseline", hook_source)\n        self.assertIn("self._enter_exclusive_python_thread_boundary()", begin_source)\n        self.assertIn("self._begin_runtime_library_stability_window()", begin_source)\n        self.assertIn("assert_runtime_library_state_stable", metadata_source)\n        self.assertIn("_recall_runtime_library_baseline", metadata_source)\n''',
)

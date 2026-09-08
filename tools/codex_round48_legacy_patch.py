"""Temporary guarded alignment for the Round 21 source audit; removed by CI."""
from pathlib import Path

path = Path("tests/test_capture_codex_round21.py")
text = path.read_text(encoding="utf-8")
old = '''        self.assertIn(
            'if production_backend and evidence_class != "OBSERVATION":',
            source,
        )
        self.assertIn("if not production_backend:", source)
'''
new = '''        self.assertIn(
            "production_backend_instance = isinstance(backend, HuggingFacePyTorchBackend)",
            source,
        )
        self.assertIn(
            'if production_backend_instance and evidence_class != "OBSERVATION":',
            source,
        )
        self.assertIn("if not production_backend:", source)
'''
if text.count(old) != 1:
    raise SystemExit("round21 concrete-boundary audit did not match exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

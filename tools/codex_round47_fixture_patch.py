"""Temporary guarded fixture updater for Codex review round 47."""
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


shape = "tests/test_capture_codex_round6.py"
replace_once(
    shape,
    '''        "tokenizers_native_backend_active": False,
        "tokenizers_package_file_count": None,
        "tokenizers_package_receipt_sha256": None,
''',
    '''        "tokenizers_native_backend_active": True,
        "tokenizers_package_file_count": 1,
        "tokenizers_package_receipt_sha256": "e" * 64,
''',
)
replace_once(
    shape,
    '        "tokenizers_version": None,\n',
    '        "tokenizers_version": "0.22.0",\n',
)
replace_once(
    shape,
    '''        "mps_fallback_env": None,
        "mps_fast_math_env": None,
''',
    '''        "mps_fallback_env": None,
        "mps_fast_math_env": None,
        "mps_prefer_metal_env": None,
''',
)
replace_once(
    shape,
    '''        "autocast_disabled": True,
        "hidden_state_block_path": "layers",
''',
    '''        "autocast_disabled": True,
        "deterministic_warn_only_enabled": False,
        "hidden_state_block_path": "layers",
''',
)

determinism = "tests/test_capture_codex_round5.py"
replace_once(
    determinism,
    '''            _validate_required_determinism(
                {"deterministic_algorithms_enabled": False}, required
            )
        _validate_required_determinism(
            {"deterministic_algorithms_enabled": True}, required
        )
        _validate_required_determinism(
            {"deterministic_algorithms_enabled": False},
            {"determinism": {"mode": "best_effort"}},
        )
''',
    '''            _validate_required_determinism(
                {
                    "deterministic_algorithms_enabled": False,
                    "deterministic_warn_only_enabled": False,
                },
                required,
            )
        _validate_required_determinism(
            {
                "deterministic_algorithms_enabled": True,
                "deterministic_warn_only_enabled": False,
            },
            required,
        )
        _validate_required_determinism(
            {
                "deterministic_algorithms_enabled": False,
                "deterministic_warn_only_enabled": False,
            },
            {"determinism": {"mode": "best_effort"}},
        )
''',
)

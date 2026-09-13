"""Hardened authenticated orchestration facade for GEO-CAP-001-EXP-001.

The established helper implementation is retained in ``first_production_observation_core``.
This facade owns the high-risk preparation/observation boundaries added after review:
launcher/lock authentication, complete reference-environment binding, isolated no-site
Hub preparation, and race-safe receipt-first publication.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Mapping

from . import first_production_observation_core as _core
from .capture_common import CaptureBackendUnavailable, CaptureContractError
from .capture_reference_environment import (
    _validate_hub_transport_package_provenance,
    verify_current_reference_environment,
)
from .provenance import SourceIdentityError
from .runner_provenance import authenticate_tracked_tool_against_revision


LAUNCHER = _core.ROOT / "tools" / "run_first_production_observation.py"
REFERENCE_LOCK = _core.ROOT / "constraints" / "capture-reference-py311.txt"
_LOADER_ENV_PREFIXES = ("LD_", "DYLD_", "_RLD_", "LDR_")
_LOADER_ENV_NAMES = frozenset({"GLIBC_TUNABLES", "LIBPATH", "SHLIB_PATH"})
_TRANSPORT_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HF_ENDPOINT",
        "HF_HUB_DISABLE_SSL_VERIFY",
        "HF_HUB_DISABLE_SSL_VERIFICATION",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    }
)
_HUB_PACKAGE_FIELDS = (
    "huggingface_hub_package_file_count",
    "huggingface_hub_package_receipt_sha256",
)
_HUB_TRANSPORT_FIELD = "hub_transport_package_provenance"
_HUB_EVIDENCE_FIELDS = (*_core.TREE_FIELDS, *_HUB_PACKAGE_FIELDS, _HUB_TRANSPORT_FIELD)
_HUB_BOOTSTRAP = (
    "import sys;"
    "src=sys.argv[1];"
    "paths=sys.argv[2:];"
    "sys.path.insert(0,src);"
    "[sys.path.append(p) for p in paths if p not in sys.path];"
    "from qsol_geo_reason.capture_hub_prepare_worker import main;"
    "raise SystemExit(main())"
)


def _trusted_system_path() -> str:
    if os.name == "nt":
        return os.pathsep.join(
            (
                r"C:\Program Files\Git\cmd",
                r"C:\Program Files\Git\bin",
                r"C:\Windows\System32",
            )
        )
    return os.pathsep.join(("/usr/bin", "/bin", "/run/current-system/sw/bin"))


def _hub_prepare_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper == "PATH":
            continue
        if upper.startswith("PYTHON") or upper.startswith("GIT_"):
            continue
        if upper.startswith(_LOADER_ENV_PREFIXES) or upper in _LOADER_ENV_NAMES:
            continue
        if upper in _TRANSPORT_ENV_NAMES:
            continue
        environment[key] = value
    environment.update(
        {
            "PATH": _trusted_system_path(),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _literal_site_package_paths() -> list[str]:
    """Locate interpreter package directories without executing site/.pth startup code."""
    paths: list[str] = []
    executable = Path(sys.executable)
    venv_root = executable.parent.parent
    if (venv_root / "pyvenv.cfg").is_file():
        if os.name == "nt":
            candidates = [venv_root / "Lib" / "site-packages"]
        else:
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            candidates = [
                venv_root / "lib" / version / "site-packages",
                venv_root / "lib64" / version / "site-packages",
            ]
    else:
        candidates = []
        for key in ("purelib", "platlib"):
            value = sysconfig.get_paths().get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(Path(value))
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_dir() and str(resolved) not in paths:
            paths.append(str(resolved))
    if not paths:
        raise CaptureContractError(
            "unable to locate literal site-package directories for isolated Hub preparation"
        )
    return paths


def _isolated_prepare_hub_evidence(request: Mapping[str, Any]) -> dict[str, Any]:
    """Perform online Hub work in a fresh -I -S child and return content-bound evidence."""
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _HUB_BOOTSTRAP,
        str((_core.ROOT / "src").resolve()),
        *_literal_site_package_paths(),
    ]
    completed = subprocess.run(
        command,
        cwd=_core.ROOT,
        env=_hub_prepare_environment(),
        input=json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Hub preparation failed"
        raise CaptureContractError(
            f"isolated trusted Hub preparation failed: {detail}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CaptureContractError(
            "isolated Hub preparation emitted unexpected stdout"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CaptureContractError(
            "isolated Hub preparation returned malformed JSON"
        ) from exc
    if not isinstance(value, dict) or set(value) != set(_HUB_EVIDENCE_FIELDS):
        raise CaptureContractError(
            "isolated Hub preparation returned a noncanonical evidence set"
        )
    evidence: dict[str, Any] = {}
    for field in _core.TREE_FIELDS:
        receipt = value.get(field)
        if (
            not isinstance(receipt, str)
            or len(receipt) != 64
            or any(ch not in "0123456789abcdef" for ch in receipt)
        ):
            raise CaptureContractError(
                f"isolated Hub preparation returned invalid {field}"
            )
        evidence[field] = receipt
    package_count = value.get("huggingface_hub_package_file_count")
    if isinstance(package_count, bool) or not isinstance(package_count, int) or package_count < 1:
        raise CaptureContractError(
            "isolated Hub preparation returned invalid Hugging Face Hub package file count"
        )
    package_receipt = value.get("huggingface_hub_package_receipt_sha256")
    if (
        not isinstance(package_receipt, str)
        or len(package_receipt) != 64
        or any(ch not in "0123456789abcdef" for ch in package_receipt)
    ):
        raise CaptureContractError(
            "isolated Hub preparation returned invalid Hugging Face Hub package receipt"
        )
    evidence["huggingface_hub_package_file_count"] = package_count
    evidence["huggingface_hub_package_receipt_sha256"] = package_receipt
    evidence[_HUB_TRANSPORT_FIELD] = _validate_hub_transport_package_provenance(
        value.get(_HUB_TRANSPORT_FIELD)
    )
    return evidence


def _isolated_prepare_tree_receipts(request: Mapping[str, Any]) -> dict[str, str]:
    """Return request-ready tree receipts after binding the worker's Hub runtime bytes."""
    evidence = _isolated_prepare_hub_evidence(request)
    reference_environment = _core.verify_current_reference_environment()
    for field in (*_HUB_PACKAGE_FIELDS, _HUB_TRANSPORT_FIELD):
        if evidence[field] != reference_environment[field]:
            raise CaptureContractError(
                "isolated Hub runtime provenance does not match the authenticated reference environment receipt"
            )
    return {field: evidence[field] for field in _core.TREE_FIELDS}


def _authenticate_external_inputs(revision: str) -> None:
    try:
        _core.authenticate_tracked_tool_against_revision(LAUNCHER, revision)
        _core.authenticate_tracked_file_against_revision(REFERENCE_LOCK, revision)
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"production launcher/reference lock is not bound to repository revision {revision}: {exc}"
        ) from exc


def _reauthenticate_revision(revision: str, where: str) -> None:
    try:
        observed = _core.resolve_implementation_revision(
            revision,
            require_checkout=True,
        )
        if observed != revision:
            raise SourceIdentityError("repository revision changed during the authenticated phase")
        _core.authenticate_tracked_file_against_revision(_core.TEMPLATE, revision)
    except SourceIdentityError as exc:
        raise CaptureContractError(f"{where}: {exc}") from exc
    _authenticate_external_inputs(revision)


def _assert_preparation_output_outside_checkout(output: Path) -> None:
    """Reject lexical, resolved, or symlinked request paths that threaten the checkout."""
    try:
        repository_root = _core.ROOT.resolve(strict=True)
        requested = Path(os.path.abspath(os.fspath(Path(output))))
    except OSError as exc:
        raise CaptureContractError(
            f"unable to resolve preparation output trust boundary: {exc}"
        ) from exc

    try:
        requested.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise CaptureContractError(
            f"preparation output must be outside the source checkout: {output}"
        )

    try:
        metadata = requested.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise CaptureContractError(
            f"unable to inspect preparation output trust boundary: {exc}"
        ) from exc
    else:
        if stat.S_ISLNK(metadata.st_mode):
            raise CaptureContractError(
                f"preparation output path must not be a symlink: {output}"
            )

    try:
        candidate = requested.resolve(strict=False)
        candidate.relative_to(repository_root)
    except ValueError:
        return
    except OSError as exc:
        raise CaptureContractError(
            f"unable to resolve preparation output trust boundary: {exc}"
        ) from exc
    raise CaptureContractError(
        f"preparation output must be outside the source checkout: {output}"
    )


def _recover_incomplete_preparation(
    output: Path,
    preparation_receipt_path: Path,
    *,
    recovery_repository_commit: str,
) -> str:
    receipt = _core._read_json(preparation_receipt_path)
    final_request = _core._reconstruct_request_from_preparation_receipt(
        receipt,
        recovery_repository_commit=recovery_repository_commit,
    )
    _core.verify_current_reference_environment(receipt["reference_environment"])
    _reauthenticate_revision(
        recovery_repository_commit,
        "incomplete preparation recovery inputs changed before publication",
    )
    _core._exclusive_write_json(output, final_request)
    return _core.sha256_json(final_request)


def _preserve_failed_attempt(
    output_root: Path,
    exc: BaseException,
    repository_commit: str,
    execution_ids: Mapping[str, str],
    preparation_receipt: Mapping[str, Any],
    *,
    replay_verdict_published: bool | None = None,
) -> Path:
    """Archive one failed attempt without misclassifying a published replay verdict."""
    if replay_verdict_published is None:
        try:
            replay_verdict_published = (output_root / "replay-verdict.json").is_file()
        except OSError:
            replay_verdict_published = False
    attempt_status = (
        "failed_after_replay_verdict_publication"
        if replay_verdict_published
        else "failed_before_replay_verdict"
    )
    interpretation = (
        "This directory records an attempt that failed after replay-verdict publication. "
        "The published verdict is retained for audit, but orchestration did not return "
        "cleanly; classify it separately from both pre-verdict failures and clean completed runs."
        if replay_verdict_published
        else (
            "This directory records an incomplete production attempt. It is not a "
            "successful replay result and must not be substituted for the final experiment."
        )
    )
    marker = {
        "schema_version": "1.0.0",
        "protocol_id": "GEO-CAP-001",
        "experiment_id": _core.EXPERIMENT_ID,
        "repository_commit": repository_commit,
        "preparation_repository_commit": preparation_receipt[
            "preparation_repository_commit"
        ],
        "preparation_receipt_sha256": preparation_receipt[
            "preparation_receipt_sha256"
        ],
        "planned_execution_ids": dict(execution_ids),
        "attempt_status": attempt_status,
        "replay_verdict_published": bool(replay_verdict_published),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "completed_run_directories": _core._completed_run_directories(output_root),
        "interpretation": interpretation,
    }
    try:
        _core._exclusive_write_json(output_root / "execution-failure.json", marker)
    except CaptureContractError:
        pass
    failed_path = output_root.with_name(
        f"{output_root.name}.failed-{os.getpid()}-{_core.secrets.token_hex(8)}"
    )
    try:
        os.rename(output_root, failed_path)
    except OSError:
        return output_root
    try:
        _core._fsync_directory(failed_path.parent)
    except OSError:
        return failed_path
    return failed_path


def prepare(
    output: Path,
    implementation_revision: str | None = None,
) -> str:
    """Prepare the frozen request with launcher, runtime, and no-site Hub provenance."""
    _assert_preparation_output_outside_checkout(output)
    preparation_receipt_path = _core._default_preparation_receipt_path(output)
    if output.exists():
        raise CaptureContractError(f"refusing to overwrite existing artifact {output}")

    try:
        preparation_repository_commit = _core.resolve_implementation_revision(
            implementation_revision,
            require_checkout=True,
        )
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"unable to bind online preparation to a clean repository revision: {exc}"
        ) from exc
    _authenticate_external_inputs(preparation_repository_commit)

    if preparation_receipt_path.exists():
        return _core._recover_incomplete_preparation(
            output,
            preparation_receipt_path,
            recovery_repository_commit=preparation_repository_commit,
        )

    reference_environment = _core.verify_current_reference_environment()
    template = _core._load_template(preparation_repository_commit)
    receipts = _core.prepare_tree_receipts(template)
    final_request = json.loads(json.dumps(template))
    final_request["model"].update(receipts)
    final_request = _core._assert_exact_experiment_request(
        final_request,
        require_receipts=True,
        repository_commit=preparation_repository_commit,
    )

    _core.verify_current_reference_environment(reference_environment)
    _reauthenticate_revision(
        preparation_repository_commit,
        "online preparation inputs changed before publication",
    )
    preparation_receipt = _core.build_preparation_receipt(
        request=final_request,
        repository_commit=preparation_repository_commit,
        experiment_id=_core.EXPERIMENT_ID,
        reference_environment_receipt=reference_environment,
    )
    _core.verify_preparation_receipt(
        preparation_receipt,
        request=final_request,
        experiment_id=_core.EXPERIMENT_ID,
    )

    _core._exclusive_write_json(preparation_receipt_path, preparation_receipt)
    _core._exclusive_write_json(output, final_request)
    return _core.sha256_json(final_request)


def observe(
    request_path: Path,
    output_root: Path,
    implementation_revision: str | None,
    preparation_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run two observations only when launcher, lock, and runtime match preparation."""
    if output_root.exists():
        raise CaptureContractError(
            f"refusing to reuse output root {output_root}; observation evidence directories are immutable"
        )
    try:
        repository_commit = _core.resolve_implementation_revision(
            implementation_revision,
            require_checkout=True,
        )
    except SourceIdentityError as exc:
        raise CaptureContractError(
            f"unable to bind production observation to a clean repository revision: {exc}"
        ) from exc
    _authenticate_external_inputs(repository_commit)

    request = _core._assert_exact_experiment_request(
        _core._read_json(request_path),
        require_receipts=True,
        repository_commit=repository_commit,
    )
    source_preparation_receipt_path = (
        Path(preparation_receipt_path)
        if preparation_receipt_path is not None
        else _core._default_preparation_receipt_path(request_path)
    )
    preparation_receipt = _core.verify_preparation_receipt(
        _core._read_json(source_preparation_receipt_path),
        request=request,
        experiment_id=_core.EXPERIMENT_ID,
    )
    expected_environment = preparation_receipt["reference_environment"]
    _core.verify_current_reference_environment(expected_environment)
    _reauthenticate_revision(
        repository_commit,
        "production observation inputs changed before attempt creation",
    )

    attempt_nonce = _core.secrets.token_hex(16)
    execution_ids = {
        "run-a": f"{_core.EXPERIMENT_ID}:{attempt_nonce}:run-a",
        "run-b": f"{_core.EXPERIMENT_ID}:{attempt_nonce}:run-b",
    }
    _core._create_output_root_durable(output_root)
    replay_verdict_published = False

    try:
        validated_request_path = output_root / "validated-request.json"
        archived_preparation_receipt_path = output_root / "preparation-receipt.json"
        _core._exclusive_write_json(validated_request_path, request)
        _core._exclusive_write_json(
            archived_preparation_receipt_path,
            preparation_receipt,
        )
        if _core._read_json(validated_request_path) != request:
            raise CaptureContractError(
                "validated request snapshot changed immediately after publication"
            )
        archived_preparation_receipt = _core.verify_preparation_receipt(
            _core._read_json(archived_preparation_receipt_path),
            request=request,
            experiment_id=_core.EXPERIMENT_ID,
        )
        if archived_preparation_receipt != preparation_receipt:
            raise CaptureContractError(
                "preparation receipt snapshot changed immediately after publication"
            )

        _core.verify_current_reference_environment(expected_environment)
        run_a = output_root / "run-a"
        run_b = output_root / "run-b"
        run_a_execution_receipt = output_root / "run-a-execution-receipt.json"
        run_b_execution_receipt = output_root / "run-b-execution-receipt.json"
        manifest_a = _core._run_capture(
            validated_request_path,
            run_a,
            repository_commit,
            execution_ids["run-a"],
            run_a_execution_receipt,
        )
        manifest_b = _core._run_capture(
            validated_request_path,
            run_b,
            repository_commit,
            execution_ids["run-b"],
            run_b_execution_receipt,
        )

        _core.verify_current_reference_environment(expected_environment)
        _reauthenticate_revision(
            repository_commit,
            "production observation inputs changed before replay-verdict publication",
        )
        verdict = _core.build_replay_verdict(
            request=request,
            validated_request_path=validated_request_path,
            preparation_receipt_path=archived_preparation_receipt_path,
            run_a_dir=run_a,
            run_b_dir=run_b,
            run_a_execution_receipt_path=run_a_execution_receipt,
            run_b_execution_receipt_path=run_b_execution_receipt,
            run_a_execution_id=execution_ids["run-a"],
            run_b_execution_id=execution_ids["run-b"],
            run_a_manifest_receipt=manifest_a,
            run_b_manifest_receipt=manifest_b,
            experiment_id=_core.EXPERIMENT_ID,
        )
        verdict_path = output_root / "replay-verdict.json"
        _core._exclusive_write_json(verdict_path, verdict)
        replay_verdict_published = True
        persisted = _core._read_json(verdict_path)
        _core.verify_replay_verdict(
            persisted,
            request=request,
            validated_request_path=validated_request_path,
            preparation_receipt_path=archived_preparation_receipt_path,
            run_a_dir=run_a,
            run_b_dir=run_b,
            run_a_execution_receipt_path=run_a_execution_receipt,
            run_b_execution_receipt_path=run_b_execution_receipt,
            run_a_execution_id=execution_ids["run-a"],
            run_b_execution_id=execution_ids["run-b"],
            run_a_manifest_receipt=manifest_a,
            run_b_manifest_receipt=manifest_b,
            experiment_id=_core.EXPERIMENT_ID,
        )
        return persisted, 0 if persisted["replay_outcome"] == "byte_identical" else 2
    except BaseException as exc:
        failed_path = _core._preserve_failed_attempt(
            output_root,
            exc,
            repository_commit,
            execution_ids,
            preparation_receipt,
            replay_verdict_published=replay_verdict_published,
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise CaptureContractError(
            f"production observation attempt failed: {exc}; incomplete evidence preserved at {failed_path}"
        ) from exc


_core.LAUNCHER = LAUNCHER
_core.REFERENCE_LOCK = REFERENCE_LOCK
_core.authenticate_tracked_tool_against_revision = authenticate_tracked_tool_against_revision
_core.verify_current_reference_environment = verify_current_reference_environment
_core.prepare_hub_evidence = _isolated_prepare_hub_evidence
_core.prepare_tree_receipts = _isolated_prepare_tree_receipts
_core._assert_preparation_output_outside_checkout = _assert_preparation_output_outside_checkout
_core._recover_incomplete_preparation = _recover_incomplete_preparation
_core._preserve_failed_attempt = _preserve_failed_attempt
_core.prepare = prepare
_core.observe = observe

if __name__ == "__main__":
    print(
        "direct module execution is not an authenticated production boundary; use the "
        "authenticated Git-blob production bootstrap documented in GEO-CAP-001-EXP-001",
        file=sys.stderr,
    )
    raise SystemExit(2)
else:
    sys.modules[__name__] = _core

"""Windows stage-0 for the installed qsol-geo-capture command.

This file is started only by qsol-geo-capture.cmd under CPython -I -S -B. It
uses stdlib metadata to locate the editable checkout without executing .pth
files, binds one Git commit through a fixed Git-for-Windows executable,
authenticates both installed Windows bootstrap files against that commit, then
loads the shared authenticated Python payload from the same Git commit.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse
import urllib.request


if (
    not sys.flags.isolated
    or not sys.flags.no_site
    or not sys.flags.no_user_site
    or not sys.flags.ignore_environment
    or not sys.dont_write_bytecode
):
    raise RuntimeError("qsol-geo-capture Windows stage0 requires CPython -I -S -B")
if os.name != "nt":
    raise RuntimeError("qsol-geo-capture-windows.py is a Windows-only stage0")
if len(sys.argv) < 2:
    raise RuntimeError("qsol-geo-capture Windows stage0 requires its installed wrapper path")

wrapper = pathlib.Path(sys.argv[1]).resolve(strict=True)
stage0 = pathlib.Path(__file__).resolve(strict=True)


def _literal_site_package_paths() -> tuple[pathlib.Path, ...]:
    # Keep the lexical interpreter path where possible so venv installs retain
    # their own Lib/site-packages. A --user wrapper is handled independently by
    # deriving <userbase>/PythonXY/site-packages from the installed Scripts path.
    executable = pathlib.Path(sys.executable)
    if not executable.is_absolute():
        raise RuntimeError("Windows isolated stage0 requires an absolute interpreter pathname")
    venv_root = executable.parent.parent
    candidates: list[pathlib.Path] = []
    if (venv_root / "pyvenv.cfg").is_file():
        candidates.append(venv_root / "Lib" / "site-packages")
    else:
        import sysconfig

        for key in ("purelib", "platlib"):
            value = sysconfig.get_paths().get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(pathlib.Path(value))

        # nt_user installs raw scripts at <userbase>/PythonXY/Scripts and
        # metadata at the sibling site-packages directory. This path is derived
        # from the installed wrapper, not HOME/PYTHONUSERBASE/PATH.
        if wrapper.parent.name.lower() == "scripts":
            candidates.append(wrapper.parent.parent / "site-packages")
    observed: list[pathlib.Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_dir() and resolved not in observed:
            observed.append(resolved)
    if not observed:
        raise RuntimeError("unable to locate installed distribution metadata without site startup")
    return tuple(observed)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _editable_checkout_root() -> pathlib.Path:
    roots: set[pathlib.Path] = set()
    for site_path in _literal_site_package_paths():
        for distribution in importlib.metadata.distributions(path=[str(site_path)]):
            name = distribution.metadata.get("Name", "")
            if _canonical_distribution_name(name) != "qsol-geo-reason":
                continue
            raw = distribution.read_text("direct_url.json")
            if raw is None:
                raise RuntimeError(
                    "qsol-geo-capture requires an editable qsol-geo-reason install to locate its authenticated checkout"
                )
            record = json.loads(raw)
            if record.get("dir_info", {}).get("editable") is not True:
                raise RuntimeError(
                    "qsol-geo-capture requires an editable qsol-geo-reason install to bind source provenance"
                )
            parsed = urllib.parse.urlsplit(record.get("url", ""))
            if parsed.scheme.lower() != "file":
                raise RuntimeError("editable qsol-geo-reason direct_url must use a file URL")
            decoded = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                decoded = f"//{parsed.netloc}{decoded}"
            roots.add(pathlib.Path(decoded).resolve(strict=True))
    if len(roots) != 1:
        raise RuntimeError(
            "qsol-geo-capture could not identify exactly one editable QSOL-GEO-REASON checkout"
        )
    return next(iter(roots))


def _hash_git(candidate: pathlib.Path) -> tuple[pathlib.Path, str]:
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"untrusted Git-for-Windows executable {resolved}")
    return resolved, hashlib.sha256(resolved.read_bytes()).hexdigest()


def _trusted_git_identity() -> tuple[pathlib.Path, str]:
    last_error: BaseException | None = None
    for candidate in (
        pathlib.Path(r"C:\Program Files\Git\cmd\git.exe"),
        pathlib.Path(r"C:\Program Files\Git\bin\git.exe"),
    ):
        try:
            return _hash_git(candidate)
        except OSError as exc:
            last_error = exc
    raise RuntimeError("qsol-geo-capture requires trusted Git for Windows") from last_error


def _git_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper == "PATH" or upper.startswith("GIT_") or upper.startswith("PYTHON"):
            continue
        environment[key] = value
    environment.update(
        {
            "PATH": os.pathsep.join(
                (
                    r"C:\Program Files\Git\cmd",
                    r"C:\Program Files\Git\bin",
                    r"C:\Windows\System32",
                )
            ),
            "LC_ALL": "C",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


root = _editable_checkout_root()
git, git_digest = _trusted_git_identity()


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        [str(git), "-C", str(root), *args],
        env=_git_environment(),
        check=True,
        capture_output=True,
    )
    observed_git, observed_digest = _hash_git(git)
    if observed_git != git or observed_digest != git_digest:
        raise RuntimeError("trusted Git-for-Windows executable changed during stage0")
    return completed.stdout


commit = _git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
    raise RuntimeError("Windows stage0 could not bind a lowercase 40-hex Git commit")

for installed, git_path in (
    (wrapper, "scripts/qsol-geo-capture.cmd"),
    (stage0, "scripts/qsol-geo-capture-windows.py"),
):
    committed = _git("cat-file", "blob", f"{commit}:{git_path}")
    if installed.read_bytes() != committed:
        raise RuntimeError(f"installed {installed.name} does not match the bound Git revision")

payload = _git("cat-file", "blob", commit + ":scripts/qsol-geo-capture-python")
if _git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip() != commit:
    raise RuntimeError("Git HEAD changed while authenticating the Windows standalone payload")

sys.argv = [str(wrapper), *sys.argv[2:]]
namespace = {
    "__name__": "qsol_standalone_authenticated_payload",
    "__file__": str(wrapper),
    "__package__": None,
    "_QSOL_STANDALONE_AUTHENTICATED_ROOT": str(root),
    "_QSOL_STANDALONE_BOOTSTRAP_GIT_PATH": "scripts/qsol-geo-capture.cmd",
}
exec(compile(payload, str(wrapper), "exec"), namespace, namespace)
raise SystemExit(namespace["main"]())
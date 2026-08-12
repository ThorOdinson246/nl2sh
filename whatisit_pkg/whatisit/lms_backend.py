"""LM Studio CLI (`lms`) backend.

Setup stores an absolute path to the `lms` binary and optional model pin.
Runtime never downloads models; it only ls / ps / load / chat / unload.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from . import config as cfg_mod

NL2SH_PREFIX = "nl2sh"
HF_MODELS = "https://huggingface.co/ThorOdinson246"


class LmsError(Exception):
    """User-facing failure talking to the lms CLI or resolving its model."""


class LmsUnavailable(LmsError):
    """Backend cannot run (binary missing, model not on disk, load failed)."""


def find_lms() -> Path | None:
    """Resolve `lms` on PATH to an absolute path, or None."""
    w = shutil.which("lms")
    return Path(w).resolve() if w else None


def lms_path(cfg: dict) -> Path:
    raw = cfg.get("lms_path")
    if not raw:
        raise LmsUnavailable("lms_path not set -- run `whatisit setup`")
    p = Path(raw)
    if not p.is_file():
        raise LmsUnavailable(f"lms_path not found: {p} -- re-run `whatisit setup`")
    return p


def _run(lms: Path, args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    try:
        # Always decode as UTF-8. On Windows, text=True alone uses the active
        # code page (often cp1252), and lms status/spinners emit bytes that
        # crash the stdout reader thread mid-flight.
        return subprocess.run(
            [str(lms), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise LmsUnavailable(f"cannot execute lms at {lms}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise LmsError(f"lms {' '.join(args)} timed out after {timeout:.0f}s") from e


def _run_json(lms: Path, args: list[str], timeout: float = 60.0) -> list | dict:
    p = _run(lms, args, timeout=timeout)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise LmsUnavailable(f"lms {' '.join(args)} failed (rc={p.returncode}): {err[-400:]}")
    text = (p.stdout or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LmsError(f"lms {' '.join(args)} returned non-JSON: {text[:200]!r}") from e


def list_models(lms: Path) -> list[dict]:
    data = _run_json(lms, ["ls", "--json"])
    if not isinstance(data, list):
        raise LmsError("lms ls --json: expected a JSON array")
    return data


def loaded_models(lms: Path) -> list[dict]:
    data = _run_json(lms, ["ps", "--json"])
    if not isinstance(data, list):
        raise LmsError("lms ps --json: expected a JSON array")
    return data


def nl2sh_keys(models: list[dict]) -> list[str]:
    keys = []
    for m in models:
        if m.get("type") and m.get("type") != "llm":
            continue
        key = m.get("modelKey") or ""
        if key.startswith(NL2SH_PREFIX):
            keys.append(key)
    return sorted(set(keys))


def resolve_model_key(cfg: dict, lms: Path | None = None) -> str:
    """Pinned lms_model, else unique nl2sh* key from ls, else raise."""
    pin = cfg.get("lms_model")
    if pin:
        return str(pin)
    lms = lms or lms_path(cfg)
    keys = nl2sh_keys(list_models(lms))
    if len(keys) == 1:
        return keys[0]
    if not keys:
        raise LmsUnavailable(
            "no nl2sh model in LM Studio -- during setup, download one with:\n"
            f"  lms get {HF_MODELS}/nl2sh-<version> -y\n"
            f"  see {HF_MODELS}"
        )
    raise LmsError(
        "multiple nl2sh models in LM Studio; pin one with "
        "`whatisit config --set lms_model=<key>`:\n  " + "\n  ".join(keys)
    )


def is_loaded(lms: Path, key: str) -> bool:
    for m in loaded_models(lms):
        if m.get("modelKey") == key or m.get("identifier") == key:
            return True
    return False


def ensure_loaded(cfg: dict, key: str | None = None) -> str:
    """Load model if needed. Returns the model key used."""
    lms = lms_path(cfg)
    key = key or resolve_model_key(cfg, lms)
    disk_keys = {m.get("modelKey") for m in list_models(lms)}
    if key not in disk_keys:
        raise LmsUnavailable(
            f"lms model {key!r} not in `lms ls` -- run setup / lms get"
        )
    if is_loaded(lms, key):
        return key
    args = ["load", key, "-y"]
    ttl = cfg.get("lms_ttl")
    if ttl is not None and ttl != "" and int(ttl) > 0:
        args.extend(["--ttl", str(int(ttl))])
    p = _run(lms, args, timeout=600.0)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise LmsUnavailable(f"lms load {key} failed (rc={p.returncode}): {err[-400:]}")
    return key


def chat(cfg: dict, system: str, prompt: str, key: str | None = None,
         timeout: float = 300.0) -> str:
    """One-shot chat; returns assistant text on stdout."""
    lms = lms_path(cfg)
    key = ensure_loaded(cfg, key)
    args = [
        "chat", key,
        "--dont-fetch-catalog", "-y",
        "-s", system or cfg_mod.SYSTEM_PROMPT,
        "-p", prompt,
    ]
    p = _run(lms, args, timeout=timeout)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise LmsError(f"lms chat failed (rc={p.returncode}): {err[-400:]}")
    return (p.stdout or "").strip()


def unload_nl2sh(cfg: dict) -> bool:
    """Unload the configured/discovered nl2sh model if loaded. True if unloaded."""
    if not cfg.get("lms_path"):
        return False
    try:
        lms = lms_path(cfg)
        key = resolve_model_key(cfg, lms)
    except LmsError:
        return False
    if not is_loaded(lms, key):
        return False
    # Prefer identifier from ps when present (unload arg is identifier).
    ident = key
    for m in loaded_models(lms):
        if m.get("modelKey") == key or m.get("identifier") == key:
            ident = m.get("identifier") or key
            break
    p = _run(lms, ["unload", ident], timeout=60.0)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise LmsError(f"lms unload {ident} failed (rc={p.returncode}): {err[-400:]}")
    return True


def model_get_instructions() -> str:
    return (
        f"Download an nl2sh model into LM Studio, then re-run setup:\n"
        f"  see {HF_MODELS}\n"
        f"  lms get {HF_MODELS}/nl2sh-<version> -y"
    )

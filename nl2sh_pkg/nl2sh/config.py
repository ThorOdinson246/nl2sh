"""Config + model resolution, following the XDG spec.

Locations (overridable by env, which is what makes the tool testable):
  config  $XDG_CONFIG_HOME/nl2sh/config.json   (~/.config/nl2sh)
  models  $XDG_DATA_HOME/nl2sh/models          (~/.local/share/nl2sh)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

MODEL_NAME = "nl2sh-1.5b-Q4_K_M.gguf"
SYSTEM_PROMPT = (
    "You are a shell command generator. Output exactly one line: a single "
    "POSIX/bash command that accomplishes the user's request. No prose, no "
    "markdown fences, no explanation."
)

DEFAULTS = {
    "threads": 0,          # 0 => auto (see resolve_threads)
    "max_tokens": 64,
    "temperature": 0.0,    # greedy: same question -> same command
    "confirm_execute": True,
}


def config_dir() -> Path:
    return Path(os.environ.get("NL2SH_CONFIG_DIR") or
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nl2sh")


def data_dir() -> Path:
    return Path(os.environ.get("NL2SH_DATA_DIR") or
                Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "nl2sh")


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    p = config_path()
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass  # a corrupt config must not brick the tool
    return cfg


def save_config(cfg: dict) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = config_path()
    p.write_text(json.dumps(cfg, indent=2) + "\n")
    return p


def resolve_threads(cfg: dict) -> int:
    """Default to half the cores, capped at 4.

    Decode is memory-bandwidth-bound (measured: ~31 GB/s saturated by both the
    1.5B and the 4B), so throughput stops scaling with cores well before the
    core count -- and on a laptop, spending every core makes the fans audible
    for no speedup. Half-cores capped at 4 is the knee of that curve.
    """
    t = int(cfg.get("threads") or 0)
    if t > 0:
        return t
    return max(1, min(4, (os.cpu_count() or 4) // 2))


def find_model() -> Path | None:
    if env := os.environ.get("NL2SH_MODEL"):
        p = Path(env)
        return p if p.exists() else None
    for cand in (data_dir() / "models" / MODEL_NAME,
                 Path.cwd() / MODEL_NAME):
        if cand.exists():
            return cand
    return None


def find_llama_cli() -> Path | None:
    if env := os.environ.get("NL2SH_LLAMA_CLI"):
        p = Path(env)
        return p if p.exists() else None
    if w := shutil.which("llama-cli"):
        return Path(w)
    cand = data_dir() / "bin" / "llama-cli"
    return cand if cand.exists() else None

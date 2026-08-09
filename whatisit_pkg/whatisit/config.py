"""Config + model resolution, following the XDG spec.

Locations (overridable by env, which is what makes the tool testable):
  config  $XDG_CONFIG_HOME/whatisit/config.json   (~/.config/whatisit)
  models  $XDG_DATA_HOME/whatisit/models          (~/.local/share/whatisit)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# The published filename on Hugging Face, not our identifier.
MODEL_NAME = "nl2sh-1.5b-Q4_K_M.gguf"

APP_NAME = "whatisit"
LEGACY_NAME = "nl2sh"
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
    # OFF by default. It is cheap (prefix-cached) but MEASURED HARMFUL on the
    # only benchmark available: 54.2% -> 45.1% pass, p=0.0004. See hostctx.py.
    "host_context": False,
}


def env(suffix: str, default: str | None = None) -> str | None:
    """WHATISIT_<suffix>, falling back to NL2SH_<suffix>.

    The fallback is permanent. Dropping it would fail silently: callers that
    set only the old name would get the defaults instead of an error.
    """
    return os.environ.get(f"WHATISIT_{suffix}",
                          os.environ.get(f"NL2SH_{suffix}", default))


def config_dir() -> Path:
    return Path(env("CONFIG_DIR") or
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME)


def data_dir() -> Path:
    return Path(env("DATA_DIR") or
                Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / APP_NAME)


def legacy_dir(kind: str) -> Path:
    base = {"config": (os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")),
            "data": (os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))}[kind]
    return Path(base) / LEGACY_NAME


def migrate_legacy_dirs(echo=print) -> list[str]:
    """Move ~/.config/nl2sh and ~/.local/share/nl2sh to their new names, once.

    Fires only when the new directory is absent and the old one is present, so
    it is idempotent and never overwrites. Skipped when the location came from
    the environment: that path was chosen deliberately.
    """
    msgs = []
    for kind, explicit, new in (("config", env("CONFIG_DIR"), config_dir()),
                                ("data", env("DATA_DIR"), data_dir())):
        old = legacy_dir(kind)
        if explicit or new.exists() or not old.is_dir():
            continue
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
        except OSError as e:
            msgs.append(f"whatisit: could not move {old} to {new}: {e}")
            continue
        msgs.append(f"whatisit: moved {old} -> {new}")
        # setup registers the model and binaries as symlinks. Absolute ones
        # survive the move; a relative one pointing outside now dangles.
        for p in sorted(new.rglob("*")):
            if p.is_symlink() and not p.exists():
                msgs.append(f"whatisit: warning: {p} points at "
                            f"{os.readlink(str(p))}, which no longer resolves")
    for m in msgs:
        echo(m)
    return msgs


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
    os.chmod(d, 0o700)
    p = config_path()
    # Create with 0600 from the start rather than write-then-chmod: the latter
    # leaves a window where the file sits at the umask default -- group-readable
    # on the shared-NFS-home clusters this targets -- readable by a co-tenant.
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    # The 0o600 above applies only at CREATION; a pre-existing file keeps
    # whatever mode it had. fchmod on the open fd closes that gap, race-free
    # because it acts on the fd rather than the path.
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(cfg, indent=2) + "\n")
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
    if v := env("MODEL"):
        p = Path(v)
        return p if p.exists() else None
    for cand in (data_dir() / "models" / MODEL_NAME,
                 Path.cwd() / MODEL_NAME):
        if cand.exists():
            return cand
    return None


def find_llama_cli() -> Path | None:
    if v := env("LLAMA_CLI"):
        p = Path(v)
        return p if p.exists() else None
    if w := shutil.which("llama-cli"):
        return Path(w)
    cand = data_dir() / "bin" / "llama-cli"
    return cand if cand.exists() else None

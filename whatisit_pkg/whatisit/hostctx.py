r"""Tell the model about the machine it is running on.

Why this exists, and why it is shaped this way.

The single most common failure in real use was the model emitting a placeholder
(`cp -r /path/to/dir /tmp`) or naming a tool the host does not have
(`netstat -tuln | grep 5000` on a box with no netstat). Both are the same defect:
the model is guessing at facts it could simply be told.

Measured on the target hardware: putting a host-facts block in the SYSTEM prompt
is nearly free, because llama-server prefix-caches it. A 685-token block cost
11.1 s on the first query and then 1.25 / 0.54 / 1.92 s -- i.e. baseline. Paid
once per session. In the same test the block changed
`netstat -tuln | grep 5000` into `ss -lptn | grep 5000`, fixing a real observed
failure, because the block said netstat was absent and ss shows pids.

Two rules follow from that measurement:

STABLE facts (distro, package manager, which tools exist, shell) go in the
system prompt so the cached prefix stays byte-identical across queries.

VOLATILE facts (cwd, directory listing, git branch) must NOT go there -- they
change per query and would invalidate the cache every time, turning a once-per-
session cost into a per-query one. They are appended to the user turn instead.

Budget: the stable block is capped (see MAX_STABLE_CHARS). The 685-token block
used in the experiment was deliberately inflated to measure the worst case; a
real one is ~150 tokens, about 2.4 s of one-time prefill.

STATUS: DISABLED BY DEFAULT, on measurement.

I originally ranked this the highest-value change on the strength of 4 hand-run
examples. A proper 295-task run reversed that, and the anecdote was simply too
small to rank on:

    no context   0.849 mean / 54.2% pass
    context      0.815 mean / 45.1% pass    d=-0.034, McNemar p=0.0004

Two implementation bugs accounted for part of it (a false working-directory claim
and a key=value format the model read as shell variables); fixing both recovered
39.0% -> 45.1% but did not close the gap. Ruled out as the cause: only 3 of 41
regressions involve a tool the context declared missing.

What actually happens is that the extra tokens make a 1.5B model produce more
elaborate answers, and elaboration breaks simple correct ones:
    `sha512sum f`            -> `echo -n "f" | sha512sum`   (hashes the NAME)
    `echo 'hello' > world.txt` -> `touch /testbed/world.txt`  (drops the content)
    `find .. | xargs wc -l`  -> `find .. -exec wc -l {} \;`  (working idiom traded away)

But the benchmark is close to blind to the BENEFIT this is for. Measured on the
task text: 73% of ALFA tasks already name a concrete target path, so context is
pure noise there, and the underspecified 27% are trivia (`ls`, `pwd`, `date`)
that need no context either. Of real typed queries, 79% are underspecified --
the opposite distribution.

So ALFA measures this feature's cost in full and its benefit barely. The measured
harm is real and sufficient reason not to enable it by default; it is NOT
sufficient reason to conclude the idea is wrong. Revisit only against an eval
built from underspecified requests.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from . import config as cfg_mod

CACHE_TTL = 24 * 3600      # installed tools change rarely
MAX_STABLE_CHARS = 900     # ~200 tokens; keeps first-query prefill near 2-3 s
MAX_ENTRIES = 12           # directory listing truncation

# Tools worth telling the model about, chosen from observed failures rather than
# from a generic "useful commands" list: each one is either a tool the model
# reached for and the host lacked, or the correct modern replacement for one.
PROBE_TOOLS = [
    "ss", "netstat", "lsof", "ip", "ifconfig",       # the port-lookup failures
    "docker", "podman", "kubectl",
    "git", "rg", "fd", "jq", "tree", "ncdu",
    "zip", "unzip", "tar", "xz", "rsync", "curl", "wget",
    "python3", "node", "shellcheck",
    "squeue", "sbatch",                              # HPC: scheduler present?
    "systemctl", "brew", "apt", "dnf", "yum", "pacman", "apk",
]

PKG_MANAGERS = [("apt", "apt"), ("dnf", "dnf"), ("yum", "yum"),
                ("pacman", "pacman"), ("apk", "apk"), ("brew", "brew"),
                ("zypper", "zypper")]


def _distro() -> str:
    try:
        data = dict(
            line.split("=", 1)
            for line in Path("/etc/os-release").read_text().splitlines()
            if "=" in line
        )
        return data.get("PRETTY_NAME", "").strip('"') or platform.system()
    except OSError:
        return platform.system()


def _cache_path() -> Path:
    return cfg_mod.data_dir() / "hostctx.json"


def _probe() -> dict:
    present = [t for t in PROBE_TOOLS if shutil.which(t)]
    missing = [t for t in PROBE_TOOLS if t not in present]
    pkg = next((name for bin_, name in PKG_MANAGERS if shutil.which(bin_)), "unknown")
    return {
        "generated": time.time(),
        "distro": _distro(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "shell": Path(os.environ.get("SHELL", "/bin/sh")).name,
        "pkg": pkg,
        "present": present,
        "missing": missing,
    }


def stable_facts(refresh: bool = False) -> dict:
    """Probe the host, cached to disk. Cheap after the first call."""
    p = _cache_path()
    if not refresh and p.exists():
        try:
            d = json.loads(p.read_text())
            if time.time() - d.get("generated", 0) < CACHE_TTL:
                return d
        except (json.JSONDecodeError, OSError):
            pass
    d = _probe()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2))
    except OSError:
        pass
    return d


def stable_block(facts: dict | None = None) -> str:
    """The part that goes in the system prompt. Must be stable across queries."""
    f = facts or stable_facts()
    lines = [
        "Facts about this machine:",
        f"It runs {f['distro']} on {f['arch']}, shell {f['shell']}, "
        f"package manager {f['pkg']}.",
        f"These tools are available: {' '.join(f['present'])}.",
        f"These are NOT available, do not use them: {' '.join(f['missing'])}.",
    ]
    # Steer to the modern tool when the legacy one is genuinely unavailable.
    if "ss" in f["present"] and "netstat" in f["missing"]:
        lines.append("For listening ports use `ss -lptn` (shows the owning pid); "
                     "netstat is unavailable.")
    if "lsof" in f["missing"] and "ss" in f["present"]:
        lines.append("lsof is unavailable; use `ss -lptn` or `fuser` instead.")
    block = "\n".join(lines)
    return block[:MAX_STABLE_CHARS]


def volatile_block(cwd: Path | None = None) -> str:
    """Per-query facts. Deliberately NOT in the system prompt -- see module docstring."""
    cwd = Path(cwd or Path.cwd())
    # Prose labels, NOT `key=value`. Measured: a `cwd=/testbed` line made the
    # model treat the key as a shell variable and emit `mkdir -p $cwd/test_dir`
    # and `for i in $(echo $cwd_entries ...)`. The context format itself was
    # teaching it to reference variables that do not exist.
    lines = [f"Working directory is {cwd}"]
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in cwd.iterdir()
                         if not p.name.startswith("."))
        shown = entries[:MAX_ENTRIES]
        more = f" (+{len(entries)-len(shown)} more)" if len(entries) > len(shown) else ""
        lines.append(f"It contains: {' '.join(shown)}{more}" if shown else "It is empty.")
    except OSError:
        pass
    if git := _git_state(cwd):
        lines.append(git)
    return "\n".join(lines)


def _git_state(cwd: Path) -> str:
    """Branch and dirtiness, or '' if not a repo. Short timeout: a slow NFS repo
    must never delay a command suggestion."""
    if not shutil.which("git"):
        return ""
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(cwd),
                           capture_output=True, text=True, timeout=1.5)
        if r.returncode != 0:
            return ""
        branch = r.stdout.strip()
        s = subprocess.run(["git", "status", "--porcelain"], cwd=str(cwd),
                           capture_output=True, text=True, timeout=1.5)
        dirty = "dirty" if s.stdout.strip() else "clean"
        return f"It is a git repo on branch {branch}, working tree {dirty}."
    except (subprocess.TimeoutExpired, OSError):
        return ""


def build(prompt: str, enabled: bool = True, cwd: Path | None = None) -> tuple[str, str]:
    """Return (system_prompt, user_message) with context folded in."""
    if not enabled:
        return cfg_mod.SYSTEM_PROMPT, prompt
    system = cfg_mod.SYSTEM_PROMPT + "\n\n" + stable_block()
    user = volatile_block(cwd) + "\n\nRequest: " + prompt
    return system, user

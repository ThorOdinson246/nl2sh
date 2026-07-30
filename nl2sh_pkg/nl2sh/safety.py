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
    # find can delete a whole tree without the word "rm -rf" appearing at all.
    # Caught by testing: `find / -type f -exec rm {} \;` was NOT flagged by the
    # rm patterns above and was only stopped by the interactive-confirm check.
    (re.compile(rf"\bfind\b\s+{CRITICAL_PATHS}(\s|$|/).*?"
                r"(?:-delete\b|-exec\w*\s+(?:sudo\s+)?(?:rm|shred|truncate)\b)"),
     "find + delete across a critical path"),
    (re.compile(rf"\b(chmod|chown)\b.*\s-R\b.*\s{CRITICAL_PATHS}(\s|$|/)"),
     "recursive permission/ownership change on a critical path"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd)"), "redirect over a block device"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "shuts the machine down"),
]

WRITES_OUTSIDE_CWD = re.compile(r"(>|>>|\b(cp|mv|rm|tee|truncate|chown|chmod)\b)[^|]*\s/(etc|usr|var|boot|lib|sbin|bin)/")

# Angle-bracket placeholders are documentation, not shell syntax. Stripping them
# first fixes a false positive found in real output: the `>` inside
# `docker exec -it <container-id> /bin/bash` was read as a redirect into /bin/.
PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]*>")

CAUTIONS = [
    # A PID taken from a parsed column is only a PID if the column was right.
    # Observed twice, both wrong: `netstat ... | awk '{print $NF}' | xargs kill -9`
    # ($NF is the connection state) and `docker ps -a | awk '{print $1}'`
    # (row 1 is the CONTAINER header).
    (re.compile(r"\b(awk|cut|sed)\b[^|]*\|\s*xargs\b[^|]*\b(kill|docker\s+rm|rm)\b"),
     "kills/removes using a field parsed from text -- verify the column is really an ID"),
    # ping with no -c never exits; the user has to notice and Ctrl-C.
    (re.compile(r"^\s*(sudo\s+)?ping\b(?!.*\s-[a-zA-Z]*c\b)(?!.*\s-c\d)"),
     "ping without -c runs until you interrupt it"),
    # This tool is offline by design; a suggestion that leaves the machine is
    # worth naming. Observed: `curl -s http://whatismyip.org | sed ...`.
    (re.compile(r"\b(curl|wget)\b[^|]*\b(https?://|[a-z0-9-]+\.[a-z]{2,})"),
     "contacts the network"),
    (re.compile(r"\bdocker\s+system\s+prune\b.*\s-a\b|\bdocker\s+system\s+prune\s+-a"),
     "docker system prune -a removes all unused images, not just dangling ones"),
]


def split_segments(command: str):
    """Split on shell operators so each clause is checked independently."""
    return [s.strip() for s in re.split(r"(?:\|\||&&|[;|])", command) if s.strip()]


def _for_analysis(command: str) -> str:
    """Blank out <placeholder> spans so they cannot fake shell metacharacters."""
    return PLACEHOLDER_RE.sub("PLACEHOLDER", command)


def check(command: str):
    """Return a list of (severity, reason) findings. Empty means nothing flagged."""
    findings = []
    # Whole-command cautions (a pipeline is the unit of meaning for these).
    whole = _for_analysis(command)
    for pat, why in CAUTIONS:
        if pat.search(whole):
            findings.append(("CAUTION", why))
    for seg in split_segments(whole):
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

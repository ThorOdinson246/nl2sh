"""Turn raw model output into one runnable command line.

This is not cosmetic. Measured on the ALFA eval set, parsing the model's output
properly rather than taking it verbatim was worth about +21 points of pass rate
-- the single largest non-training lever found in the whole project. Small
instruct models leak markdown fences, `$` prompts, and reasoning preambles even
when the system prompt forbids them.
"""
from __future__ import annotations

import re

# Model output is untrusted and goes straight to a terminal. Escape sequences
# can conceal or repaint the command a user is about to approve (e.g. \x1b[8m
# hides text), so they are removed before the string is ever printed,
# safety-checked, or executed.
CONTROL_RE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b-\x1f\x7f]")

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)(?:```|\Z)", re.DOTALL)
PROMPT_RE = re.compile(r"^[\$#>]\s+")
# Prose that small models prepend even when told not to.
PREAMBLE_RE = re.compile(
    r"^(?:sure|certainly|here(?:'s| is)|you can|to do (?:this|that)|the command"
    r"|this command|answer|command)\b[^\n]*?[:\n]", re.IGNORECASE)


def extract(raw: str) -> str:
    if not raw:
        return ""
    text = THINK_RE.sub("", CONTROL_RE.sub("", raw)).strip()
    # Unterminated <think> means the model never stopped reasoning -> no answer.
    if "<think>" in text:
        text = text.split("<think>")[0].strip()
        if not text:
            return ""

    if m := FENCE_RE.search(text):
        text = m.group(1).strip()

    text = PREAMBLE_RE.sub("", text, count=1).strip()

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = PROMPT_RE.sub("", line).strip()
        # Strip surrounding backticks from an inline-code answer.
        if len(line) > 1 and line.startswith("`") and line.endswith("`"):
            line = line[1:-1].strip()
        if line:
            return line
    return ""

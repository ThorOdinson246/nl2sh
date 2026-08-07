# Contributing

This file governs commit hygiene going forward. It does not apply
retroactively — the existing history stays exactly as it is.

## Commit message format

```
<type>(<scope>): <subject>

<body>
```

- **Subject line**: imperative mood ("add", "fix", "close" — not "added",
  "fixes", "closed"), no trailing period, **72 characters or fewer**.
- **Body** (optional but encouraged for anything non-trivial): explain *why*
  the change was made and what it does to observable behavior — not a
  restatement of the diff. If there's a measured before/after number
  (accuracy, latency, test count), put it here or in the subject.
- Wrap body lines at ~72 characters. Leave a blank line between subject and
  body.

### Types

| type | use for |
|---|---|
| `feat` | a new user-facing capability |
| `fix` | correcting broken or wrong behavior (a bug, a security bypass, a wrong number) |
| `chore` | maintenance with no source-behavior change: gitignore, tracked-file cleanup, dependency bumps |
| `docs` | documentation only |
| `test` | adding or changing tests without changing production behavior |
| `refactor` | restructuring code with no behavior change (includes moving/renaming files) |
| `perf` | a performance improvement, ideally with a measured number |
| `ci` | CI/build/packaging configuration |

If a change is hard to place, pick the type that describes the *primary*
effect a reader cares about — a bug fix that happens to touch tests is
still `fix`, not `test`.

### Scopes

Use one of these when the change is localized; omit the scope for anything
repo-wide (e.g. a `.gitignore` chore, a root README update).

| scope | covers |
|---|---|
| `cli` | `nl2sh_pkg/nl2sh/cli.py`, `__main__.py`, argument parsing, user-facing output |
| `safety` | `nl2sh_pkg/nl2sh/safety.py` and its test suite |
| `engine` | `nl2sh_pkg/nl2sh/engine.py`, `hostctx.py`, `extract.py`, `config.py` — the model-serving/inference path |
| `eval` | `scripts/eval/`, benchmark harness, scoring |
| `data` | `scripts/data/`, dataset assembly and cleaning |
| `train` | `scripts/train/`, `scripts/distill/` — training and distillation |
| `docs` | the root `README.md`, package `README.md`, contributor guides |
| `ci` | `.github/`, `pyproject.toml`, packaging/release config |

## Examples, drawn from this project's own history

**Bad** (an actual past subject line — vague, no type, no scope, describes
the author's intent rather than the effect):

```
Queue the retrain and benchmark chain
```

**Good** — same change, made production-shaped:

```
chore(train): queue retrain-and-benchmark run on ALFA-augmented pool

Training data grew from 104k to 123.7k rows after ingesting the
paper's published ALFA train split and removing 34 leaking rows.
Queues the retrain + official-harness rerun to measure the effect
before deciding whether to adopt it.
```

---

**Bad** (result buried in a long, run-on subject with no type/scope):

```
Retrain on the ALFA-augmented pool: 0.560 -> 0.617 official, p=0.043
```

**Good**:

```
feat(train): retrain on ALFA-augmented pool, 0.560 -> 0.617 official

Paired McNemar on the same 300 tasks: b=40 new wins, c=23 old wins,
p=0.043. The 941 MB 1.5B model is now indistinguishable from the
untuned 7B (0.620 vs 0.613, p=0.91). Per-task outputs for both arms
are in scripts/icalfa_udocker/, including the 23 regressions.
```

---

**Bad** (mixes scope into a free-text prefix instead of the conventional
`type(scope):` slot):

```
safety: fix 18+ bypasses found by a third adversarial audit
```

**Good**:

```
fix(safety): close 18 bypasses found by third adversarial audit

Includes wrapper-verb bypasses (/bin/rm, sudo, env, nice, nohup),
shell-string indirection (bash -c "rm -rf /"), .. traversal, and
critical-file targets for shred/unlink. Regression suite:
nl2sh_pkg/tests/test_safety.py, 95/95 at the time of this commit.
```

## Using the commit template

A `.gitmessage` template with the format above is checked in at the repo
root. To have `git commit` (no `-m`) open it by default:

```bash
git config commit.template .gitmessage
```

This is local-only (it edits your own `.git/config`, not anything checked
in), so each contributor opts in individually.

## Optional commit-msg hook

`hooks/commit-msg` can check that a commit subject matches
`type(scope): subject` (or `type: subject` for repo-wide changes) and warn
(not block) otherwise. It is **not** installed automatically — nothing in
this repo touches `.git/hooks` on your behalf. To opt in:

```bash
cp hooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

Remove it the same way (`rm .git/hooks/commit-msg`) at any time.

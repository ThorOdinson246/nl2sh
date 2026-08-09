# Contributing

Thanks for taking a look. Bug reports and small, focused pull requests are
both welcome.

## Getting set up

```bash
git clone https://github.com/ThorOdinson246/whatisit-nl2sh
cd whatisit-nl2sh/whatisit_pkg
pip install -e ".[dev]"
pytest
```

You do not need the model or a `llama.cpp` build to run the test suite. Every
test either mocks the HTTP layer or works on strings, so the whole thing runs
in a couple of seconds on any machine.

To exercise the tool end to end you do need both. See the README's install
section, then `whatisit doctor` to check what is missing.

## Before you open a pull request

```bash
ruff check .
pytest -q --cov=whatisit
```

CI runs the same two commands on Python 3.9 through 3.12 on Linux, plus 3.12
on macOS. It also builds the wheel and sdist, checks the packaging metadata
with `twine`, installs both into clean virtualenvs, and asserts that the CLI
starts and exits with the right codes when no model is present.

Two things that will fail CI and are easy to miss:

- Coverage has a floor. Adding a substantial untested code path will drop it
  below the threshold.
- Do not run `ruff format`. The codebase is hand-formatted and the safety
  tests in particular are laid out deliberately; a reformat produces a huge
  diff that buries the actual change.

## The safety checker

`whatisit_pkg/whatisit/safety.py` decides whether a generated command gets
flagged `DANGER` (never auto-run) or `CAUTION` (warned, still the user's call).
It has 304 regression cases in `tests/test_safety.py`, and it is the one file
where extra care is expected:

- Every rule in it came from a command the model actually produced. If you add
  a rule, add the command that motivated it as a test case.
- Adding a case is always welcome, especially a false positive — a benign
  command that gets flagged is a real bug, because it trains people to ignore
  the warning.
- It is a denylist over a Turing-complete language, not a sandbox. A pull
  request that describes it as making the tool "safe" will get pushback on the
  wording, not the code.

If you find a bypass, a plain issue with the exact command is enough. You do
not need to write the fix.

## Commit messages

```
<type>(<scope>): <subject>
```

Imperative mood ("add", "fix", not "added", "fixes"), no trailing period, 72
characters or fewer. A body is optional; use it to explain why the change was
made and what it does to observable behaviour, rather than restating the diff.
If there is a measured before/after number, put it there.

| type | use for |
|---|---|
| `feat` | a new user-facing capability |
| `fix` | correcting broken or wrong behaviour |
| `docs` | documentation only |
| `test` | tests, without changing production behaviour |
| `refactor` | restructuring with no behaviour change, including renames |
| `perf` | a performance improvement, ideally measured |
| `chore` | maintenance with no source-behaviour change |
| `ci` | CI, build and packaging config |

Scopes are `cli`, `safety`, `engine`, `docs` and `ci`. Omit the scope for
anything repo-wide. If a change is hard to place, pick the type describing the
primary effect a reader cares about — a bug fix that touches tests is still
`fix`.

An example:

```
fix(safety): close 18 bypasses found by an adversarial audit

Covers wrapper verbs (/bin/rm, sudo, env, nice, nohup), shell-string
indirection (bash -c "rm -rf /"), .. traversal, and critical-file
targets for shred and unlink. Each has a regression case.
```

## Scope of the project

This is a small tool with a deliberately narrow remit: turn one plain-English
request into one shell command, locally, with no network. Things like
multi-turn conversation, shell state and remote model backends are out of
scope by design rather than by omission.

The training and evaluation pipeline is not part of this repository.

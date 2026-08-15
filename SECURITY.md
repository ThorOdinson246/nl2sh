# Security Policy

## Supported versions

Only the latest release on PyPI gets fixes. Please upgrade before reporting.

## Reporting a vulnerability

Use [private vulnerability reporting](https://github.com/ThorOdinson246/whatisit-nl2sh/security/advisories/new).
Please do not open a public issue for a security problem.

Include what you ran, what happened, and the output of `whatisit doctor`.
I will acknowledge within a week.

## What counts as a vulnerability

This tool turns natural language into shell commands using a local model. Some
things that look alarming are working as designed, and some are not.

In scope:

- A way to make `whatisit` run a command without the confirmation prompt
- A way for another user on the same machine to read your prompts, reach the
  model server, or make it answer in place of the real one
- Config, token, or log files written with permissions that let others read them
- A path that sends your prompts somewhere you did not configure
- Command injection through a file path, environment variable, or config value

Out of scope:

- The model generating a wrong or destructive command. It is a 1.5B model. The
  safety check is a seatbelt, not a sandbox, and nothing runs without you
  confirming it.
- The safety check missing a dangerous command. Report these as normal issues,
  they are useful, but they are not vulnerabilities.
- Anything that requires you to already have root, or to have deliberately
  pointed the tool at a hostile binary or endpoint.

## Things worth knowing

- **Do not run `whatisit` through `sudo`.** It executes binaries whose paths come
  from environment variables you control. Across a privilege boundary, that is a
  way to run anything as root.
- The model server listens on a UNIX socket inside a `0700` directory with a
  per-run bearer token, so other users on the machine cannot reach it. Setting
  `WHATISIT_FORCE_TCP=1` gives that up in exchange for working on systems where
  the socket path fails to bind.
- If you configure an OpenAI-compatible endpoint, your prompts go to that
  endpoint. That is the point of the feature, but it is worth saying out loud.
- `host_context` is off by default. With it on, the prompt includes your current
  directory and its file listing, which then goes wherever the prompt goes.

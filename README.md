# nl2sh

**Ask for a shell command in plain English. Runs entirely on your own machine,
on CPU — no GPU, no API key, no network.** Answers in about a second.

```console
$ nl2sh find files bigger than 100MB in this folder
find . -size +100M -exec ls -lh {} \;

$ nl2sh compress the logs directory into a tarball
tar -czf logs.tar.gz logs/

$ nl2sh delete everything in the root directory
  !! DANGER  recursive force-delete of a critical path
rm -rf /
```

The model is 941 MB and runs locally through `llama.cpp`. Nothing you type is
ever sent anywhere, which also means it works offline and on machines where
sending shell context to a cloud API would not be acceptable.

## Install

Everything below runs on CPU. No GPU, no API key, and no network at inference
time.

```bash
# 1. the CLI
git clone https://github.com/ThorOdinson246/nl2sh
cd nl2sh && pip install ./nl2sh_pkg

# 2. the model (941 MB)
pip install huggingface_hub
hf download ThorOdinson246/nl2sh-1.5b-Q4_K_M nl2sh-1.5b-Q4_K_M.gguf --local-dir .

# 3. a llama.cpp runtime -- prebuilt binaries from
#    https://github.com/ggml-org/llama.cpp/releases (llama-server, llama-cli)

# 4. wire them together
nl2sh setup --model ./nl2sh-1.5b-Q4_K_M.gguf --bin-dir /path/to/llama.cpp/bin
nl2sh doctor
```

Python 3.9 or newer, on Linux or macOS. There is no PyPI release yet, so the
CLI installs from source.

## Use

Type the request as plain arguments — no quoting needed:

```bash
nl2sh list files changed in the last week
```

| flag | what it does |
|---|---|
| `-e`, `--execute` | run the command after you confirm it |
| `-n N` | show N alternative commands instead of one |
| `-q`, `--quiet` | print only the bare command, for `$(...)` substitution |
| `-t`, `--timing` | report how long generation took |

Commands are never executed unless you pass `-e` and confirm at the prompt,
and anything flagged `DANGER` is never auto-run at all.

```bash
# use the result inline
cd "$(nl2sh -q the directory holding the largest log file)"

# review, then run
nl2sh -e remove every .pyc file under this tree
```

Other subcommands: `nl2sh stop` shuts down the resident model server,
`nl2sh config --set threads=4` changes settings.

## How it works

The first call starts a small `llama.cpp` server that stays resident, so
subsequent calls skip model loading and answer in roughly a second (31.7 tok/s
measured on a Xeon Gold 6426Y using 3 threads). Generation is greedy —
temperature 0 — so the same question gives the same command every time.

The model is Qwen2.5-Coder-1.5B-Instruct with a LoRA fine-tune trained on
125,770 natural-language/shell-command pairs, merged and quantized to GGUF
Q4_K_M. Weights:
[ThorOdinson246/nl2sh-1.5b-Q4_K_M](https://huggingface.co/ThorOdinson246/nl2sh-1.5b-Q4_K_M).

## How good is it

Measured on [InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA),
the official benchmark for this task. It scores a command by *running* it in a
container and comparing the resulting filesystem, file contents and stdout
against a reference. A task passes only on an exact match, across 300 tasks.

| model | size on disk | pass rate |
|---|---|---|
| GPT-4o — cloud API † | — | 0.73 |
| **nl2sh (this tool)** | **941 MB** | **0.620** |
| Qwen2.5-Coder-7B, untuned | 4.4 GB | 0.613 |
| Qwen2.5-Coder-1.5B, untuned — the base this is built on | 941 MB | 0.540 |

**Fine-tuning is what makes a small model usable here.** The same 1.5B base
goes from 0.540 to 0.620 on the identical 300 tasks (+0.080, p = 0.004, exact
McNemar on paired outcomes).

**It holds its own against a model five times its size.** 0.620 against 0.613
for the untuned 7B is a difference of 0.007, 95% CI [−0.050, +0.063],
p = 0.91 — statistically indistinguishable. That is a bound, not a claim of
equality: 300 tasks can only rule out gaps larger than about 5 points. But
shrinking to 941 MB and moving to CPU costs far less than the size gap
suggests.

GPT-4o is ahead, by about 11 points. It is also a cloud service you send your
shell requests to. nl2sh is the local option that gets closest.

<sub>† The GPT-4o figure is the one published by the benchmark's authors; the
other rows were measured with the unmodified upstream scorer at temperature 0,
on all 300 tasks, using paired per-task comparisons.</sub>

## Safety

Every command is checked before it is shown. nl2sh flags recursive deletes of
critical paths, writes to raw block devices, `chmod -R 777 /`, fork bombs,
curl-piped-to-shell, and the same patterns hidden behind `sudo`, `env`,
`nohup`, quoting tricks or `..` traversal. Findings are labelled `DANGER`
(never auto-run) or `CAUTION` (warned, still yours to approve). The checker
ships with a 150-case regression suite run in CI on Python 3.9 through 3.12.

It is a denylist over a Turing-complete language, not a sandbox. It raises the
cost of the common destructive mistakes; it cannot catch every one. On a
held-out set of ordinary prompts it flagged one of three genuinely destructive
outputs. **Read the command before you run it.**

## Limitations

- Single-turn. It does not remember your last command or track shell state.
- It cannot see your filesystem, so requests that depend on what is actually
  on disk ("delete the older backup") may guess wrong.
- Output is capped at 64 tokens, enough for a command and not for a script.
- Quality is measured on one benchmark of 300 tasks. It is not a complete
  measure of shell competence, and it is English-only.
- Trained from one base model family; nothing here shows the recipe carries to
  others.

## Development

```bash
cd nl2sh_pkg
pip install -e ".[dev]"
pytest
```

Tests cover the CLI, the extraction parser, the engine and the safety checker,
and run in CI on Python 3.9 through 3.12.

## Licence

Apache-2.0 for this code. The model derives from Qwen2.5-Coder-1.5B-Instruct,
also Apache-2.0. Training data is assembled from tldr-pages (CC-BY 4.0),
NL2Bash (GPLv3), Fig autocomplete specs (MIT), Warp workflows (Apache 2.0) and
the NL2SH-ALFA published training split (MIT), deduplicated and audited for
contamination against the benchmark test set.

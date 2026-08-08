# nl2sh

Ask for a shell command in plain English. Runs on your own machine, on CPU.
No GPU, no API key, no network. Answers in about a second.

```console
$ nl2sh find files bigger than 100MB in this folder
find . -size +100M -exec ls -lh {} \;

$ nl2sh compress the logs directory into a tarball
tar -czf logs.tar.gz logs/

$ nl2sh delete everything in the root directory
  !! DANGER  recursive force-delete of a critical path
rm -rf /
```

The model is 941 MB and runs through `llama.cpp`. Nothing you type leaves the
machine, so it works offline, and on boxes where piping shell context to a
cloud API isn't allowed.

## Install

Three pieces: the CLI, the model file, and a `llama.cpp` build to run it.

**1. The CLI.**

```bash
git clone https://github.com/ThorOdinson246/nl2sh
cd nl2sh && pip install ./nl2sh_pkg
```

**2. The model, 941 MB.**

```bash
pip install -U huggingface_hub
hf download ThorOdinson246/nl2sh-1.5b-Q4_K_M nl2sh-1.5b-Q4_K_M.gguf --local-dir .
```

If `hf` isn't found, your `huggingface_hub` predates the rename. Either upgrade
it, or use `huggingface-cli download` with the same arguments.

**3. A llama.cpp build.** Grab the release archive for your platform from
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and unzip
it. You need `llama-server` (and `llama-cli` for the fallback path). If you'd
rather build from source or already have it via Homebrew, that's fine too, just
note where `llama-server` ended up.

**4. Point nl2sh at both.**

```bash
nl2sh setup --model ./nl2sh-1.5b-Q4_K_M.gguf --bin-dir /path/to/llama.cpp/bin
nl2sh doctor
```

`--bin-dir` is the directory containing `llama-server`, not the binary itself.
`doctor` tells you which of the three pieces is missing if something's off.

Python 3.9+, Linux or macOS. No PyPI release yet, so the CLI installs from
source.

## Use

Type the request as plain arguments. No quoting needed:

```bash
nl2sh list files changed in the last week
```

| flag | what it does |
|---|---|
| `-e`, `--execute` | run the command after you confirm it |
| `-n N` | show N alternative commands instead of one |
| `-q`, `--quiet` | print only the bare command, for `$(...)` substitution |
| `-t`, `--timing` | report how long generation took |

Nothing runs unless you pass `-e` and confirm at the prompt. Anything flagged
`DANGER` is never auto-run at all.

```bash
# use the result inline
cd "$(nl2sh -q the directory holding the largest log file)"

# review, then run
nl2sh -e remove every .pyc file under this tree
```

`nl2sh stop` shuts down the resident model server. `nl2sh config --set threads=4`
changes settings.

## How it works

First call starts a small `llama.cpp` server and leaves it resident, so later
calls skip model loading and come back in about a second, around 32 tok/s on 3
threads. Three threads is the point: it's a 1.5B at Q4_K_M, so it's bound by
how many cores you give it rather than what machine they're in, and it needs
under 2 GB of RAM. Decoding is greedy at temperature 0, so the same question
always gives the same command.

## Training setup

For anyone who wants to reproduce or fork this.

Base is [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).
LoRA fine-tune, merged into the base weights in bf16, converted to f16 GGUF,
then quantized to Q4_K_M.

| | |
|---|---|
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| target modules | all linear (`q,k,v,o,gate,up,down`) |
| LR / schedule | 2e-4, cosine, 3% warmup |
| epochs | 2, packing off |
| batch | 16 x 2 grad accum, seq len 512 |
| precision | bf16, seed 42 |
| hardware | one A100 80GB, about an hour |
| data | 125,770 NL/command pairs |

A few things that surprised me and might save you time:

**Targeting all linear layers mattered more than rank.** The NAACL 2025 NL2SH
paper fine-tuned this exact base model with r=64 and got *worse* results
(0.21 to 0.19). I used half their rank but hit the MLP layers too, with
alpha/r = 2.0 instead of 0.5 and a 20x higher LR. That flipped the outcome.

**Raising rank past 32 did nothing.** I swept r=64, 128 and 256 holding
everything else fixed. None of them beat r=32 by more than noise (r=128 was
+0.007, McNemar p = 1.0). Biderman et al.'s "code needs r=256" finding did not
transfer here.

**Don't select checkpoints on eval loss.** Eval loss and benchmark accuracy
only correlate at about rho 0.4 on this task. The r=64 run had the *best* eval
loss and nearly the worst accuracy, so `load_best_model_at_end` would have
shipped the wrong model.

**Output parsing is worth a lot on the untuned model and nothing on the tuned
one.** Stripping markdown fences and prose is worth about 15 points on base
Qwen2.5-Coder-1.5B. On the fine-tuned model it's worth zero, because the
fine-tune already taught it to emit a bare command. Nice side effect: none of
the gain below is a post-processing artifact.

**Quantization is a cliff, not a slope.** f16 scores 0.637 and Q4_K_M scores
0.620. Q5_K_M and Q6_K recover none of that 1.7 points, so Q4_K_M is the right
stop.

There's a 3B version of the same recipe that scores 0.657, mostly by being
better on the hard tasks (+9 points there vs +0 on easy ones). It's 2.3x
slower, which is why the 1.5B is the default.

## Benchmarks

Measured on [InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA),
the benchmark from the NAACL 2025 NL2SH paper. It runs each generated command
in a container and diffs the resulting filesystem and stdout against a
reference command. 300 tasks, pass or fail per task.

| model | size on disk | pass rate |
|---|---|---|
| GPT-4o, cloud API † | | 0.73 |
| **nl2sh (this tool)** | **941 MB** | **0.620** |
| Qwen2.5-Coder-7B, untuned | 4.4 GB | 0.613 |
| Qwen2.5-Coder-1.5B, untuned (the base) | 941 MB | 0.540 |

Same base, same 300 tasks, 0.540 to 0.620. That's +0.080 paired, p = 0.004 on
an exact McNemar test.

Against the untuned 7B the difference is 0.007, which 300 tasks can't resolve
(95% CI -0.050 to +0.063). The honest reading is "roughly a 7B", not "beats a
7B". GPT-4o is about 11 points ahead and it's a cloud service you hand your
shell context to.

<sub>† GPT-4o's number is the one published by the benchmark authors. Every
other row I measured myself with the unmodified upstream scorer at temperature
0, `max_tokens=64`, embedding heuristic at threshold 0.75, icalfa 0.3.6.</sub>

## What it gets wrong

Worth knowing before you trust it:

- **It inverts things.** Ask for "smallest first" and you may get largest
  first. Ask for "case sensitive" and get `grep -i`. In adversarial testing
  roughly 1 in 7 commands flipped some aspect of the request. These run
  cleanly and look right, which is the worst kind of wrong.
- On ordinary everyday requests, about 1 in 8 outputs is just wrong.
- It's single-turn. No memory of your last command, no shell state.
- It can't see your filesystem, so "delete the older backup" is a guess.
- Output caps at 64 tokens. That's a command, not a script.
- English only, and measured on one 300-task benchmark, which is not the same
  thing as being good at shell.

## Safety

Every generated command is checked before it's printed. The checker flags
recursive deletes of critical paths, writes to raw block devices, chmod and
chown across system paths, fork bombs, curl piped into a shell, crontab wipes,
firewall flushes, private key exposure, and the same patterns hidden behind
`sudo`, `env`, `nohup`, quoting tricks or `..` traversal. Findings come back as
`DANGER` (never auto-run) or `CAUTION` (warned, still yours to approve). 191
regression cases run in CI on Python 3.9 through 3.12.

It's a denylist over a Turing-complete language, not a sandbox. Every rule in
it came from a command this model actually produced during testing, which means
it covers the mistakes I've seen and not the ones I haven't. Read the command
before you run it.

## Development

```bash
cd nl2sh_pkg
pip install -e ".[dev]"
pytest
```

Tests cover the CLI, the extraction parser, the engine and the safety checker.

## Licence

Apache-2.0 for this code. The model derives from Qwen2.5-Coder-1.5B-Instruct,
also Apache-2.0.

Training data, by measured row share of the 125,770-row pool:

| source | share | licence |
|---|---|---|
| Fig autocomplete specs | 32.8% | MIT |
| tldr-pages | 23.1% | CC-BY-4.0 |
| NL2SH-ALFA training split | 18.0% | MIT |
| cli-commands-explained | 11.8% | CC0-1.0 *(declared, unverified)* |
| command-generation | 7.3% | Apache-2.0 *(declared, unverified)* |
| git-instruction | 7.1% | MIT *(declared, unverified)* |

5.67% is verbatim NL2Bash arriving via the ALFA split. Its `data/bash` is MIT,
not GPL. Warp workflows are not used. The three *declared* sources have
licences I couldn't independently confirm. Deduplicated, with 0 exact and 0
fuzzy matches against the benchmark test set.

**Attribution:** includes content from
[tldr-pages](https://github.com/tldr-pages/tldr) under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (page content is
CC-BY-4.0, only `scripts/` is MIT).

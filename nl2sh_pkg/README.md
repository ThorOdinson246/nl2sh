# nl2sh

Natural language to shell command. Fully local, no API key, no network call.

```console
$ nl2sh look at current queued tasks in slurm
squeue -u $USER

$ nl2sh find files bigger than 100MB in this folder
find . -size +100M -exec ls -lh {} \;

$ nl2sh delete everything in the root directory
  !! DANGER  recursive force-delete of a critical path
rm -rf /
```

## Install

```bash
# not on PyPI yet -- install from source
git clone https://github.com/ThorOdinson246/nl2sh
cd nl2sh && pip install ./nl2sh_pkg

# the model (941 MB)
hf download ThorOdinson246/nl2sh-1.5b-Q4_K_M nl2sh-1.5b-Q4_K_M.gguf --local-dir .

# a llama.cpp runtime: prebuilt binaries from
# https://github.com/ggml-org/llama.cpp/releases

nl2sh setup --model ./nl2sh-1.5b-Q4_K_M.gguf --bin-dir /path/to/llama.cpp/bin
nl2sh doctor
```

`setup` does not fetch anything itself yet; point it at the files above.
`nl2sh doctor` reports exactly what is missing.

## Use

```bash
nl2sh <your request>              # unquoted is fine
nl2sh -n 3 compress this folder   # show 3 alternatives
nl2sh -e count lines in every py file   # run it, after confirming
eval "$(nl2sh -q show disk usage)"      # bare output, for scripting
```

| command | what it does |
|---|---|
| `nl2sh setup --model <gguf>` | register a local model file |
| `nl2sh doctor` | check the install and report what's missing |
| `nl2sh stop` | unload the model from memory |
| `nl2sh config --set threads=3` | change settings |

## Security notes

**Never run `nl2sh` through `sudo`.** Three environment variables
(`NL2SH_LLAMA_SERVER`, `NL2SH_LLAMA_CLI`, `NL2SH_RUNTIME_LIB`) point at the
binaries and shared libraries it executes. That is harmless when it is your own
environment running as you, but across a privilege boundary -- `sudo -E`, a cron
or setuid wrapper -- they become a way to run an arbitrary binary as root.

The model server listens on a **UNIX socket** inside a `0700` directory, not a
TCP port, and requires a per-run bearer token. On a shared machine loopback is
reachable by every other user, so a TCP port would let a co-tenant use your model
or -- worse -- claim the port first and answer in place of the real server with a
command of their choosing. Config, query log, pid, token and server log are all
`0600`.

**The safety check is a seatbelt, not a sandbox.** It is a denylist over a
Turing-complete language: `eval`, base64 indirection and aliasing defeat any
static check. The real protection is that nothing runs unless you ask it to.

## Design notes

**It never runs anything on its own.** The default prints the command and
stops. `-e` runs it, but only after an interactive confirmation, and it
*refuses* on anything flagged `DANGER` — you have to copy those yourself.
Compound commands are split on `;`, `&&`, `||` and `|` and every segment is
checked independently, because a plausible first clause followed by a
destructive one is a real observed failure mode.

**The model stays resident.** The first query starts a small local server that
holds the model in RAM; later queries reuse it. Reloading a ~1 GB model per
invocation costs several seconds and is most of what makes local tools feel
slow. `nl2sh stop` unloads it.

**Threads default to half your cores, capped at 4.** Decoding is limited by
memory bandwidth, not compute — measured at ~31 GB/s saturated by both a 1.5B
and a 4B model — so throughput stops improving well before your core count,
and spending every core only spins the fans.

**Zero Python dependencies.** `pip install nl2sh` cannot disturb anything else
in your environment.

## Model

A 1.5B-parameter model fine-tuned for this one job and quantised to Q4_K_M
(941 MB): [ThorOdinson246/nl2sh-1.5b-Q4_K_M](https://huggingface.co/ThorOdinson246/nl2sh-1.5b-Q4_K_M).

On [InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA) — 300 tasks
scored by *executing* each command in a container and comparing filesystem
state, file contents and stdout against a reference — it scores **0.620**,
against 0.540 for the untuned base it was fine-tuned from (+0.080, p = 0.004).
That puts it level with an untuned Qwen2.5-Coder-7B at 0.613: a difference of
0.007, p = 0.91, statistically indistinguishable at roughly a fifth the size.

It is not as good as a frontier model — GPT-4o is about 11 points ahead on the
same benchmark — and it is weakest on multi-stage pipelines. It is meant for
the one-liner you'd otherwise go and look up.

## Licence

Apache-2.0, as is the Qwen2.5-Coder base model it derives from.

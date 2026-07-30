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
pip install nl2sh
nl2sh setup --model /path/to/nl2sh-1.5b-Q4_K_M.gguf
```

> **Not yet on PyPI, and `setup` does not download the model yet.** Point it at
> a GGUF you already have; a released build would fetch it from a model hub.
> `nl2sh doctor` tells you exactly what is missing.

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
(~940 MB). On a benchmark of 294 real shell tasks scored by *executing* the
commands and comparing filesystem and stdout effects, it matches a 4B general
model at 2.55× smaller and 2.56× faster — and it is the only one of the two
that answers inside the 1–2 s budget this tool is built around.

It is not as good as a frontier model, and it is weakest on multi-stage
pipelines. It is meant for the one-liner you'd otherwise go and look up.

## License

Apache-2.0.

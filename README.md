# nl2sh

Natural language to shell command, running entirely on your own CPU. You
type what you want in plain English; it prints (or, if you ask, runs) the
shell command that does it — no API key, no network call, no data leaving
your machine.

```console
$ nl2sh find files bigger than 100MB in this folder
find . -size +100M -exec ls -lh {} \;

$ nl2sh delete everything in the root directory
  !! DANGER  recursive force-delete of a critical path
rm -rf /
```

## Headline result

On [InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA) (the
official evaluation harness from the NL2SH paper — 300 tasks, scored by
*executing* each generated command and comparing filesystem effects, file
content and stdout against a reference; `reward == 1.0` is a strict pass,
no partial credit):

| system | params | accuracy |
|---|---|---|
| gpt-4o-2024-08-06 (their SOTA, closed) | — | 0.74 |
| **this model** (Qwen2.5-Coder-1.5B + LoRA SFT, Q4_K_M, 941 MB) | **1.5B** | **0.617** |
| their Qwen2.5-Coder-7B, untuned | 7B | 0.61 |
| their 3B fine-tune (their shipped Ollama model) | 3B | 0.51 |
| their 7B fine-tune | 7B | 0.51 |
| their 1.5B fine-tune — same base model, same size as this one | 1.5B | 0.19 |

The model is **941 MB on disk, CPU-only, and answers a realistic query in
about 1 second** (31.7 tok/s generation, measured on a Xeon Gold 6426Y with
3 threads). It edges an untuned 7B model at roughly 1/5th the size, and
beats a same-size, same-base fine-tune from the paper this benchmark comes
from by more than 3x. See `docs/PRIOR_ART.md` for the full methodology,
instrument-validation numbers, and everything documented about how this
comparison was made fair.

## Honest limitations

- **Weak on multi-constraint requests.** It reliably handles requests that
  map to a single flag (`git reset --soft`, `ping -c 3`). It is
  considerably less reliable when a request requires composing a pipeline
  (e.g. "biggest files, sorted" needing `sort` on top of `find`) or stacking
  several constraints at once ("recursively, excluding hidden files, case
  insensitive"). This is the model's most significant, measured weakness —
  see the "compositional gap" analysis in `docs/STATUS.md`.
- **Roughly 7 in 10 on everyday requests.** On 150 ordinary developer
  prompts (git, file organisation, search, archives, disk cleanup,
  processes and ports, docker, python/node workflow, permissions,
  networking, text processing, system info) it was correct 70.7% of the
  time and correct-or-workable 80.0%. 59 of those verdicts were confirmed
  by actually running the command against fixtures. Worth knowing *where*
  the other 30% falls: not exotic edge cases, but routine commands --
  `git status` answered with `git remote -v`, `npm run` with no script
  name, a `sort` missing `-n` so "9.5" ranks above "55.0". Those are
  exactly the commands least likely to be double-checked before running.
- **Measured 36% on adversarial prompts.** The 0.617 figure above is the
  published benchmark. A separate audit wrote 136 prompts specifically to
  attack known weak spots and the model answered 36% of them correctly:
  93% on non-English and typo'd input, 57% on slang, but 23% on requests
  carrying three or more constraints, 18% on one-word queries, and 8% on
  genuinely ambiguous requests like "clean up this mess" — where it invents
  a specific interpretation rather than asking. Treat the benchmark number as
  best-case and this as the floor.
- **Prompt injection is not resisted at the model level.** Nothing about
  the model's training makes it robust to adversarial input trying to
  manipulate what command it produces. Do not point it at untrusted text.
- **The safety checker is a seatbelt, not a sandbox.** It statically
  flags destructive patterns (`rm -rf /`, `dd` over a device, curl-pipe-bash,
  and dozens of known bypasses of each) before anything runs. It is a
  denylist over a Turing-complete language: `eval`, base64 indirection, and
  enough aliasing will defeat any static check. The real protection this
  tool offers is that **nothing runs unless you explicitly ask it to** —
  the default is print-only, and `-e` still refuses anything flagged
  `DANGER` outright.

## Install / quickstart

```bash
pip install nl2sh          # not yet on PyPI — see nl2sh_pkg/README.md
nl2sh setup --model /path/to/nl2sh-1.5b-Q4_K_M.gguf
nl2sh find files bigger than 100MB in this folder
```

`setup` does not download the model for you yet — point it at a GGUF you
already have. `nl2sh doctor` reports exactly what's missing. Full install,
usage, and security notes are in `nl2sh_pkg/README.md` (the package-level
README shipped with `pip install nl2sh`).

## Repo layout

```
nl2sh_pkg/          the installable package: cli, engine, safety checker, tests
  nl2sh/            source
  tests/            test_safety.py — 140/140 passing regression suite
scripts/
  data/             dataset assembly, cleaning, contamination checks
  distill/          teacher-data generation for distillation
  train/            LoRA training and merging
  export/           GGUF export and CPU benchmarking
  eval/             the InterCode-ALFA benchmark harness and scoring
  icalfa_udocker/   rootless-container plumbing for running the official
                    icalfa harness on a cluster without a Docker daemon
  icalfa_feasibility/  one-off spike that established the udocker approach
  infra/            environment/shim scripts
docs/               design decisions, measurements, and project history
  STATUS.md         single-page current state (start here)
  PRIOR_ART.md       full benchmark methodology and comparison to the paper
  ARCHITECTURE_REVIEW.md, DATASETS.md, DISTILLATION.md, EVAL_HARNESS.md,
  TEACHER.md, LOG.md, FINAL_REPORT.md  deeper dives on each subsystem
```

Scripts under `scripts/` were written for a specific SLURM cluster and are
kept as-is for reproducibility and auditability, not as a portable
pipeline — several `.sbatch` files hardcode cluster paths and partition
names.

## Reproducing the benchmark

The official harness setup and every deviation from the paper's own
Docker-based scoring (container transport, working-directory defaults,
provisioning quirks) is documented in `docs/PRIOR_ART.md` sections 6-7,
along with the exact per-fixture breakdown and instrument-validation
(gold-vs-gold) numbers. `docs/EVAL_HARNESS.md` covers the harness
internals; `scripts/eval/` and `scripts/icalfa_udocker/` contain the code
that ran it.

## Training

`docs/DATASETS.md` documents where the training pool comes from and how it
was cleaned and de-contaminated; `docs/DISTILLATION.md` covers the
teacher-data generation approach and the literature it draws on.
`scripts/train/train_lora.py` (driven by the `.sbatch` files alongside it)
runs the LoRA SFT; `scripts/train/merge_lora.py` merges the adapter back
into the base model before GGUF export via `scripts/export/`.

## Further reading

Everything in `docs/` — start with `docs/STATUS.md` for a single current
snapshot, then follow its links into whichever subsystem you want depth
on.

## License

Apache-2.0.

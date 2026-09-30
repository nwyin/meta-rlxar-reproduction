# Meta blog XAR reproduction

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
Muse Spark 1.1 writes missing paper sections, generates rubrics and judges; Kimi K2.6 rewrites
the rubric meta prompt. One trajectory covers 52 sections from 8 training and 5 validation
papers, with 7 prompt updates. [METHOD.md](METHOD.md) describes the data, the procedure, how
this differs from the blog, and the limitations.

## Result

The research run finished on September 29, 2026.

- The checkpoint selected on training was P1. At P1 the validation gap (author score minus
  model score) was -0.16, up from -1.04 at the initial prompt. The last checkpoint, P7, had a
  validation gap of -0.37. The gap stayed negative at every checkpoint.
- The blog's reversal, from -4.2 to +2.76, was not reproduced.
- The paired improvement from P0 to P1 on validation was 0.88, with a 95% whole-paper
  bootstrap interval of 0.37 to 1.35. Author scores rose from 7.91 to 8.38 and model scores
  fell from 8.95 to 8.53.
- 12 of the 20 validation sections met the ±15% length target. On those 12 the gap went from
  -1.12 to -0.26.
- The research run cost $55.60 for 1,389 requests and took 75 minutes. The pilot cost $2.42.

`run.py report` writes the full numbers to `reports/results.md`.

## Setup

You need Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and the Hugging Face `hf`
CLI (used to download the tokenizers).

```sh
uv sync --locked
uv run python run.py prepare-tokenizers
uv run python run.py prepare-data
uv run python run.py validate-data
uv run pytest -q
```

`prepare-tokenizers` downloads the Kimi and Qwen tokenizers pinned in
`configs/tokenizers.json` into `data/tokenizers/`. `prepare-data` downloads the 20 arXiv HTML
papers listed in `data/source_manifest.json` into `data/raw/` and rebuilds
`data/examples.jsonl`. It fails if the result does not match the saved split and manifest,
for example because arXiv changed a page.

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## Run

```sh
uv run python run.py all --dry-run
uv run python run.py all --budget-usd 100 --total-budget-usd 100
```

`--dry-run` prints a cost estimate and sends nothing. `all` runs `pilot` and then
`reproduce`, which you can also run on their own:

- `run.py pilot` runs one update on the two pilot papers to check the pipeline end to end.
- `run.py reproduce` audits the pilot, runs the research trajectory, and writes `reports/`.

Paid runs need both limits: `--budget-usd` for this run and `--total-budget-usd` for all runs
that share `runs/budget_ledger.json`. `--concurrency` sets the number of parallel requests
(default 4, from `configs/experiments.yaml`). `--resume` continues an interrupted run; it
stops if anything other than the budgets has changed since the run started.

## Audit

```sh
uv run python run.py audit-run --source-run runs/meta-blog-seed0
uv run python scripts/audit_completed_run.py
uv run python run.py report
```

`audit-run` rebuilds a run's checkpoint table from its saved raw responses and checks the
request settings, the anonymous grading inputs, the scoring arithmetic, the training-only
selection and that validation came after it. `scripts/audit_completed_run.py` goes further on
the completed research run: it checks the manifest's code hashes against the commit that
produced the run, re-derives the writer inputs, optimizer feedback and proposal checks with
the current code, and writes the cost, request, latency and length-subset numbers to
`reports/completion_audit.json`. Neither makes API calls. `report` runs the audit and then writes the report.

## Outputs

| Path | Contents |
| --- | --- |
| `runs/pilot-meta-blog-attested/` | the pilot run |
| `runs/meta-blog-seed0/` | the research run: `manifest.json`, every request and response under `requests/`, sections in `generations/`, `rubrics/`, `scores/`, optimizer `feedback/`, `prompts/`, `freeze.json`, `costs.json` |
| `runs/budget_ledger.json` | reservations and charges for every run |
| `reports/` | `results.md`, `checkpoints.csv`, `gap_curves.png` and `.svg`, `blog_comparison.json`, `audit.json` |

`runs/`, `reports/`, the paper text and the tokenizers are not in git.

## Versions

The git tag `meta-blog-seed0` is the commit that produced the research run. The code has been
cleaned up since; `scripts/audit_completed_run.py` checks that the saved run still re-audits
with the current code. The tag `alternative-model-study` holds an earlier study with other
models, which did not go beyond pilots.

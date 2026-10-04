# Meta blog XAR reproduction

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
The reproduction uses 2 scripts: `fetch_data.py` restores the frozen paper corpus, and
`run.py` writes missing sections, optimizes the rubric meta prompt, and evaluates the result.

The current settings use Muse Spark 1.3, on OpenRouter's contributor tier, as writer,
rubric generator and judge, and Kimi K2.6 as optimizer. The corpus holds 56 peer-reviewed
papers from 2016 to 2021 across 9 fields, giving 153 sections. [METHOD.md](METHOD.md)
describes the experiment and its limits. [LOG.md](LOG.md) records the runs and changes.

## Historical result

The research run finished on September 29, 2026. It used the earlier cs.CL corpus, archived in
`data/archive/arxiv-2609-cs-cl/`.

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

The saved results remain in `reports/` and `runs/`. The `meta-blog-seed0` Git tag holds
the code for this historical run. The current scripts support fresh runs.

## Setup and data

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run python fetch_data.py
```

`fetch_data.py` reads `data/source_manifest.json` and `data/splits.json`, downloads missing
raw ar5iv HTML, and rebuilds `data/examples.jsonl`. It checks the saved paper and dataset
hashes and keeps the original provenance. Cached HTML is reused. A changed source or
conflicting local file stops the script. The selected papers and splits stay fixed.

Discovery and tokenizer downloads were part of building the corpus. They are no longer
setup steps. The corpus-construction code remains in Git history. To check local data
without downloads or writes:

```sh
uv run python fetch_data.py --check
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## Run

Estimate a full run before spending money, then use a fresh output directory:

```sh
uv run python run.py --dry-run
uv run python run.py --output runs/reproduction-01
```

For a small pilot on 2 training papers with 1 prompt update:

```sh
uv run python run.py --pilot --dry-run
uv run python run.py --pilot --output runs/pilot-01
```

The pilot uses 1 paper for training and 1 for validation. Its scores stay separate from a
full run. `--seed` controls grading order and bootstrap sampling, not model decoding.
`--concurrency` controls parallel requests. Other settings come from
`configs/experiments.yaml` and `configs/models.yaml`; prompts stay in `prompts/`.
`uv run python run.py --help` lists the options.

`--dry-run` estimates cost from saved prices and sends no requests. Spending is capped by
the OpenRouter key's limit, set on openrouter.ai. Each run records raw requests, responses,
billed costs and unresolved sends. The code enforces no account budget of its own.

Each run requires a new output directory. Interrupted runs keep their evidence; start a new
run with a different directory. The runner has no resume or historical-report command.

## Outputs

The run directory contains the whole result:

| Path | Contents |
| --- | --- |
| `results.md`, `checkpoints.csv`, `summary.json` | The result summary, checkpoint scores and uncertainty estimates. |
| `manifest.json` | The settings, model endpoints, seed and source hashes. |
| `requests/` | Raw requests, responses and transport attempts. |
| `generations/`, `rubrics/`, `scores/` | Fixed writer drafts, rubrics and blind grades. |
| `feedback/`, `prompts/`, `freeze.json` | Training feedback, prompt history and selection before validation. |
| `costs.json`, `status.json` | Billed costs, unresolved sends and run status. |

`runs/`, `reports/`, paper text and raw HTML stay local. `runs/budget_ledger.json` remains
as the record of runs before 2026-10-01.

## Repository layout

```text
fetch_data.py          Restore and check the frozen corpus from raw paper HTML.
run.py                 OpenRouter calls, optimization, validation and result summary.
configs/               Experiment settings, models and saved endpoint listings.
prompts/               Writer, rubric, judge and optimizer prompts.
data/                  Frozen manifests, splits and local paper text.
tools/data-viewer/     Optional local viewer for saved runs.
METHOD.md              Method and limitations.
LOG.md                 Experiment history.
```

The optional viewer runs with `uv run python tools/data-viewer/serve.py`;
see [its README](tools/data-viewer/README.md).

```sh
uv run ruff check .
```

Validate code changes with temporary checks during the session, then discard them. The
repository keeps no test suite. `AGENTS.md` describes this workflow. Use
`fetch_data.py --check` to verify local corpus files against their frozen hashes.

The tags `meta-blog-seed0` and `alternative-model-study` preserve earlier implementations
and experiments.

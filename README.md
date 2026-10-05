# Meta blog XAR reproduction

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
The reproduction uses 3 scripts. `fetch_data.py` restores the frozen paper corpus.
`fetch_fiction.py` builds and restores the blog's story-continuation corpus from Project
Gutenberg. `run.py` writes the model's sections or continuations, optimizes the rubric meta
prompt, and evaluates the result on either corpus.

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

### Fiction

```sh
uv run python fetch_fiction.py           # download the frozen books and build data/fiction/examples.jsonl
uv run python fetch_fiction.py --check   # verify local files without downloads or writes
```

Building the corpus from scratch takes 3 steps. `discover` reads the Gutenberg catalogs,
chooses lesser-read novels by the authors in `data/fiction/authors.json` and proposes 6 cuts
per book. `probe` is paid: the writer continues every cut at temperature 0, and `freeze` drops
the passages it reproduced and fills the splits. [METHOD.md](METHOD.md#fiction-corpus) gives
the rules.

```sh
uv run python fetch_fiction.py discover
uv run python fetch_fiction.py probe --output runs/probe-fiction-<date>
uv run python fetch_fiction.py freeze
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## Run

A config file defines each run: its output directory, split, seed, iterations, dataset,
concurrency and limits. `run.py` takes the path to one and has no other options. Set a new
`output` directory in the file first, because a run refuses to start in a directory that exists.

`configs/arxiv.yaml` defines the paper run and is the default. `configs/fiction.yaml` defines
the fiction run:

```sh
uv run python run.py
uv run python run.py configs/fiction.yaml
```

For a dry run, edit the config temporarily: set `split: pilot`, `iterations: 1` and a new
`output`. The run then trains on the first training source (a paper or a book), validates on
the second, and makes 1 prompt update. Restore the config afterward. `seed` controls grading
order, not model decoding. The config's `roles` section sets each role's model, provider and decoding
settings, and its `prompts` key names the prompt directory, `prompts/papers` or `prompts/fiction`. The run copies its config into its directory as `config.yaml`.

The runner trusts the prepared examples and their split labels. It enforces the initial
meta-prompt word limit. It sends the configured model, provider and settings directly to
OpenRouter and records the returned model and provider. Saved endpoint listings are
historical records; provider changes can affect reproducibility.

OpenRouter calculates billing and caps spending at the key's limit, set on openrouter.ai.
The report uses recorded charges after the run. Each run saves raw requests and responses,
including OpenRouter's usage fields and generation IDs. Missing billing fields leave an
otherwise valid response usable.

After scoring, the runner looks up each unique generation ID through OpenRouter's API,
including IDs from retried requests. It sums the returned `total_cost` values and writes
them to `costs.json`, `summary.json` and `results.md`. The report marks the total as partial
when a lookup fails or a send has no generation ID. `costs.json` records each retrieved
charge, lookup error and attempt missing an ID. Connection failures before sending are
excluded. Billing lookup failures leave the experiment results available.

You can also check [OpenRouter Activity](https://openrouter.ai/activity), or query its
API with the same bearer key:

- `GET https://openrouter.ai/api/v1/key` returns `data.usage` and `data.limit_remaining`
  for the key. These totals cover all activity on that key.
  [Key API](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key).
- `GET https://openrouter.ai/api/v1/generation?id=gen-…` returns `data.total_cost` for a
  generation. Use `response_id` from a saved `requests/*/attempt_*.json` record; older
  records keep the ID in `response.id`. A request whose connection failed before an ID
  arrived needs investigation in OpenRouter Activity.
  [Generation API](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation).

Each run requires a new output directory. Interrupted runs keep their evidence; start a new
run with a different directory. The runner has no resume or historical-report command.

## Outputs

The run directory contains the whole result:

| Path | Contents |
| --- | --- |
| `results.md`, `checkpoints.csv`, `summary.json` | The result summary and the author mean, model mean and gap at each checkpoint. |
| `config.yaml` | A copy of the config file that defined the run. |
| `manifest.json` | The config, configured models and providers, and the dataset hash. |
| `requests/` | Raw requests, responses and transport attempts. |
| `generations/`, `rubrics/`, `scores/` | Fixed writer drafts, rubrics and blind grades. |
| `feedback/`, `prompts/`, `freeze.json` | Training feedback, prompt history and selection before validation. |
| `costs.json` | Charges retrieved from OpenRouter, lookup errors and sends missing generation IDs. |
| `status.json` | Run status and any stopping error. |

`runs/`, `reports/`, paper text, raw HTML, book text and both `examples.jsonl` files stay local. `runs/budget_ledger.json` remains
as the record of runs before 2026-10-01.

## Repository layout

```text
fetch_data.py          Restore and check the frozen paper corpus from raw paper HTML.
fetch_fiction.py       Build, freeze, restore and check the Gutenberg fiction corpus.
run.py                 OpenRouter calls, optimization, validation and result summary.
configs/               Experiment settings, models and saved endpoint listings.
prompts/papers/        Writer, rubric, judge and optimizer prompts for papers.
prompts/fiction/       The same prompts for story continuation.
data/                  Frozen paper manifests, splits and local paper text.
data/fiction/          Fiction authors, discovery, probe, manifests and splits; local book text.
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
`fetch_data.py --check` and `fetch_fiction.py --check` to verify local corpus files against
their frozen hashes.

The tags `meta-blog-seed0` and `alternative-model-study` preserve earlier implementations
and experiments.

# Meta blog XAR reproduction

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
The core reproduction uses 3 scripts. `fetch_data.py` restores the frozen paper corpus.
`fetch_fiction.py` builds and restores the blog's story-continuation corpus from Project
Gutenberg. `run.py` writes the model's sections or continuations, optimizes the rubric meta
prompt, and evaluates the result on either corpus.

The current settings use Muse Spark 1.3, on OpenRouter's contributor tier, as writer,
rubric generator and judge, and Kimi K2.6 as optimizer. The corpus holds 56 peer-reviewed
papers from 2016 to 2021 across 9 fields, giving 153 sections. [METHOD.md](METHOD.md)
describes the experiment and its limits. [LOG.md](LOG.md) records the runs and changes.

## Reduced experiments (October 7–9, 2026)

The cloud search resumed on October 9 after the OpenRouter account was funded again.
The preceding funding stop occurred at $102.16 of key usage; account credits and the key's
spending allowance are separate. The first completed batch favored human stories in 3 of
3 trials and human paper sections in 2 of 3 trials, with 1 tie. The second batch is running.
Consistent reproduction across the required 2 batches remains unestablished.
[PR #7](https://github.com/nwyin/meta-rlxar-reproduction/pull/7) records the reduced search;
[PR #8](https://github.com/nwyin/meta-rlxar-reproduction/pull/8) records the revised paper
method. [LOG.md](LOG.md) records the checks and funding history.

The project has reopened for a budget-limited search of meta-optimization methods.
Both default configs now select 10% of each active split, rounded to the nearest
example: papers use 10 training and 4 validation sections; fiction uses 3 training
and 2 validation passages. Sampling rotates across shuffled sources within each
frozen split. `sample_seed` controls this selection independently of grading order.
The frozen datasets remain intact. Each run records the selected IDs, code hash,
and complete input prompt snapshots.

The loop now supports `feedback_policy: full_context` (nonpositive pairs with their
visible context) and `all_pairs_context` (the lowest-gap pairs, including positive
pairs), alongside the original `failing_gap_without_context` policy. `failure_examples`
bounds detailed pairs. Contexts are sent whole. `train_only: true` excludes validation
from generation and grading while searching methods. `reuse_candidates: runs/<run>`
reuses drafts only after checking their writer configuration, context and text hashes.
Runs still require fresh output directories. Optional `best_parent` and `optimizer_history`
keep the search anchored to its strongest training checkpoint. `proposal_rationale_first`
and `proposal_plaintext` compare output formats; `balanced_feedback` rotates across section
types when choosing detailed examples. `replicate.py` runs 3 independent reduced repetitions
of 2 frozen method configs and reports validation consistency with source-level intervals.

The initial campaign compares current settings with Muse Spark 1.1 and contextual
feedback. Prompt variants live in `prompts/experiments/`. Configs, logs, raw calls and
results live locally under `runs/campaign-20261007/`. The current configs retain Muse
Spark 1.3; every experimental model change appears in its saved config.

The historical results below remain evidence from the earlier investigation.
New results require separate reporting, including failures and repeat evaluations.

## Continuing cloud search

`configs/search-arxiv.yaml` and `configs/search-fiction.yaml` define the current Muse
Spark 1.1 / Kimi search methods. The default corpus configs still use Muse Spark 1.3.
All new default and search runs require complete drafts within ±15% of the target.
Length revisions use the same model as a focused editor of its complete draft.

```sh
uv run python campaign.py runs/cloud-search-NEW configs/search-arxiv.yaml configs/search-fiction.yaml
```

This is a paid, continuing command. The controller runs 3 fresh reduced trials per domain,
then uses training evidence to revise optimizer instructions and loop settings. A passing
method stays fixed for another 3 trials. It stops successfully after 2 consecutive passing
batches per domain, stops when the OpenRouter budget blocks calls, or records a technical
block after bounded recovery fails. It never changes the key's spending limit or substitutes
models. Validation controls stopping and remains outside the instruction designer's input.
Repeated validation checks are exploratory; confirmation remains unused.

The cloud controller writes `runs/cloud-search-20261007/report.md`, `status.json`,
`budget.json`, and per-run requests, prompts, scores, and costs. These files remain local
to the cloud workspace. The diagnostic report is `reports/2026-10-07-diagnostics.md`.

## Historical result

We closed the project on October 5, 2026 with a null result. No run reproduced the blog's
reversal, in which the validation gap (author score minus model score) went from -4.2 to
+2.76. On papers the optimized rubrics narrowed a negative gap but never made it reliably
positive. On fiction the author led before optimization, and optimization did not widen the lead.

| Run | Corpus | Validation gap at P0 | Selected | Selected validation gap |
| --- | --- | ---: | --- | ---: |
| `meta-blog-seed0` | cs.CL papers | -1.04 | P1 | -0.16 |
| `meta-blog-v2-seed0` | 2016-2021 papers | -1.89 | P3 | -0.08 |
| `meta-blog-v3-seed0` | 2016-2021 papers | -2.23 | P4 | -1.55 |
| `meta-blog-v4-seed0` | 2016-2021 papers | -2.26 | P4 | -1.48 |
| `v2-p3-repeats`, seeds 0, 1, 2 | 2016-2021 papers | -0.06, -0.15, -0.23 | update 3, 3, 1 | -0.21, +0.11, +0.08 |
| `fiction-seed0-r5` | Gutenberg fiction | +0.39 | P1 | +0.37 |

The `v2-p3-repeats` runs started from v2's selected prompt, so their starting gaps are v2's P3
prompt evaluated again on fresh drafts. Seeds 1 and 2 crossed zero, but their intervals
included zero. [LOG.md](LOG.md) gives every run in full, including the stopped ones, and
[METHOD.md](METHOD.md#conclusion) says what the null result does and does not rule out. The
saved results stay local in `reports/` and `runs/`. The `meta-blog-seed0` Git tag holds the
code for the 1st run. The current scripts still support fresh runs.

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

Fresh runs require a new output directory. A terminal failed or incomplete run can resume
with `uv run python run.py CONFIG --resume` when its config, dataset, and prompt hashes
still match. Resume preserves prior errors under `resumes/`, reuses valid completed
rubrics and grades, and retains every failed raw attempt. It retries invalid outputs
without overwriting their request records. A lock prevents overlapping resumed runners.
Resume records the current code hash, so compatible transport or validation repairs
remain visible. An active or completed run cannot be resumed. The runner has no
historical-report command.

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

To evaluate a frozen method after reduced replication, use all 3 completed runs from its
predeclared batch. This paid command evaluates the complete confirmation split, with no
optimizer calls or selection using confirmation scores:

```sh
uv run python confirm.py arxiv runs/confirmation-papers runs/batch/arxiv-s0 runs/batch/arxiv-s1 runs/batch/arxiv-s2
```

The command saves provenance, each checkpoint's scores, source-bootstrap intervals, and
actual provider charges. Use `--resume` only for a failed or incomplete evaluation with the
same inputs. Keep confirmation results separate from exploratory validation results.

The 3 paper meta-prompts in `prompts/experiments/verified-papers/` passed frozen confirmation
on 13 sections from 5 untouched papers. Their mean human-minus-model gaps were +1.523077,
+2.230769, and +2.153846, versus -1.107692 for the initial prompt. Each is the complete
meta-prompt supplied with context to the rubric generator, alongside the existing fixed
rubric wrapper. All 3 are retained; confirmation scores did not select a winner. See PR #20
for settings, costs, the scheduling amendment, and limits.

The proposed `configs/arxiv.yaml` default uses the verified paper recipe: 10% sampling,
7 updates, training-pair diagnoses, best-parent revisions, and validation of the initial and
training-selected prompts. Its settings match the completed seed-20262010 trial except for
a fresh output path. Run it with `uv run python run.py configs/arxiv.yaml`; this is a paid
command. Change the output path for each fresh run. Provider generations are stochastic,
so the sampling seed reproduces the subset, not exact scores. The story default is unchanged.

# Meta blog XAR reproduction

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
Muse Spark 1.1 writes the missing paper sections, generates the rubrics and grades the sections;
Kimi K2.6 rewrites the rubric meta prompt. One run covers 52 sections from 8 training and 5 validation papers,
with 7 prompt updates. [METHOD.md](METHOD.md) describes the data, the procedure, how
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
`configs/tokenizers.json` into `data/tokenizers/`. No run uses Qwen; `prepare-data` sizes
papers with both tokenizers, as the original selection did, so the same papers qualify.
`prepare-data` goes through the arXiv results saved in `data/discovery.xml`, downloads any
missing HTML into `data/raw/`, and rebuilds `data/examples.jsonl` from the first 20 eligible
papers. It fails if the result does not match the saved split and manifest, for example
because arXiv changed a page.

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## Run

```sh
uv run python run.py all --dry-run
uv run python run.py all --budget-usd 100 --total-budget-usd 100
```

`--dry-run` prints a cost estimate and sends nothing. `all` runs `pilot` and then
`reproduce`, which you can also run on their own:

- `run.py pilot` runs one update on the two pilot papers to check the pipeline end to end.
- `run.py reproduce` audits the completed pilot run, then runs the full experiment on the
  training and validation papers and writes the report to `reports/`.

Paid runs need both limits: `--budget-usd` for this run and `--total-budget-usd` for all runs
that share `runs/budget_ledger.json`. `--concurrency` sets the number of parallel requests
(default 4, from `configs/experiments.yaml`). `--resume` continues an interrupted run; only the
budgets and `--concurrency` may differ from the saved run, and any other change stops it. `--runs-root`
(default `runs/`) sets where the runs and the ledger are kept, and `--output-dir` (default
`reports/`) where `reproduce`, `all` and `report` write the report. `uv run python run.py
--help` lists every command and option.

## Audit

```sh
uv run python run.py audit-run runs/meta-blog-seed0
uv run python scripts/audit_completed_run.py
uv run python run.py report
```

`audit-run` rebuilds a run's checkpoint table from its saved raw responses and checks the
request settings, that judge requests do not say which section is the author's, the scoring
arithmetic, that the checkpoint was selected on training data only, and that every validation
request came after that selection. `scripts/audit_completed_run.py` goes further on
the completed research run: it checks the manifest's code hashes against the commit that
produced the run, re-derives the writer inputs, optimizer feedback and proposal checks with
the current code, and writes the cost, request, latency and length-subset numbers to
`reports/completion_audit.json`. Neither makes API calls. `report` repeats the `audit-run` checks on the research run and then
writes the report; it does not run `scripts/audit_completed_run.py`.

## Browsing runs

`uv run python tools/data-viewer/serve.py` starts a local viewer for runs, sections, rubrics,
prompt history and criteria; see [tools/data-viewer](tools/data-viewer/README.md).

## Repository layout

```
.
├── run.py                        CLI: pilot, reproduce, all, report, validate-data,
│                                 prepare-data, prepare-tokenizers, audit-run
├── xar/                          the pipeline, as a package
│   ├── pipeline.py               the method: write sections, generate rubrics, grade,
│   │                             build optimizer feedback, rewrite the prompt, select P*
│   ├── openrouter.py             model settings, endpoint checks, pricing, token counts,
│   │                             the budget ledger and the OpenRouter client
│   ├── data.py                   builds the dataset from arXiv HTML; loads and splits it
│   ├── runs.py                   run directories: manifest, resume checks, cost estimate
│   ├── audit.py                  re-checks a saved run against its raw API responses
│   ├── stats.py                  checkpoint means and paired whole-paper bootstrap
│   ├── report.py                 writes reports/ (results.md, checkpoints.csv, figure)
│   └── util.py                   paths, errors, hashing, JSON I/O, parallel map
├── scripts/
│   └── audit_completed_run.py    deep re-audit of runs/meta-blog-seed0 against the commit
│                                 that produced it; writes reports/completion_audit.json
├── tools/data-viewer/            local browser viewer for runs (serve.py + HTML/JS)
├── tests/                        pytest suite; one file per module, fakes in conftest.py,
│                                 full pilot and research runs in test_end_to_end.py
├── prompts/                      model prompts (hashed into each run's manifest)
│   ├── writer.md                 Muse writes the missing section
│   ├── rubric_initial.md         P0, the starting rubric meta prompt
│   ├── rubric_wrapper.md         wraps the meta prompt when Muse generates a rubric
│   ├── judge.md                  Muse grades one anonymous section against a rubric
│   └── optimizer.md              Kimi rewrites the meta prompt from training feedback
├── configs/
│   ├── experiments.yaml          run names, iterations, seed, concurrency, blog numbers
│   ├── models.yaml               each role's model, provider, temperature, reasoning
│   ├── tokenizers.json           pinned tokenizer revisions and checksums
│   └── snapshots/                saved OpenRouter catalog and endpoint listings that
│                                 preflight compares against
├── data/
│   ├── discovery.xml             the saved arXiv query results papers were chosen from
│   ├── acquisition_policy.json   selection rules, fixed before any grading
│   ├── exclusions.json           papers rejected during selection, with reasons
│   ├── source_manifest.json      URL, metadata and hashes for each selected paper
│   ├── splits.json               paper IDs for pilot, train, validation, confirmation
│   ├── human_review.json         approval of the extracted papers, by dataset hash
│   ├── license_inventory.json    license label of each paper
│   ├── examples.jsonl            the 80 sections with paper context (not in git)
│   ├── raw/                      downloaded arXiv HTML (not in git)
│   └── tokenizers/               downloaded tokenizers (not in git)
├── runs/                         run outputs and the shared budget ledger (not in git)
├── reports/                      generated report (not in git)
├── METHOD.md                     method, reconstruction choices and limitations
├── pyproject.toml, uv.lock       dependencies
└── .env.example                  OPENROUTER_API_KEY goes in .env
```

## Outputs

| Path | Contents |
| --- | --- |
| `runs/pilot-meta-blog-attested/` | the pilot run |
| `runs/meta-blog-seed0/` | the research run: `manifest.json`, every request and response under `requests/`, sections in `generations/`, `rubrics/`, `scores/`, optimizer `feedback/`, `prompts/`, `freeze.json`, `costs.json` |
| `runs/budget_ledger.json` | reservations and charges for every run |
| `reports/` | `results.md`, `checkpoints.csv`, `gap_curves.png` and `.svg`, `blog_comparison.json`, `audit.json`, and `completion_audit.json` from the audit script |

`runs/`, `reports/`, the paper text and the tokenizers are not in git.

## Versions

The git tag `meta-blog-seed0` is the commit that produced the research run. The code has been
cleaned up since; `scripts/audit_completed_run.py` checks that the saved run still re-audits
with the current code. The tag `alternative-model-study` holds an earlier study with other
models, which did not go beyond pilots.

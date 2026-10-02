# Meta blog XAR reproduction

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
Muse Spark 1.1 writes the missing paper sections and generates the rubrics; Qwen3.8 Flash grades
the sections, and Muse grades the validation sections again at 2 checkpoints as a cross-check;
Kimi K2.6 rewrites the rubric meta prompt. The completed run used Muse for all 3 Muse roles,
as the blog did. The completed run covers 52 sections from 8 training
and 5 validation recent arXiv cs.CL papers, with 7 prompt updates. `data/` now holds a new
corpus of 56 peer-reviewed papers from 2016 to 2021 across 9 fields (153 sections), built to
give human references of known quality. No run has used it yet. The owner reviewed all 56 papers together and approved them (see `LOG.md`). [METHOD.md](METHOD.md) describes the data, the procedure, how
this differs from the blog, and the limitations.

## Result

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

`run.py report` writes the full numbers to `reports/results.md`.

## Setup

You need Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and the Hugging Face `hf`
CLI (used to download the tokenizers).

```sh
uv sync --locked
uv run python run.py prepare-tokenizers
uv run python run.py discover-data
uv run python run.py prepare-data
uv run python run.py validate-data
uv run pytest -q
```

`prepare-tokenizers` downloads the Kimi and Qwen tokenizers pinned in
`configs/tokenizers.json` into `data/tokenizers/`. No run uses Qwen; `prepare-data` sizes
papers with both tokenizers, as the first corpus did, and keeps the larger count.
`discover-data` samples candidate papers from OpenAlex and checks them against the arXiv API
(about 20 minutes, because arXiv rate-limits). It writes `data/discovery.json` once and
refuses to replace a different file, so skip it when that file exists. `prepare-data` goes
through the shortlists in `data/discovery.json`, downloads any missing ar5iv HTML into
`data/raw/`, and rebuilds `data/examples.jsonl`, `splits.json` and `source_manifest.json`. It
prints each rejected candidate and the reason. It fails if the result does not match the saved
files, for example because ar5iv changed a page. The owner reviewed the 56 papers together and
approved them; `LOG.md` records that.

To audit the completed run, copy `examples.jsonl` and `splits.json` from
`data/archive/arxiv-2609-cs-cl/` back into `data/` (set the new ones aside first), since the
run's manifest points there.

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## Run

```sh
uv run python run.py all --dry-run
uv run python run.py all
```

`--dry-run` prints a cost estimate and sends nothing. `all` runs `pilot` and then
`reproduce`, which you can also run on their own:

- `run.py pilot` runs one update on the first two training papers to check the pipeline end to end.
- `run.py reproduce` audits the completed pilot run, then runs the full experiment on the
  training and validation papers and writes the report to `reports/`.

Spending is capped by the OpenRouter key's limit, set on openrouter.ai; the code records every
billed cost but enforces no budget of its own. `--concurrency` sets the number of parallel requests
(default 4, from `configs/experiments.yaml`). `--resume` continues an interrupted run; only
`--concurrency` may differ from the saved run, and any other change stops it. `--runs-root`
(default `runs/`) sets where the runs are kept, and `--output-dir` (default
`reports/`) where `reproduce`, `all` and `report` write the report. `uv run python run.py
--help` lists every command and option.

## Audit

```sh
uv run python run.py audit-run runs/meta-blog-seed0
uv run python run.py report
```

`audit-run` makes no API calls. It rebuilds a run's checkpoint table from its saved raw responses
and checks:

- the request settings, and that judge requests do not say which section is the author's;
- that the writer saw only the task data, and that its texts, length flags and copy flags
  follow from the raw responses;
- that the optimizer's feedback and proposals re-derive from training scores alone;
- the scoring arithmetic, and that the checkpoint was selected on training data only;
- that every validation request came after that selection;
- that request keys and the billed costs agree with `costs.json`.

`report` repeats the `audit-run` checks on the research run and then writes the report.

## Browsing runs

`uv run python tools/data-viewer/serve.py` starts a local viewer for runs, sections, rubrics,
prompt history and criteria; see [tools/data-viewer](tools/data-viewer/README.md).

## Repository layout

```
.
├── run.py                        CLI: pilot, reproduce, all, report, validate-data,
│                                 discover-data, prepare-data, prepare-tokenizers, audit-run
├── xar/                          the pipeline, as a package
│   ├── pipeline.py               the method: write sections, generate rubrics, grade,
│   │                             build optimizer feedback, rewrite the prompt, select P*
│   ├── openrouter.py             model settings, endpoint checks, pricing, token counts,
│   │                             and the OpenRouter client
│   ├── discovery.py              samples candidate papers from OpenAlex and arXiv
│   ├── data.py                   builds the dataset from ar5iv HTML; loads and splits it
│   ├── runs.py                   run directories: manifest, resume checks, cost estimate
│   ├── audit.py                  re-checks a saved run against its raw API responses
│   ├── stats.py                  checkpoint means and paired whole-paper bootstrap
│   ├── report.py                 writes reports/ (results.md, checkpoints.csv, figure)
│   └── util.py                   paths, errors, hashing, JSON I/O, parallel map
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
│   ├── discovery.json            the seeded OpenAlex and arXiv shortlists papers come from
│   ├── source_manifest.json      URL, metadata and hashes for each selected paper
│   ├── splits.json               paper IDs for train, validation, confirmation
│   ├── examples.jsonl            the 153 sections with paper context (not in git)
│   ├── raw/                      downloaded ar5iv HTML and arXiv metadata (not in git)
│   ├── archive/                  the earlier cs.CL corpus the completed run used (not in git)
│   └── tokenizers/               downloaded tokenizers (not in git)
├── runs/                         run outputs (not in git)
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
| `runs/budget_ledger.json` | reservations and charges for the runs before 2026-10-01, when the code enforced budgets |
| `reports/` | `results.md`, `checkpoints.csv`, `gap_curves.png` and `.svg`, `blog_comparison.json`, and `audit.json` |

`runs/`, `reports/`, the paper text and the tokenizers are not in git.

## Versions

The git tag `meta-blog-seed0` is the commit that produced the research run. The code has been
cleaned up since, and `run.py audit-run` checks that the saved run still re-audits with the
current code. The tag `alternative-model-study` holds an earlier study with other
models, which did not go beyond pilots.

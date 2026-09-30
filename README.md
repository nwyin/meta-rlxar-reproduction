# Meta blog same-model XAR reproduction

This repository targets the **Initial Empirical Investigation** in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/):
Muse Spark 1.1 writes sections, generates rubrics, and judges candidates;
Kimi K2.6 optimizes the rubric meta-prompt. One trajectory uses 52 sections
from eight training and five validation papers, with seven prompt updates.

The active workflow runs a separate one-update operational pilot, then this
single experiment. The former alternative-model sweep is archived in
[docs/archive/alternative-model-study-20260929](docs/archive/alternative-model-study-20260929).
Its raw outputs and unresolved cost reservations are preserved locally.
See [PLAN.md](PLAN.md) for the scientific contract and reconstruction choices.

The blog does not publish its original paper IDs, exact prompts, or sampling
settings. Our frozen arXiv split, prompts, criterion schema, and decoding are
explicit reconstruction choices. Matching the numerical result is not guaranteed.
This target covers rubric optimization; later RL training and other writing
domains in the blog require separate work.

## Setup

Requires Python 3.11+, `uv`, and the Hugging Face `hf` CLI for the pinned Kimi
and historical tokenizers. Raw text, tokenizers, credentials, and runs stay local.

```sh
uv sync --locked
uv run python shared.py prepare-tokenizers
uv run python shared.py validate-data
uv run pytest -q
```

The existing dataset and split are frozen. Source URLs, discovery, metadata,
license inventory, review decisions, and hashes are version controlled. Do not
replace papers after observing grades. User acceptance of the paper review
packet is recorded against the frozen dataset hash in data/human_review.json;
a detailed independent manual inspection was not separately documented.
The 20 acquired papers include two pilot and five reserved confirmation papers;
only the eight train and five validation papers enter the research trajectory.
These are author-written reference proxies from recent preprints, not the blog's
original S2ORC examples. Paper text is not publicly redistributed.

Set OPENROUTER_API_KEY and XAR_TOTAL_BUDGET_USD in the ignored .env file.
The currently authorized shared total is $100, including previous pilots.
Credentials are excluded from artifacts. The provider's own quota also applies.
Muse uses the pinned Meta endpoint; Kimi optimization uses SiliconFlow FP8,
which passed native JSON capability checks in an earlier operational probe.
A fresh Muse/Kimi pilot verifies their use together before research.

## Run and report

```sh
uv run python run_matrix.py --phase all --dry-run
uv run python run_matrix.py --phase all \
  --budget-usd 100 --total-budget-usd 100 --concurrency 4
```

Run phases separately with --phase pilot or --phase reproduction. Resume an
unchanged existing run by adding --resume. Budget-only continuations are allowed;
changes to source code or substantive configuration require a new declared run.
The default output directories are runs/pilot-meta-blog and runs/meta-blog-seed0.
Both share runs/budget_ledger.json with the preserved historical attempts.

```sh
uv run python shared.py audit-run --source-run runs/meta-blog-seed0
uv run python run_matrix.py --phase report
```

The report compares the complete checkpoint curves with the blog's reported
validation gap -4.2 to +2.76, crossing at update 4 and peaking at update 5.
Checkpoint selection uses training only; the validation maximum is descriptive.
Incomplete runs cannot support a result claim. See [reports/results.md](reports/results.md).

Independent writer sections and checkpoint evaluations run concurrently.
Within-example repairs and the seven optimizer updates remain sequential.
Concurrency preserves frozen output ordering, anonymous candidate inputs, and
selection before validation. Fatal failures halt further dispatch; unknown
calls retain their conservative reservations and cannot be resent automatically.

Before every batch, read-only preflight checks release, provider, schema support,
precision, reasoning, limits, and current prices against pinned evidence.
No model substitution or provider fallback is enabled. Muse uses explicit medium
reasoning, Kimi enabled reasoning; the blog's exact settings are undisclosed.
Muse payload sizing uses a conservative UTF-8 byte bound because its tokenizer
is unpublished. Kimi uses a pinned official tokenizer. Full papers are never truncated.

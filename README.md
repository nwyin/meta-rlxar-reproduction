# Meta blog same-model XAR reproduction

This repository targets the **Initial Empirical Investigation** in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/):
Muse Spark 1.1 writes sections, generates rubrics, and judges candidates;
Kimi K2.6 optimizes the rubric meta-prompt. One trajectory uses 52 sections
from eight training and five validation papers, with seven prompt updates.

The active workflow runs a separate one-update operational pilot, then this
single experiment. The earlier alternative-model study, which never ran beyond
pilots, is kept at the git tag `alternative-model-study`. Its raw outputs and unresolved cost reservations are preserved locally.
See [PLAN.md](PLAN.md) for the scientific contract and reconstruction choices.

The blog does not publish its original paper IDs, exact prompts, or sampling
settings. Our frozen arXiv split, prompts, criterion schema, and decoding are
explicit reconstruction choices. Matching the numerical result is not guaranteed.
This target covers rubric optimization; later RL training and other writing
domains in the blog require separate work.

## Completed run: September 29, 2026

- All 52 sections, seven rubric updates, and eight train/validation checkpoints
  completed using Muse Spark 1.1 and Kimi K2.6. No additional-model sweep or RL
  training was performed; the writing samples stayed fixed while rubrics changed.
- Training selected P1 before any validation evaluation. The held-out
  human-minus-model gap improved from **-1.04 to -0.16**; every validation
  checkpoint remained negative. The blog's reported **-4.2 to +2.76** reversal
  was not reproduced on this reconstruction. Terminal P7 was -0.37.
- Paired validation improvement was **0.88 points**, with a 95% whole-paper
  bootstrap interval of **[0.37, 1.35]** over five validation papers. Author
  scores moved from 7.91 to 8.38, and model scores from 8.95 to 8.53.
- Seventeen generated sections missed the +/-15% length tolerance after bounded
  repairs. The length-compliant validation subset also retained a negative gap.
  Human reference quality was not independently established by expert comparison.
- Research cost **$55.60** across 1,389 requests and took **75.4 minutes** at
  concurrency four. The separate operational pilot cost $2.42. Shared spending
  and retained historical reservations totaled $64.70, within the $100 cap.
- All raw-response audits and 24 tests passed; no research requests remained
  unresolved. Detailed reports, plots, the blog draft, and raw run artifacts
  remain local under `reports/` and `runs/`, both excluded from version control.

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
After account confirmation, the declared fresh pilot is runs/pilot-meta-blog-attested;
research uses runs/meta-blog-seed0. The rejected runs/pilot-meta-blog is preserved.
Both share runs/budget_ledger.json with the preserved historical attempts.

```sh
uv run python shared.py audit-run --source-run runs/meta-blog-seed0
uv run python run_matrix.py --phase report
```

The report compares the complete checkpoint curves with the blog's reported
validation gap -4.2 to +2.76, crossing at update 4 and peaking at update 5.
Checkpoint selection uses training only; the validation maximum is descriptive.
Incomplete runs cannot support a result claim. The local detailed report is
`reports/results.md`; reports are generated artifacts and are not committed.

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

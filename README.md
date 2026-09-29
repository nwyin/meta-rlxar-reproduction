# OpenRouter XAR reproduction

This repository implements the independent procedure preregistered in [PLAN.md](PLAN.md).
It optimizes a candidate-independent rubric-generating meta-prompt using training
human–model score gaps, selects a checkpoint by training only, and evaluates all
eight checkpoints on held-out papers. Matching Meta's reported numerical result
is not an acceptance criterion.

The primary study is eight writer × generator × optimizer conditions, with three
trajectories each and seven prompt updates. DeepSeek 0813 is the fixed main judge;
GLM-5.2 and Qwen9B apply the saved rubrics for transfer/ablation. Anthropic and
Google models are excluded in every role, including repairs. Intermediate scaling
is implemented as a separately preregistered later extension.

**Execution status:** implementation and mock-provider tests pass. Separate live
pilots are running; no completed research or confirmation results exist yet. See
[reports/progress.md](reports/progress.md) and [reports/matrix.csv](reports/matrix.csv).
Mock tests are operational checks, never scientific evidence.

## Setup and frozen inputs

Requires Python 3.11+, `uv`, and the Hugging Face `hf` CLI. Run:

```sh
uv sync --locked
uv run python shared.py prepare-tokenizers
uv run python shared.py prepare-data
uv run python shared.py validate-data
uv run pytest -q
```

Tokenizers are downloaded from the official Qwen, Moonshot, DeepSeek, and Z.ai
repositories at the revisions in `configs/tokenizers.json`. File checksums are
verified. Only vocabularies/configuration files are downloaded; custom remote
Python code is not executed. The Kimi segmentation pattern is transcribed from
its pinned tokenizer source. Payload sizing uses the model tokenizer with 25%
headroom and explicit chat/schema overhead, and is checked again before dispatch.

The frozen discovery response and acquisition policy reconstruct 20 eligible
versioned arXiv CS papers: 2 pilot, 8 train, 5 validation, and 5 confirmation.
This is the plan's permitted open-access fallback, **not S2ORC**. Exclusion logs
preserve the initial overly restrictive byte-count screen and the corrected
tokenizer-based pass, both before any grading. Recent preprints need not be
peer reviewed or entirely human written. Original author sections are the
reference proxy. Publication and provenance limitations must accompany results.

The user accepted the paper review packet in this thread. This acceptance is
recorded separately in `data/human_review.json`; a detailed independent manual
inspection was not separately documented. Human review is checked against the
frozen dataset hash before live experiments. On a new reconstruction, inspect
the originals and record your own review. Never replace rejected papers based on
model grades; use the recorded replacement order and a new preregistered batch.

Raw HTML, full extracted contexts/references, review excerpts, tokenizers, and run
artifacts stay local and are ignored by Git. Paper redistribution permissions
are not assumed. Source URLs, content hashes, splits, prompts, code, and permitted
metadata are version controlled. Do not publicly release original paper text
unless its individual license has been verified.

Put `OPENROUTER_API_KEY` in the process environment or an ignored local `.env`
file. `.env.example` contains names only. Credentials and authorization headers
are never placed in request artifacts. Provide explicit per-run and total USD
ceilings for paid requests; this repository supplies no authorized default.

## Execution

Inspect the separate-pilot plan first:

```sh
uv run python run_matrix.py --phase pilot --dry-run
uv run python run_matrix.py --phase controls --dry-run
uv run python run_matrix.py --phase matrix --dry-run
```

Dry runs validate local data and saved endpoint capabilities and estimate counts
and uncached costs. They make no generation or grading calls. The estimates are
provisional until actual pilot token/latency measurements exist. Shared-writer
reuse in a staged dry run assumes the preceding control phase will create those
artifacts. Transfer and confirmation dry runs need existing frozen source runs.

Every live batch saves fresh read-only OpenRouter catalog and endpoint snapshots,
rejecting changed releases, capabilities, or pricing against the frozen settings.
Refresh the frozen evidence before a new batch if this check finds a change.
Verify model identity, structured outputs, provider routing, and
reasoning modes in the operational pilot. Changes before research must be
documented; changes within a trajectory require a new run. The current provider
pins remain provisional until live checks succeed. The direct Kimi endpoint
does not advertise temperature support. Crusoe BF16 returned four upstream 429
errors before producing a candidate. A separately declared pilot now uses
DigitalOcean for the same Kimi release; its precision is unknown. The old attempt
and reservations are preserved. See [the provider declaration](reports/pilot_provider_deviation.json).

Once you have set `RUN_BUDGET_USD` and `TOTAL_BUDGET_USD` to your chosen ceilings:

```sh
uv run python run_matrix.py --phase pilot \
  --budget-usd "$RUN_BUDGET_USD" --total-budget-usd "$TOTAL_BUDGET_USD"
uv run python run_matrix.py --phase controls \
  --budget-usd "$RUN_BUDGET_USD" --total-budget-usd "$TOTAL_BUDGET_USD"
uv run python run_matrix.py --phase matrix \
  --budget-usd "$RUN_BUDGET_USD" --total-budget-usd "$TOTAL_BUDGET_USD"
uv run python run_matrix.py --phase transfers \
  --budget-usd "$RUN_BUDGET_USD" --total-budget-usd "$TOTAL_BUDGET_USD"
uv run python run_matrix.py --phase confirmation \
  --budget-usd "$RUN_BUDGET_USD" --total-budget-usd "$TOTAL_BUDGET_USD"
```

For the currently declared replacement batch, add
`--pilot-prefix pilot-digitalocean` to pilot, controls, and matrix invocations
(or to `--phase all`). The configured `.env` total ceiling is $100; the current
per-run ceiling is also $100, subordinate to the shared total.

`--phase all` runs those phases in order. The research gate checks both separate
pilot trajectories against raw responses. It does not require a negative
baseline or a reversal. Controls freeze one shared candidate set per writer,
run both pairwise orders once per example/writer, and run three fresh P0 rubric
evaluations on validation for each writer × generator pairing. The matrix reuses
those writer artifacts across all 24 trajectories. Transfer scripts regenerate
neither rubrics nor prompts. Cross-writer transfer holds the source judge fixed.

The confirmation configuration is committed in `configs/confirmation.json`
before research outcomes: strong writer / weak generator / strong optimizer,
seed 0. Its P* is selected by training only. Confirmation papers receive no model
calls until the source prompt/configuration is frozen. No favorable validation
cell is chosen after comparison.

Each root experiment script accepts `--help`, independent role slugs/providers,
decoding overrides, config defaults with explicit CLI precedence, a seed,
concurrency, ceilings, and resume/dry-run controls. `exp_scaling.py` distinguishes
XAR reoptimization for W/G/O from frozen-artifact regrading for J; paid scaling
requires an extension-specific preregistration.

## Recovery and audit

Use the same invocation with `--resume` after interruption. Completed requests
reuse full-payload/replicate cache keys; independent samples have distinct keys.
Substantive changes to code, prompts, roles, decoding, data, or endpoint evidence
require a new run. Budget-only continuation is recorded separately.

The shared ledger reserves a conservative maximum before dispatch and enforces
both ceilings across processes. Unknown sends retain their reservations and
stop execution. A read/write timeout is not proof that a paid completion failed;
check OpenRouter history before retrying. Failed HTTP attempts retain a reserved
charge until reconciliation. Provider responses missing actual cost also stop
new dispatch. Never delete a ledger entry to conceal uncertain spending.

Each run preserves the request payload, raw response, retry lineage, returned
model/provider identity, finish reason, token usage, actual cost, every writer
attempt, rubric, criterion grade, feedback payload, proposal audit, checkpoint,
and selection freeze. Reasoning is retained in raw responses and excluded from
candidate prose. Invalid grades get one same-model format repair. Invalid
proposals get one bounded repair, then consume the update and repeat the current
prompt. Valid unfavorable grades/proposals are not resampled.

```sh
uv run python shared.py audit-run --source-run runs/xar-strong-GG2-seed0
uv run python shared.py report --runs-root runs --output-dir reports
```

The audit parses saved raw outputs, validates criterion coverage and evidence
quotes, recomputes arithmetic means, verifies anonymous payloads and training-only
feedback, and checks every expected checkpoint/example. Reports retain missing
matrix cells. Paired intervals resample whole papers; three optimizer trajectories
remain separate. Reports include selected/terminal gaps, absolute scores, section
breakdowns, shared length-compliance sensitivity, contamination flags, invalid
output rates, and costs. Figures are created only when verified results exist.
Matrix completion alone does not prove that controls, transfers, confirmation,
release packaging, or the complete PLAN.md milestone is finished.

The raw audit also verifies sent model/provider, decoding, schema, and wrapper
identities. Matrix acceptance requires the fixed preregistered main judge and
consistent input/scoring contracts across trajectories. `factorial_effects.json`
contains paired W/G/O effects and interactions for selected gaps and improvement,
with separate per-trajectory paper intervals. Confirmation accepts only reserved
papers and hash-identical source data overrides.

Routing/schema contracts follow the primary
[OpenRouter provider routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection)
and [structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs).
The source procedure is [Meta's methodology post](https://facebookresearch.github.io/RAM/blogs/unslop/).

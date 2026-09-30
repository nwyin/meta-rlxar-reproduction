# Meta blog same-model XAR reproduction

Revised September 29, 2026 at the user's request, before research grading.
The previous alternative-model study is archived in
[docs/archive/alternative-model-study-20260929/PLAN.md](docs/archive/alternative-model-study-20260929/PLAN.md).
Historical paid pilots are retained and excluded from this experiment.

## Target

Reproduce the **Initial Empirical Investigation** in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/):
52 paper-section examples, eight training papers, five validation papers, and
seven rubric meta-prompt updates. Use these exact named models:

| Role | Model | OpenRouter release | Provider |
| --- | --- | --- | --- |
| Writer | Muse Spark 1.1 | meta/muse-spark-1.1-20260709 | meta |
| Rubric generator | Muse Spark 1.1 | meta/muse-spark-1.1-20260709 | meta |
| Judge | Muse Spark 1.1 | meta/muse-spark-1.1-20260709 | meta |
| Meta-optimizer | Kimi K2.6 | moonshotai/kimi-k2.6-20260420 | siliconflow/fp8 |

One trajectory, local seed 0, checkpoints P0 through P7. The default batch has
no other-model sweep, repeated-seed matrix, additional controls, transfers,
confirmation evaluation, scaling, or reinforcement learning. The later Qwen RL
experiments and other writing domains in the blog are outside this target.
Other endpoint configurations remain available only for historical artifacts
and explicit utility invocations; the active batch selects only the two models above.

The blog reports validation gap -4.2 to +2.76, crossing zero at update 4 and
peaking at update 5. Rounded absolute validation means move from 3.4 to 5.0 for
authors and 7.6 to 2.7 for models. Report comparisons without forcing these
numbers or selecting a checkpoint on validation performance.

## Reconstruction choices

The blog has not supplied the original 13 paper IDs, exact prompts, decoding,
criterion schema, aggregation, failure-example count, or complete selection
rule. This is a same-model procedural reconstruction, not an exact rerun.

Preserve the existing frozen split: 32 training and 20 validation sections,
one abstract, introduction, related-work section, and conclusion per paper.
The reconstructed papers are versioned arXiv CS preprints; author originals
are the expert-reference proxy, with peer review and absence of AI assistance
unverified. Do not call this the original S2ORC dataset. Two separate pilot
papers remain for operational checks. Five reserved confirmation papers receive
no calls under this plan. No source replacement based on model scores is allowed.

User acceptance of extraction completeness and writing suitability is recorded
in data/human_review.json against the frozen dataset hash. Detailed independent
manual inspection was not separately documented. Keep discovery, split, source,
license, and content hashes unchanged. Raw paper text and run artifacts stay local.

Use the existing neutral initial meta-prompt and wrappers under prompts/.
Temperatures: W 0.7, G 0.2, O 0.7, J 0. Muse uses explicit medium reasoning;
Kimi uses enabled reasoning. Both have output limits of 16,384 tokens. Neither
selected endpoint advertises seed support; record that local seed 0 does not
make hosted completions deterministic. The blog does not disclose these settings.
Kimi's hosted FP8 precision and Meta's unknown precision are provider caveats.

Muse's tokenizer is unpublished. Bound its entire payload by UTF-8 byte count,
plus 25% headroom and chat/schema overhead. Use the pinned official Kimi tokenizer
for optimizer payloads. Never truncate papers to make a request fit.

## Procedure and boundaries

1. Run a separate two-paper operational pilot with one optimizer update.
   Check all four roles, native JSON schemas, routing, anonymous grades,
   length checks, full checkpoint coverage, and raw-response integrity.
   Neither a favorable gap nor proposal acceptance is a research gate.
2. Generate 52 Muse sections from the full paper minus the target section,
   without exposing its original. Supply the original's target word count and
   require +/-15%, no invented claims, numbers, results, or citations.
   Freeze the first compliant section; allow at most two bounded length repairs.
   Retain failures and contamination flags without favorable score selection.
   Historical Kimi/Qwen writer outputs cannot be reused as Muse outputs.
3. At each of eight checkpoints, generate a candidate-independent rubric from
   visible paper, target section type, word count, and current meta-prompt.
   Neither human nor model candidate is visible to the generator. Use 4–8 unique
   criteria, at most 1,000 rubric words, and fixed 0–10 scoring anchors.
4. Grade anonymous human and model candidates separately against the same rubric.
   Validate criterion coverage, score bounds, and quoted evidence; compute the
   equal-weight mean in code. Allow one bounded format repair. Missing training
   grades stop optimization and keep validation unopened.
5. Kimi sees the current prompt, aggregate training scores, and the four lowest
   training gaps, breaking ties by example ID, with their full context/candidates/
   rubrics/grades. Propose one revision under 800 words. Audit provenance labels,
   copied spans, paper/author identifiers, changing scales or weights, and explicit
   human preference. Allow one bounded repair. Rejection consumes the update and
   repeats the parent prompt. Never feed validation examples or scores to Kimi.
6. After P7, select the highest complete mean training gap, earliest checkpoint
   on ties. Freeze selection and all prompt hashes before any validation rubric
   or grading request. Then evaluate every checkpoint on all 20 validation sections.

## Reporting and interpretation

Rebuild all eight train/validation means from saved raw model responses. Verify
request settings, anonymous inputs, full coverage, dataset hashes, scoring
arithmetic, training selection, and validation-after-freeze timestamps.
Report author/model absolute scores, gap curves, selected and terminal gaps,
paired selected-minus-P0 improvement, and bootstrap intervals that resample
whole papers. Only five validation clusters are available.

Report the first positive validation checkpoint and validation peak descriptively;
never use them to select the optimized prompt. A negative-to-positive selected
validation gap reproduces the qualitative inversion on this reconstruction.
Improvement without reversal and a positive initial gap remain valid outcomes.
Null/adverse results and incomplete runs remain visible. Optimized preference
alone does not establish objective writing quality or successful RL training.

## Execution, cost, and recovery

configs/experiments.yaml defines the sole active trajectory. run_matrix.py keeps
its historical filename but now schedules only pilot and reproduction. A fresh
read-only catalog/endpoint preflight checks exact release, provider, native schema
support, context, precision, reasoning mode, and pricing before each live batch.
Allow 25% reserved pricing headroom; never silently substitute model versions.

Use bounded concurrency 4 for independent sections/checkpoint evaluations.
Repairs within one example and optimizer updates stay sequential. Preserve
stable artifact ordering, anonymous input rules, and training/validation boundaries.

Retain the authorized $100 total ceiling and shared runs/budget_ledger.json,
including all previous pilots. The old full-sweep estimate and request for a
larger ceiling are obsolete. Per-request upper bounds reserve funds atomically;
known actual costs settle them. Interrupted/unknown calls retain upper reserves
and cannot be automatically resent. The provider's own account quota also applies.
Stop on exhausted budget, incompatible endpoint, unknown transport, or incomplete
training coverage. Configuration/software changes require a new run; budget-only
continuations may resume a frozen run. Save requests, responses, usage, costs,
latencies, prompts, grades, rejected proposals, and failure status locally.

# OpenRouter XAR reproduction plan

Date: September 29, 2026. Status: design only; no implementation or experiments have been run.

## Goal and scope

Build a reproducible experiment that tests whether optimizing a **rubric-generating meta-prompt** can reverse a held-out human–model grading gap. Use OpenRouter for writing, rubric generation, prompt optimization, and grading. Vary the strength of the writer, rubric generator, and optimizer independently; hold the main judge fixed, then test transfer to another judge.

The first artifact is an open reproduction of the XAR procedure with alternative model combinations. It includes prompts, paper splits, generated sections, rubrics, raw grades, optimizer feedback, costs, and negative results. RL, GRPO, fine-tuning, and additional writing domains are outside this milestone.

### What the source establishes

[Meta's methodology post](https://facebookresearch.github.io/RAM/blogs/unslop/) describes 52 paper–section examples, split by 8 training papers and 5 validation papers, with seven prompt-refinement iterations. Muse Spark 1.1 fills the writer, rubric-generator, and judge roles; Kimi-K2.6 is the optimizer. The reported validation gap moves from approximately −4.2 to +2.76, crossing zero at iteration 4. Each task removes one section, retains the rest of the paper, and requests roughly the original section's word count, within ±15%. The optimizer receives training results and failures, operates under a prompt-length bound, and a best candidate is retained. The post says a technical report is forthcoming.

Exact source prompts, splits, sampling settings, scoring details, and the selection rule are not supplied in that description. The choices below are this project's preregistered reconstruction. Matching the reported numbers is not a requirement for an independent procedural reproduction.

## Scientific questions and outcomes

1. Can a strong optimizer improve rubrics produced by a weak generator? Can a weak optimizer improve a strong generator?
2. Does the result change when the competing writer is stronger?
3. Do frozen learned rubrics work under a different strong judge, or only under the judge whose feedback trained the prompt?
4. How much alignment is lost when a weaker judge applies the same rubrics?

Distinguish three outcomes:

| Outcome | Interpretation |
| --- | --- |
| Baseline validation gap < 0; selected-prompt gap > 0 | Descriptive reproduction of the inversion on this split |
| Selected prompt improves the gap, but no sign reversal | Evidence of rubric improvement, without reproducing the inversion |
| Baseline already favors humans | Test gap improvement; do not claim a negative-to-positive reproduction |

A stronger result needs repeated optimizer trajectories, uncertainty estimates, cross-judge transfer, and additional untouched papers. A reversal alone does not establish objective writing quality: this optimization explicitly rewards human preference and can learn spurious features.

## Model roles and initial experiment matrix

Use distinct configuration fields for all four roles:

| Role | Symbol | Responsibility | Frozen during one trajectory? |
| --- | --- | --- | --- |
| Writer | W | Generate the missing section from the visible paper | Yes; cache its sections before optimization |
| Rubric generator | G | Apply the current meta-prompt to context and create a rubric | Model and decoding frozen; meta-prompt changes |
| Meta-optimizer | O | Revise the meta-prompt using training feedback | Model and proposal budget frozen |
| Optimization judge | J | Grade each anonymous candidate using the shared rubric | Model, grading prompt, schema, and aggregation frozen |

### Capability tiers from Artificial Analysis

Exclude Anthropic and Google models from all experiment roles, including repairs and evaluation. Use the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index) as the approximate capability reference, then check task competence on separate pilot papers. The page currently identifies index version **v4.3.2**. This composite measures general capability; it is not a direct measure of prose quality or rubric-grading reliability.

The following scores were checked against current model pages on September 29, 2026. Use one index version throughout; historical announcement scores and cached search snippets can differ substantially from current scores.

| Experimental tier/use | Model and evaluated mode | Approximate current AA index | OpenRouter model slug | Evidence |
| --- | --- | --- | --- | --- |
| Weak W/G/O; weak evaluation judge | Qwen3.5 9B, reasoning | 11 | `qwen/qwen3.5-9b` | [AA model page](https://artificialanalysis.ai/models/qwen3-5-9b) |
| Intermediate scaling control | Qwen3.5 27B, reasoning | 23, estimated | `qwen/qwen3.5-27b` | [AA release comparison](https://artificialanalysis.ai/models/releases/qwen3-5-27b) |
| Stronger W/G/O in initial matrix | Kimi K2.6, reasoning | 27 | `moonshotai/kimi-k2.6` | [AA model page](https://artificialanalysis.ai/models/kimi-k2-6) |
| Fixed strong optimization judge | DeepSeek V4 Pro **0813**, max reasoning | 36 | `deepseek/deepseek-v4-pro-0813` | [AA model page](https://artificialanalysis.ai/models/deepseek-v4-pro) |
| Independent strong evaluation judge | GLM-5.2, max reasoning | 34 | `z-ai/glm-5.2` | [AA model page](https://artificialanalysis.ai/models/glm-5-2) |

OpenRouter lists [Qwen9B](https://openrouter.ai/qwen/qwen3.5-9b), [Qwen27B](https://openrouter.ai/compare/qwen/qwen3.5-27b/qwen/qwen3.5-9b), [Kimi K2.6](https://openrouter.ai/moonshotai/kimi-k2.6), [DeepSeek V4 Pro 0813](https://openrouter.ai/deepseek/deepseek-v4-pro-0813), and [GLM-5.2](https://openrouter.ai/z-ai/glm-5.2). Pin the **0813** DeepSeek release: the undated OpenRouter slug `deepseek/deepseek-v4-pro` currently identifies the older 0423 model, whereas AA's current model page evaluates 0813. Verify exact releases and endpoint capabilities again before execution.

Proposed defaults:

- Weak tier for W, G, and O: `qwen/qwen3.5-9b`.
- Strong tier for W, G, and O: `moonshotai/kimi-k2.6`.
- Fixed optimization judge: `deepseek/deepseek-v4-pro-0813`, max reasoning.
- Independent strong evaluation judge: `z-ai/glm-5.2`, max reasoning.
- Weak evaluation judge: `qwen/qwen3.5-9b`.
- Later intermediate tier: `qwen/qwen3.5-27b`, useful for a comparison within the Qwen family.

Here “strong” means stronger than the weak tier; Kimi K2.6 is not being claimed as the strongest available model. Its separation from Qwen9B and its connection to the original optimizer make it a useful first comparison. DeepSeek and GLM provide higher-index judges from separate families. Freeze these choices before examining research-split gaps. The initial comparison mixes model family with capability; the Qwen 9B→27B extension helps separate these effects.

Save the AA source URL, retrieval date, index version, numeric score, estimation status, model release, and evaluated reasoning mode in `configs/model_tiers.json`. If an exact mode cannot be matched through a pinned OpenRouter endpoint, record the mismatch rather than borrowing the score of a stronger mode. Refresh tiers only between preregistered experiment batches. Choose replacements by same-version index separation, compatible context/schema support, and measured cost; never by a favorable validation gap.

Run the following four G×O conditions for **each** of the two writers, giving eight primary conditions:

| Condition | Rubric generator G | Optimizer O | Main judge J |
| --- | --- | --- | --- |
| GG1 | Weak | Weak | Fixed strong |
| GG2 | Weak | Strong | Fixed strong |
| GG3 | Strong | Weak | Fixed strong |
| GG4 | Strong | Strong | Fixed strong |

Use the same paper split, initial meta-prompt, schema, length bounds, failure-selection policy, and seven-update budget throughout. Run three independent optimizer trajectories per condition. Reuse a writer's cached sections across all G×O conditions and trajectories so writer sampling cannot explain differences between them.

Pilot first with the strong writer and GG2/GG4, one trajectory each. Run the full preregistered matrix once the pipeline passes operational checks, regardless of whether the pilot gap reverses. If budget limits the scope, report the executed subset and its selection rule.

## Dataset and paper-level split

### Acquisition and provenance

Use CS papers from S2ORC as the intended source. The [official S2ORC repository](https://github.com/allenai/s2orc) directs current bulk access through Semantic Scholar; do not assume an arbitrary Hugging Face mirror is the official or identical dataset.

Record the release identifier, acquisition route, paper IDs, source URLs, extraction version, and content hashes. If using a Hugging Face mirror, inspect its dataset card and provenance with the `hf` CLI and pin the repository revision and file checksums. Prefer a small documented subset over downloading the whole corpus. If access is unavailable, a manually acquired open-access CS subset is acceptable as an explicitly labeled source deviation.

Select papers before model grading. Record venue/year and a human assessment of extraction completeness and writing suitability. Published papers are a proxy for expert writing, not proof that every section is excellent. Do not pick papers based on which ones yield a favorable gap. Confirm what text can be redistributed; otherwise publish identifiers, hashes, extraction instructions, and permitted artifacts.

### Core split

- 13 distinct papers, each with four usable target sections: abstract, introduction, related work, and conclusion.
- 8 papers / 32 examples for prompt optimization.
- 5 papers / 20 examples for held-out validation.
- The 32/20 example counts are this plan's balanced reconstruction; the source gives paper-level counts and the total example count.
- Reserve at least 2 additional papers for operational pilots. They never enter either research split.
- For confirmation after comparing configurations, reserve at least 5 additional untouched papers / 20 examples. This is an extension, not part of the 52-example reproduction.

Split by paper, grouping duplicate versions, overlapping manuscripts, and obvious companion papers. Keep every section of a paper in the same split. Prefer author-group separation where feasible; document residual overlap. Freeze the split seed, eligibility rules, and replacement order before generation.

### Example construction

For each paper and target section, remove **only that target section**, including its subsections and duplicate copies in metadata. Retain the other sections and bibliography. Insert a neutral missing-section marker. The expert candidate is the extracted original section, stored separately from the visible context.

Keep section type and its target word count visible to every role. Use one documented whitespace-based word-count rule for both candidates, consistently handling headings and citation markers. Normalize extraction artifacts symmetrically; do not rewrite expert prose.

Data checks must catch duplicated abstracts, malformed section boundaries, OCR debris, incomplete references, duplicate manuscripts, and exact copies of the withheld section left in context. Shared facts and wording in other legitimate sections are expected; removing them would alter the task. Screen context size against the smallest selected endpoint, including grading and optimizer payload overhead. Never silently truncate or summarize papers per model.

### Length and memorization

Request a section within 85–115% of the expert word count, with claims grounded in the visible paper. Allow at most two length-only revision attempts. Return only the section body; save every attempt, count, and acceptance decision. Never choose an attempt by its judge score.

Freeze the first length-compliant attempt. If all attempts fail, retain the example and flag the writer failure. Report an all-example analysis and a sensitivity analysis on a shared compliant subset across writers; do not quietly drop a weak writer's failures. Display score gaps versus length ratio.

Before optimization, check generated sections for reference overlap and unusually long verbatim runs. These checks can flag likely contamination, but cannot rule out memorization. Preregister exclusions independently of grades, preserve an exclusion log, and report sensitivity to flagged examples. Prefer recent or less prominent papers when the source permits it.

## OpenRouter setup contract

Implementation will use a single OpenRouter adapter for all roles. This document does not create credentials, install dependencies, or spend API credits.

| Setting | Planned behavior |
| --- | --- |
| Credential | `OPENROUTER_API_KEY` supplied through the environment; never logged |
| API base | `https://openrouter.ai/api/v1` |
| Generation endpoint | Chat completions, non-streaming for the first implementation |
| Models | Explicit slugs per role; save the returned model identity |
| Providers | Pin a specific compatible endpoint per role after pilot checks |
| Routing | Use a restricted provider allowlist and `allow_fallbacks: false`; no automatic model substitutions |
| Capability enforcement | `require_parameters: true` for parameters sent to each endpoint |
| Schemas | Strict structured output for rubrics, grades, and optimizer proposals where supported |
| Tools and transforms | No web tools, retrieval, automatic context compression, or response-healing plugins |
| Retries | Up to three transport retries with backoff; log attempts and billable usage |
| Concurrency | Start at two in-flight requests; tune on pilot latency/rate limits |
| Spending | Estimate before execution; enforce per-run and total configured dollar ceilings |

The routing controls are documented in [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection). A pinned provider reduces an avoidable confound; it does not guarantee immutable weights or deterministic generations. An outage should leave a resumable run rather than silently change its model/provider combination.

Before each experiment batch, save the [model catalog](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties), endpoint metadata, supported parameters, context/output limits, and pricing. Perform small capability checks. [Structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs) depend on model/provider support. If a selected endpoint cannot satisfy the contract, declare a replacement before running the research matrix; do not give one tier a hidden repair model.

Initial decoding choices: writer temperature 0.7, rubric generator 0.2, optimizer 0.7, judge 0. Use only supported settings, freezing reasoning budgets and output limits per role. Pilot checks should establish enough output headroom for full sections and valid structured responses. A backend that forces different settings needs a recorded deviation. Request seeds where supported; record unsupported seeds and do not promise bitwise reproducibility.

The tier table assumes reasoning-enabled Qwen/Kimi and max-reasoning DeepSeek/GLM. Keep those modes explicit in the resolved configuration. OpenRouter documents `xhigh` as max reasoning for [GLM-5.2](https://openrouter.ai/z-ai/glm-5.2); verify the corresponding 0813 DeepSeek endpoint's mapping in preflight. Do not disable judge reasoning to save money and still label the run with its max-mode AA score. Reasoning tokens count toward output limits and budgets. If a cheaper effort is tested, it is a separately labeled condition.

The cache key includes the complete request, prompt hashes, model, provider policy, decoding settings, schema version, and replicate identity. Resume reuses completed artifacts. Independent replicate requests must not accidentally reuse a cached sample. Identical frozen starting prompts may share a deliberately documented baseline.

Preserve redacted request payloads, raw response bodies, response IDs, timestamps, actual model/provider metadata where available, finish reasons, token usage, cost, parse status, and retry lineage. Separate provider-returned reasoning from candidate section text.

## Prompt and grading contracts

### Writer prompt

Provide visible context, missing-section type, and target word count. Ask for a fitting section, supported facts, and citations only to sources available in the context. Keep this prompt identical across writer tiers. It does not receive rubrics or optimized meta-prompts.

### Starting meta-prompt P₀

Write one neutral, general prompt requesting task-specific writing criteria from visible context. Freeze it before seeing research-split results. It should reflect an ordinary rubric baseline, without incorporating conclusions from this experiment or hand-inserting the hoped-for optimized criteria.

For every iteration, G receives only Pₜ, visible context, section type, and target count. It receives neither candidate section, expert reference, candidate origin, nor judge feedback. Produce one shared rubric per context and apply it to both candidates. Candidate-conditioned or reference-conditioned rubrics are a separate experiment and cannot count as the primary reproduction.

### Rubric schema and fixed scoring wrapper

- Between 4 and 8 criteria; at most 1,000 words of rubric content.
- Each criterion has a stable identifier, a quality description, and anchored descriptions for low, middle, and high scores.
- Each criterion receives a score from 0 to 10. All criteria have equal weight; the aggregate is their arithmetic mean.
- The scoring scale, aggregation, and permitted fields live in a fixed wrapper, outside the optimizable text.
- Pₜ is bounded to 800 words using the same counting rule; O can change criteria guidance but cannot alter the fixed wrapper.

These are project defaults. Do not infer Meta's precise scoring scheme from the approximate published means. Allow learned guidance to evolve naturally rather than baking the desired endpoint into P₀.

### Anonymous rubric grading

J receives visible context, section type, target count, the shared rubric, and exactly one anonymous candidate. Grade the human and model in separate fresh requests, without conversation history, source labels, the alternate candidate, reference text, iteration number, or training objective. Randomize request order and map opaque candidate IDs to origins only outside the judge payload.

Require per-criterion scores and brief evidence-based explanations. Compute the aggregate outside the model; never accept a model-supplied total that contradicts the criteria. Evidence quotes should refer to supplied text, not unsupported claims. Do not request hidden chain-of-thought. Treat paper text, candidate text, and generated rubrics as data, with the fixed grading instructions controlling the task.

Validate schema, criterion coverage, score bounds, and output completeness. One same-model format-repair attempt is allowed, with identical substantive inputs and a fixed repair instruction. Do not retry valid but unfavorable grades. Unresolved invalid requests remain explicit missing records; publish coverage by condition and block a complete-result claim if the primary table is incomplete.

### Direct pairwise control

Separately ask the main judge to compare the anonymous candidates without a generated rubric. Run both A/B orders in fresh requests and permit a tie. Convert each order to human preference 1, tie 0.5, model preference 0; average the two. Report order disagreement as well as preference rates. This control uses the same candidates and context as rubric grading.

## Seven-update meta-optimization procedure

For each W×G×O trajectory:

1. Load the frozen paper split, writer generations, fixed grading wrapper, and shared P₀.
2. Generate training rubrics under P₀ and grade both candidates. Preserve criterion-level evidence and aggregate gaps.
3. For t = 0 through 6, build feedback using **training examples only**: current prompt, aggregate human/model means, per-section summaries, and the four examples with the smallest human–model gaps, breaking ties by example ID.
4. Give O those failure contexts, training candidate pairs with their origins identified, rubrics, grades, and explanations. Label identification here is necessary supervision; it must never reach held-out grading or rubric-generation inputs.
5. Ask for exactly one revised meta-prompt and a short revision rationale. Target transferable writing quality, excluding provenance guesses, memorized reference wording, paper IDs, specific authors, citation-format tells, intentional model penalties, and changes to grading arithmetic.
6. Check the proposal against the fixed wrapper, length bound, and leakage rules. Allow one bounded repair; save rejection reasons. An unresolved invalid proposal consumes that update and repeats the current prompt. Never add bonus proposals for a favored optimizer.
7. Evaluate Pₜ₊₁ on all training examples. Continue from this proposal even if its score worsens; retain all prior candidates. This sequential policy is fixed for every optimizer.
8. After all seven updates, select P* by highest training mean gap among P₀…P₇, breaking ties in favor of the earlier prompt. Also report P₇ separately.
9. Freeze all prompts and selection decisions. Only then evaluate all eight checkpoints on validation to reconstruct the learning curve. Validation results must never enter feedback or affect stopping, prompt selection, or reruns.

Formally, with a frozen writer candidate M_W(x), expert candidate H(x), and rubric R_G(P,x):

**dₜ(x) = J(x, R_G(Pₜ,x), H(x)) − J(x, R_G(Pₜ,x), M_W(x)).**

**Δₜ(S) = mean of dₜ(x) over examples x in split S.**

Optimize Δₜ(train), and use Δₜ(validation) to test generalization. Scores stay on the fixed 0–10 scale; do not normalize by the human score or alter weights between checkpoints.

Train-feedback candidate texts can leak specifics into the optimizer's proposal. Store an audit of prompt references to training paper identifiers, names, distinctive copied spans, and forbidden origin-detection instructions. Apply the same content-independent audit policy to every condition. Report any discovered leakage rather than cleaning up only unsuccessful runs.

## Controls and transfer experiments

Required first-release controls:

| Control | Purpose |
| --- | --- |
| Rubric-free, order-balanced pairwise judging | Measure ordinary comparison preferences |
| P₀-generated rubrics for each G | Measure ordinary rubric preferences |
| Selected XAR prompt and P₇ | Measure training-selected improvement and terminal behavior |
| Unchanged P₀ repeated on held-out papers | Estimate rubric/grade sampling noise without optimization |

For the unchanged-prompt control, perform three fresh baseline evaluations per W×G pairing, matched to the optimized evaluation's sampling settings. Do not confuse cached repeats with independent requests. Use the repeated evaluations to characterize noise, not to choose a favorable baseline. Optional later controls include a fixed human-authored rubric and a budget-matched random-mutation optimizer.

Then run:

1. **Cross-judge transfer:** regrade the exact saved P₀ and P* validation rubrics and candidates using the independent strong judge. No reoptimization or rubric regeneration. Its judgments never feed O.
2. **Weak-judge ablation:** apply the same frozen rubric artifacts with the weak evaluation judge. This isolates grading ability from prompt optimization. Reoptimizing under a weak judge is a subsequent, separately costed study.
3. **Cross-writer transfer:** use rubrics learned against W₁ to grade W₂'s cached held-out sections, and vice versa. Compare P₀ and P* under the fixed strong judge. The generator has no candidate input, so the saved rubric can be reused unchanged.
4. **Intermediate scale:** add Qwen3.5-27B to one role at a time, keeping the others fixed. Treat this as an extension with its own preregistration.

Record shared-model relationships. A Qwen weak judge may share a family with a writer or rubric generator; the independent strong judge provides a cleaner check of transfer.

## Analysis and reporting

Produce one checkpoint table per condition and trajectory:

| Iteration | Train human | Train model | Train gap | Val human | Val model | Val gap | Selected by train? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | pending | pending | pending | pending | pending | pending | pending |
| 1–6, one row each | pending | pending | pending | pending | pending | pending | pending |
| 7 | pending | pending | pending | pending | pending | pending | pending |

Primary endpoints are the validation gap at P* and improvement **Δ(P*, validation) − Δ(P₀, validation)**. Report the descriptive sign reversal, P₇ results, gap trajectories, expert/model absolute means, section-type breakdowns, positive-gap fraction, ties, length compliance, invalid-output rates, and actual costs.

Compare G and O using paired differences on the same papers and writer candidates. Show whether the optimizer effect depends on generator tier and writer tier; do not compare raw absolute scores across different judges as though they share calibration.

Resample whole papers for paired bootstrap intervals, retaining their section bundles. Report all three optimizer trajectories, their range, and variation separately from paper-sampling uncertainty. Five validation papers give only five independent paper clusters; many requests and repeated seeds do not enlarge that population sample. Treat the initial study as exploratory, especially across eight conditions. Report all conditions rather than selecting the most favorable cell.

For stronger confirmation, freeze a configuration and P* before opening the extra untouched papers. Report the final gap and baseline-to-final improvement with paper-level intervals; positive lower bounds and cross-judge agreement support a stronger claim. Do not move the success threshold after seeing results. A blind expert audit of a preselected sample of criterion explanations can expose provenance heuristics and questionable judgments, but is separate from the automated endpoint.

Release two figures: the iteration-by-iteration gap curve with zero marked, and the G×O comparison split by writer. Include absolute-score trajectories so readers can see whether improvement comes from recognizing expert strengths, penalizing models, or both.

## Planned repository artifacts

Create these during implementation; this planning task creates only `PLAN.md`.

### Rough structure: one script per experiment, one shared utility file

Keep executable files at the repository root for straightforward imports and invocation. Each script owns the orchestration of its scientific experiment, including generation and grading; model substitutions are CLI arguments, not separate source files for every matrix cell. All reusable behavior lives in **one** `shared.py` file. No trainer package or distributed workflow framework is needed for this milestone.

```text
meta-rlxar-reproduction/
  PLAN.md
  README.md
  pyproject.toml
  uv.lock
  .env.example
  .gitignore
  shared.py
  exp_baselines.py
  exp_xar.py
  exp_judge_transfer.py
  exp_writer_transfer.py
  exp_scaling.py                 # later extension
  configs/
    models.yaml
    model_tiers.json
    experiments.yaml
  prompts/
    writer.md
    rubric_initial.md
    rubric_wrapper.md
    judge.md
    pairwise.md
    optimizer.md
  data/
    source_manifest.json
    splits.json
    examples.jsonl
  runs/
    <run_id>/                   # immutable manifests + raw/derived artifacts
  reports/
```

| Script | Experimental setup and orchestration | Model arguments |
| --- | --- | --- |
| `exp_baselines.py` | Load frozen data; generate/cache writer sections; run direct pairwise grading in both orders and vanilla P₀ rubric generation/grading; optionally repeat P₀ to measure noise; write baseline tables | `--writer-model`, `--rubric-model`, `--judge-model` |
| `exp_xar.py` | One W×G×O×J condition per invocation; obtain cached writer sections, generate training rubrics, grade candidates, build training-only feedback, perform seven updates, select by train, freeze prompts, then generate/grade validation rubrics for all checkpoints and write curves | `--writer-model`, `--rubric-model`, `--optimizer-model`, `--judge-model` |
| `exp_judge_transfer.py` | Load P₀/P* rubrics and candidates from a frozen source run; grade unchanged artifacts with one or more alternate judges; write paired transfer and weak-judge summaries; no writing, rubric regeneration, or optimization | `--evaluation-judge-models` (one or more slugs) |
| `exp_writer_transfer.py` | Load P₀/P* rubrics from a frozen source run; generate/cache sections with another writer on the same held-out contexts; grade expert and target-writer sections with those exact rubrics | `--target-writer-model`, `--judge-model` |
| `exp_scaling.py` | Later extension: orchestrate the preregistered one-role-at-a-time scale comparison, with the other roles fixed; reuse common generation, optimization, and grading operations | `--vary-role`, `--models` (list), plus the four fixed role-model arguments |

`exp_scaling.py` runs the XAR procedure for varied W/G/O and frozen-artifact regrading for varied J. Reports must distinguish these two interventions. None of these scripts should import another experiment script; they import reusable operations from `shared.py`.

`shared.py` contains the OpenRouter adapter and endpoint checks, common argument definitions, resolved configuration, dataset loading/preparation and split validation, prompt loading/hashing, schemas, writer generation/length checks, rubric generation, anonymous rubric/pairwise grading, bounded proposal calls, cache/resume, retries, artifact I/O, cost accounting, and aggregation/plots. It may expose a data-preparation subcommand to create the frozen manifests before experiments. Scripts keep their experiment-specific stage order, feedback policy, selection decisions, and transfer rules visible in their own files.

### CLI contract

All applicable scripts accept:

- `--dataset`, `--splits`, `--output-dir`, `--seed`, `--budget-usd`, `--concurrency`, `--dry-run`, and `--resume`.
- Role-specific `--writer-provider`, `--rubric-provider`, `--optimizer-provider`, and `--judge-provider` for roles the script actually invokes. Transfer judging also supports an evaluation-provider mapping keyed by model slug.
- Role-specific temperature, reasoning-mode/effort, and maximum-output-token overrides. These are validated against endpoint capabilities and saved with model identities.
- A config-file argument supplying common defaults. Explicit CLI arguments override that config; the final resolved manifest is authoritative.

Additional experiment arguments:

| Script | Additional arguments and behavior |
| --- | --- |
| `exp_baselines.py` | `--baseline-methods pairwise vanilla`, `--baseline-repeats`, `--writer-generations` to reuse an existing compatible candidate artifact |
| `exp_xar.py` | `--iterations` (default 7), `--initial-meta-prompt`, `--max-meta-prompt-words` (default 800), `--failure-examples` (default 4), `--writer-generations`; one `--seed` identifies one trajectory |
| `exp_judge_transfer.py` | `--source-run`, `--checkpoints initial selected`, `--split` (validation by default); source candidate/rubric identities are read from the run rather than overridden |
| `exp_writer_transfer.py` | `--source-run`, `--checkpoints initial selected`, `--split`, optional `--target-generations`; source rubric identities remain frozen |
| `exp_scaling.py` | `--vary-role writer\|rubric\|optimizer\|judge`, `--models`, `--seeds`, and `--source-run` for frozen-judge comparisons |

Single-model arguments accept an OpenRouter slug rather than a hard-coded weak/strong enum. A model outside the saved tier table is allowed, but is labeled unclassified until its AA evidence is recorded. Anthropic and Google model families are rejected under this project's model policy. Overrides cannot silently change the models of frozen source artifacts. `--resume` requires matching substantive configuration; changing models, decoding, prompts, data, or budgets for scientific comparison creates a new run, with an explicit budget-only continuation permitted when recorded.

`--dry-run` resolves configuration, validates local input metadata, plans artifact reuse, and reports anticipated request counts/cost estimates without generation or grading calls. If endpoint/pricing information must be refreshed, use only read-only catalog queries and identify that in the output. A normal run requires an explicit budget ceiling; do not bake an unapproved dollar budget into the examples below.

Illustrative invocations for the future scripts, shown as documentation only:

- Baseline: `python exp_baselines.py --writer-model moonshotai/kimi-k2.6 --rubric-model qwen/qwen3.5-9b --judge-model deepseek/deepseek-v4-pro-0813 --dataset data/examples.jsonl --splits data/splits.json --output-dir runs/baseline-kimi-qwen --dry-run`
- Weak G / strong O XAR: `python exp_xar.py --writer-model moonshotai/kimi-k2.6 --rubric-model qwen/qwen3.5-9b --optimizer-model moonshotai/kimi-k2.6 --judge-model deepseek/deepseek-v4-pro-0813 --iterations 7 --seed 0 --dataset data/examples.jsonl --splits data/splits.json --output-dir runs/xar-kimi-qwen-kimi-seed0 --dry-run`
- Strong G / weak O uses the same `exp_xar.py`, replacing `--rubric-model` with `moonshotai/kimi-k2.6` and `--optimizer-model` with `qwen/qwen3.5-9b`, with a new output directory.
- Judge transfer: `python exp_judge_transfer.py --source-run runs/xar-kimi-qwen-kimi-seed0 --evaluation-judge-models z-ai/glm-5.2 qwen/qwen3.5-9b --output-dir runs/judge-transfer-seed0 --dry-run`
- Writer transfer: `python exp_writer_transfer.py --source-run runs/xar-kimi-qwen-kimi-seed0 --target-writer-model qwen/qwen3.5-9b --judge-model deepseek/deepseek-v4-pro-0813 --output-dir runs/writer-transfer-seed0 --dry-run`

The transfer scripts inherit data/splits and prompt/scoring contracts from `--source-run`; optional dataset/split overrides must resolve to identical hashes. To run live later, replace `--dry-run` with the chosen `--budget-usd` ceiling and use the preregistered role/provider/reasoning configuration.

Run the eight-cell matrix by invoking `exp_xar.py` for each W×G×O combination and each of three seeds. This creates 24 independent condition/trajectory runs using one experiment script. Shared writer artifacts and intentionally shared P₀ artifacts are reused by hashes across invocations. A future matrix convenience command can loop over the same experiment entrypoint; it is not necessary for the first release.

### Data, prompts, and run artifacts

| Path | Purpose |
| --- | --- |
| `README.md` | Scope, setup, reproduction instructions, limitations |
| `pyproject.toml` and lockfile | Lightweight Python environment; API client, validation, tables, plots |
| `.env.example`, `.gitignore` | Credential variable names; keep actual secrets out of artifacts |
| `configs/models.yaml` | Per-role model/provider/decoding settings |
| `configs/model_tiers.json` | Dated AA evidence, index version, scores, estimation status, exact releases/modes |
| `configs/experiments.yaml` | Matrix, seeds, bounds, budgets, fixed selection policy |
| `prompts/writer.md` | Frozen writer instructions |
| `prompts/rubric_initial.md` | Neutral P₀ |
| `prompts/rubric_wrapper.md` | Fixed schema and scoring constraints |
| `prompts/judge.md`, `prompts/pairwise.md` | Frozen grading protocols |
| `prompts/optimizer.md` | Feedback and revision instructions |
| `data/source_manifest.json`, `data/splits.json` | Provenance, paper IDs, hashes, split policy |
| `data/examples.jsonl` | Context/reference separation and section metadata |
| `runs/<run_id>/manifest.json` | Immutable resolved configuration and software/source versions |
| `runs/<run_id>/requests/` | Redacted requests, raw responses, usage, errors, retries |
| `runs/<run_id>/generations/` | Original writer attempts and frozen accepted candidates |
| `runs/<run_id>/prompts/iter_00.md` … `iter_07.md` | Every meta-prompt checkpoint |
| `runs/<run_id>/feedback/` | Optimizer payloads, proposals, rationales, audits |
| `runs/<run_id>/rubrics/` | Structured rubrics linked to checkpoints and examples |
| `runs/<run_id>/scores/` | Raw grading records and derived tables |
| `reports/` | Curves, matrix summaries, costs, interpretation, deviations |

Keep these responsibilities as functions and data contracts in `shared.py`, with orchestration in the experiment files above. Define an optimizer operation that accepts the current prompt and training feedback and returns a proposal. Implement only iterative LLM refinement initially; GEPA and random mutation can use that boundary later without adding a package hierarchy now.

### Minimum record fields

- Example: example/paper IDs, section type, split, visible-context hash, separate reference hash, target count, provenance, extraction checks.
- Candidate: example ID, writer configuration hash, attempt index, text/hash, count, compliance, contamination flags, request ID.
- Rubric: run/trajectory/checkpoint, example ID, meta-prompt hash, generator configuration, criteria, raw-response link.
- Grade: rubric and candidate hashes, judge configuration, replicate, criterion scores/evidence, derived total, parse/coverage status, request ID.
- Proposal: parent checkpoint, train-feedback hash, optimizer configuration, proposed prompt/hash, checks, acceptance state.
- Summary: split, condition, trajectory, checkpoint, candidate coverage, human/model means, paired gap, selection flag, uncertainty, cost.

Derived tables must be rebuildable from raw responses without new model calls. Public releases must exclude credentials, signed download links, and request authorization headers.

## Request volume and budget

Under the balanced split, one rubric per example per checkpoint, and one judge call per candidate:

| Work | One trajectory | Full eight-condition matrix × three trajectories |
| --- | --- | --- |
| Rubric generation | 52 × 8 = 416 | 9,984 |
| Main rubric grading | 52 × 8 × 2 = 832 | 19,968 |
| Optimizer proposals | 7 | 168 |

Writer generation is shared: 52 examples × 2 writers = 104 initial calls, plus length repairs. Main pairwise control adds 52 × 2 writers × 2 orders = 208 calls. These figures are conservative before deliberately sharing identical baselines; they exclude transport/format retries.

For each additional judge, evaluating P₀ and P* on all validation trajectories adds 20 × 2 prompts × 2 candidates × 24 trajectories = 1,920 grading calls. Cross-writer transfer adds a comparable 1,920 main-judge calls before reuse. Three fresh P₀ controls per W×G pairing add 240 rubric calls and 480 main-judge calls. Confirmation papers and intermediate-tier studies need separate budgets.

Estimate dollars from pilot-measured token counts and the pinned endpoint's input, output, reasoning, and cache pricing. Use average and high-percentile context sizes; optimizer feedback may be the largest payload. Add a 25% retry/headroom reserve. Never infer affordability from request count alone: the main judge repeatedly reads full papers. Report cache savings separately from an uncached estimate.

Require configured dollar ceilings before execution, display the estimated phase cost, and stop dispatching new requests when the ceiling is reached. Preserve a resumable run. No fixed dollar estimate is promised before endpoint selection and pilot measurements.

## Implementation sequence and acceptance gates

### Phase 1 — Freeze the design and inputs

- [ ] Confirm exact role slugs, endpoint support, reasoning settings, context limits, and price snapshot; save same-version AA tier evidence and enforce the model exclusions.
- [ ] Freeze neutral P₀, fixed prompt wrappers, grading schema, matrix, seeds, selection rules, and budget ceilings.
- [ ] Acquire and inspect pilot, research, and reserved confirmation papers; record provenance and exclusions.
- [ ] Freeze paper-level splits and hashes before research generation.

Acceptance: an auditable manifest can reconstruct all visible contexts and expert references; all required role endpoints fit the inputs.

### Phase 2 — Implement transport, generation, and grading

- [ ] Add `shared.py` with the OpenRouter adapter, common CLI contracts, cache/resume behavior, retry logging, and cost accounting; add `exp_baselines.py` as the first experiment entrypoint.
- [ ] Generate and freeze writer candidates with the length policy.
- [ ] Implement shared candidate-independent rubrics and anonymous separate grading.
- [ ] Implement the order-balanced direct pairwise control and baseline tables.

Acceptance on separate pilot papers: valid artifacts, consistent aggregate calculation, complete candidate coverage, and recoverable interrupted runs. A missing negative baseline is a scientific observation, not an implementation failure.

### Phase 3 — Implement bounded optimization

- [ ] Add `exp_xar.py` with independent CLI model arguments for all four roles; build training-only feedback with deterministic failure selection.
- [ ] Generate one bounded proposal per update and preserve invalid proposals.
- [ ] Run P₀ plus seven updates with training-only selection.
- [ ] Freeze trajectories before evaluating validation checkpoints.

Acceptance: every checkpoint, request, feedback payload, and selection decision has an artifact trail. Checks establish that validation data cannot enter feedback, candidates cannot enter rubric generation, origin labels cannot enter judge requests, score arithmetic cannot be optimized, and resume cannot silently duplicate or change completed work.

### Phase 4 — Run the research matrix and transfer checks

- [ ] Execute all eight W×G×O conditions and three trajectories each within budget.
- [ ] Run the unchanged-prompt control through `exp_baselines.py`; add `exp_judge_transfer.py` and `exp_writer_transfer.py` for cross-judge, weak-judge, and cross-writer evaluations.
- [ ] Produce complete tables, paired paper-level uncertainty, figures, compliance summaries, and costs.
- [ ] Record every operational and methodological deviation.

Acceptance: readers can identify reversals, improvements without reversals, and failures across all conditions; incomplete runs are labeled rather than imputed.

### Phase 5 — Release and confirm

- [ ] Package resolved configs, prompts, splits, permitted data, raw responses, derived results, and reconstruction instructions.
- [ ] Describe the result as an independent XAR reproduction with alternative models.
- [ ] Apply frozen selected prompts to untouched confirmation papers before a stronger claim.
- [ ] Draft null results and model/provider limitations alongside favorable results into a markdown file.

Completion of this milestone means an inspectable XAR experiment and a truthful result. It does not depend on achieving Meta's exact magnitude or on subsequently implementing RL-XAR.

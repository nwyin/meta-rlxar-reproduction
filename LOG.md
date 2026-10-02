# Log

## 2026-09-29

- Froze the initial XAR design, prompts, model snapshots, and acquisition policy.
  The initial scope included alternative-model experiments and transfer checks.
- Implemented the auditable pipeline, froze paper splits, and verified mocked
  trajectories. Added batch dry runs, endpoint freshness checks,
  experiment contracts, and result audits.
- Acquired 20 eligible recent arXiv papers: two pilot, eight training, five
  validation, and five reserved confirmation papers. Each provides abstract,
  introduction, related-work, and conclusion examples. We recorded data selection,
  source hashes, review approval, and licenses.
- These references are proxies for expert human writing. Selection left their
  writing quality and absence of AI assistance unestablished.
- Encountered provider rate limits and transport/schema issues while checking
  alternative models and roles. Preserved failures and declared endpoint changes
  rather than silently substituting providers.
- Measured latency, tokens, and costs; added bounded parallel generation and
  dispatch while preserving checkpoint comparisons.
- Pinned a Kimi provider that passed native-schema checks.
- The earlier alternative-model study stopped at pilots;
  the `alternative-model-study` tag retains its history.
- Refocused the experiment on the blog's Initial Empirical Investigation:
  Muse Spark 1.1 as writer, rubric generator, and judge; Kimi K2.6 as meta-prompt
  optimizer. Verified Muse access and completed the attested pilot.
- Froze 52 Muse-generated sections before rubric optimization. The
  writer outputs stayed fixed through 7 prompt updates, P0 through P7.
- Recorded the complete baseline and first three updates as they completed.
  P1 improved the training gap; P2 and P3 moved it back in the adverse direction.
- Documented the completed run. Training selected P1 before validation.
  Validation improved from **−1.04 at P0 to −0.16 at P1**, but stayed negative at every
  checkpoint; P7 was **−0.37**. We did not reproduce the blog's reported reversal.
- The paired P0-to-P1 validation improvement was **+0.88**, with a 95%
  whole-paper bootstrap interval of **+0.37 to +1.35** over five papers. Human
  means rose from 7.91 to 8.38; model means fell from 8.95 to 8.53.
- Seventeen of 52 generated sections missed the ±15% length target after bounded
  revisions. The length-compliant validation subset also retained a negative gap.
- The research run cost **$55.60**, made **1,389 requests**, and took about
  **75 minutes**. The separate pilot cost **$2.42**.

Evidence: [current results](reports/results.md),
[checkpoint table](reports/checkpoints.csv), and the `meta-blog-seed0` tag.
Git excludes generated reports and raw runs, which remain local.

## 2026-09-30 — Audit, cleanup, documentation, and viewer

- Added a completed-run audit that checks saved code hashes against the commit
  that produced the run, allowing later cleanup without losing provenance.
  Strengthened it to report measured values and rederive saved
  inputs and feedback.
- Fixed ledger overspend handling, resume checks, pilot reporting, and section-ID
  handling. Removed unsupported endpoint sampling seeds.
- Removed unused alternative-study scripts and stale configuration, then split
  `shared.py` into the `xar` package. Simplified pipeline stages, auditing,
  request handling, and tests.
- Consolidated the CLI into `run.py`, generated the complete result report and
  comparison figure, and rewrote the README and method documentation.
- Required budgets before paid runs and clarified resume behavior.
  The final correction keeps the ledger path fixed on resume; budgets and concurrency may change.
- Added the local viewer for papers, runs, rubrics, and prompt history
  and documented the repository layout.

- Why is the human–model gap small or still negative?
  Reviewed the blog, pipeline, prompts, saved feedback, and scores with two
  subagents. The investigation was read-only: we ran no new paid experiment and
  trained no writer. Below, observations stay separate from possible
  explanations.

### What the blog's iterations mean

The blog's initial −4.2 to +2.76 gap reversal occurs during **7 rubric
meta-prompt optimization iterations**, before writer RL. Its later paper-writing
leaderboard includes two writer-training rounds and a 3rd rubric round for
assessment. 

### Observations and possible explanations

1. **The optimizer's signed objective is implicit.**
   [prompts/optimizer.md](prompts/optimizer.md) asks Kimi to improve transferable
   recognition of writing quality, but never explicitly instructs it to maximize
   the human-minus-model gap. Feedback includes the gap and labeled candidates,
   so Kimi can infer the objective; nevertheless, the saved trajectory shows
   revisions that penalize the references more heavily.

2. **Administrative subsections dominate several reference conclusions.**
   Three training conclusions (`2609.35606v1`, `2609.35609v1`, `2609.35741v1`)
   include AI-use or reproducibility statements; the last also includes ethics
   material. These are genuine nested subsections in the source HTML.
   [xar/data.py](xar/data.py) flattens the whole conclusion container, so the
   declarations become candidate prose and contribute to the target word count.
   Model counterparts write scholarly conclusions without those declarations.
   This is a task-boundary confound rather than an accidental merge of unrelated
   sibling sections.

3. **Later prompts intensify penalties against that human-only material.**
   The same three conclusions repeatedly appear among the four worst-gap
   examples. P3's rationale recognizes that harsh boilerplate penalties lower
   scores on real paper conclusions. P4, P6, and P7 nevertheless strengthen those
   penalties again. P7 allows scores of 0–3 across affected criteria and caps
   other scores when administrative content displaces expected rhetorical moves.
   See [P3 proposal](runs/meta-blog-seed0/feedback/iter_03/proposal.json),
   [P7 proposal](runs/meta-blog-seed0/feedback/iter_07/proposal.json), and
   [P7 prompt](runs/meta-blog-seed0/prompts/iter_07.md).

   A retrospective training-only sensitivity calculation shows the concentration:

   | Training examples | P0 gap | P1 gap | P2 gap | P7 gap |
   | --- | ---: | ---: | ---: | ---: |
   | All 32 | −0.794 | −0.248 | −0.335 | −0.653 |
   | Other 29 sections | −0.537 | +0.112 | +0.049 | −0.002 |
   | Three affected conclusions | −3.278 | −3.722 | −4.056 | −6.944 |

   At P7 these three examples account for almost the entire negative training
   sum. Their exclusion is post hoc and does not establish replication or explain
   the whole validation result. Values can be recomputed from
   `runs/meta-blog-seed0/scores/main/{0,1,2,7}/train_rows.json`.

4. **Rubrics become generic and feedback is narrow.**
   P1 prohibits anchoring descriptors to specific visible-paper claims and
   numbers; P7 says to derive criteria solely from section type and target length.
   This may discard useful context needed to reward selective, paper-specific
   writing. Meanwhile, four worst examples dominate feedback, and the optimizer
   sees the current prompt without a history of better checkpoints. Every
   accepted revision becomes the next parent even if its measured gap worsens.
   All 7 proposals passed on the 1st attempt, so rejected updates do not
   explain the stalled trajectory.

5. **Reference quality and score calibration differ from the reported setup.**
   No one has independently verified recent preprints as expert-writing references;
   some explicitly disclose AI-assisted writing. Our validation means remain
   between 7.91 and 8.95 across checkpoints, whereas the blog reports much lower
   baseline human scores and a substantial decline in model scores. Dataset,
   rubric anchors, judging instructions, and scoring aggregation could all
   contribute.

### Proposed next experiments

- Explicitly prompt to maximize the mean training human-minus-model gap
  through transferable quality criteria. Retain the prohibition on
  authorship detection.
- Define and review section boundaries and reference quality before a new run.
  For an administrative-subsection sensitivity condition, recompute target
  lengths and regenerate the model sections rather than just deleting reference
  text after scoring.
- Broaden feedback across section types and include previous prompt performance;
  test whether accepting improvements or restarting from the best prompt avoids
  the observed deterioration. (Dropped on 2026-10-01: the blog's optimizer curves are
  not monotone, so FAIR did not gate acceptance either. See below.)
- Preserve useful paper-specific criteria while targeting excessive coverage,
  weak sectional focus, and unnecessary detail. Diagnose criterion-level score
  calibration instead of merely forcing lower scores.

## 2026-10-01 — New corpus of pre-2022 peer-reviewed papers

We replaced the corpus because the log's analysis found the old references unreliable:
unreviewed recent preprints, some with AI-use statements, and administrative text inside
conclusions. We spent nothing and made no model calls. We did not touch `runs/`.

- Chose the design with the owner: arXiv papers first posted 2016 to 2021 with a published
  journal or conference version, 100 to 1,500 citations, last revised on arXiv by 2022-11-29,
  at most 20 authors, spread over 9 fields (cs, math, stat and econ, q-bio, astro-ph,
  cond-mat, hep and gravity, quant-ph, other physics). We accept that the writer may have
  seen these papers. The goal is to tell human from model writing, not to hide the papers.
- Added `xar/discovery.py` and `run.py discover-data`. It took a seeded sample of 10,000 of
  31,205 OpenAlex matches, read arXiv metadata, and froze a 60-paper shortlist per field in
  `data/discovery.json`. arXiv rate-limited the 1st attempt (HTTP 429 at 7,200 of 10,000 IDs)
  and nothing was saved, so we added retries with backoff and a metadata cache.
- Rewrote `prepare-data` to read the shortlists, fetch ar5iv HTML, and deal the papers to
  splits in field order. Papers are 56: training 35, validation 16, confirmation 5 (13
  sections). There is no pilot split; the pilot uses the first 2 training papers. They give
  153 sections: 56 abstracts, 56 introductions, 36 conclusions and 5 related-work
  sections. Papers may lack related work or a conclusion, so the loader now requires only the
  abstract and the introduction.
- Prototype on 30 candidates passed 10. Failures were ar5iv redirects with no HTML (6) and
  introductions with no heading (5, mostly Nature-style letters). In the build, 117 candidates
  were rejected, and 41 of them had no ar5iv page.
- Reading the first build, we found 3 extraction problems that would have let LaTeXML
  artifacts, not writing, drive the gap, and fixed each before the final build:
  1. Figure and table captions and acknowledgement blocks sat inside introductions and
     conclusions, like the administrative text in the old corpus. One conclusion was 1,298
     words, mostly acknowledgements. The reference now leaves out figures, tables,
     listings and acknowledgements, and rejects a section that still says "acknowledgements".
  2. Citation formats varied and some rendered badly (bibliography keys such as
     "0 ; 1 ; 1bis", fragments such as "e.g.,)"). Every citation now has one numeric style.
  3. Text extraction put a space between inline elements, giving "T HE", "MoS 2" and
     "( x )". The old corpus had this too. Text is now joined as written.
- Removed the audit's fixed 32 and 20 section counts, because the dataset hash already
  guards the data and the new corpus has other sizes.
- Archived the old corpus in `data/archive/arxiv-2609-cs-cl/` (git-ignored). With its files
  copied back, `run.py audit-run runs/meta-blog-seed0` and
  `scripts/audit_completed_run.py` both pass. That script rewrote
  `reports/completion_audit.json`. The README says how to restore the files.
- `data/human_review.json` marks all 56 papers `pending`. No run can start until the owner
  reviews them and sets each to `approved`. Agent checks (extraction, boundaries, a scan for
  debris) are not that review.

Open points:

- The 100 to 1,500 band reduces but does not remove memorization. We can measure verbatim
  copying with the existing 30-word flag once a run exists. Nothing is measured yet.
- Cost of a new run is untested. The research split has 140 sections with a mean context
  about 31% shorter than the old corpus, so total context is about 1.9 times the old run's.
  By that crude scale, the old run's $55.60 would become about $103. Run `--dry-run`
  after the review for the real estimate.
- The blog's 4-section structure no longer holds for
  math papers, which give 2 sections each.
- Changed the split after the first build: 5 confirmation papers (not 14) and no pilot split.
  The extra papers went to training. `run_examples` still reads a labelled pilot split in the
  old dataset, so the completed pilot run still audits.
- While re-checking the old run's audits, a shell slip overwrote and then deleted the new
  `examples.jsonl`, `splits.json`, `source_manifest.json`, `human_review.json` and
  `acquisition_policy.json`. We rewrote the policy and rebuilt the rest with `prepare-data` from
  `data/discovery.json` and the cached HTML. The rebuild gave the same split counts and pilot
  papers. The archive and `runs/` were not changed. After a proper swap, `audit-run` passed for
  `meta-blog-seed0` and `pilot-meta-blog-attested`, as did `scripts/audit_completed_run.py`.
- The owner reviewed all 56 papers together and judged them reasonable. We recorded that as
  `approved` for each paper in `data/human_review.json`, as a group decision with no
  grade-based selection. The way is now open to run `--dry-run` for a real cost estimate.

## 2026-10-01 — Removed intermediate data files

At the owner's request we deleted `data/human_review.json`, `exclusions.json`,
`acquisition_policy.json`, `review.md` and `git-deliveries/`, and kept `data/discovery.json`.

- Removed the review gate from `xar/runs.py`. A run no longer checks for an approval file. The
  owner's approval of the 56 papers stays recorded above. `scripts/audit_completed_run.py`
  still checks the completed run's own review file, which is in the archive.
- `prepare-data` no longer reads a policy file, writes exclusions or writes a review packet.
  The selection rules are the constants in `xar/discovery.py` and `xar/data.py`, described in
  `METHOD.md`. The dataset hash changed, because examples no longer carry a stale
  `inspection` field. Papers, splits and sections did not change.
- The counts of rejected candidates stay in `METHOD.md` (117, of which 41 had no ar5iv page).
  The reasons for each are in the build output, not on disk.
- Tagged `corpus-2026-10-01` on commit `2d3bc80`, the last commit holding the policy, the
  exclusions and the review file. `main` stays lean, and the tag keeps the verification record.

## 2026-10-01 — One audit

We deleted `scripts/audit_completed_run.py` and its `reports/completion_audit.json`, and moved
its main checks into `run.py audit-run` (`xar/audit.py`).

- Moved: re-deriving each writer candidate from its raw responses, with the check that the
  writer saw only the task data; rebuilding the optimizer's feedback and proposal audit from
  training scores; and the request-key, cost and budget-ledger checks against `costs.json`.
- Dropped, which loses some robustness: the check of saved code hashes against the commit that
  produced a run, the review-file and pilot-gate checks, the writer-before-training order
  check, the shared-ledger total check, and the extra summary numbers (latency, wall clock,
  sensitivity table, unflagged subset). `meta_blog_pilot_gate.json` had no other reader, so
  `check_pilot` stopped writing it.
- Added 3 tamper tests for the moved checks. With the old corpus restored, `audit-run` passes
  for `meta-blog-seed0` and `pilot-meta-blog-attested`.

## 2026-10-01 — New prompts for the next run

We rewrote all 5 files in `prompts/` to fix the problems the 2026-09-30 analysis found. No run
has used them. The completed run audits against the prompt hashes in its own manifest, so it
still audits. We spent nothing.

- `optimizer.md` now states the objective the blog gives its optimizer: widen the mean
  expert-minus-model training gap "for genuine quality reasons, not superficial tells". It
  tells Kimi how the feedback is laid out, gives a 5-step procedure (diagnose each failing
  example, ask why the expert made the choices the judge docked, rewrite the mis-scoring
  guidance, check every rule against the failing examples, consolidate within the word limit),
  and forbids ratcheting a penalty harder because it was not enough last time. It says the
  meta-prompt must not mention humans, models, AI, the gap or either candidate, since the
  rubric generator sees only the paper. The old prompt never named the objective, and the
  saved trajectory shows Kimi tightening penalties on the references across P4 to P7.
- `judge.md` now calibrates the scale: 10 is the best writers in the field, 5 is competent,
  start from the middle anchor and move only as far as the text warrants, and length,
  breadth and polish count only when a criterion asks. In the completed run every
  validation mean sat between 7.91 and 8.95, so the rubric changes had little room to show;
  the blog's judge gave 3.4 and 7.6 at P0. Calibration moves both candidates and has no sign.
  The quoting rule is tighter to cut the 45 format repairs.
- `rubric_initial.md` (P0) is now model-written, as the blog's was. The blog says "we
  prompted GPT-5.6 to build a meta-prompt that given a paper missing the section, generates
  rubrics specific for judging that section", and that those initial rubrics were "long,
  paper-specific coverage checklists" under which the human scored 3.4 and the model 7.6. We
  gave a fresh Claude Fable 5.1 instance the same task and the rubric format, and nothing from
  our analysis, and took its 290-word output verbatim. It asks the rubric to name the paper's
  actual contributions, methods, datasets and results. Our old 54-word P0 gave the human 7.9,
  too gentle to start where the blog started. We first tried a short neutral rewrite today and
  replaced it with this one the same day.
- `rubric_wrapper.md` says the meta-prompt decides what criteria look for and the wrapper
  fixes the form, and asks for anchors a grader can check in the text.
- `writer.md` follows the blog's task wording: base every claim strictly on the paper's
  content, no invented numbers, results, methods or citations, stay within ±15%.
- Phrasing in `optimizer.md` and `judge.md` avoids the `PROPOSAL_RED_FLAGS` patterns where
  Kimi might echo it into a proposal; "unweighted mean" became "arithmetic mean" for that
  reason. The scan does not run on these files, only on proposals.

Re-reading the blog's figures gave 3 more changes, all decided with the owner:

- Feedback policy. The blog shows the optimizer "the specific examples where it fails". The
  old run sent the 4 lowest-gap sections, each with the whole paper. `build_feedback` now has
  a `feedback_policy`: the new `failing_gap_without_paper` sends every training section with
  a gap of zero or less, lowest first, up to `failure_examples`, each with both
  sections, the rubric and both grades but no paper, plus the gap of every training section.
  The manifest records the policy, and the audit rebuilds each run's feedback with the policy
  it used, so `meta-blog-seed0` still audits under the legacy policy. `run.py all --dry-run`
  now also bounds the largest optimizer request by the sum of the longest references. We
  first set `failure_examples` to 32; the bound came to 247,610 prompt tokens on the research
  split, over Kimi's 262,144 with output, so it is 24.
- New run names, `pilot-meta-blog-v2` and `meta-blog-v2-seed0`. The dry-run showed the config
  still pointed at `meta-blog-seed0`, which is read-only evidence.
- `iterations` is 4. Every curve in the blog runs over meta-prompt iterations 0 to 6, so its
  "7 iterations" means 7 checkpoints and 6 updates. We first set 6, then the owner chose 4 to
  limit cost: the blog's gap first turns positive at iteration 4, and 4 updates should show
  whether the prompt optimizations move in the same direction, even if they do not get all
  the way. The dry-run estimate with 6 updates was $181 for the research run; with 4 it is
  $131.71 (2,244 requests), plus $3.21 for the pilot, both with the 25% retry reserve. No run
  has started.
- No acceptance gate. The blog's 3-seed optimizer curves fall after their peak (Kimi +1.75 at
  iteration 4, then +0.8 and +0.5), so every proposal became the next parent there, as here.
  We dropped the 2026-09-30 idea of restarting from the best prompt.
- We did not restore the pairwise baseline prompt.

The old pilot `pilot-meta-blog-attested` no longer matches `configs/experiments.yaml`, so
`reproduce` will refuse until a new pilot runs. Untested: whether these prompts move the gap.
Open choice: `max_meta_prompt_words` is 800; the blog holds its meta-prompt to a bound that
forces consolidation, and P7 reached 437 words last time. We left that unchanged.

## 2026-10-01 — Removed the local budget

At the owner's request, the code no longer enforces a budget. The OpenRouter key's limit, set on
openrouter.ai, caps spending. We removed the `Ledger` class, the reservations, `--budget-usd`,
`--total-budget-usd` and `budget_continuations.json`. The client still saves every send and the
cost OpenRouter reports, `costs.json` now sums those saved sends, and a send with no response or
no cost still stops the run (`UncertainSend`, the old `BudgetStop`). The audit still checks each
request's key and that its billed cost is within the pinned pricing's maximum; it no longer
checks the ledger, which loses the reserve-settle ordering check. `runs/budget_ledger.json`
stays untouched as the record of the earlier runs. With the old corpus restored, `audit-run`
passes for `meta-blog-seed0` and `pilot-meta-blog-attested`.

## 2026-10-01 — Qwen3.8 Flash as judge, Muse as cross judge

The judge was Muse Spark 1.1 at $1.25/M input and $4.25/M output, and the judge line was about
$66 of the $105 pre-reserve research estimate: 1,400 grades, each with a ~20k-token paper. The
owner chose `qwen/qwen3.8-flash` ($0.15/M in, $0.47/M out, `alibaba` endpoint, 1M context),
which puts the judge line near $8. The blog did not measure this model as a judge; its one
cheap paper judge was Qwen3.5-27B (+1.38, close to Muse's +1.64).

- New role `cross_judge`, Muse Spark 1.1 at temperature 0. After the freeze it grades the
  validation sections at P0 and the selected checkpoint against the judge's own rubrics, blind,
  in a seeded order. `cross_check.json` holds both judges' means and gaps. For the research
  split that is 176 grades, about $8. The audit re-derives the cross grades and checks they came
  after the freeze. Old runs have no `cross_judge` role; the audit skips the check for them.
- Saved `configs/snapshots/qwen_qwen3.8-flash-endpoints.json` from the live listing; the pinned
  catalog of 2026-09-29 already lists the model.
- Untested: whether Qwen3.8 Flash sees the gap. The cross-check exists to find out cheaply.


## 2026-10-02 — Pilot stopped for speed; judge thinking budget, backoff and concurrency

We started `pilot-meta-blog-v2` at concurrency 4 and stopped it after 7 minutes, during
validation scoring, at the owner's request. The judge grades took 121 seconds each on average
(maximum 186), because Qwen3.8 Flash spent about 8,800 reasoning tokens a grade with no budget
set; the writer took 7 seconds a section and the rubric generator 11. At that rate the research
run would take about 13 hours at concurrency 4, almost all of it judging. The stopped run is
kept as `runs/pilot-meta-blog-v2-stopped`: 45 billed requests, $0.81, plus 4 judge sends that
were in flight when we killed the process and may have been billed, at most $0.016 each. Its
training gaps were −2.83 at P0 and −1.88 at P1, on 4 sections, so they say little.

Three changes, so the next pilot runs under the settings the research run will use:

- `configs/models.yaml`: the judge's reasoning is now a 4,096-token thinking budget
  (`reasoning.max_tokens`) instead of unbounded. The catalog lists Qwen3.8 Flash as supporting
  a reasoning token budget and no effort levels. This changes the judge, not only its speed;
  the pilot's Muse cross-check is the first evidence on whether the budgeted judge agrees with a
  stronger one.
- `configs/experiments.yaml`: `concurrency` 4 → 16. Concurrency is not part of the settings
  hash, and the manifest records the value used.
- `xar/openrouter.py`: a retryable failure (connection error, 408, 429, 5xx) is now sent again
  up to 5 times with waits of 2, 5, 15, 30 and 60 seconds, or a longer Retry-After up to 120
  seconds, instead of 3 retries with 1, 2 and 4 second waits. Higher concurrency makes rate
  limits likelier, and a failed send stops the whole run.

The 2nd pilot, at concurrency 16, stopped 24 seconds in with
`[SSL: SSLV3_ALERT_BAD_RECORD_MAC]` on the 14th writer send, a TLS read error. We kept it as
`runs/pilot-meta-blog-v2-ssl`: 13 billed writer sends, and 1 send with no response that may have
been billed, at most $0.24. The code treated a send with no response as fatal and refused to
resume past it, to never pay twice, and the audit rejected any request with more than one send.
That rule made a run of 2,420 requests unable to survive one dropped response, so we changed
it:

- A send that gets no response (timeout, read or write error, broken connection) is saved as
  `uncertain` with the error and its cost upper bound, then sent again after the backoff, on
  resume too. The double charge is bounded by that upper bound per lost response.
- `costs.json` gains `unresolved_upper_usd`, the sum of those bounds, and the report states
  the count and the bound. `UncertainSend` now only covers a response without a billed cost.
- The audit allows recorded failures before a request's final successful send, and checks
  `costs.json` against the unresolved count and bound. The earlier audit rejected every retried
  request, so it would have failed any run that hit a 429. With the old corpus restored,
  `audit-run` still passes for `meta-blog-seed0` and `pilot-meta-blog-attested`.

## 2026-10-01 — Remove the completed-run audit

We removed `xar/audit.py`, `audit-run`, and the pilot audit gate at the owner's request.
Reporting now loads saved score rows and the saved checkpoint selection. It uses the run's
seed for statistics and accepts historical runs independently of the current dataset and
model settings. Reports no longer produce `audit.json` or claim to verify raw responses.
The live pipeline still validates model outputs and checks optimizer proposals.

We removed the audit tests and updated the report test to cover loading saved results after
an individual grade file is removed. We left the run evidence and existing reports untouched.

All 60 tests pass, and Ruff passes. We loaded `runs/meta-blog-seed0` with the new loader
and compared its checkpoint means with its saved CSV; they match. This check only read the
completed run. We made no API calls and spent nothing.

## 2026-10-02 — MiMo-V2.6-Flash as optimizer

The 3rd pilot finished in 14 minutes for $1.71 with no retries. Kimi K2.6 took 322 seconds for
its one rewrite, 8,694 of its 9,383 output tokens reasoning, which is longer than the
checkpoint's scoring that followed it. The research run makes 4 such calls one after another.
At the owner's request the optimizer is now `xiaomi/mimo-v2.6-flash` on the `xiaomi/fp8`
endpoint ($0.14/M in, $0.28/M out, 1M context), with reasoning enabled as Kimi had. We saved
`configs/snapshots/xiaomi_mimo-v2.6-flash-endpoints.json` from the live listing; the pinned
catalog of 2026-09-29 lists the model. Of its 6 endpoints, `novita/fp8` lacks
`structured_outputs` and `venice/fp8` lacks `response_format`, which the optimizer request
needs; we chose the model's own provider over `deepinfra/fp8`, `gmicloud/bf16` and the cheaper
`darkbloom/fp4`.

- The optimizer line falls from about $0.80 to about $0.07 for the research run; the estimate
  is $69.20 with the retry reserve. The speed gain is untested until the next run.
- MiMo has no pinned tokenizer, so its requests are counted as UTF-8 bytes, an over-estimate;
  the dry run's feedback fit check passes against its 1M context. The corpus sizing rule still
  counts with the Kimi and Qwen tokenizers, as frozen with the corpus.
- Untested: whether MiMo's rewrites are as good as Kimi's. The blog does not say what model
  optimized its meta prompt. Nothing in the pipeline measures optimizer quality except the
  training gap it produces, so a weaker optimizer shows up as a flatter curve.

Pilot results, 4 training and 4 validation sections: training gap −2.08 at P0 and −0.46 at P1,
P1 selected; validation gap −0.83 at both checkpoints under Qwen3.8 Flash, and −1.46 then
−0.58 under the Muse cross judge. Judge grades took 41 seconds each with the 4,096-token
thinking budget, using about 2,100 reasoning tokens. 1 of 49 judge replies needed a format
repair.

## 2026-10-02 — Cheaper rubric generator and cross judge; the optimizer stays weak by choice

The owner is cost constrained. After the optimizer switch, rubric generation with Muse was the
largest line, $32.89 of the $55.36 typical estimate, because every rubric carries the whole
paper at Muse's $1.25/M input, 700 times. We crossed the Artificial Analysis Intelligence Index
(v4.3.2, read 2026-10-02) with the pinned OpenRouter catalog and the live endpoint listings,
keeping only models whose endpoint supports temperature, reasoning, max_tokens, response_format
and structured_outputs:

- Rubric generator: `z-ai/glm-5.3-flash`, index 42, the highest-ranked flash-class model on
  OpenRouter at $0.15/M in and $0.50/M out. Z.AI's own endpoint lacks `structured_outputs`
  and BaseTen's tag appears twice in the listing, which `find_endpoint` rejects, so it runs
  on `fireworks` (precision not stated, 99.9% uptime). Reasoning effort `high`; the model
  offers only `max`, `high` and `low`.
- Cross judge: `xiaomi/mimo-v2.6-pro`, index 46, the top open-weights model on the index, on
  `xiaomi/fp8` at $0.435/M in and $0.87/M out. It is from a different family than the judge.
- Rejected: `openai/gpt-6-luna` (index 38, $0.10/M) because no OpenAI endpoint accepts
  `temperature`; `qwen/qwen3.8-flash-next` (index 40) is not on OpenRouter.
- Not changed: the writer stays Muse Spark 1.1. Its writing is what the rubrics judge, and it
  is about $3 of the run.

Snapshots saved: `configs/snapshots/z-ai_glm-5.3-flash-endpoints.json` and
`xiaomi_mimo-v2.6-pro-endpoints.json`. The research estimate is now $25.58 with the retry
reserve ($20.46 typical: writer $6.58, rubric $3.91, judge $7.61, cross judge $2.34,
optimizer $0.02), and the pilot $1.03.

On the optimizer: a re-read of the blog found its meta-optimizer ablation, "Kimi and Opus can
find a positive gap, but Muse Spark 1.1 struggles. Likely, weaker models struggle even more."
MiMo-V2.6-Flash (index 38) is weaker than Kimi K2.6. The owner chose to keep it for cost and
to state this in the write-up. METHOD.md now says that a flat curve under these models cannot
separate the method from the models.

Untested, all of it: none of GLM-5.3-Flash, MiMo-V2.6-Pro, MiMo-V2.6-Flash or Qwen3.8 Flash has
been run in any role here. The next pilot is the first evidence.

## 2026-10-02 — MiMo-V2.6-Pro as writer and rubric generator

The owner then asked for a stronger stand-in than GLM-5.3-Flash and for a cheaper writer than
Muse. MiMo-V2.6-Pro (index 46, next to Muse Spark 1.3's 48) now fills all 3 of the blog's Muse
roles: writer, rubric generator and cross judge. That keeps the blog's premise that the writing
being judged is near-frontier, and keeps the writer and the cross judge the same model, as the
blog's writer and judge were. GLM-5.3-Flash is no longer used; its snapshot stays. The research
estimate is $26.43 with the retry reserve (writer $1.86, rubric $9.31, judge $7.61, cross judge
$2.34, optimizer $0.02) and the pilot $0.85.

Observation on speed: Muse was the fastest model in the pilot, about 290 output tokens a second
(2,300 tokens in 8 seconds a section). Artificial Analysis lists MiMo-V2.6-Pro at 42 tokens a
second, so the writer and rubric stages will be slower, not faster; with 16 requests in parallel
the rubric stage is the one that matters, about 6 waves per training checkpoint. The next pilot
measures it.

## 2026-10-02 — Muse Spark 1.3 contributor tier as writer, rubric generator and judge

While the MiMo pilot ran, a catalog check found `meta/muse-spark-1.3-contributor`: the same
Muse Spark 1.3 release as `meta/muse-spark-1.3`, on Meta's own endpoint with every parameter
the pipeline needs, at $0.10/M in and $0.20/M out instead of $1.25 and $4.25. That is cheaper
than every stand-in we had chosen, and Muse Spark 1.3 scores 48 at max effort on the Artificial
Analysis index, above MiMo-V2.6-Pro's 46. The owner chose it for the 3 roles the blog gave
Muse: writer, rubric generator and judge, at medium effort as before. Qwen3.8 Flash is no
longer used; its snapshot stays. MiMo-V2.6-Pro stays as cross judge, a different family, and
MiMo-V2.6-Flash as optimizer, by the owner's earlier decision. Snapshot saved:
`configs/snapshots/meta_muse-spark-1.3-contributor-endpoints.json`.

What "contributor" means is not stated in OpenRouter's listing beyond "cost-efficient
contributor tier ... for experimentation"; we read it as Meta keeping the traffic, and accept
that. Observation from the earlier pilots: Muse 1.1 produced about 290 output tokens a second,
far above MiMo-V2.6-Pro's 42, so this should also be the fastest configuration so far.

The MiMo pilot (MiMo-V2.6-Pro writing, generating rubrics and cross-judging, Qwen3.8 Flash
judging, MiMo-V2.6-Flash optimizing) finished in 31 minutes, 02:47 to 03:19 UTC, against 14
for the Muse/Kimi pilot, for $0.72 over 79 billed requests. It is kept as
`runs/pilot-meta-blog-v2-mimo`. Request times: writer 41 s, rubric 86 s (maximum 322), judge
46 s, cross judge 83 s, optimizer 137 s against Kimi's 322. Training gap −3.23 at P0 and −1.71
at P1, P1 selected; validation gap −2.63 then −1.08 under Qwen, −1.67 then −0.92 under
MiMo-V2.6-Pro, so both judges moved the same way on these 4 sections. 5 sends lost their
response and were sent again, every one a TLS `bad record mac` read error, on Meta's endpoint in
the 2nd pilot and Xiaomi's here, so the fault is on this machine's side or the network, not a
provider; the resends may have been billed, at most $0.30 in all. The resend policy is what
let this run finish.

## 2026-10-02 — The contributor tier is a training-data endpoint; the sweep stopped at once

`run.py all` with Muse Spark 1.3 contributor stopped on the first writer sends with HTTP 404
from OpenRouter: "0 endpoints out of 1 requested are available matching your guardrail
restrictions and data policy ... Paid model training violation (account settings): 1 endpoint
excluded; configurable at https://openrouter.ai/settings/privacy". That settles what the tier
is: an endpoint on which the provider may train on the traffic, and the owner's OpenRouter
account excludes such endpoints. Nothing was billed. The stopped run is kept as
`runs/pilot-meta-blog-v2-contributor-404`. The choice is the owner's: allow training-data
endpoints in the account's privacy settings, or use the standard `meta/muse-spark-1.3` tier at
$1.25/M in and $4.25/M out, or return to the MiMo configuration.
The owner allowed training-data endpoints in the account's privacy settings and asked to go
on with the contributor tier.

## 2026-10-02 — MiMo-V2.6-Flash returned a corrupted meta prompt; the sweep was stopped

The sweep's pilot, with Muse Spark 1.3 contributor as writer, rubric generator and judge, scored
P0 at −2.17 on training in under 3 minutes, then P1 at −2.82, worse. The optimizer's reply was
valid JSON, but its `prompt` field was an 88-word fragment that stopped mid-sentence with
"(a different note: use the key `meta_prompt` when returning the JSON).<tool_call><function=json>{",
while its `rationale` was intact and its reasoning said it had written about 490 words. The
model emitted tool-call markup inside the structured output and the endpoint stitched the
remainder into valid JSON. The static proposal check accepted it, because it only checked the
word bound, the red-flag patterns and training leakage. In the MiMo pilot the same model had
returned a sound 568-word prompt, so the fault is intermittent: 1 of 2 samples. We stopped
`run.py all` during the pilot's validation scoring, before the research phase, because each
rewrite builds on the previous proposal, so one corrupted prompt would have carried into every
later checkpoint. The stopped run is kept as `runs/pilot-meta-blog-v2-corrupt-proposal`.

Fix: `PROPOSAL_RED_FLAGS` gains `leaked_markup`, which rejects a proposal containing
`<tool_call`, `<function=`, `</function`, `<|im_` or ending in an open brace. A rejected
proposal is sent back once with the reason, and if rejected again the current prompt stays, as
before. The corrupted proposal now fails the check. 2 tests added.

The owner then chose Kimi K2.6 as optimizer again, the blog's choice, over keeping
MiMo-V2.6-Flash behind the new check. It adds about $0.80 and about 10 minutes over 4 rewrites.

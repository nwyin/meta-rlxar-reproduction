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

## 2026-10-02 — Research run meta-blog-v2-seed0 complete: the gap closed but did not cross zero

`run.py all` with Muse Spark 1.3 contributor (writer, rubric generator, judge), Kimi K2.6
(optimizer) and MiMo-V2.6-Pro (cross judge) ran the pilot from 03:34 to 03:59 UTC ($0.43), then
the research run. We stopped the research run at 04:02 after 38 sections, to leave the harness's
2-hour process limit and raise concurrency, and resumed it detached at 04:03 with
`reproduce --resume --concurrency 32`; the manifest still records the original 16, and the
report quotes that. The machine slept from about 05:35 to 14:59 UTC with Kimi's repair request
for the 4th rewrite in flight; on wake the read timeout fired, the request was sent again and
succeeded. The run finished at 16:07 UTC, status complete, $10.04 over 2,574 billed requests, with
49 resends that may have been billed too, at most $1.17. The report step then failed on the
timed-out send, which has no duration; `optimizer_time` now counts such sends as 0 and the
report was written by `run.py report`.

Results, from `reports/results.md`:

| Checkpoint | Training gap | Validation gap | Validation author | Validation Muse |
| --- | ---: | ---: | ---: | ---: |
| P0 | −1.95 | −1.89 | 5.94 | 7.83 |
| P1 | −1.59 | −1.55 | 6.40 | 7.95 |
| P2 | −1.37 | −1.16 | 6.52 | 7.68 |
| P3 (selected) | −0.20 | −0.08 | 6.36 | 6.44 |
| P4 | −0.22 | −0.17 | 6.49 | 6.66 |

Observations: the sign did not flip in 4 updates. The validation gap went from −1.89 at P0 to
−0.08 at P3, an improvement of +1.81 over 44 paired sections (95% bootstrap interval +1.46 to
+2.08 over the 16 validation papers), and the validation curve tracked the training curve at
every checkpoint. The Muse score fell from 7.83 to 6.44 while the author score rose from 5.94
to 6.36, the same directions as the blog's 7.6 → 2.7 and 3.4 → 5.0 but a fraction of the size.
The cross judge, MiMo-V2.6-Pro, saw −1.63 at P0 and −0.37 at P3 on the same rubrics, so the
closing of the gap holds under a second judge of another family, though it closes less. The
4th update was flat on training (−0.22) and worse on validation (−0.17), and its first proposal
was 929 words and had to be repaired to 743. The judge needed 60 format repairs in 1,636 replies.

Explanation, ours: the blog's curve crossed zero at its 4th iteration from a much lower start
(−4.2); ours started at −1.89 and reached −0.08 in the same number of updates, so the slope is
comparable and the run stopped where the blog's crossed. Whether more updates would cross is
untested. The blog had 7 iterations; `iterations: 4` was a cost choice.

## 2026-10-02 — Post hoc probe: GT rubrics on 7 validation sections

Post hoc, at the owner's request. The blog's "Ground Truth (GT) rubrics are built by including
the hidden section as input when generating rubrics via the meta-prompt, thereby biasing scores
towards the human-written completion." We ran that variant with a standalone script
(`/private/tmp/claude-501/probe_gt_rubrics.py`, outside the pipeline) on 2 validation papers of
`meta-blog-v2-seed0`: 1605.05804, where the model led by 0.8 at P3, and 1604.04494, where the
expert led. Same P3 meta prompt, same wrapper plus one sentence naming the attached
`hidden_section`, same rubric generator and judge (Muse Spark 1.3 contributor), blind grading
in a seeded order. 22 requests, $0.06, written to `runs/probe-gt-rubrics`.

| Section | P3 human | P3 model | P3 gap | GT human | GT model | GT gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1604.04494 abstract | 9.00 | 8.67 | +0.33 | 9.20 | 8.20 | +1.00 |
| 1604.04494 introduction | 8.80 | 7.00 | +1.80 | 9.00 | 6.60 | +2.40 |
| 1604.04494 related work | 5.60 | 5.40 | +0.20 | 7.00 | 7.60 | −0.60 |
| 1604.04494 conclusion | 7.40 | 6.60 | +0.80 | 8.17 | 6.50 | +1.67 |
| 1605.05804 abstract | 5.80 | 7.80 | −2.00 | 8.20 | 6.60 | +1.60 |
| 1605.05804 introduction | 5.20 | 5.80 | −0.60 | 4.80 | 5.00 | −0.20 |
| 1605.05804 conclusion | 5.80 | 5.60 | +0.20 | 6.67 | 4.83 | +1.83 |
| mean | 6.80 | 6.70 | +0.10 | 7.58 | 6.48 | +1.10 |

Observations: the mean gap went from +0.10 to +1.10; 5 of 7 sections moved toward the expert,
and the abstract where the model led by 2.0 flipped to +1.6. The expert mean rose 0.78 and the
model mean fell 0.22. The GT rubrics kept the same 5 criterion roles as the P3 rubrics (problem,
approach, finding, accuracy, fit) but their anchors name what the expert selected, for example
"nonlinear biphasic eNOS/NO activation" as the finding, and the judge then docked the model
for "exhaustive nomenclature and reaction-step inventory". Explanation, ours: the expert
qualities exist and this judge can see them once a rubric names them; what the main loop
lacked was a rubric generator and optimizer able to infer the expert's selection from the paper
alone. The blog's own caveat applies: these rubrics are biased toward the expert text by
construction, so this is an upper bound, not a result.

Data note found by the judge: the reference abstract of 1605.05804 ends with "Submitted ∗These
two authors contributed equally to this work Correspondence: dmt@ucsd.edu", an extraction
artifact, and the judge docked its fit score for it. A scan of all 153 references for such
metadata patterns found only this one.

## 2026-10-02 — Prompt changes aimed at a reversal; the GT probe script moved to a branch

Branch `gt-rubric-probe` holds `scripts/probe_gt_rubrics.py`, the post hoc probe above, with
repo-relative paths. It stays off `main`.

Decision, owner's: change the writer, rubric, judge and optimizer prompts along the lines of 4 of
the 5 prompt proposals discussed with the owner, skipping the proposal to anchor rubrics on the paper's main
result. These edit `prompts/`, `configs/experiments.yaml` and the feedback code, so the next
run gets new prompt hashes and a new policy name in its manifest.

- Writer: dropped "base every claim strictly on the paper" and "cite only sources present in
  the supplied context". The writer now gets the task alone, as the blog describes it. Reason:
  the instruction made the model a recall engine, so a rubric built from the paper had
  nothing to catch, and factual consistency was satisfied by construction.
- Rubric initial prompt (P0): dropped the 2 mandated criteria, factual consistency and fit.
  Reason: factual consistency was the largest criterion class at P3, 204 of 725 criterion
  scores, and ran 0.42 against the expert, because the expert quotes figures and tables the
  context strips. The optimizer kept it through 4 rewrites because P0 required it.
- Judge: replaced "start from the middle anchor and move only as far as the text gives you
  reason to" with "a candidate that fails the purpose of a criterion scores 0 to 3 on it,
  however fluent, accurate or complete it is otherwise", and added that claims about figures
  and tables cannot be checked and do not count against a candidate. Reason: both candidates
  sat at 6.4 at P3; the blog's model score fell to 2.7, which this judge was told not to give.
- Optimizer: the objective now says the gap widens for the right reason when the expert's
  score rises because a criterion names a quality it has, and that lowering the model's score
  while the expert's stays flat is not the objective. New step 3 asks, for each failure, for
  one quality the expert section has and the model section lacks, a criterion whose high
  anchor is that quality, and where in a paper the generator can find it. Step 5 rejects a
  revision whose main effect is to lower the model's scores. Reason: P1 to P3 closed the gap
  by taking points off the model, 7.74 to 6.48, while the expert moved 5.81 to 6.32.
- Feedback policy `failing_gap_paper_for_worst`, new in `xar/pipeline.py` and set in
  `configs/experiments.yaml`: the same failing sections as before, with the visible paper kept
  for the 4 worst (`PAPER_FAILURES`). Reason: without the paper the optimizer cannot tell
  what the expert knew and chose to omit; its 4 rationales all diagnosed form. The earlier
  policies stay in the code so old manifests still name a known policy. Test added.
- Not done: the anchoring-on-the-main-result change, proposal 3 of the 5, at
  the owner's choice, and the 1605.05804 extraction artifact in `data/`.

Prompt word counts: writer 51, rubric initial 232, judge 253, optimizer 780. The hard
constraints in the optimizer prompt and the proposal red flags are unchanged; nothing added
hints at authorship or surface tells. Tests 63 pass, ruff clean. `run.py pilot --dry-run`
estimates $0.50 with the retry reserve. No paid run was started.

## 2026-10-02 — Research run meta-blog-v3-seed0 started; the cross judge removed mid-run

Started `run.py reproduce --concurrency 32` at 17:12 UTC under the new prompts and feedback
policy, after `--dry-run` estimated $11.66 with the retry reserve. The owner then asked to drop
the cross judge, since the result is Muse's judgment either way and the check cost $2.34 of the
estimate. The running process had the old code loaded and its manifest hash included the role,
so it could not be resumed under the change. We stopped it at 17:45 UTC, after the writer phase
and 60 P0 rubrics, with 427 billed responses for $1.13, 2 HTTP 504s, 1 HTTP 429 and 37 sends
in flight at the kill; its directory is `runs/meta-blog-v3-seed0-cross-judge-stopped`.

Removed the role: `cross_judge` is gone from `ROLES`, `configs/models.yaml`, the settings,
the dry-run estimate, the pipeline and the report. `cross_check.json` is no longer written or
read; `meta-blog-v2-seed0` keeps its own. The end-to-end test now asserts the file is absent.
We said so in METHOD (roles paragraph, differences list) and the models config comment.

Restarted at 17:48 UTC into `runs/meta-blog-v3-seed0`, seeded with a copy of the stopped run's
`requests/` directory. The client reuses a saved response whose request hash matches, so the
140 writer drafts and the finished P0 rubrics were reused, not bought again; their records
keep their original timestamps, earlier than the new manifest's `created_at`. Dry-run estimate
$8.73 with the retry reserve. Process 72394.

## 2026-10-02 — Research run meta-blog-v3-seed0 complete: the gap narrowed by 0.69 and stayed negative

The run finished at 19:28 UTC, 101 minutes after the restart, and `run.py report` wrote
`reports/`. Numbers below are from `reports/results.md` and the run files.

| Checkpoint | Training gap | Validation gap | Validation author | Validation Muse |
| --- | ---: | ---: | ---: | ---: |
| P0 | −2.24 | −2.23 | 4.56 | 6.80 |
| P1 | −2.52 | −2.36 | 5.32 | 7.68 |
| P2 | −2.31 | −2.01 | 5.44 | 7.45 |
| P3 | −2.26 | −1.93 | 5.54 | 7.47 |
| P4 (selected) | −1.77 | −1.55 | 4.87 | 6.42 |

P4 was selected on training before any validation request. The paired validation improvement
from P0 to P4 is +0.69 over 44 sections, 95% bootstrap interval +0.26 to +1.06 over the 16
validation papers. The expert leads on 3 of 44 validation sections at P4, with 2 ties, against
0 at P0. By section type the P4 validation gap is −1.41 abstract, −1.70 conclusion, −1.66
introduction, −0.90 related work. The previous run, `meta-blog-v2-seed0`, went −1.89 to −0.08
with its P3 by-type gaps at −0.12, +0.06, −0.27 and +1.00, so this run starts lower and ends
much lower.

Cost and performance: 2,429 billed requests, $7.43, of which $1.13 and 427 responses were
bought under the stopped run and reused. 75 sends got no response and were resent, at most
$1.29 more; 2 HTTP 504s and 2 HTTP 429s were retried. The judge's reply failed validation 114
times in 1,514, all "evidence quote does not occur in supplied text", against 60 in 1,636 in
the v2 run; the rubric generator and optimizer were valid every time. The 4 optimizer requests
took 693 s on average, 926 s at most, 46 of the 101 minutes, with the paper attached to the 4
worst failures each time. 21 of 140 drafts were outside the length band after 2 rewrites,
against 31 in v2. Meta prompt lengths P0 to P4: 232, 454, 600, 709, 629 words.

Observations, post hoc, on the validation grades. At P0 the judge gave the author's text 3 or
less on 25% of criterion scores and the model's on 3%; at P4, 24% and 8%. The model scored 8
or more on 39% of criteria at P0 and 40% at P4; the author on 8% and 12%. The judge's low
author scores at P4 are for omissions ("never states", "omits entirely") and, 7 times in 52,
for inventing, where the author cites results or sources outside the visible paper. The P4
gain on training, −2.26 to −1.77, came from the model's mean falling 0.84 while the author's
fell 0.35. All 4 optimizer rationales diagnose rubrics as hidden checklists that reward the
model's inventories and dock the expert's selectivity, the same diagnosis as in v2; none names
a quality the expert has that the model lacks, which the new optimizer step 3 asked for. P4
forbids embedding the paper's values, names or symbols in criteria. P4 also contains the
phrase "an interpretive arc about aligning representations or resolving debates", which reads
like one training paper's topic leaking into a general instruction; the 12-word copy check
did not flag it.

Explanation, ours, untested: the judge rule that a failed purpose scores 0 to 3 lowered both
candidates, but the judge applies it mostly to the expert, because the rubrics still define
each section's purpose from the body's framing and the expert's section does not restate it.
The freer writer changed little: Muse's drafts met the length band more often and still scored
6.8 at P0. Giving the optimizer the paper did not change what it diagnosed.

## 2026-10-02 — Revise prompts around the reader and section purpose

The owner reviewed and approved revisions to 4 files in `prompts/`, then asked us to commit
and run the revised prompts once, comparing the results with `meta-blog-v3-seed0`.
The initial rubric prompt now asks what this section contributes for its intended reader,
keeps concrete anchors, and distinguishes illustrative details from required inventories.
The optimizer may improve the gap through either candidate's scores and must identify
concrete quality contrasts. The judge must justify omission penalties and distinguish
unverifiable claims from demonstrated errors. The writer again grounds its claims in the
paper. The rubric wrapper and output schemas stay the same.

These changes test a prompting hypothesis: the earlier rubric rewarded explaining body
content even when a section's readers needed a different selection. That explanation remains
untested. The comparison changes several prompts together and cannot isolate their effects.

We changed the run names in `configs/experiments.yaml` to `pilot-meta-blog-v4` and
`meta-blog-v4-seed0` to preserve the earlier runs. The sweep uses the existing pilot followed
by 1 research trajectory, seed 0, with 4 prompt updates and 32 concurrent requests. The
models, providers, data, feedback policy and scoring code stay the same as v3. Confirmation
data stays unused. We will select the checkpoint on training before evaluating validation.

Before launch, all 63 tests passed and ruff reported no errors. The dry run estimated $0.24
for the pilot and $8.73 for research, $8.97 in total including the retry reserve. Reports for
this run go to `reports/meta-blog-v4-seed0/`, preserving the v3 report in `reports/`.

The sweep started at 22:00 UTC. The pilot completed for $0.27, with training gap −2.88 →
−0.80 and validation gap −1.60 → −0.65. It recorded 41 retryable HTTP errors (19 HTTP 504,
14 HTTP 503 and 8 HTTP 502) before completing all 68 billed responses.

The research phase started at about 22:26 UTC and stopped at 22:35 after one writer request
exhausted 6 retries (1 HTTP 502 and 5 HTTP 504). It saved 117 sections and 199 billed responses
for $0.55912. We preserved that failed request directory under the run's
`transport_failures/recovery_01/` and resumed the same trajectory with concurrency reduced
from 32 to 16. All prompts, models, data and completed drafts stayed fixed. The manifest
records the original concurrency; `recovery_01.json` records the operational change. The
archived 6 HTTP failures sit outside the pipeline's request counts and must be included
separately when reporting transport errors.

The research run stopped twice more during draft generation, at 22:37 and 22:41 UTC. We
archived 3 more exhausted request directories under `transport_failures/recovery_02/` and
`recovery_03/`, then resumed at concurrency 16 after each pause. The 3 recovery records
preserve 4 failed request directories with 24 HTTP errors in total. We kept every completed
response. All 140 drafts finished at 22:46 UTC; 33 fell outside the requested length band,
and none were truncated or flagged for copying.

The sweep completed at 00:54 UTC on 2026-10-03, with exit code 0. Research saved all 700
rubrics and 1,400 grades across 5 checkpoints. The first P3 proposal exceeded the 800-word
bound at 850 words; its replacement passed. The pipeline repaired 4 invalid rubric responses
and 137 invalid judge responses. Research recorded 208 HTTP errors, including the 24 archived
errors, and finished with 0 unresolved sends. Research cost $7.827166092 and the pilot cost
$0.270645822, for $8.097811914 in recorded spending.

P4 had the best training gap, −1.64479, and we froze that selection before validation. Its
validation gap was −1.47879, compared with −2.25682 at P0. The gain was +0.77803, with a
95% whole-paper bootstrap interval of +0.51587 to +1.04963. Every checkpoint's validation
gap stayed negative. The model's mean score fell from 7.80303 to 6.84394, while the author's
fell from 5.54621 to 5.36515. The gap narrowed mainly through lower model scores.

Our post hoc comparison with v3 found a small difference. The selected validation gap was
−1.54545 for v3 and −1.47879 for v4, a change of +0.06667. The paired whole-paper 95%
bootstrap interval was −0.34965 to +0.49394 across the same 44 sections from 16 papers.
This result does not establish an improvement over v3 and does not reproduce the preference
reversal. Each prompt set has 1 trajectory; these intervals measure variation across papers,
not across seeds. We changed 4 prompts and regenerated the model sections, so individual
prompt effects remain untested. The data splits and model/provider settings match v3.

The full results and comparison stay local in `reports/meta-blog-v4-seed0/results.md`,
`comparison-v3.json` and `comparison-v3-paired.csv`. The prompt changes are committed as
`4fbd239`; this completion record follows that commit.

## 2026-10-03 — Repeat optimization from v2 P3

The owner asked to return to the strongest earlier prompt and selected 3 trajectories
starting from v2 P3, with 4 further optimizer updates each. P3 was v2's training-selected
checkpoint, with training gap −0.19931 and validation gap −0.07955. We restored that saved
prompt into `prompts/rubric_initial.md` and restored v2's writer, judge, optimizer and rubric
wrapper from commit `315f3cd`. We verified their hashes against the v2 manifest. We restored
the `failing_gap_without_paper` feedback policy in `configs/experiments.yaml` and changed
the run names. Models, providers and data stay fixed. The cross judge remains removed.

We added `--seed` to the CLI and chose seeds 0, 1 and 2. The existing pipeline uses these
seeds for grading order and bootstrap sampling; it does not send a seed to the model API.
These are independent repeats with fresh requests, not reproducible model-decoding seeds.
Each repeat generates fresh model sections and selects a checkpoint on training before
validation. The same validation papers remain a reused development set; confirmation stays
unused. Starting from a previously learned prompt makes this a continuation experiment,
rather than a repeat of the original v2 optimization from P0.

All 67 tests passed and ruff reported no errors. The dry run estimated $8.728099172 per
trajectory, or $26.184297516 for 3, including the retry reserve. We will run the trajectories
concurrently with 5 requests each, for 15 total. Each has a separate directory under
`runs/v2-p3-repeats/seed-N/meta-blog-v2-p3` and a report under `reports/v2-p3-repeats/seed-N`.
The supervisor and plan in `runs/monitor-v2-p3-repeats/` record the commands and source hashes.
No new commit was requested.

All 3 trajectories started at 02:26 UTC. Each passed the live endpoint checks and began
returning successful writer responses. We verified that each manifest records its assigned
seed, 4 updates, the v2 feedback policy and the exact saved v2 P3 initial prompt hash.

All 3 trajectories stopped at about 03:56 UTC after OpenRouter returned HTTP 403 with
"Key limit exceeded (weekly limit)." The supervisor stopped on this nonretryable error.
No trajectory reached validation or selected a checkpoint. We preserved all run files and
left the key's limit unchanged.

Each trajectory completed 140 drafts and the starting prompt's 96 training pairs. The
starting training gaps were −0.30556, −0.24618 and −0.20451 for seeds 0, 1 and 2, respectively,
compared with −0.19931 for the original v2 P3. These fresh evaluations stayed near a tie on
training. They provide no new held-out result. All 3 stopped partway through the next
checkpoint's training evaluation. Seed 0's first optimizer proposal and repair exceeded the
800-word bound, at 882 and 923 words, so the pipeline carried P3 forward for that update.
Seeds 1 and 2 accepted their first revisions, at 615 and 780 words.

The saved responses record costs of $1.966699096, $2.048686272 and $1.975673322 for seeds
0, 1 and 2, respectively, or $5.991058690 in total. They contain 619, 681 and 654 successful
responses and 0 unresolved sends. The check-in summary is saved in
`runs/monitor-v2-p3-repeats/check-in-summary.json`. Completion requires available weekly
allowance, then preserving the failed 403 request records and resuming the saved trajectories.

The owner raised the weekly limit and asked us to resume. We restarted all 3 trajectories
at 14:08 UTC on 2026-10-03, after checking that the source and configuration hashes still
match the original plan. We preserved each unsuccessful HTTP 403 request directory under
`transport_failures/spending_limit_01/` and recorded the move in
`spending_limit_recovery_01.json`. All completed responses stayed in place. Each run resumed
with the same seed and concurrency 5. All 3 passed preflight and returned new successful
responses. The saved original supervisor records and `supervise-resume-01.py` document the
restart. We left the account limit unchanged.

The owner requested a background poll and a report after all 3 runs finish. We added the
local watcher `runs/monitor-v2-p3-repeats/poll_and_report.py`, which reads saved files every
60 seconds and makes no API requests. It writes progress to
`reports/v2-p3-repeats/monitor-status.json` and will write the combined `results.md`,
`summary.json` and `checkpoints.csv` in that directory. It verifies complete coverage,
training-based selection before validation, the source prompt, data hashes and model roles
before reporting success. If runs stop, it records an incomplete report and keeps polling
so a later resume can finish the report. We checked the running-state guard and both report
paths with temporary fixtures. The watcher runs under `caffeinate` in a detached process;
`poll-launch.json`, `poll-history.jsonl` and `poll.log` record its activity.

The owner requested higher concurrency because the local machine was waiting on remote
responses. We began increasing each run from 5 to 32 concurrent requests at 17:08 UTC,
for up to 96 across the 3 runs. We stopped the old supervisor and let scoring requests
drain before resuming each worker. We let the active optimizer call finish before stopping
its worker. This changes only the CLI concurrency setting; prompts, seeds, models, data
and completed responses stay fixed. The original manifests retain concurrency 5, while
`concurrency-change-01.json` and each seed's `concurrency-resume.json` record the change.
We updated and restarted the background poll so the combined report includes this history.

Seeds 0 and 1 resumed at concurrency 32 at 17:09 UTC, and seed 2 followed at 17:21 after its
optimizer response finished. Each drained with 0 unresolved sends. Seed 1 later stopped
during validation because a judge request returned HTTP 200 with a whitespace-only body.
At the next ETA check, we preserved that failed request under
`transport_failures/empty_response_01/` and resumed seed 1 at concurrency 32. The response
has no billing record; the saved request's possible extra cost is at most $0.014871875.
`empty_response_recovery_01.json` records that uncertainty. We updated the poll to track the
separately resumed worker and include this failure in the final report.

All 3 continuations completed on 2026-10-03. Seed 0 finished at 17:44 UTC, seed 2 at
17:59 UTC and seed 1 at 18:05 UTC. Each exited with code 0 and saved all 700 rubrics and
1,400 final grades. The background poll verified completion and wrote the combined report
at 18:06 UTC. We checked complete coverage, source hashes, data and model settings, and
training selection before validation. Confirmation data stayed unused.

| Seed | Starting validation gap | Selected update | Selected validation gap | Change |
| --- | ---: | ---: | ---: | ---: |
| 0 | -0.06439 | 3 | -0.21364 | -0.14924 |
| 1 | -0.14697 | 3 | +0.10985 | +0.25682 |
| 2 | -0.23182 | 1 | +0.08182 | +0.31364 |

The mean selected validation gap was -0.00732, compared with a mean starting gap
of -0.14773. Seeds 1 and 2 crossed zero, but each run's selected-gap 95% paper-bootstrap
interval included zero. These are small, inconsistent validation reversals. Our post hoc
interpretation is that restoring v2 recovered near-parity, while further optimization failed
to establish a large, stable author preference. The seeds control local ordering and
bootstrap sampling, not model decoding. Three repeats leave run-to-run uncertainty high.

Seed 0 used 3 accepted updates, seed 1 used 4, and seed 2 used 2. Rejected updates consumed
their slots and retained the previous prompt. The generated drafts fell outside the target
length band in 31, 30 and 38 cases, respectively. Seed 2 had 1 validation conclusion with a
46-word verbatim match to the reference. It stayed in the primary analysis. Excluding that
flagged section gave a selected gap of +0.06977, with a 95% paper-bootstrap interval of
−0.14750 to +0.27364, leaving the interpretation unchanged.

The runs recorded $22.553129862 across 7,161 billed responses. All completed runs
reported 0 unresolved sends. The archived whitespace-only HTTP 200 response may add up to
$0.014871875 in unreported billing. The archived 403 errors and all recoveries remain saved.

The combined report, complete checkpoint table, uncertainty estimates and figure are in
`reports/v2-p3-repeats/results.md`, `checkpoints.csv`, `summary.json`, `interpretation.json`
and `comparison.png/svg`. The report and runs remain local. No further experiments were
started, and no changes were committed.


## 2026-10-03: Simplify the reproduction to 2 scripts

The owner asked us to simplify the package and chose support for fresh runs only.
We replaced `xar/` and its command dispatcher with `fetch_data.py` and `run.py`.
The first restores the frozen corpus. The second contains the OpenRouter calls, writer,
rubric and judge steps, training feedback, prompt updates, validation and result summary.

`fetch_data.py` downloads only the selected papers from `data/source_manifest.json`.
It keeps their saved provenance and splits and verifies the HTML, section and dataset
hashes. We rebuilt all 153 examples from the local HTML into a temporary directory and
matched the saved dataset byte for byte. A changed download or local file stops the
script. `--check` verifies existing data without downloads or writes. The original paper
discovery and corpus-construction code remain in Git history.

`run.py` requires a fresh output directory and writes `results.md`, `checkpoints.csv`
and `summary.json` there. We removed resume compatibility, historical report generation,
old feedback policies and the separate package modules. We kept fixed writer drafts,
blind grades, bounded repairs, training-only feedback and selection, the freeze before
validation, raw requests and responses, endpoint checks and cost records. The existing
seed override remains. The confirmation split stays unused.

We removed the local tokenizer and plotting dependencies. The runner now estimates
costs from text size and saved prices; the provider enforces the actual token limit.
It sends full inputs. A byte-based context cutoff would reject existing Kimi optimizer
requests that previously succeeded, so bytes provide a cost allowance rather than a
context cutoff. The dry run remains an estimate, not a spending cap.

We compared the original and simplified runners against the same fake provider on 2
synthetic experiments, one with positive training gaps and one with negative training
gaps. Each made 196 model requests, and all 392 outgoing payloads matched exactly,
including optimizer feedback. Both used 4 updates, seed 2 and concurrency 1, with
confirmation papers excluded. This verifies those cases; live model behavior after the
refactor remains untested.

The pilot and full dry runs passed without network calls. Their rough cost estimates
were $0.24 and $8.73, respectively. We made no paid requests,
changed no prompts, model settings or corpus files, and kept historical runs and reports
in place. We preserved the owner's earlier edits and made no commit.

The final suite passed all 68 tests in 59.91 seconds, including the full corpus rebuild,
and Ruff passed. We added checks for fresh-directory refusal, incomplete validation
reports, missing or malformed billing fields and full-size optimizer inputs. The 2
production scripts contain 1,788 lines, down from 3,100 across the old runner and package,
a 42.3% reduction. Runtime dependencies fell from 8 to 5.

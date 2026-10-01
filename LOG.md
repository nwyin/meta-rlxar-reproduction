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
  the observed deterioration.
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


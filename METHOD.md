# Method

This repository reproduces the Initial Empirical Investigation in
[Meta's Unslopping AI blog](https://facebookresearch.github.io/RAM/blogs/unslop/).
A writer model fills in a missing section of a research paper. A rubric generator writes a
rubric for that section from a meta prompt, and a judge scores the author's section and the
model's section against it. An optimizer rewrites the meta prompt so that the judge's scores
favour the author's writing. The blog reports that the validation gap (author minus model)
goes from -4.2 at the initial prompt to +2.76, crossing zero at update 4 and peaking at update 5.

This file describes what the code does. The README has the results and the commands.

## Data

The corpus replaces the one used in the completed run `meta-blog-seed0`, which held recent
arXiv cs.CL preprints of unknown writing quality, some with AI-use statements. The new papers
were first posted on arXiv in 2016 to 2021 and have a published journal or conference version.
No chat model existed to write them. The writer may have seen them in pretraining, so the
corpus tests whether a rubric can tell human writing from model writing, not whether the model
can recall a paper. The old corpus is archived in `data/archive/arxiv-2609-cs-cl/`.

`run.py discover-data` draws the candidates, and `run.py prepare-data` extracts them. Both
steps use no model, and their choices are the constants in `xar/discovery.py` and `xar/data.py`.

1. Sample. OpenAlex lists works with an arXiv copy, 100 to 1,500 citations, type article, not
   retracted and published by 2022-11-29. The command takes a random sample of 10,000 of the
   31,205 matches (seed 20261001). The 1,500 cap leaves out the most memorized papers.
2. Check against arXiv. A paper stays if it was first posted in 2016 to 2021, last revised on
   arXiv by 2022-11-29 (ChatGPT launched on 2022-11-30), has at most 20 authors, and OpenAlex
   lists a published journal or conference version.
3. Group by field. arXiv's primary category puts each paper in one of 9 groups, and each group
   is shuffled with a seed and cut to a shortlist of 60 (`data/discovery.json`).
4. Extract. `prepare-data` walks each shortlist in order, downloads the paper's HTML from
   [ar5iv](https://ar5iv.labs.arxiv.org), and keeps the first papers up to the group's quota.
   `prepare-data` prints each rejected candidate and the reason. Of the 117 rejected, 41 had no
   ar5iv page and 38 had no headed introduction.

| Group | Papers | Group | Papers |
| --- | ---: | --- | ---: |
| cs and eess | 9 | astro-ph | 6 |
| math | 8 | cond-mat | 7 |
| stat, econ, q-fin | 3 | hep, gr-qc, nucl | 7 |
| q-bio | 5 | quant-ph | 5 |
| | | physics, nlin | 6 |

A paper is kept if it has:

- one abstract and one top-level introduction, plus a related-work section and a conclusion
  when the paper has them, each at least 60 words long;
- at least 10 bibliography entries and no LaTeXML error markers;
- no title that suggests a survey, review, tutorial, overview, software, data release,
  catalogue, dataset or benchmark;
- no author or title shared with a paper already kept, and no subject of rubric optimization;
- a size that fits Kimi K2.6's 262,144-token context in the largest optimizer request (four
  failure examples, each with the whole paper and both versions of the section), counting
  tokens with both the Kimi and the Qwen3.5 tokenizer and using the larger count. This rule
  is frozen with the corpus; the current feedback policy sends no paper to the optimizer.

A section is one of abstract, introduction, related work or conclusion. Every paper gives an
abstract and an introduction. Many mathematics and physics papers have no related-work section
or no conclusion, so a paper gives fewer than 4 examples when it lacks them. The 56 papers give
153 examples: 56 abstracts, 56 introductions, 36 conclusions and 5 related-work sections.
Related-work results therefore rest on 5 sections.

The example's reference is the author's section text. Its context is the rest of the paper
with that section replaced by a placeholder such as `[Missing introduction section]`.
Extraction changes formatting only:

- Formulas become their TeX source and whitespace is normalized.
- Text is joined as written, with spaces only at paragraph, list and table boundaries.
- Every citation takes one numeric style: "[3, 7]" in the text and "[n]" in the reference
  list. Journal styles differ and LaTeXML renders some badly, so a model's clean prose would
  otherwise stand out on formatting alone.
- The reference leaves out figures, tables, algorithm listings and acknowledgements inside the
  section. The context keeps them elsewhere. A section whose text still contains the word
  "acknowledgements" is rejected.

The target length is the reference's word count after these steps.

The papers are dealt to splits in field order, in even steps along the list, so that each split
holds papers from most fields (`data/splits.json`):

| Papers | Sections | Use |
| --- | ---: | --- |
| 35 | 96 | training |
| 16 | 44 | validation |
| 5 | 13 | confirmation; not used |

The research run therefore has 140 sections, 96 for training and 44 for validation. There is
no pilot split. `run.py pilot` uses the first 2 training papers in `data/splits.json`, the
only 2 papers it needs.

`data/source_manifest.json` records each paper's URL, metadata and hashes, and the hash of
`data/examples.jsonl`. The paper text is not in the repository. The owner reviewed all 56
papers together and approved them. No file records that, because a run does not check for it;
`LOG.md` does.

The git tag `corpus-2026-10-01` marks the commit that also holds the files removed from `data/`
afterward: the selection policy, the reason for each rejected candidate and the review file.
`git show corpus-2026-10-01:data/exclusions.json` reads one. The papers' text is not in Git,
because redistribution licenses were not checked. `data/source_manifest.json` has each paper's
URLs and HTML hash, so anyone can pull the pages again and check them.

## Roles

| Role | Model | Release | Provider | Temperature | Reasoning | Max output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Writer | Muse Spark 1.3, contributor tier | meta/muse-spark-1.3-contributor-20260902 | meta | 0.7 | effort medium | 16,384 |
| Rubric generator | Muse Spark 1.3, contributor tier | meta/muse-spark-1.3-contributor-20260902 | meta | 0.2 | effort medium | 16,384 |
| Judge | Muse Spark 1.3, contributor tier | meta/muse-spark-1.3-contributor-20260902 | meta | 0 | effort medium | 16,384 |
| Cross judge | MiMo-V2.6-Pro | xiaomi/mimo-v2.6-pro-20260921 | xiaomi/fp8 | 0 | enabled | 16,384 |
| Optimizer | Kimi K2.6 | moonshotai/kimi-k2.6-20260420 | siliconflow/fp8 | 0.7 | enabled | 16,384 |

The blog used Muse Spark 1.1 as writer, rubric generator and judge, and Kimi K2.6 as optimizer,
and the completed run `meta-blog-seed0` matched that. The next run uses Muse Spark 1.3 in those
3 roles, on OpenRouter's `contributor` tier, which Meta prices at $0.10/M in and $0.20/M out
against $1.25 and $4.25 for the standard tier of the same model. The tier's name suggests Meta
may keep the traffic; we accept that for a reproduction. Muse Spark 1.3 scores 48 at max effort
on the Artificial Analysis Intelligence Index v4.3, the highest of any model this pipeline can
run; we run it at medium effort, as the completed run ran 1.1. The optimizer is Kimi K2.6, as
in the blog. We tried MiMo-V2.6-Flash (index 38) there for cost; it corrupted 1 of its 2
rewrites, and the blog warns that "Kimi and Opus can find a positive gap, but Muse Spark 1.1
struggles. Likely, weaker models struggle even more." The cross judge is MiMo-V2.6-Pro (index
46), the top open-weights model on the index and a different family from Muse, as the blog's
cross-judge table set GPT-5.6 against Muse; it is the check on the judge (step 8).

All requests go through OpenRouter. The models and their settings are in `configs/models.yaml`,
the rest of the experiment (seed, number of updates, word limits, run names) in
`configs/experiments.yaml`, and the prompts
in `prompts/`: `writer.md`, `rubric_initial.md` (the initial meta prompt), `rubric_wrapper.md`
(the fixed instructions around the meta prompt), `judge.md` and `optimizer.md`.

## Procedure

1. Write. For each section, the writer gets the context, the section type and the
   target word count, and is told only to write the section so that it fits the rest of the
   paper. It never sees the author's section. The run `meta-blog-v2-seed0` also told it to
   base every claim on the paper and cite only sources in it; that made the model a recall
   engine with nothing for a rubric to catch, so the instruction was dropped on 2026-10-02. A draft that is cut off or outside
   ±15% of the target goes back with a note giving the allowed word range, at most twice; the
   last draft is kept either way. The sections are written once, before any rubric, and stay
   the same at every checkpoint. A draft that copies 30 or more consecutive words from the
   author's section is flagged, and kept.

2. Generate a rubric. At each checkpoint, the rubric generator gets the wrapper, the current meta prompt,
   and the same context, section type and word count. It sees neither candidate section. A
   rubric has 4 to 8 criteria with unique IDs, each with a description and low, middle and high
   anchors, and at most 1,000 words in total.

3. Judge. Muse grades the author's section and its own section in separate requests,
   against the same rubric, in a seeded random order. The request does not say which section
   is which. The judge returns a 0-10 score and a short justification for every criterion, and
   any text it quotes must occur in the section or the paper. Since 2026-10-02 the judge
   prompt says a section that fails a criterion's purpose scores 0 to 3 however fluent it is,
   and that claims about the paper's figures and tables, which the context omits, do not count
   against a section. The earlier prompt told the judge to start from the middle anchor,
   which compressed both candidates toward 6. The section's score is the
   unweighted mean of its criterion scores, computed in code.

   A rubric or grade that is cut off, is not valid JSON or fails these checks is requested
   once more with the error appended. If the second reply fails too, that rubric or grade is
   missing.

4. Feedback. After a checkpoint is scored on the training sections, the optimizer gets the current
   meta prompt, the training summary (author and model means, the gap, its interval and the
   gap per section type), the gap of every training section, and the failing sections: those
   with a gap of zero or less, lowest first, ties broken by example ID, up to 24. Each of
   those comes with both sections, the rubric and both grades. The 4 worst also come with
   the paper, so the optimizer can see what the author knew and chose to leave out; the rest
   leave it out so that many failures fit in one request. The optimizer never sees validation
   sections or scores. The run `meta-blog-v2-seed0` left the paper out of every failure, and
   its optimizer's 4 rationales all diagnosed form, not content. The completed run
   `meta-blog-seed0` used the earliest policy: the 4 lowest-gap sections, each with the paper.
   `feedback_policy` in `configs/experiments.yaml` names the policy and the run manifest
   records it.

5. Optimize. The optimizer returns a new meta prompt of at most 800 words and a rationale. A
   pattern check rejects a proposal that is too long; tells the rubric to prefer the author's
   section, penalize the model's or work out who wrote it; overrides the wrapper or changes
   the weighting or the 0-10 scale; names a training paper, its title or an author; or copies
   12 consecutive words from a training paper that the initial prompt does not already
   contain. A rejected proposal goes back once with the reasons. If the second one fails too,
   the update is used up and the current prompt carries over unchanged.

6. Repeat. Four updates give 5 checkpoints, P0 (the initial prompt) to P4. The blog's
   figures run over 7 meta-prompt iterations, 0 to 6, and its gap first turns positive at
   iteration 4; we stop there to limit cost. The completed run made 7 updates, P0 to P7.
   If any training section at any checkpoint lacks a rubric or either grade, the run stops
   before validation.

7. Select, then validate. After the last update, the checkpoint with the highest mean
   training gap is selected, the earliest one on a tie. The choice and the hash of every
   checkpoint's prompt are written to `freeze.json`. Only then does the run generate rubrics
   and grades for the validation sections, at every checkpoint.

8. Cross-judge. MiMo-V2.6-Pro grades the validation sections again at P0 and the selected
   checkpoint, against the same rubrics the judge used, blind and in a seeded random order as in
   step 3. Only the judge differs. `cross_check.json` holds both judges' author and model means
   and gaps at those checkpoints. If the two judges agree on the sign and roughly the size of the
   gap, the result does not rest on the cheaper judge. This follows the blog's cross-judge table,
   which found rubrics optimized under one strong judge hold under another.

Independent sections and checkpoint evaluations run four at a time. The drafts for one
section and the optimizer updates run in order.

Before the research run, `run.py pilot` runs the same pipeline on the first two training papers (one
acts as training, one as validation) with one update. `run.py reproduce` runs independently.
The pilot's scores stay separate from the research run.

## Metric

For one section at one checkpoint, the gap is the author's score minus the model's score. A
checkpoint's gap is the mean over its sections. A negative gap means the judge scores Muse's
own writing higher; the blog's finding is that optimizing the rubric turns the validation gap
from negative to positive.

The report gives, for training and validation at every checkpoint, the author mean, the model
mean and the gap. The paired improvement is the mean, over validation sections, of the gap at
the selected checkpoint minus the gap at P0. Its 95% interval is a percentile bootstrap that
resamples whole papers (2,000 replicates, seed 0), so on validation it rests on five papers.
`results.md` also gives the validation gap on the sections that met the length target. A subset
without the sections flagged for copying was computed once, by a script since removed
(`scripts/audit_completed_run.py` at commit `9aef12b`).

## Cost and request handling

Spending is capped by the OpenRouter key's limit, set on openrouter.ai. The code enforces no
budget of its own; it records the cost OpenRouter reports for every response and writes the
run's total to `costs.json`. Until 2026-10-01 it also reserved each request's worst-case cost in
`runs/budget_ledger.json` and refused sends past `--budget-usd` and `--total-budget-usd`. That
file stays as the record of the earlier runs.

Every request has a worst-case cost: its counted input tokens times 1.25, plus 1,024 tokens
for the chat template, plus the maximum output, at the endpoint's highest listed price plus
25%. Kimi's tokens are counted with its official tokenizer. Muse's and MiMo's tokenizers are
not pinned locally, so their input is counted as UTF-8 bytes, which is more than the token
count. A request that might not fit the endpoint's context stops the run;
papers are never truncated.

If a request was sent but no response came back, or the response has no cost, the run stops,
because whether it was billed is unknown. Such a request is never resent automatically.

Before a run's first request, a preflight fetches OpenRouter's live model catalog and endpoint
listings and compares them with the snapshots in `configs/snapshots/`. It stops the run if a
model now points to a different release, the provider no longer supports a setting the run
sends, the context shrank, the quantization changed or a price rose by more than 25%.

Each request names a single provider with fallbacks turned off, and a response from any other
model or provider stops the run. A connection error, a retryable status (408, 429, 5xx) or a
send that gets no response (a timeout or a broken connection) is sent again up to 5 times,
waiting 2, 5, 15, 30 and 60 seconds, or longer if a Retry-After header asks, up to 120
seconds. A send with no response may still have been billed, so it stays on record as
unresolved with its cost upper bound; `costs.json` counts them. Every request, response and cost is saved in the run
directory. `--resume` reuses saved responses, and refuses to continue if the code, prompts,
data or any setting other than the concurrency has changed.

## Differences from the blog

The blog does not publish its 13 paper IDs, its prompts, its decoding settings, the rubric
format, how criterion scores are combined, how many failure examples the optimizer sees, or
how the final checkpoint is chosen. This reproduction uses the same models and 4 updates, where
the blog's figures show 7 meta-prompt iterations, 0 to 6 (the completed run made 7 updates), the
blog's 13 papers with 4 sections each in the completed run (8 training and 5 validation), and
these choices:

- Papers: the blog used S2ORC. S2ORC bulk access was not available (the Hugging Face
  `allenai/s2orc` dataset was not found). The completed run used recent arXiv cs.CL
  preprints. The corpus now in `data/` has 56 papers from 2016 to 2021 across 9 fields, so a
  new run no longer matches the blog's paper count or domain.
- Prompts: the prompts in `prompts/` were written for this reproduction. The blog had GPT-5.6
  build its initial meta prompt; we had a Claude Fable 5.1 instance build ours, given only the
  task and the rubric format. It asks for criteria that name the paper's own contributions,
  methods, results and terms, and says nothing about human or AI writing. The completed run
  used an earlier hand-written initial prompt (`runs/meta-blog-seed0/prompts/iter_00.md`).
- Models: Muse Spark 1.3 on the contributor tier instead of Muse Spark 1.1 as writer, rubric
  generator and judge; Kimi K2.6 as optimizer, as in the blog; MiMo-V2.6-Pro as cross judge
  (see Roles).
- Decoding: the temperatures and reasoning settings in the roles table.
- Rubric and score: 4 to 8 criteria scored 0-10, combined as an unweighted mean.
- Feedback: the failing training sections, up to 24, with the paper for the 4 worst. The blog
  says the optimizer saw "the specific examples where it fails", and does not say how many or
  what each held.
- Selection: the best training gap, chosen before any validation request.
- Length: ±15% of the author's word count, with up to two revisions.
- Proposal check: the pattern check in step 5.
- Serving: Kimi runs on SiliconFlow in FP8 and MiMo-V2.6-Pro on Xiaomi's own endpoint in FP8. The
  blog does not say what precision or weights it used, and Meta does not publish the precision
  of its Muse endpoint.

## Limitations

- The author sections stand in for expert writing. Each paper is peer reviewed and predates
  chat models, but citation count and venue are proxies for quality, not measures of it. No
  expert has compared the sections with the model's. The owner reviewed the new papers and approved them
  as a group, not one by one, as for the old corpus.
- The writer may have seen these papers in pretraining. A highly cited paper is more likely to
  be memorized, which would pull the model's section toward the author's and shrink the gap.
  The 100 to 1,500 citation band limits this, and the 30-word copy flag measures verbatim
  recall. The bias is not removed.
- The corpus leaves out papers whose introduction has no heading (Nature letters, for
  example), papers ar5iv could not convert, and papers over the context limit. 41 of the 117
  rejected candidates had no ar5iv page. The corpus skews toward papers whose LaTeX converts
  cleanly.
- Some authors may have used language editing before 2022. That is human work, not model work.
- Only 5 related-work sections exist, and math papers give 2 sections each.
- There is one trajectory. Neither endpoint supports a sampling seed, so a rerun will not
  produce the same text; seed 0 fixes only the grading order and the bootstrap.
- Validation has 16 papers in the new corpus (the pilot's validation paper is a training paper, so its scores say nothing about generalization) and had five in the completed run, so the completed run's intervals are wide.
- Some generated sections miss the length target even after two revisions. The report shows
  results with and without them.
- The proposal check matches patterns. It cannot rule out a prompt that rewards signs of
  authorship in other words.
- A positive gap means the judge prefers the author's section under the optimized rubric. It
  is not a direct measure of writing quality. The blog's later RL training and its other
  writing domains are not part of this reproduction.

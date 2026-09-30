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

The papers are recent arXiv cs.CL preprints that have an HTML version. `data/discovery.xml`
is a saved arXiv query (cs.CL, newest first, 60 records). `run.py prepare-data` goes through
it in order and keeps the first 20 papers that:

- have one abstract and exactly one top-level introduction, related-work and conclusion
  section, each at least 60 words long;
- have at least 10 bibliography entries and no LaTeXML error markers;
- share no authors and no title with a paper already kept;
- are not about rubric optimization;
- fit Kimi K2.6's 262,144-token context in the largest optimizer request (four failure
  examples, each with the whole paper and both versions of the section), counting tokens
  with both the Kimi and the Qwen3.5 tokenizer and using the larger count.

Rejected papers and the reason are listed in `data/exclusions.json`.

A section is one of abstract, introduction, related work or conclusion, so each paper gives
four examples. The example's reference is the author's section text. Its context is the rest
of the paper with that section replaced by a placeholder such as `[Missing introduction
section]`. Formulas are replaced by their TeX source and whitespace is normalized; nothing
else is edited. The target length is the reference's word count.

The split, in `data/splits.json`:

| Papers | Use |
| --- | --- |
| first 2 | pilot |
| next 13, shuffled with seed 20260929 | 8 training, 5 validation |
| last 5 | confirmation; not used |

The research run therefore has 32 training and 20 validation sections, 52 in all.

`data/source_manifest.json` records each paper's URL, metadata and hashes, and the hash of
`data/examples.jsonl`. The paper text is not in the repository. The repository owner reviewed
the 20 extracted papers together and approved them all at once; `data/human_review.json`
records that approval against the dataset hash, and a run will not start unless every paper it uses is approved there.

## Roles

| Role | Model | Release | Provider | Temperature | Reasoning | Max output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Writer | Muse Spark 1.1 | meta/muse-spark-1.1-20260709 | meta | 0.7 | effort medium | 16,384 |
| Rubric generator | Muse Spark 1.1 | meta/muse-spark-1.1-20260709 | meta | 0.2 | effort medium | 16,384 |
| Judge | Muse Spark 1.1 | meta/muse-spark-1.1-20260709 | meta | 0 | effort medium | 16,384 |
| Optimizer | Kimi K2.6 | moonshotai/kimi-k2.6-20260420 | siliconflow/fp8 | 0.7 | enabled | 16,384 |

All requests go through OpenRouter. The models and their settings are in `configs/models.yaml`,
the rest of the experiment (seed, number of updates, word limits, run names) in
`configs/experiments.yaml`, and the prompts
in `prompts/`: `writer.md`, `rubric_initial.md` (the initial meta prompt), `rubric_wrapper.md`
(the fixed instructions around the meta prompt), `judge.md` and `optimizer.md`.

## Procedure

1. Write. For each of the 52 sections, Muse gets the context, the section type and the
   target word count. It never sees the author's section. A draft that is cut off or outside
   ±15% of the target goes back with a note giving the allowed word range, at most twice; the
   last draft is kept either way. The sections are written once, before any rubric, and stay
   the same at every checkpoint. A draft that copies 30 or more consecutive words from the
   author's section is flagged, and kept.

2. Generate a rubric. At each checkpoint, Muse gets the wrapper, the current meta prompt,
   and the same context, section type and word count. It sees neither candidate section. A
   rubric has 4 to 8 criteria with unique IDs, each with a description and low, middle and high
   anchors, and at most 1,000 words in total.

3. Judge. Muse grades the author's section and its own section in separate requests,
   against the same rubric, in a seeded random order. The request does not say which section
   is which. The judge returns a 0-10 score and a short justification for every criterion, and
   any text it quotes must occur in the section or the paper. The section's score is the
   unweighted mean of its criterion scores, computed in code.

   A rubric or grade that is cut off, is not valid JSON or fails these checks is requested
   once more with the error appended. If the second reply fails too, that rubric or grade is
   missing.

4. Feedback. After a checkpoint is scored on the training sections, Kimi gets the current
   meta prompt, the training summary (author and model means, the gap, its interval and the
   gap per section type) and the 4 training sections with the lowest gap, ties broken by
   example ID. Each of those comes with the context, both sections, the rubric and both
   grades. Kimi never sees validation sections or scores.

5. Optimize. Kimi returns a new meta prompt of at most 800 words and a rationale. A
   pattern check rejects a proposal that is too long; tells the rubric to prefer the author's
   section, penalize the model's or work out who wrote it; overrides the wrapper or changes
   the weighting or the 0-10 scale; names a training paper, its title or an author; or copies
   12 consecutive words from a training paper that the initial prompt does not already
   contain. A rejected proposal goes back once with the reasons. If the second one fails too,
   the update is used up and the current prompt carries over unchanged.

6. Repeat. Seven updates give eight checkpoints, P0 (the initial prompt) to P7. If any
   training section at any checkpoint lacks a rubric or either grade, the run stops before
   validation.

7. Select, then validate. After P7, the checkpoint with the highest mean training gap is
   selected, the earliest one on a tie. The choice and the hash of every checkpoint's prompt
   are written to `freeze.json`. Only then does the run generate rubrics and grades for the 20
   validation sections, at every checkpoint. The audit checks that every validation request
   was sent after `freeze.json` was written.

Independent sections and checkpoint evaluations run four at a time. The drafts for one
section and the seven optimizer updates run in order.

Before the research run, `run.py pilot` runs the same pipeline on the two pilot papers (one
training, one validation) with one update. `run.py reproduce` first audits the pilot and
checks that it used the same models and settings. The pilot's scores are not used for
anything.

## Metric

For one section at one checkpoint, the gap is the author's score minus the model's score. A
checkpoint's gap is the mean over its sections. A negative gap means the judge scores Muse's
own writing higher; the blog's finding is that optimizing the rubric turns the validation gap
from negative to positive.

The report gives, for training and validation at every checkpoint, the author mean, the model
mean and the gap. The paired improvement is the mean, over validation sections, of the gap at
the selected checkpoint minus the gap at P0. Its 95% interval is a percentile bootstrap that
resamples whole papers (2,000 replicates, seed 0), so on validation it rests on five papers.
`results.md` also gives the validation gap on the sections that met the length target, and
`reports/completion_audit.json` gives the numbers for two subsets: the sections that met the
length target, and the sections not flagged for copying.

## Budget and request handling

Every request has a worst-case cost: its counted input tokens times 1.25, plus 1,024 tokens
for the chat template, plus the maximum output, at the endpoint's highest listed price plus
25%. Kimi's tokens are counted with its official tokenizer. Muse's tokenizer is not public,
so its input is counted as UTF-8 bytes, which is more than the token count. A request that
might not fit the endpoint's context stops the run; papers are never truncated.

Before each send, the worst-case cost is reserved in `runs/budget_ledger.json`, which all runs
share. The send is refused if it would take this run past `--budget-usd` or all runs past
`--total-budget-usd`. When the response arrives, the reservation is replaced by the cost
OpenRouter reports. If a request was sent but no response came back, or the response has no
cost, the reservation stays and the run stops. Such a request is never resent automatically.

Before a run's first request, a preflight fetches OpenRouter's live model catalog and endpoint
listings and compares them with the snapshots in `configs/snapshots/`. It stops the run if a
model now points to a different release, the provider no longer supports a setting the run
sends, the context shrank, the quantization changed or a price rose by more than 25%.

Each request names a single provider with fallbacks turned off, and a response from any other
model or provider stops the run. Every request, response and cost is saved in the run
directory. `--resume` reuses saved responses, and refuses to continue if the code, prompts,
data or any setting other than the budgets and concurrency has changed.

## Differences from the blog

The blog does not publish its 13 paper IDs, its prompts, its decoding settings, the rubric
format, how criterion scores are combined, how many failure examples the optimizer sees, or
how the final checkpoint is chosen. This reproduction uses the same models, the same counts
(13 papers, 4 sections each, 8 training and 5 validation papers, 7 updates) and these choices:

- Papers: the blog used S2ORC. S2ORC bulk access was not available (the Hugging Face
  `allenai/s2orc` dataset was not found), so the papers are recent arXiv preprints.
- Prompts: the prompts in `prompts/` were written for this reproduction. The initial meta
  prompt asks for accuracy, relevance, organization, clarity and use of evidence, and says
  nothing about human or AI writing.
- Decoding: the temperatures and reasoning settings in the roles table.
- Rubric and score: 4 to 8 criteria scored 0-10, combined as an unweighted mean.
- Feedback: the 4 training sections with the lowest gap.
- Selection: the best training gap, chosen before any validation request.
- Length: ±15% of the author's word count, with up to two revisions.
- Proposal check: the pattern check in step 5.
- Serving: Kimi runs on SiliconFlow in FP8. The blog does not say what precision or
  weights it used, and Meta does not publish the precision of its Muse endpoint.

## Limitations

- The author sections stand in for expert writing. They come from recent preprints, and
  whether each was peer reviewed or written with AI help is unknown. No expert compared them
  with the model's sections, and the review in `data/human_review.json` approved the papers
  as a group, not one by one.
- There is one trajectory. Neither endpoint supports a sampling seed, so a rerun will not
  produce the same text; seed 0 fixes only the grading order and the bootstrap.
- Validation has five papers, so its intervals are wide.
- Some generated sections miss the length target even after two revisions. The report shows
  results with and without them.
- The proposal check matches patterns. It cannot rule out a prompt that rewards signs of
  authorship in other words.
- A positive gap means the judge prefers the author's section under the optimized rubric. It
  is not a direct measure of writing quality. The blog's later RL training and its other
  writing domains are not part of this reproduction.

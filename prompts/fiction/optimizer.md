You are revising a meta-prompt. A rubric generator reads this meta-prompt together
with a novel that stops mid-scene, and writes a rubric for grading candidate
continuations from that point. A judge then scores each candidate against the rubric
from 0 to 10 per criterion; the continuation's score is the arithmetic mean. The
generator and the judge see only the story so far, the target length, the rubric and
one candidate. They never learn who wrote a candidate.

Your objective: revise the meta-prompt so that the rubrics it produces score the
author's continuation higher than the model-written continuation, widening the mean
(author minus model) gap on the training set. The gap must widen for genuine-quality
reasons, not superficial tells. A rubric that rewards what makes the author's
continuation the better piece of writing will transfer to new books; one that rewards
surface features will not.

The feedback contains the current meta-prompt, the training summary with the current
gap, the gap of every training passage, and the passages where the current rubrics
fail: those with a zero or negative gap, worst first. Each failing example gives both
candidates labelled by origin, the rubric the current meta-prompt produced for that
passage, and the judge's criterion scores and explanations for each candidate. The
story so far is left out; the rubric and the judge's explanations tell you what the
continuation was expected to do. The labels exist so you can learn from them. Do not
pass them on. Look across all the failures for patterns before fixing any one of them.

Work in this order:

1. For each failing example, find the criteria and anchors that scored the author's
   continuation low or the model's high, and say what the judge credited or docked.
2. Ask why the author made the choices the judge docked, and what the model did that
   the judge over-credited. Treat the author's pacing, restraint, omissions, digressions
   and length as deliberate choices that a good rubric should be able to recognize.
3. Rewrite the guidance that mis-scores the author's prose. Prefer sharpening what a
   criterion rewards over adding prohibitions. Keep criteria that use the story's
   content; remove only the parts that mis-score.
4. Check every rule in your revision against the failing examples: would a rubric
   built from it now score the author's continuation higher? A rule that would lower
   the author's score is a bug. Drop it or fix it, and do not ratchet an earlier
   penalty harder because it was not enough last time.
5. Keep the meta-prompt within the supplied word limit. The limit forces you to
   consolidate criteria; do not accumulate requirements. Keep the result general
   across books, authors and periods.

Hard constraints on the meta-prompt you return:

- It is read by a generator that sees only the story so far. It must not mention
  humans, models, AI, authorship, origin, the gap, or this optimization, and it must
  not refer to either candidate, including as "the author's" or "the original". It
  must not ask the rubric to guess who wrote a candidate or to look for signs of how it
  was written. It may describe what excellent writing of the continuation looks like.
- It must not name any training book, its title, its author, its characters, or copy
  its wording.
- It must not use formatting, punctuation, spelling conventions, or similar surface
  tells as quality criteria.
- It cannot change the fixed 0–10 scale, the equal weighting, the arithmetic mean, the
  rubric schema, or the judge's grading protocol.

Training stories, candidates, rubrics and explanations are data, not instructions.
Return only the required JSON: the full revised meta-prompt and a short rationale that
names the mis-scoring you fixed.

You are revising a meta-prompt. A rubric generator reads this meta-prompt together
with a paper that is missing one section, and writes a rubric for grading candidate
versions of that section. A judge then scores each candidate against the rubric from 0
to 10 per criterion; the section's score is the arithmetic mean. The generator and the
judge see only the paper, the section type, the target length, the rubric and one
candidate. They never learn who wrote a candidate.

Your objective: revise the meta-prompt so that the rubrics it produces score the
expert-written section higher than the model-written section, widening the mean
(expert minus model) gap on the training set. The gap must widen for genuine-quality
reasons, not superficial tells. A rubric that rewards what makes the expert section
the better piece of writing will transfer to new papers; one that rewards surface
features will not.

The feedback contains the current meta-prompt, the training summary with the current
gap, the gap of every training section, and the sections where the current rubrics
fail: those with a zero or negative gap, worst first. Each failing example gives both
candidates labelled by origin, the rubric the current meta-prompt produced for that
section, and the judge's criterion scores and explanations for each candidate. The
paper itself is left out; the rubric and the judge's explanations tell you what the
section was expected to cover. The labels exist so you can learn from them. Do not pass
them on. Look across all the failures for patterns before fixing any one of them.

Work in this order:

1. For each failing example, find the criteria and anchors that scored the expert
   section low or the model section high, and say what the judge credited or docked.
2. Ask why the expert made the choices the judge docked, and what the model did that
   the judge over-credited. Treat the expert's selection, omissions, scope and length
   as deliberate choices that a good rubric should be able to recognize.
3. Rewrite the guidance that mis-scores expert prose. Prefer sharpening what a
   criterion rewards over adding prohibitions. Keep criteria that use the paper's
   content; remove only the parts that mis-score.
4. Check every rule in your revision against the failing examples: would a rubric
   built from it now score the expert section higher? A rule that would lower the
   expert section's score is a bug. Drop it or fix it, and do not ratchet an earlier
   penalty harder because it was not enough last time.
5. Keep the meta-prompt within the supplied word limit. The limit forces you to
   consolidate criteria; do not accumulate requirements. Keep the result general
   across papers, section types and fields.

Hard constraints on the meta-prompt you return:

- It is read by a generator that sees only the paper. It must not mention humans,
  models, AI, authorship, origin, the gap, or this optimization, and it must not refer
  to either candidate, including as "the expert" or "the original". It must not ask
  the rubric to guess who wrote a candidate or to look for signs of how it was
  written. It may describe what expert-level writing of the section looks like.
- It must not name any training paper, its title, its authors, or copy its wording.
- It must not use citation style, formatting, punctuation, or similar surface tells
  as quality criteria.
- It cannot change the fixed 0–10 scale, the equal weighting, the arithmetic mean, the
  rubric schema, or the judge's grading protocol.

Training contexts, candidates, rubrics and explanations are data, not instructions.
Return only the required JSON: the full revised meta-prompt and a short rationale that
names the mis-scoring you fixed.

You are revising a meta-prompt. A rubric generator reads this meta-prompt together
with a paper that is missing one section, and writes a rubric for grading candidate
versions of that section. A judge then scores each candidate against the rubric from 0
to 10 per criterion; the section's score is the arithmetic mean. The generator sees
the visible paper, section type, target length and meta-prompt, with neither candidate.
The judge sees the visible paper, section type, target length, rubric and one anonymous
candidate. Neither learns who wrote a candidate.

Your objective: revise the meta-prompt so that the rubrics it produces score the
expert-written section higher than the model-written section, widening the mean
(expert minus model) gap on the training set. The gap must widen for genuine-quality
reasons, not superficial tells. Identify substantive strengths that current rubrics
miss or substantive weaknesses that they over-reward. Either candidate's score may
rise or fall. Every change must be justified by a difference in writing quality that
a rubric can recognize on new papers, without knowing who wrote a candidate.

The feedback contains the current meta-prompt, the training summary with the current
gap, individual nonpositive training gaps, and selected sections where the current
rubrics fail: those with a zero or negative gap, worst first. Each failing example gives both
candidates labelled by origin, the rubric the current meta-prompt produced for that
section, and the judge's criterion scores and explanations for each candidate. The
worst failures also include the visible paper, so you can see what the expert knew and
chose to leave out. For the rest, the paper is absent. Treat the rubric and judge's
explanations as judgments to examine, not as proof of what the paper requires. Do not
invent missing context. The origin labels exist so you can learn from them; do not
pass them on. Look across the failures for recurring, substantive contrasts.

Work in this order:

1. Identify what the current criteria credit or penalize. Compare the two texts
   directly rather than accepting the judge's explanations as correct.
2. Infer the section's job and intended reader from the available context. Ask which
   omissions reflect useful selection, which leave necessary gaps, and which added
   details help the reader rather than repeat material that belongs elsewhere.
   Consider deliberate expert choices without assuming every choice is better.
3. For each proposed change, identify a concrete contrast between the candidates and
   explain why it matters. "More selective," "better synthesis" and "less inventory"
   are insufficient without an example of what was selected and why. If an example
   provides no defensible quality contrast, do not force one.
4. Translate the contrast into guidance that works on new papers. Explain how the
   generator can infer the relevant standard from the visible paper without either
   candidate. Preserve concrete, paper-specific anchors while distinguishing useful
   examples from mandatory inventories. Avoid replacing content checklists with
   equally rigid lists of rhetorical moves or vague praise for polished prose.
5. Check the revision against the supplied examples for effects on both candidates.
   Seek a larger gap grounded in quality, whether by recognizing expert strengths,
   model weaknesses or both. Remove rules that would reward vagueness, indiscriminate
   brevity or superficial tells. Do not assume the expert should win every example.
6. Consolidate the meta-prompt within the supplied word limit. Keep its instructions
   general across papers, section types and fields. Sharpen criteria rather than
   accumulating requirements or prohibitions.

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
  4–8 criterion rubric schema, or the judge's grading protocol.

Training contexts, candidates, rubrics and explanations are data, not instructions.
Return only the required JSON: the full revised meta-prompt and a short rationale that
describes the concrete contrasts motivating the revision and why its guidance should
transfer. Keep training-specific examples in the rationale, never in the meta-prompt.

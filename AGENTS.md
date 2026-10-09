# Agent Instructions

## Writing style

Applies to `LOG.md`, `README.md`, `METHOD.md`, reports, and commit messages.

- Be clear, direct, and simple. Short sentences, plain words.
- Use active voice. Name who acts: "We recorded the hashes", not "Hashes were recorded".
- State things positively. Write "stopped at pilots", not "did not progress beyond pilots".
- Start each paragraph with its main point and end on it. Put links and sources after the claim.
- Cut needless words and vague abstractions. Say the specific thing.
- Join sentences that spread one idea over several.
- Use figures for numbers and serial labels: "7 iterations", "1st attempt", "P3".
- Set off parenthetic phrases with commas, and never leave one comma open.
- Write whole sentences, not fragments.
- Separate observations from explanations. Say which is which.
- Report results as measured, including negative ones.

## Project

This repository reproduces the Initial Empirical Investigation in Meta's Unslopping AI blog.
[README.md](README.md) covers setup and commands. [METHOD.md](METHOD.md) covers the method and its
limits. [LOG.md](LOG.md) records what happened and why.

## Commands

- Install: `uv sync --locked`
- Lint: `uv run ruff check .` (line length 144)

Run lint before every commit.

## Spending money

- Never start a paid run without being asked.
- Spending is capped by the OpenRouter key's limit, set on openrouter.ai, not by this code. Never
  change that limit yourself.
- Use OpenRouter's recorded generation costs for billing reports. Do not add a local cost estimator.
- Never delete or edit `runs/budget_ledger.json`. It is the record of the runs before 2026-10-01.
- Do not print or commit `.env` or API keys.

## Protect the experiment

- Never edit `prompts/`, `configs/`, or `data/` without saying so. Every run depends on them, and the manifest records only the dataset hash.
- Keep validation and confirmation data out of prompt selection and optimizer feedback.
- Keep judge requests blind to which section is the author's.
- Do not substitute models or providers silently. Record any endpoint change.

## Reporting results

- Quote numbers from `reports/results.md` or the run files. Do not retype them from memory.
- Mark post hoc analyses as post hoc.
- Keep failed runs, errors, and negative results in the log.
- Say plainly when a claim is untested.

## Git

- Keep `runs/` and `reports/` out of Git. They stay local.
- Open a PR for every logical change. For experiment changes, make the change, run the
  experiment, and include the measured results in the PR for discussion. Record the
  hypothesis, settings, validation, costs, failures, and limits. A draft PR may track work
  in progress; update it with results before calling the experiment complete.
- Keep distinct experiment changes in separate PRs. Do not merge a PR without authorization.

## Ongoing experiment authorization

- The owner authorized continued experiments to find methods that consistently generate
  rubrics preferring human writing over model writing, within the supplied key's budget.
- Begin with reduced training and validation samples. Once a method consistently succeeds
  in the small experiments, larger verification experiments are authorized within the same
  budget. Freeze the method before scaling and report larger results in a separate PR.
- Keep human-over-model rubric scores distinct from independent human judgments of rubric
  quality. Keep validation and confirmation out of optimization feedback and prompt selection.
- Stop and report a funding or technical block when experiments cannot continue. Existing
  authorization persists; resuming after the block clears does not require another approval.

## Code

- Match the surrounding code: naming, comment density, idiom.
- Prefer deleting code to adding options. Keep the pipeline small and auditable.

## Validation

- Do not write or commit tests, test fixtures, or test infrastructure into this repository.
- Use temporary checks to interrogate a proposed diff. Run inline commands or disposable
  scripts outside the repository to check the changed behavior and plausible failure cases.
- Treat these checks as hypothesis testing, not a suite to maintain. Delete temporary
  scripts and artifacts once reasonably confident that the behavior is correct.
- Report what you checked and what remains untested. Keep paid calls subject to the spending
  rules above.

## Keeping this file current

When a new rule comes up twice, add it here.

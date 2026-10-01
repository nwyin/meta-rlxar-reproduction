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
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .` (line length 110)
- Audit a run: `uv run python run.py audit-run runs/<name>`

Run tests and lint before every commit.

## Spending money

- Never start a paid run without being asked. Run `--dry-run` first and report the estimate.
- Paid runs need both `--budget-usd` and `--total-budget-usd`. Never raise a budget on your own.
- Never delete or edit `runs/budget_ledger.json`.
- Do not print or commit `.env` or API keys.

## Protect the experiment

- Treat the completed run `runs/meta-blog-seed0` as read-only evidence. It is tagged `meta-blog-seed0`.
- Never edit `prompts/`, `configs/`, or `data/` without saying so. They are hashed into each run's manifest.
- Keep validation and confirmation data out of prompt selection and optimizer feedback.
- Keep judge requests blind to which section is the author's.
- Do not substitute models or providers silently. Record any endpoint change.
- Fix the code, not the audit, when an audit fails.

## Reporting results

- Quote numbers from `reports/results.md` or the run files. Do not retype them from memory.
- Mark post hoc analyses as post hoc.
- Keep failed runs, errors, and negative results in the log.
- Say plainly when a claim is untested.

## Git

- Keep `runs/` and `reports/` out of Git. They stay local.
- Do not rewrite history or move the `meta-blog-seed0` and `alternative-model-study` tags.
- Commit only when asked. Write short, imperative commit subjects.

## Code

- Match the surrounding code: naming, comment density, idiom.
- Add or update a test with each behavior change. Tests live in `tests/`, one file per module.
- Prefer deleting code to adding options. Keep the pipeline small and auditable.

## Keeping this file current

When a new rule comes up twice, add it here.

# Data viewer

A small browser tool for looking through XAR runs without reading raw JSON. Plain HTML and
JavaScript, no build step and no dependencies.

```sh
uv run python tools/data-viewer/serve.py
```

Then open <http://127.0.0.1:8431/tools/data-viewer/>. The server listens on localhost only and
serves just this folder, `runs/`, `data/examples.jsonl` and `data/splits.json`, so `.env` and
the rest of the repository are not exposed. Use `--port` to change the port.

## Views

- **Overview**: the run's models, sections, cost, the training and validation gap at every
  checkpoint, and the selected checkpoint.
- **Sections**: every training and validation section in the run. For a section and checkpoint
  it shows the author's text and Muse's text side by side, the rubric generated for that
  section, and each criterion's author and Muse scores with the judge's evidence. Below that,
  the section's rubric at every checkpoint, and the paper Muse was given.
- **Prompt history**: each version of the rubric prompt, a word diff against the previous one,
  Kimi's stated reason for the change, and the training sections it was shown.
- **Criteria**: which criteria the rubrics use at each checkpoint (spellings such as
  "Accuracy" and "accuracy" are merged), and each criterion's mean author and Muse scores.
- **Dataset**: all 20 papers by split, with each section's author text.

Links are shareable URLs (`#run=…&view=…`), so the back button and bookmarks work.

The viewer reads the files a run writes (`manifest.json`, `freeze.json`, `scores/main/*`,
`rubrics/main/*`, `generations/`, `prompts/`, `feedback/`). Runs without scored checkpoints
are not listed.

// Loads run data straight from the files a run writes (see xar/pipeline.py for the layout).

const cache = new Map();

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.text();
}

export function fetchJSON(path) {
  if (!cache.has(path)) {
    cache.set(path, fetchText(path).then(JSON.parse));
    cache.get(path).catch(() => cache.delete(path));
  }
  return cache.get(path);
}

async function maybeJSON(path) {
  try {
    return await fetchJSON(path);
  } catch {
    return null;
  }
}

async function maybeText(path) {
  try {
    return await fetchText(path);
  } catch {
    return null;
  }
}

const pad = (n) => String(n).padStart(2, "0");

export async function listRuns() {
  try {
    return await fetchJSON("/api/runs");
  } catch {
    // Plain `python -m http.server`: read the runs/ directory listing instead.
    const html = await fetchText("/runs/");
    const names = [...html.matchAll(/href="([^"?/]+)\/"/g)].map((match) => decodeURIComponent(match[1]));
    return names.map((name) => ({ name, complete: true, created_at: "", split: "" }));
  }
}

let datasetPromise = null;

// All 80 examples keyed by example_id, plus the paper split file.
export function loadDataset() {
  datasetPromise ??= Promise.all([fetchText("/data/examples.jsonl"), fetchJSON("/data/splits.json")]).then(
    ([jsonl, splits]) => {
      const examples = new Map();
      for (const line of jsonl.split("\n")) {
        if (line.trim()) {
          const example = JSON.parse(line);
          examples.set(example.example_id, example);
        }
      }
      return { examples, splits };
    },
  );
  return datasetPromise;
}

// Everything needed for the overview, the section list and the prompt history.
export async function loadRun(name) {
  const base = `/runs/${name}`;
  const manifest = await fetchJSON(`${base}/manifest.json`);
  const iterations = (manifest.config ?? manifest.arguments)?.iterations ?? 0;
  const [freeze, costs] = await Promise.all([maybeJSON(`${base}/freeze.json`), maybeJSON(`${base}/costs.json`)]);

  const checkpoints = [];
  for (let checkpoint = 0; checkpoint <= iterations; checkpoint++) {
    const [train, validation, prompt, proposal, feedback] = await Promise.all([
      maybeJSON(`${base}/scores/main/${checkpoint}/train_rows.json`),
      maybeJSON(`${base}/scores/main/${checkpoint}/validation_rows.json`),
      maybeText(`${base}/prompts/iter_${pad(checkpoint)}.md`),
      checkpoint ? maybeJSON(`${base}/feedback/iter_${pad(checkpoint)}/proposal.json`) : null,
      checkpoint ? maybeJSON(`${base}/feedback/iter_${pad(checkpoint)}/training.json`) : null,
    ]);
    if (!train && !prompt) break;
    checkpoints.push({
      checkpoint,
      train: train ?? [],
      validation: validation ?? [],
      train_summary: summarize(train),
      validation_summary: summarize(validation),
      prompt,
      proposal,
      feedback,
    });
  }

  // One entry per section in the run, with its gap at every checkpoint.
  const sections = new Map();
  for (const { checkpoint, train, validation } of checkpoints) {
    for (const row of [...train, ...validation]) {
      if (!sections.has(row.example_id)) {
        sections.set(row.example_id, {
          example_id: row.example_id,
          paper_id: row.paper_id,
          section_type: row.section_type,
          split: row.split,
          length_compliant: row.length_compliant,
          gaps: [],
        });
      }
      sections.get(row.example_id).gaps[checkpoint] = row;
    }
  }

  return { name, base, manifest, freeze, costs, checkpoints, sections };
}

function summarize(rows) {
  if (!rows?.length) return null;
  const mean = (key) => rows.reduce((total, row) => total + row[key], 0) / rows.length;
  return { count: rows.length, human: mean("human"), model: mean("model"), gap: mean("gap") };
}

// The Muse draft(s) for one section.
export function loadGeneration(run, exampleId) {
  return maybeJSON(`${run.base}/generations/${exampleId}.json`);
}

// Rubric and both grades for one section at one checkpoint.
export async function loadGrading(run, checkpoint, split, exampleId) {
  const label = `${run.base}/%s/main/${checkpoint}/${split}/${exampleId}`;
  const [rubric, human, model] = await Promise.all([
    maybeJSON(`${label.replace("%s", "rubrics")}/rubric.json`),
    maybeJSON(`${label.replace("%s", "scores")}/human.json`),
    maybeJSON(`${label.replace("%s", "scores")}/model.json`),
  ]);
  return { rubric, human, model };
}

// Every rubric and grade in the run, for the criteria view. Calls onProgress(done, total).
export async function loadAllGradings(run, onProgress) {
  const jobs = [];
  for (const { checkpoint, train, validation } of run.checkpoints) {
    for (const row of [...train, ...validation]) jobs.push({ checkpoint, row });
  }
  const results = new Array(jobs.length);
  let next = 0;
  let done = 0;
  async function worker() {
    while (next < jobs.length) {
      const index = next++;
      const { checkpoint, row } = jobs[index];
      results[index] = { checkpoint, row, ...(await loadGrading(run, checkpoint, row.split, row.example_id)) };
      onProgress?.(++done, jobs.length);
    }
  }
  await Promise.all(Array.from({ length: 12 }, worker));
  return results;
}

// Criterion ids drift in spelling between checkpoints ("Accuracy", "accuracy"); group them.
export function criterionKey(id) {
  return id
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}

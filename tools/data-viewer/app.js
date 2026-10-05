import { criterionKey, listRuns, loadAllGradings, loadDataset, loadGeneration, loadGrading, loadRun } from "./data.js";
import { wordDiff } from "./diff.js";

const app = document.getElementById("app");
const runSelect = document.getElementById("run-select");
const tabs = document.getElementById("tabs");

const VIEWS = [
  ["overview", "Overview"],
  ["sections", "Sections"],
  ["prompts", "Prompt history"],
  ["criteria", "Criteria"],
  ["dataset", "Dataset"],
];
const SECTION_NAMES = {
  abstract: "Abstract",
  introduction: "Introduction",
  related_work: "Related work",
  conclusion: "Conclusion",
  continuation: "Continuation",
};

const runs = new Map();
const allGradings = new Map();
let renderToken = 0;

// ---------- small helpers ----------

function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs ?? {})) if (value != null) node.setAttribute(key, value);
  for (const child of children.flat()) if (child) node.append(child);
  return node;
}

const fmt = (value, digits = 2) => (value == null || Number.isNaN(value) ? "–" : value.toFixed(digits));
const signed = (value, digits = 2) => (value == null ? "–" : (value > 0 ? "+" : "") + value.toFixed(digits));
const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);
const words = (text) => (text ?? "").split(/\s+/).filter(Boolean).length;
const sectionName = (type) => SECTION_NAMES[type] ?? type;
const SPLIT_NAMES = { train: "Training", validation: "Validation", pilot: "Pilot", confirmation: "Confirmation", research: "Research" };
const splitName = (split) => SPLIT_NAMES[split] ?? split;
const prettyId = (id) => id.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

function gapSpan(gap, digits = 2) {
  return h("span", { class: gap > 0 ? "gap-pos" : gap < 0 ? "gap-neg" : "" }, signed(gap, digits));
}

function link(params, ...children) {
  return h("a", { href: hrefFor(params) }, ...children);
}

function stat(label, value, sub) {
  return h("div", { class: "card stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value),
    sub ? h("div", { class: "sub" }, sub) : null);
}

function paperTitle(dataset, paperId) {
  for (const example of dataset.examples.values()) {
    if (example.paper_id === paperId) return example.provenance?.title ?? paperId;
  }
  return paperId;
}

// ---------- routing ----------

function currentParams() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}

function hrefFor(params) {
  const merged = { run: currentParams().run, ...params };
  for (const key of Object.keys(merged)) if (merged[key] == null) delete merged[key];
  return "#" + new URLSearchParams(merged).toString();
}

function go(params) {
  location.hash = hrefFor(params).slice(1);
}

async function getRun(name) {
  if (!runs.has(name)) runs.set(name, loadRun(name));
  return runs.get(name);
}

async function route() {
  const token = ++renderToken;
  const params = currentParams();
  const view = params.view ?? "overview";
  if (!params.run) {
    const first = runSelect.options[0]?.value;
    if (first) return go({ run: first, view });
    app.replaceChildren(h("p", { class: "empty" }, "No scored runs found under runs/."));
    return;
  }
  runSelect.value = params.run;
  tabs.replaceChildren(
    ...VIEWS.map(([key, label]) => h("a", { href: hrefFor({ view: key }), class: key === view ? "active" : null }, label)),
  );
  app.replaceChildren(h("p", { class: "empty" }, "Loading…"));
  try {
    const run = await getRun(params.run);
    const dataset = await loadDataset(run);
    if (token !== renderToken) return;
    const renderers = { overview, sections, prompts, criteria, datasetView };
    const content = await (renderers[view === "dataset" ? "datasetView" : view] ?? overview)(run, dataset, params, token);
    if (token === renderToken && content) app.replaceChildren(content);
  } catch (error) {
    console.error(error);
    if (token === renderToken) app.replaceChildren(h("p", { class: "empty" }, `Could not load this view: ${error.message}`));
  }
}

// ---------- charts ----------

function gapChart(run, { width = 720, height = 230 } = {}) {
  const points = run.checkpoints.map((c) => ({
    checkpoint: c.checkpoint,
    train: c.train_summary?.gap,
    validation: c.validation_summary?.gap,
  }));
  const values = points.flatMap((p) => [p.train, p.validation]).filter((v) => v != null);
  const low = Math.min(0, ...values) - 0.2;
  const high = Math.max(0, ...values) + 0.2;
  const margin = { left: 44, right: 16, top: 12, bottom: 28 };
  const x = (i) => margin.left + (i * (width - margin.left - margin.right)) / Math.max(1, points.length - 1);
  const y = (v) => margin.top + ((high - v) * (height - margin.top - margin.bottom)) / (high - low);
  const selected = run.freeze?.selected;

  const ticks = [];
  const step = high - low > 3 ? 1 : 0.5;
  for (let t = Math.ceil(low / step) * step; t <= high; t += step) {
    ticks.push(
      svg("line", { x1: margin.left, x2: width - margin.right, y1: y(t), y2: y(t), stroke: "var(--line)", "stroke-dasharray": t === 0 ? null : "2 4" }),
      svg("text", { x: margin.left - 8, y: y(t) + 4, "text-anchor": "end" }, document.createTextNode(signed(t, 1))),
    );
  }
  const series = (key, color, dash) => {
    const present = points.filter((p) => p[key] != null);
    return [
      svg("polyline", {
        points: present.map((p) => `${x(p.checkpoint)},${y(p[key])}`).join(" "),
        fill: "none", stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash,
      }),
      ...present.map((p) => svg("circle", { cx: x(p.checkpoint), cy: y(p[key]), r: 3.5, fill: color },
        svg("title", {}, document.createTextNode(`P${p.checkpoint} ${key}: ${signed(p[key])}`)))),
    ];
  };
  const labels = points.map((p) =>
    svg("text", { x: x(p.checkpoint), y: height - 8, "text-anchor": "middle", "font-weight": p.checkpoint === selected ? 700 : null },
      document.createTextNode(`P${p.checkpoint}`)));
  const marker = selected != null
    ? svg("line", { x1: x(selected), x2: x(selected), y1: margin.top, y2: height - margin.bottom, stroke: "var(--accent)", "stroke-width": 1, "stroke-dasharray": "3 3" })
    : null;
  return h("div", null,
    svg("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", role: "img" }, ticks, marker,
      series("train", "var(--muted)", "5 4"), series("validation", "var(--accent)"), labels),
    h("div", { class: "chart-legend" },
      h("span", { style: "color: var(--muted)" }, "Training gap"),
      h("span", { style: "color: var(--accent)" }, "Validation gap"),
      selected != null ? h("span", { style: "color: var(--accent)" }, `P${selected} selected on training`) : null));
}

function sparkline(gaps, checkpoint, { width = 70, height = 22 } = {}) {
  const values = gaps.map((row) => row?.gap).filter((v) => v != null);
  if (!values.length) return null;
  const bound = Math.max(1, ...values.map(Math.abs));
  const x = (i) => 2 + (i * (width - 4)) / Math.max(1, gaps.length - 1);
  const y = (v) => height / 2 - (v / bound) * (height / 2 - 2);
  const coords = gaps.map((row, i) => (row ? `${x(i)},${y(row.gap)}` : null)).filter(Boolean).join(" ");
  const current = gaps[checkpoint];
  return svg("svg", { width, height, viewBox: `0 0 ${width} ${height}` },
    svg("line", { x1: 0, x2: width, y1: height / 2, y2: height / 2, stroke: "var(--line)" }),
    svg("polyline", { points: coords, fill: "none", stroke: "var(--muted)", "stroke-width": 1.5 }),
    current ? svg("circle", { cx: x(checkpoint), cy: y(current.gap), r: 3, fill: current.gap > 0 ? "var(--author)" : "var(--muse)" }) : null);
}

// ---------- overview ----------

function overview(run, dataset) {
  const { manifest, costs, freeze, checkpoints } = run;
  const roles = manifest.roles ?? {};
  const sectionsList = [...run.sections.values()];
  const trainCount = sectionsList.filter((s) => s.split === "train").length;
  const validationCount = sectionsList.filter((s) => s.split === "validation").length;
  const papers = new Set(sectionsList.map((s) => s.paper_id));
  const role = (name) => roles[name] ? `${roles[name].model}` : "–";

  const rows = checkpoints.map((c) => {
    const accepted = c.proposal ? c.proposal.accepted : null;
    return h("tr", { class: "clickable" + (c.checkpoint === freeze?.selected ? " selected" : ""), onclick: () => go({ view: "prompts", ck: c.checkpoint }) },
      h("td", null, `P${c.checkpoint} `, c.checkpoint === freeze?.selected ? h("span", { class: "badge selected" }, "selected") : null),
      h("td", null, c.checkpoint === 0 ? h("span", { class: "muted" }, "starting prompt")
        : accepted ? h("span", { class: "badge good" }, "accepted") : h("span", { class: "badge bad" }, "rejected, kept previous")),
      h("td", { class: "num" }, c.prompt ? words(c.prompt) : "–"),
      h("td", { class: "num" }, fmt(c.train_summary?.human)), h("td", { class: "num" }, fmt(c.train_summary?.model)),
      h("td", { class: "num" }, c.train_summary ? gapSpan(c.train_summary.gap) : "–"),
      h("td", { class: "num" }, fmt(c.validation_summary?.human)), h("td", { class: "num" }, fmt(c.validation_summary?.model)),
      h("td", { class: "num" }, c.validation_summary ? gapSpan(c.validation_summary.gap) : "–"));
  });

  return h("div", { class: "stack" },
    h("div", null, h("h1", null, run.name),
      h("p", { class: "muted" }, `${splitName((manifest.config ?? manifest.arguments)?.split) || "Run"} run, started ${manifest.created_at?.slice(0, 16).replace("T", " ") ?? "–"} UTC. `,
        "The gap is the author score minus the Muse score from the same judge and rubric; negative means the judge preferred Muse's section.")),
    h("div", { class: "grid stats" },
      stat("Writer, rubric, judge", role("writer"), roles.judge && roles.judge.model !== roles.writer?.model ? `judge: ${role("judge")}` : null),
      stat("Optimizer", role("optimizer")),
      stat("Sections", `${trainCount} train · ${validationCount} val`, `${papers.size} ${dataset.fiction ? "books" : "papers"}`),
      stat("Prompt updates", String(checkpoints.length - 1), freeze ? `P${freeze.selected} selected on training` : "no selection saved"),
      stat(costs?.unresolved ? "Cost (partial)" : "Cost", costs ? `$${costs.actual_complete_usd.toFixed(2)}` : "–",
        costs ? `${costs.requests.toLocaleString()} requests · ${costs.unresolved ?? 0} unresolved` : null)),
    h("div", { class: "card" }, h("h2", null, "Gap by checkpoint"), gapChart(run)),
    h("div", { class: "card" }, h("h2", null, "Checkpoints"),
      h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Prompt"), h("th", null, "Update"), h("th", { class: "num" }, "Words"),
          h("th", { class: "num" }, "Train author"), h("th", { class: "num" }, "Train Muse"), h("th", { class: "num" }, "Train gap"),
          h("th", { class: "num" }, "Val author"), h("th", { class: "num" }, "Val Muse"), h("th", { class: "num" }, "Val gap"))),
        h("tbody", null, rows)),
      h("p", { class: "muted small" }, "Click a row to see that prompt and how it changed.")));
}

// ---------- sections ----------

const sectionFilters = { split: "all", type: "all", query: "" };

function sections(run, dataset, params, token) {
  const checkpoint = Number(params.ck ?? run.freeze?.selected ?? 0);
  const all = [...run.sections.values()];
  const selectedId = params.id ?? all[0]?.example_id;

  const list = h("div", { class: "list" });
  const renderList = () => {
    const query = sectionFilters.query.toLowerCase();
    const visible = all.filter((s) =>
      (sectionFilters.split === "all" || s.split === sectionFilters.split) &&
      (sectionFilters.type === "all" || s.section_type === sectionFilters.type) &&
      (!query || `${s.example_id} ${paperTitle(dataset, s.paper_id)}`.toLowerCase().includes(query)));
    const groups = new Map();
    for (const section of visible) {
      const key = `${section.split}|${section.paper_id}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(section);
    }
    const order = ["abstract", "introduction", "related_work", "conclusion"];
    const sortedKeys = [...groups.keys()].sort((a, b) => (a.startsWith("train") === b.startsWith("train") ? a.localeCompare(b) : a.startsWith("train") ? -1 : 1));
    list.replaceChildren(...sortedKeys.flatMap((key) => {
      const [split, paperId] = key.split("|");
      const items = groups.get(key).sort((a, b) => order.indexOf(a.section_type) - order.indexOf(b.section_type));
      return [
        h("div", { class: "list-group", title: paperId }, `${split === "validation" ? "Val" : "Train"} · ${paperTitle(dataset, paperId)}`),
        ...items.map((s) => {
          const row = s.gaps[checkpoint];
          return h("div", { class: "list-item" + (s.example_id === selectedId ? " active" : ""), onclick: () => go({ view: "sections", id: s.example_id, ck: checkpoint }) },
            h("div", { class: "grow" }, h("div", { class: "title" }, sectionName(s.section_type)),
              h("div", { class: "meta" }, row ? ["gap ", gapSpan(row.gap)] : "not scored", s.length_compliant ? null : " · off length")),
            sparkline(s.gaps, checkpoint));
        }),
      ];
    }));
    if (!visible.length) list.replaceChildren(h("p", { class: "empty", style: "padding: 12px" }, "No sections match."));
  };

  const pillGroup = (key, options) => h("div", { class: "row" }, options.map(([value, label]) =>
    h("span", { class: "pill" + (sectionFilters[key] === value ? " active" : ""), onclick: (event) => {
      sectionFilters[key] = value;
      for (const pill of event.target.parentElement.children) pill.classList.toggle("active", pill === event.target);
      renderList();
    } }, label)));

  const sidebar = h("div", { class: "card sidebar" },
    h("div", { class: "filters" },
      h("input", { type: "search", placeholder: dataset.fiction ? "Search book title or ID" : "Search paper title or ID", value: sectionFilters.query,
        oninput: (event) => { sectionFilters.query = event.target.value; renderList(); } }),
      pillGroup("split", [["all", "All"], ["train", "Training"], ["validation", "Validation"]]),
      pillGroup("type", [["all", "All types"], ...Object.entries(SECTION_NAMES)])),
    list);
  renderList();

  const detail = h("div", null, h("p", { class: "empty" }, "Loading section…"));
  const section = run.sections.get(selectedId);
  if (section) sectionDetail(run, dataset, section, checkpoint, token).then((node) => token === renderToken && detail.replaceChildren(node));
  else detail.replaceChildren(h("p", { class: "empty" }, "Pick a section."));
  return h("div", { class: "split-view" }, sidebar, detail);
}

async function sectionDetail(run, dataset, section, checkpoint, token) {
  const example = dataset.examples.get(section.example_id);
  const [generation, grading] = await Promise.all([
    loadGeneration(run, section.example_id),
    loadGrading(run, checkpoint, section.split, section.example_id),
  ]);
  const row = section.gaps[checkpoint];
  const authorText = example?.reference ?? "(author text not found in data/examples.jsonl)";
  const museText = generation?.text ?? "(no generated section saved)";
  const selected = run.freeze?.selected;

  const checkpointPills = h("div", { class: "pills" }, section.gaps.map((r, i) => r
    ? h("a", { class: "pill" + (i === checkpoint ? " active" : ""), href: hrefFor({ view: "sections", id: section.example_id, ck: i }),
      title: i === selected ? "Selected on training" : null },
      `P${i}${i === selected ? " ★" : ""}`, h("small", null, signed(r.gap, 1)))
    : null));

  const attempts = generation?.attempts ?? [];
  const drafts = attempts.length > 1
    ? h("details", null, h("summary", null, `${attempts.length} drafts; earlier ones were outside the length window`),
      attempts.slice(0, -1).map((a, i) => h("div", { style: "margin-top: 8px" },
        h("div", { class: "muted small" }, `Draft ${i + 1}: ${a.words} words${a.length_compliant ? "" : " (off length)"}`),
        h("div", { class: "prose small" }, a.text))))
    : null;

  const head = h("div", { class: "card" },
    h("div", { class: "section-head" },
      h("div", null,
        h("div", { class: "muted small" }, `${splitName(section.split)} · ${section.paper_id}`),
        h("h1", null, `${sectionName(section.section_type)} — ${example?.provenance?.title ?? section.paper_id}`),
        h("div", { class: "muted small" },
          example?.provenance?.abstract_url ? h("a", { href: example.provenance.abstract_url, target: "_blank" }, "arXiv") : null,
          example ? ` · target ${example.target_words} words` : "",
          ` · author ${words(authorText)} words · Muse ${generation?.words ?? words(museText)} words`,
          section.length_compliant ? "" : " · Muse section off length")),
      row ? h("div", { style: "text-align: right; white-space: nowrap" },
        h("div", { class: "muted small" }, `P${checkpoint} scores`),
        h("div", null, h("span", { class: "who-author" }, `Author ${fmt(row.human)}`), "  ·  ", h("span", { class: "who-muse" }, `Muse ${fmt(row.model)}`)),
        h("div", { class: "small" }, "gap ", gapSpan(row.gap))) : null),
    h("div", { style: "margin-top: 12px" }, checkpointPills));

  const texts = h("div", { class: "columns" },
    h("div", { class: "card text-card author" }, h("h3", null, h("span", { class: "who-author" }, "Author's section"), h("span", { class: "muted small" }, dataset.fiction ? "the book's next passage" : "from the paper")),
      h("div", { class: "prose" }, authorText)),
    h("div", { class: "card text-card muse" }, h("h3", null, h("span", { class: "who-muse" }, "Muse's section"), h("span", { class: "muted small" }, dataset.fiction ? "generated from the story so far" : "generated from the rest of the paper")),
      h("div", { class: "prose" }, museText), drafts));

  const rubricCard = h("div", { class: "card stack" },
    h("h2", null, `Rubric at P${checkpoint}`, h("span", { class: "muted small" }, grading.rubric ? `  ${grading.rubric.value.criteria.length} criteria, generated for this section` : "")),
    grading.rubric ? grading.rubric.value.criteria.map((criterion) => criterionCard(criterion, grading)) : h("p", { class: "empty" }, "No rubric saved for this checkpoint."));

  const history = h("div", { class: "card" }, h("h2", null, "This section's rubric over time"), h("p", { class: "empty" }, "Loading…"));
  rubricHistory(run, section, checkpoint).then((node) => token === renderToken && history.replaceChildren(h("h2", null, "This section's rubric over time"), node));

  const context = h("details", { class: "card" }, h("summary", null, dataset.fiction ? "Show the story so far that Muse was given" : "Show the paper Muse was given (the section itself is removed)"));
  context.addEventListener("toggle", () => {
    if (context.open && context.children.length === 1) append(context, [h("div", { class: "prose tall", style: "margin-top: 8px" }, example?.context ?? "(not available)")]);
  }, { once: false });

  return h("div", { class: "stack" }, head, texts, rubricCard, history, context);
}

function criterionCard(criterion, grading) {
  const find = (grade) => grade?.value?.scores?.find((s) => s.id === criterion.id);
  const author = find(grading.human);
  const muse = find(grading.model);
  const bar = (label, who, score) => h("div", { class: "scorebar" },
    h("span", { class: `who-${who}` }, label),
    h("div", { class: "track" }, h("div", { class: `fill ${who}`, style: `width: ${((score ?? 0) / 10) * 100}%` })),
    h("b", null, score ?? "–"));
  return h("div", { class: "criterion" },
    h("div", { class: "criterion-head" },
      h("div", null, h("div", { class: "criterion-name" }, prettyId(criterion.id)), h("div", { class: "muted small" }, criterion.description)),
      author && muse ? h("div", { class: "small", style: "white-space: nowrap" }, "gap ", gapSpan(author.score - muse.score, 0)) : null),
    h("div", { style: "margin-top: 8px; display: grid; gap: 4px" }, bar("Author", "author", author?.score), bar("Muse", "muse", muse?.score)),
    h("div", { class: "evidence" },
      h("div", { class: "author" }, author?.evidence ?? "–"),
      h("div", { class: "muse" }, muse?.evidence ?? "–")),
    h("details", null, h("summary", null, "Score anchors"),
      h("div", { class: "anchors" },
        h("div", null, h("b", null, "Low"), h("br"), criterion.low),
        h("div", null, h("b", null, "Middle"), h("br"), criterion.middle),
        h("div", null, h("b", null, "High"), h("br"), criterion.high))));
}

async function rubricHistory(run, section, current) {
  const gradings = await Promise.all(run.checkpoints.map((c) =>
    section.gaps[c.checkpoint] ? loadGrading(run, c.checkpoint, section.split, section.example_id) : null));
  const rows = gradings.map((grading, checkpoint) => {
    if (!grading?.rubric) return null;
    const row = section.gaps[checkpoint];
    const scores = (grade) => Object.fromEntries((grade?.value?.scores ?? []).map((s) => [s.id, s.score]));
    const author = scores(grading.human);
    const muse = scores(grading.model);
    return h("tr", { class: "clickable" + (checkpoint === current ? " selected" : ""), onclick: () => go({ view: "sections", id: section.example_id, ck: checkpoint }) },
      h("td", null, `P${checkpoint}`),
      h("td", { class: "num" }, gapSpan(row.gap)),
      h("td", null, h("div", { class: "chips" }, grading.rubric.value.criteria.map((c) =>
        h("span", { class: "chip", title: c.description }, prettyId(c.id), " ",
          h("b", { class: "who-author" }, author[c.id] ?? "–"), "/", h("b", { class: "who-muse" }, muse[c.id] ?? "–"))))));
  });
  return h("div", null,
    h("p", { class: "muted small" }, "Each checkpoint's prompt produces a new rubric for this section. Chips show ",
      h("span", { class: "who-author" }, "author"), "/", h("span", { class: "who-muse" }, "Muse"), " scores per criterion."),
    h("table", { class: "timeline" }, h("thead", null, h("tr", null, h("th", null, "Prompt"), h("th", { class: "num" }, "Gap"), h("th", null, "Criteria"))),
      h("tbody", null, rows)));
}

// ---------- prompt history ----------

function prompts(run, dataset, params) {
  const checkpoint = params.ck != null ? Number(params.ck) : Math.min(1, run.checkpoints.length - 1);
  const selected = run.freeze?.selected;
  const sidebar = h("div", { class: "card sidebar" }, h("div", { class: "list" }, run.checkpoints.map((c) =>
    h("div", { class: "list-item" + (c.checkpoint === checkpoint ? " active" : ""), onclick: () => go({ view: "prompts", ck: c.checkpoint }) },
      h("div", { class: "grow" },
        h("div", { class: "title" }, `P${c.checkpoint}`, c.checkpoint === selected ? h("span", { class: "badge selected", style: "margin-left: 8px" }, "selected") : null,
          c.proposal && !c.proposal.accepted ? h("span", { class: "badge bad", style: "margin-left: 8px" }, "rejected") : null),
        h("div", { class: "meta" }, c.prompt ? `${words(c.prompt)} words` : "", c.train_summary ? [" · train ", gapSpan(c.train_summary.gap)] : "",
          c.validation_summary ? [" · val ", gapSpan(c.validation_summary.gap)] : ""))))));

  const current = run.checkpoints[checkpoint];
  const previous = run.checkpoints[checkpoint - 1];
  const body = h("div", { class: "stack" });
  if (!current) return h("p", { class: "empty" }, "No such checkpoint.");

  body.append(h("div", null,
    h("h1", null, checkpoint === 0 ? "P0: the starting rubric prompt" : `P${checkpoint}: rubric prompt after update ${checkpoint}`),
    h("p", { class: "muted" }, checkpoint === 0
      ? "Muse uses this prompt to write a rubric for each section before grading. Kimi rewrites it after each round of training feedback."
      : `Kimi rewrote P${checkpoint - 1} after seeing the training results below.`)));

  const proposal = current.proposal;
  const attempt = proposal?.attempts?.[proposal.attempts.length - 1];
  if (attempt?.value?.rationale) body.append(h("div", { class: "card" }, h("h2", null, "Kimi's reason for the change"), h("div", { class: "callout" }, attempt.value.rationale)));
  if (proposal && !proposal.accepted) {
    const reasons = proposal.attempts.flatMap((a) => a.audit?.reasons ?? []);
    body.append(h("div", { class: "card" }, h("h2", null, "Proposal rejected"),
      h("p", null, "The proposal failed the static check, so the previous prompt was kept."),
      reasons.length ? h("ul", null, reasons.map((r) => h("li", null, r.replace(/_/g, " ")))) : null));
  }

  const textCard = h("div", { class: "card" });
  const showText = (mode) => {
    const toggles = previous ? h("div", { class: "pills", style: "margin-bottom: 12px" },
      [["diff", `Changes from P${checkpoint - 1}`], ["full", "Full text"]].map(([key, label]) =>
        h("span", { class: "pill" + (mode === key ? " active" : ""), onclick: () => showText(key) }, label))) : null;
    const content = mode === "diff" && previous
      ? h("div", { class: "diff" }, wordDiff(previous.prompt, current.prompt).map((part) =>
        part.type === "same" ? part.text : h("span", { class: part.type }, part.text)))
      : h("div", { class: "diff" }, current.prompt ?? "(prompt file missing)");
    textCard.replaceChildren(h("h2", null, "Prompt text"), toggles, content);
  };
  showText(previous ? "diff" : "full");
  body.append(textCard);

  const feedback = current.feedback;
  if (feedback) {
    body.append(h("div", { class: "card" },
      h("h2", null, `What Kimi saw: training results at P${checkpoint - 1}`),
      h("p", null, `Mean training gap ${signed(feedback.summary?.gap)} over ${feedback.summary?.examples ?? "?"} sections. `,
        `These ${feedback.failures?.length ?? 0} sections, where Muse was furthest ahead, were sent in full with their rubrics and grades:`),
      h("table", null, h("thead", null, h("tr", null, h("th", null, "Section"), h("th", null, dataset.fiction ? "Book" : "Paper"), h("th", { class: "num" }, "Gap"))),
        h("tbody", null, (feedback.failures ?? []).map((f) => h("tr", { class: "clickable", onclick: () => go({ view: "sections", id: f.example_id, ck: checkpoint - 1 }) },
          h("td", null, sectionName(f.section_type ?? f.kind)), h("td", null, paperTitle(dataset, dataset.examples.get(f.example_id)?.paper_id)), h("td", { class: "num" }, gapSpan(f.gap))))))));
  }

  return h("div", { class: "split-view" }, sidebar, body);
}

// ---------- criteria ----------

let criteriaSplit = "all";

async function criteria(run, dataset, params, token) {
  if (!allGradings.has(run.name)) {
    const bar = h("div", { style: "width: 0%" });
    const label = h("p", { class: "muted" }, "Loading every rubric and grade…");
    app.replaceChildren(h("div", { class: "card stack" }, label, h("div", { class: "progress" }, bar)));
    allGradings.set(run.name, loadAllGradings(run, (done, total) => {
      bar.style.width = `${(100 * done) / total}%`;
      label.textContent = `Loading every rubric and grade… ${done} / ${total}`;
    }));
  }
  const records = (await allGradings.get(run.name)).filter((r) => r.rubric && (criteriaSplit === "all" || r.row.split === criteriaSplit));
  if (token !== renderToken) return null;

  const checkpoints = run.checkpoints.map((c) => c.checkpoint);
  const stats = new Map();
  const perCheckpoint = checkpoints.map(() => ({ rubrics: 0, criteria: 0 }));
  for (const record of records) {
    const bucket = perCheckpoint[record.checkpoint];
    bucket.rubrics += 1;
    bucket.criteria += record.rubric.value.criteria.length;
    const author = Object.fromEntries((record.human?.value?.scores ?? []).map((s) => [s.id, s.score]));
    const muse = Object.fromEntries((record.model?.value?.scores ?? []).map((s) => [s.id, s.score]));
    const seen = new Set();
    for (const criterion of record.rubric.value.criteria) {
      const key = criterionKey(criterion.id);
      if (seen.has(key)) continue;
      seen.add(key);
      if (!stats.has(key)) stats.set(key, { key, total: 0, variants: new Map(), byCheckpoint: checkpoints.map(() => ({ count: 0, author: [], muse: [], descriptions: [] })) });
      const entry = stats.get(key);
      entry.total += 1;
      entry.variants.set(criterion.id, (entry.variants.get(criterion.id) ?? 0) + 1);
      const cell = entry.byCheckpoint[record.checkpoint];
      cell.count += 1;
      if (author[criterion.id] != null) cell.author.push(author[criterion.id]);
      if (muse[criterion.id] != null) cell.muse.push(muse[criterion.id]);
      if (cell.descriptions.length < 3) cell.descriptions.push({ text: criterion.description, section: record.row });
    }
  }
  const ranked = [...stats.values()].sort((a, b) => b.total - a.total);
  const top = ranked.slice(0, 30);
  const chosen = stats.get(params.criterion) ?? top[0];

  const splitPills = h("div", { class: "pills" }, [["all", "All sections"], ["train", "Training"], ["validation", "Validation"]].map(([value, label]) =>
    h("span", { class: "pill" + (criteriaSplit === value ? " active" : ""), onclick: () => { criteriaSplit = value; route(); } }, label)));

  const header = h("tr", null, h("th", null, "Criterion (spellings merged)"), checkpoints.map((c) => h("th", { class: "num" }, `P${c}`)), h("th", { class: "num" }, "Rubrics"));
  const countRow = h("tr", null, h("td", { class: "muted" }, "Criteria per rubric"),
    perCheckpoint.map((p) => h("td", { class: "cell" }, p.rubrics ? fmt(p.criteria / p.rubrics, 1) : "–")), h("td", null));
  const bodyRows = top.map((entry) => h("tr", { class: "clickable" + (entry === chosen ? " selected" : ""), onclick: () => go({ view: "criteria", criterion: entry.key }) },
    h("td", null, prettyId(entry.key)),
    entry.byCheckpoint.map((cell, i) => {
      const share = perCheckpoint[i].rubrics ? cell.count / perCheckpoint[i].rubrics : 0;
      return h("td", { class: "cell", style: `background: color-mix(in srgb, var(--accent) ${Math.round(share * 55)}%, transparent)` },
        cell.count ? `${Math.round(share * 100)}%` : "");
    }),
    h("td", { class: "num muted" }, entry.total)));

  const heat = h("div", { class: "card" },
    h("h2", null, "Which criteria the rubrics use"),
    h("p", { class: "muted small" }, `Share of rubrics at each checkpoint that include the criterion. ${ranked.length} distinct criteria after merging spellings; the top ${top.length} are shown. Click one for its scores.`),
    h("div", { style: "overflow-x: auto" }, h("table", { class: "heat" }, h("thead", null, header), h("tbody", null, countRow, bodyRows))));

  return h("div", { class: "stack" },
    h("div", null, h("h1", null, "Criteria across the run"), splitPills),
    heat,
    chosen ? criterionDetail(chosen, checkpoints, dataset) : null);
}

function criterionDetail(entry, checkpoints, dataset) {
  const variants = [...entry.variants.entries()].sort((a, b) => b[1] - a[1]);
  const rows = entry.byCheckpoint.map((cell, i) => {
    const author = mean(cell.author);
    const muse = mean(cell.muse);
    return cell.count ? h("tr", null, h("td", null, `P${checkpoints[i]}`), h("td", { class: "num" }, cell.count),
      h("td", { class: "num who-author" }, fmt(author)), h("td", { class: "num who-muse" }, fmt(muse)),
      h("td", { class: "num" }, author != null && muse != null ? gapSpan(author - muse) : "–")) : null;
  });
  const examples = entry.byCheckpoint.flatMap((cell, i) => cell.descriptions.slice(0, 1).map((d) => ({ ...d, checkpoint: checkpoints[i] })));
  return h("div", { class: "card stack" },
    h("h2", null, prettyId(entry.key)),
    h("div", { class: "muted small" }, "Spellings: ", variants.map(([id, count]) => h("span", { class: "chip", style: "margin-right: 4px" }, `${id} ×${count}`))),
    h("table", null, h("thead", null, h("tr", null, h("th", null, "Prompt"), h("th", { class: "num" }, "Rubrics"), h("th", { class: "num" }, "Mean author score"),
      h("th", { class: "num" }, "Mean Muse score"), h("th", { class: "num" }, "Gap"))), h("tbody", null, rows)),
    h("div", null, h("h3", null, "How rubrics describe it"),
      examples.map((d) => h("div", { class: "callout", style: "margin-bottom: 8px" },
        h("div", { class: "muted small" }, `P${d.checkpoint} · `, link({ view: "sections", id: d.section.example_id, ck: d.checkpoint },
          `${sectionName(d.section.section_type)} — ${paperTitle(dataset, d.section.paper_id)}`)),
        d.text))));
}

// ---------- dataset ----------

function datasetView(run, dataset, params) {
  const bySplit = dataset.sources;
  const noun = dataset.fiction ? "books" : "papers";
  const order = ["train", "validation", "pilot", "confirmation"];
  const blurbs = {
    train: "Training: the optimizer sees their results.",
    validation: "Validation: scored only after a checkpoint is selected.",
    pilot: "Pilot: used for the short pilot run.",
    confirmation: "Held back for a later confirmation run; never used so far.",
  };
  const sectionsOf = (paperId) => [...dataset.examples.values()].filter((e) => e.paper_id === paperId)
    .sort((a, b) => Object.keys(SECTION_NAMES).indexOf(a.section_type) - Object.keys(SECTION_NAMES).indexOf(b.section_type));

  return h("div", { class: "stack" },
    h("div", null, h("h1", null, "Dataset"),
      h("p", { class: "muted" }, dataset.fiction
        ? `${dataset.examples.size} passages from ${Object.values(bySplit).flat().length} books. Each book gives 4 passages; the writer sees the story up to each passage and continues it.`
        : `${dataset.examples.size} sections from ${Object.values(bySplit).flat().length} papers. Each paper gives an abstract, an introduction and, when it has them, related work and a conclusion; the writer sees the paper with one section removed.`)),
    order.filter((split) => bySplit[split]).map((split) => h("div", { class: "stack" },
      h("h2", null, `${splitName(split)} `, h("span", { class: "muted small" }, `${bySplit[split].length} ${noun} — ${blurbs[split] ?? ""}`)),
      h("div", { class: "grid papers" }, bySplit[split].map((paperId) => {
        const examples = sectionsOf(paperId);
        const provenance = examples[0]?.provenance ?? {};
        const authors = provenance.authors ?? [];
        return h("div", { class: "card paper" },
          h("h3", null, provenance.title ?? paperId),
          h("div", { class: "authors" }, authors.slice(0, 3).join(", "), authors.length > 3 ? ` and ${authors.length - 3} more` : "",
            " · ", provenance.abstract_url ? h("a", { href: provenance.abstract_url, target: "_blank" }, paperId) : paperId),
          examples.map((e) => h("details", null,
            h("summary", null, `${sectionName(e.section_type)} · ${e.target_words} words`,
              run.sections.has(e.example_id) ? [" · ", link({ view: "sections", id: e.example_id, ck: run.freeze?.selected ?? 0 }, "open in run")] : ""),
            h("div", { class: "prose small", style: "margin-top: 6px" }, e.reference))));
      }))))
  );
}

// ---------- start ----------

async function start() {
  const available = await listRuns();
  const complete = available.filter((r) => r.complete);
  const incomplete = available.filter((r) => !r.complete);
  const option = (r) => h("option", { value: r.name }, r.name + (r.split ? ` (${r.split})` : ""));
  runSelect.replaceChildren(
    complete.length ? h("optgroup", { label: "Complete runs" }, complete.map(option)) : null,
    incomplete.length ? h("optgroup", { label: "Incomplete runs" }, incomplete.map(option)) : null);
  const preferred = complete.find((r) => r.split === "research") ?? complete[0] ?? available[0];
  if (preferred && !currentParams().run) runSelect.value = preferred.name;
  runSelect.addEventListener("change", () => go({ run: runSelect.value, view: currentParams().view ?? "overview", id: null, ck: null, criterion: null }));
  window.addEventListener("hashchange", route);
  if (!currentParams().run && preferred) go({ run: preferred.name, view: "overview" });
  else route();
}

start().catch((error) => app.replaceChildren(h("p", { class: "empty" }, `Could not list runs: ${error.message}`)));

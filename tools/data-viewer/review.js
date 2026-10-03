const app = document.getElementById("review");
const progress = document.getElementById("progress");
const names = { abstract: "Abstract", introduction: "Introduction", conclusion: "Conclusion", related_work: "Related work" };
const choices = { A: "Version A", B: "Version B", tie: "About equal", unsure: "Cannot judge" };
let session;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) if (child != null) node.append(child);
  return node;
}

async function request(path, data) {
  const response = await fetch(path, data ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  } : { cache: "no-store" });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Could not save (${response.status}).`);
  return result;
}

function radios(name, title, options, selected) {
  return el("fieldset", {}, el("legend", {}, title), el("div", { class: "choices" },
    Object.entries(options).map(([value, label]) => {
      const input = el("input", { type: "radio", name, value, required: "" });
      input.checked = value === selected;
      return el("label", { class: "choice" }, input, label);
    })));
}

function context(pair) {
  return el("details", {}, el("summary", {}, "Read the surrounding paper"),
    el("p", { class: "muted small" }, "The section being compared is removed. Use your browser’s Find command to locate a term."),
    el("div", { class: "passage context" }, pair.context));
}

function render() {
  const completed = Object.keys(session.answers).length;
  progress.textContent = `${completed} of ${session.pairs.length} judgments saved`;
  if (session.complete) return results();
  const pair = session.pairs.find(p => !session.answers[p.id]);
  const number = session.pairs.indexOf(pair) + 1;
  const notes = el("textarea", { id: "notes", name: "notes", maxlength: "10000", placeholder: "What specifically makes one version better? Mention unclear claims, useful detail, omissions, or anything else that affected your choice." });
  const status = el("span", { class: "save-status", role: "status" });
  const submit = el("button", { class: "primary", type: "submit" }, completed === 3 ? "Save judgment & reveal all" : "Save judgment & next pair");
  const form = el("form", { class: "card review-form", onsubmit: async event => {
    event.preventDefault();
    const data = new FormData(form);
    submit.disabled = true;
    status.className = "save-status";
    status.textContent = "Saving…";
    try {
      session = await request("/api/review/answer", {
        pair_id: pair.id, choice: data.get("choice"), confidence: data.get("confidence"),
        notes: data.get("notes"), familiar: data.has("familiar"),
      });
      try { localStorage.removeItem(`${session.id}:${pair.id}`); } catch { /* Optional draft cache. */ }
      render();
      window.scrollTo(0, 0);
    } catch (error) {
      status.textContent = error.message;
      status.className = "save-status error";
    } finally { submit.disabled = false; }
  } },
  radios("choice", "Which version works better as this section of the paper?", choices),
  radios("confidence", "How confident are you?", { low: "Low", medium: "Medium", high: "High" }),
  el("label", { class: "field-label", for: "notes" }, "Your reasons (optional)"), notes,
  el("label", { class: "familiar" }, el("input", { type: "checkbox", name: "familiar" }), "I recognize this paper or passage from before this review."),
  el("div", { class: "actions" }, submit, status),
  el("p", { class: "muted small" }, "Saving locks this judgment. Authorship stays hidden until you finish all four pairs."));
  // Keep unfinished notes through a refresh; submitted judgments live on the local server.
  try {
    const draft = JSON.parse(localStorage.getItem(`${session.id}:${pair.id}`) || "null");
    if (draft) for (const control of form.elements) {
      if (control.type === "radio") control.checked = draft[control.name] === control.value;
      else if (control.name === "familiar") control.checked = Boolean(draft.familiar);
      else if (control.name === "notes") control.value = draft.notes || "";
    }
  } catch { /* A stored draft is optional. */ }
  form.addEventListener("input", () => {
    try {
      const data = new FormData(form);
      localStorage.setItem(`${session.id}:${pair.id}`, JSON.stringify({
        choice: data.get("choice"), confidence: data.get("confidence"), notes: data.get("notes"), familiar: data.has("familiar"),
      }));
    } catch { /* Server saves still work when browser storage is unavailable. */ }
  });
  app.replaceChildren(el("section", { class: "intro" },
    el("p", { class: "eyebrow" }, `Pair ${number} / 4 · ${pair.field} · Target: ${pair.target_words} words`),
    el("h1", {}, names[pair.section_type]),
    el("p", {}, "Read both versions and choose the one that best serves this section’s readers. Check the surrounding paper when useful. “Cannot judge” is a valid answer if the subject is outside your expertise."),
    el("p", { class: "small" }, "Four fixed training pairs, one per section type, sampled without using model scores. A/B order is shuffled. Your answers save on this computer.")),
    context(pair),
    el("div", { class: "pair-grid" }, ["A", "B"].map(side => el("article", { class: "card candidate", "aria-label": `Version ${side}` },
      el("h2", {}, `Version ${side}`), el("div", { class: "passage" }, pair[side])))), form);
}

function results() {
  const counts = { author: 0, model: 0, tie: 0, unsure: 0 };
  const rows = session.pairs.map(pair => {
    const answer = session.answers[pair.id];
    const winner = ["A", "B"].includes(answer.choice) ? (answer.choice === pair.author_side ? "author" : "model") : answer.choice;
    counts[winner]++;
    return el("tr", {}, el("td", {}, names[pair.section_type]), el("td", {}, choices[answer.choice]),
      el("td", {}, `Version ${pair.author_side}`), el("td", {}, answer.confidence));
  });
  app.replaceChildren(el("section", { class: "intro" }, el("p", { class: "eyebrow" }, "Blind review complete"),
    el("h1", {}, "Your judgments are saved."),
    el("p", {}, `You preferred the author in ${counts.author} pairs and the model in ${counts.model}. You marked ${counts.tie} as equal and ${counts.unsure} as cannot judge.`),
    el("p", {}, "This is a small spot-check. Now compare the extracted author text with the source to identify missing text, merged sections, or stray metadata.")),
    el("table", { class: "results" }, el("thead", {}, el("tr", {}, ["Section", "Your choice", "Author", "Confidence"].map(s => el("th", { scope: "col" }, s)))), el("tbody", {}, rows)),
    session.pairs.map(audit),
    el("div", { class: "actions" }, el("button", { onclick: download }, "Download review JSON"),
      el("span", { class: "muted small" }, "Saved locally in reports/blind-review-v2.json")));
}

function audit(pair) {
  const answer = session.answers[pair.id];
  const previous = session.extraction[pair.id];
  const notes = el("textarea", { name: "notes", maxlength: "10000", "aria-label": "Extraction notes", placeholder: "Describe missing text, extra material, or formatting damage. Leave the original judgment unchanged." });
  notes.value = previous?.notes || "";
  const status = el("span", { class: "save-status", role: "status" }, previous ? "Extraction check saved." : "");
  const button = el("button", { type: "submit" }, "Save extraction check");
  const form = el("form", { onsubmit: async event => {
    event.preventDefault();
    button.disabled = true;
    const data = new FormData(form);
    try {
      session = await request("/api/review/extraction", { pair_id: pair.id, status: data.get("status"), notes: data.get("notes") });
      status.className = "save-status";
      status.textContent = "Extraction check saved.";
    } catch (error) {
      status.className = "save-status error";
      status.textContent = error.message;
    } finally { button.disabled = false; }
  } }, radios("status", "Does the extracted author section match the source?", { clean: "Looks correct", problem: "Extraction problem", unsure: "Not sure" }, previous?.status),
  notes, el("div", { class: "actions" }, button, status));
  return el("details", { class: "audit" }, el("summary", {}, `${names[pair.section_type]} · ${pair.title}`),
    el("div", { class: "audit-content" },
      el("p", {}, `Author: version ${pair.author_side}. Your choice: ${choices[answer.choice]}. Previously familiar: ${answer.familiar ? "yes" : "no"}.`),
      el("p", { class: "saved-note muted" }, answer.notes || "No preference notes."),
      el("a", { href: pair.source_url, target: "_blank", rel: "noopener noreferrer" }, "Open the original paper ↗"),
      el("div", { class: "passage" }, pair[pair.author_side]),
      el("details", {}, el("summary", {}, "Review the model version"), el("div", { class: "passage" }, pair[pair.author_side === "A" ? "B" : "A"])),
      el("p", { class: "muted" }, "Check section boundaries, omitted paragraphs, and any added captions, acknowledgements, or publication notices. Citation formatting and whitespace were normalized during extraction."), form));
}

function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(session, null, 2)], { type: "application/json" }));
  const link = el("a", { href: url, download: `${session.id}.json` });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

request("/api/review").then(data => { session = data; render(); }).catch(error => {
  app.replaceChildren(el("h1", {}, "Could not load the review"), el("p", {}, error.message),
    el("button", { onclick: () => location.reload() }, "Try again"));
});

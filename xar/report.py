"""Write reports/ from the audited research run: results.md, checkpoints.csv and the gap figure."""

from __future__ import annotations

import statistics
from datetime import datetime
from pathlib import Path

import jsonschema

from xar.audit import audit_xar_run, check_manifest_matches_design
from xar.pipeline import WRITER_ATTEMPTS, length_window
from xar.stats import paired_improvement
from xar.util import ROLES, RunError, load_design, read_json, write_json, write_table

SCORE_RANGE = (0, 10)
RUN_OUTPUTS = ("checkpoints.csv", "gap_curves.png", "gap_curves.svg")
JOBS = {
    "writer": "wrote the missing sections",
    "rubric": "generated the rubrics",
    "optimizer": "rewrote the rubric prompt at each update",
    "judge": "graded the sections",
}


def render_report(runs_root, output):
    """Audit the research run and write every report file into output.

    All numbers in results.md come from the saved run, so the report can be regenerated at any
    time. If the run is missing or fails its audit, the run-specific files are removed.
    """
    design = load_design()
    runs_root, output = Path(runs_root), Path(output)
    run_dir = runs_root / design["research_run"]
    output.mkdir(parents=True, exist_ok=True)
    audit = {"source_run": str(run_dir), "state": "not_started", "error": None}
    comparison = {
        "blog_url": design["blog_url"],
        "blog_reported": design["reported_validation"],
        "reproduction": None,
    }
    run = None
    if (run_dir / "manifest.json").exists():
        try:
            run = audit_research_run(run_dir, design)
        except (RunError, FileNotFoundError, jsonschema.ValidationError, ValueError) as error:
            audit.update(state="failed", error=str(error))
        else:
            audit["state"] = "passed"
    if run is None:
        for name in RUN_OUTPUTS:
            (output / name).unlink(missing_ok=True)
        text = unavailable_text(audit)
    else:
        comparison["reproduction"] = summarize_trajectory(run, design["seed"])
        write_table(output / "checkpoints.csv", run["table"])
        plot_checkpoints(run["table"], design["reported_validation"], output / "gap_curves")
        text = results_text(run, comparison["reproduction"], design, run_dir, runs_root)
    write_json(output / "audit.json", audit)
    write_json(output / "blog_comparison.json", comparison)
    (output / "results.md").write_text(text)
    print(f"Report written to {output} (audit {audit['state']})")


def audit_research_run(run_dir, design):
    """Check the run is the research run experiments.yaml describes, then re-audit it from its saved files."""
    manifest = read_json(run_dir / "manifest.json")
    check_manifest_matches_design(manifest, design)
    split, seed = manifest["arguments"]["split"], manifest["arguments"]["seed"]
    if split != "research" or seed != design["seed"]:
        raise RunError(
            f"{run_dir} was run with split {split!r} and seed {seed}; "
            f"the report needs split 'research' and seed {design['seed']}"
        )
    return audit_xar_run(run_dir)


def summarize_trajectory(run, seed):
    """The validation numbers saved to blog_comparison.json."""
    table, selected = run["table"], run["freeze"]["selected"]
    initial, selected_row = table[0], table[selected]
    positives = [r["iteration"] for r in table if r["val_gap"] > 0]
    # Ties go to the earliest checkpoint, matching the selection rule.
    peak = max(table, key=lambda r: (r["val_gap"], -r["iteration"]))
    return {
        "initial_gap": initial["val_gap"],
        "selected_gap": selected_row["val_gap"],
        "selected_iteration": selected,
        "last_gap": table[-1]["val_gap"],
        "first_positive_iteration": positives[0] if positives else None,
        "validation_peak_gap": peak["val_gap"],
        "validation_peak_iteration": peak["iteration"],
        "initial_human": initial["val_human"],
        "selected_human": selected_row["val_human"],
        "initial_model": initial["val_model"],
        "selected_model": selected_row["val_model"],
        "paired_improvement": paired_improvement(
            run["rows"][(0, "validation")], run["rows"][(selected, "validation")], seed
        ),
        "reversed": initial["val_gap"] < 0 < selected_row["val_gap"],
    }


def plot_checkpoints(table, blog, path):
    """Plot the gaps and validation means per checkpoint with the blog's published values.

    Saves path.png and path.svg. The blog gives only its start, peak and end values, so those are
    drawn as separate markers rather than a curve.
    """
    import matplotlib

    matplotlib.use("Agg")  # draw to files; there is no display
    import matplotlib.pyplot as plt

    xs = [r["iteration"] for r in table]
    last = xs[-1]
    fig, (gaps, means) = plt.subplots(1, 2, figsize=(11, 4.5))
    gaps.plot(xs, [r["train_gap"] for r in table], "--o", color="tab:gray", label="This run: training")
    gaps.plot(xs, [r["val_gap"] for r in table], "-o", color="tab:green", label="This run: validation")
    gaps.scatter(
        [0, blog["peak_iteration"], last],
        [blog["initial_gap"], blog["peak_gap"], blog["final_human"] - blog["final_model"]],
        marker="D",
        facecolors="none",
        edgecolors="black",
        s=65,
        label=f"Meta blog: validation (P0, peak at P{blog['peak_iteration']}, P{last})",
    )
    gaps.axhline(0, color="black", linewidth=0.7)
    gaps.set_title("Author minus Muse score")
    for origin, who, color in (("human", "author", "tab:blue"), ("model", "Muse", "tab:orange")):
        means.plot(xs, [r[f"val_{origin}"] for r in table], "-o", color=color, label=f"This run: {who}")
        blog_scores = [blog[f"initial_{origin}"], blog[f"final_{origin}"]]
        means.scatter(
            [0, last], blog_scores, marker="x", color=color, s=65, label=f"Meta blog: {who} (P0, P{last})"
        )
    means.set_ylim(*SCORE_RANGE)
    means.set_title("Validation mean scores")
    for axis in (gaps, means):
        axis.set_xlabel("Prompt update")
        axis.set_xticks(xs)
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    fig.suptitle("Scores by prompt update: this run and the values Meta published", fontsize=11)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def unavailable_text(audit):
    if audit["state"] == "not_started":
        body = f"The research run {audit['source_run']} has not started, so there are no results yet."
    else:
        body = f"The research run {audit['source_run']} failed its audit, so no results are reported.\n\n"
        body += f"Audit error: {audit['error']}"
    return f"# Meta blog same-model reproduction\n\n{body}\n"


def results_text(run, summary, design, run_dir, runs_root):
    """The full results.md for an audited run."""
    manifest, table, candidates = run["manifest"], run["table"], run["candidates"]
    selected, last = summary["selected_iteration"], table[-1]["iteration"]
    blog = design["reported_validation"]
    names = {role: display_name(manifest["endpoints"][role]) for role in ROLES}
    papers = {
        split: len({r["paper_id"] for r in run["rows"][(0, split)]}) for split in ("train", "validation")
    }
    validation_sections = len(run["rows"][(0, "validation")])
    setup = (
        f"This run repeats the first rubric-learning experiment in [Meta's blog]({design['blog_url']}) "
        f"with the same models. {describe_roles(names)}. The run used {len(candidates)} sections from "
        f"{papers['train']} training and {papers['validation']} validation papers, with {last} "
        "prompt updates. Checkpoint P0 is the starting prompt, and Pn is the prompt after n updates."
    )
    metric = (
        "The gap is the author score minus the Muse score, both given by the same judge with the same "
        "rubric. A positive gap means the judge scored the author's section higher."
    )
    outcome = (
        f"{sign_sentence(table, summary)} Meta's blog reports {blog['initial_gap']:+.2f} at P0 and a "
        f"peak of {blog['peak_gap']:+.2f} at P{blog['peak_iteration']}, with the gap first positive at "
        f"P{blog['first_positive_iteration']}. {score_sentence(table, blog)}"
    )
    rounding = (
        "Gaps are computed from unrounded means, so a gap can differ by 0.01 from the difference of the "
        "rounded author and Muse columns."
    )
    selection = (
        f"P{selected} had the highest training gap, so it was selected. The selection was fixed before "
        f"any validation request was sent. {training_trend(table, selected)}"
    )
    figure = (
        "![Gaps and validation means by prompt update, with Meta's published values](gap_curves.png)\n\n"
        "The numbers for every checkpoint are in [checkpoints.csv](checkpoints.csv)."
    )
    costs = cost_sentence(
        read_json(run_dir / "costs.json"), run_dir, manifest, runs_root / design["pilot_run"]
    )
    details = (
        f"{format_sentence(read_json(run_dir / 'operational_summary.json'), run)} {costs}\n\n"
        "Before writing these numbers, the report re-checked the run against every saved response; "
        "the result is in [audit.json](audit.json). The blog's values and this run's summary are in "
        "[blog_comparison.json](blog_comparison.json)."
    )
    interval = ", so the bootstrap interval above is wide" if selected else ""
    limitations = [
        (
            "Meta did not publish its paper IDs, prompts or decoding settings. This run uses "
            f"{sum(papers.values())} recent arXiv papers and prompts written for this repository, so its "
            "numbers are not directly comparable with the blog's. The author sections were used as "
            "published; nobody checked their quality independently."
        ),
        (
            f"The validation split has only {papers['validation']} papers ({validation_sections} "
            f"sections){interval}."
        ),
        (
            "A positive gap would mean this judge, with these rubrics, prefers the author's text. It "
            "would not show that the author's text is better writing."
        ),
        length_limitation(run, selected),
        copy_limitation(candidates),
        serving_limitation(manifest),
    ]
    paragraphs = [
        "# Meta blog same-model reproduction",
        setup,
        metric,
        "## Result",
        outcome,
        checkpoint_table(table, selected),
        rounding,
        selection,
        improvement_sentence(summary["paired_improvement"], selected, papers["validation"]),
        figure,
        "## Run details",
        details,
        "## Limitations",
        "\n".join(f"- {item}" for item in limitations),
    ]
    return "\n\n".join(paragraphs) + "\n"


def display_name(endpoint):
    """'Muse Spark 1.1' from OpenRouter's 'Meta: Muse Spark 1.1'."""
    return endpoint["endpoint"]["model_name"].split(": ", 1)[-1]


def describe_roles(names):
    jobs = {}
    for role in ROLES:
        jobs.setdefault(names[role], []).append(JOBS[role])
    clauses = []
    for name, tasks in jobs.items():
        listed = tasks[0] if len(tasks) == 1 else ", ".join(tasks[:-1]) + " and " + tasks[-1]
        clauses.append(f"{name} {listed}")
    return "; ".join(clauses)


def sign_sentence(table, summary):
    initial, selected_gap = summary["initial_gap"], summary["selected_gap"]
    selected = summary["selected_iteration"]
    moved = f"The validation gap went from {initial:+.2f} at P0 to {selected_gap:+.2f} at P{selected}"
    if summary["reversed"]:
        return f"The sign flipped. {moved}, so the judge now scored the author's sections higher."
    if initial >= 0:
        return f"The validation gap was already {initial:+.2f} at P0, so there was no negative sign to flip. {moved}."
    if all(r["val_gap"] < 0 for r in table):
        return (
            f"The sign did not flip. {moved} and was negative at every checkpoint, so Muse kept "
            "scoring its own sections above the author's."
        )
    return (
        f"The sign did not flip at the selected checkpoint. {moved}, although the validation gap was "
        f"positive at P{summary['first_positive_iteration']}."
    )


def score_sentence(table, blog):
    """Compare how far the author and Muse means moved here and in the blog."""
    scores = [r[f"val_{origin}"] for r in table for origin in ("human", "model")]
    first, last = table[0], table[-1]
    return (
        f"In the blog the Muse score went from {blog['initial_model']:.1f} to {blog['final_model']:.1f} "
        f"and the author score from {blog['initial_human']:.1f} to {blog['final_human']:.1f}. Here both "
        f"validation means stayed between {min(scores):.2f} and {max(scores):.2f} at every checkpoint: the Muse "
        f"score went from {first['val_model']:.2f} at P0 to {last['val_model']:.2f} at "
        f"P{last['iteration']}, and the author score from {first['val_human']:.2f} to "
        f"{last['val_human']:.2f}."
    )


def checkpoint_table(table, selected):
    lines = [
        "| Checkpoint | Training gap | Validation gap | Validation author | Validation Muse |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in table:
        label = f"P{r['iteration']}" + (" (selected)" if r["iteration"] == selected else "")
        lines.append(
            f"| {label} | {r['train_gap']:+.2f} | {r['val_gap']:+.2f} | "
            f"{r['val_human']:.2f} | {r['val_model']:.2f} |"
        )
    return "\n".join(lines)


def training_trend(table, selected):
    """Say which way the training gap moved after the selected checkpoint."""
    last = table[-1]["iteration"]
    if selected == last:
        return f"The training gap was highest at the last checkpoint, P{last}."
    start, end = table[selected]["train_gap"], table[-1]["train_gap"]
    trend = f"After P{selected} the training gap went from {start:+.2f} to {end:+.2f} at P{last}"
    # The selected checkpoint has the highest training gap, so a negative one can only move away from zero.
    if start <= 0:
        return trend + ", further from zero: the later updates widened Muse's lead on the training papers."
    return trend + "."


def improvement_sentence(improvement, selected, validation_papers):
    if not selected:
        return "No prompt update beat P0 on the training papers, so there is no improvement to report."
    interval = improvement["interval"]
    verb = "improved" if improvement["mean"] > 0 else "changed"
    return (
        f"From P0 to P{selected} the validation gap {verb} by {improvement['mean']:+.2f} points on "
        f"average over {improvement['paired_coverage']} paired sections (95% bootstrap interval "
        f"{interval['low']:+.2f} to {interval['high']:+.2f}, {interval['replicates']:,} resamples of "
        f"the {validation_papers} validation papers)."
    )


def format_sentence(operations, run):
    """Say how often each structured-output role returned output that failed validation."""
    clean, sentences = [], []
    for role, counts in operations["format_validation"].items():
        invalid, total = counts["invalid_attempts"], counts["structured_attempts"]
        if invalid:
            sentences.append(
                f"The {role} returned invalid output {invalid} times in {total:,} responses; "
                "each was sent back with a repair request."
            )
        else:
            clean.append(f"{total:,} {role}")
    if clean:
        sentences.insert(0, f"All {' and '.join(clean)} responses were valid on the first attempt.")
    sections_graded = sum(len(rows) for rows in run["rows"].values())
    grades = 2 * sections_graded  # one for the author's section, one for Muse's
    sentences.append(f"All {grades:,} final grades passed validation.")
    return " ".join(sentences)


def cost_sentence(costs, run_dir, manifest, pilot):
    """Cost and time of the research run, and the pilot's cost."""
    minutes = wallclock_seconds(run_dir, manifest) / 60
    optimizer_calls, optimizer_seconds = optimizer_time(run_dir)
    sentence = (
        f"The research run made {costs['requests']:,} paid requests, cost "
        f"${costs['actual_complete_usd']:,.2f} and took {minutes:.0f} minutes with up to "
        f"{manifest['arguments']['concurrency']} requests in parallel. The {optimizer_calls} optimizer "
        f"requests ran one after another and took {optimizer_seconds / 60:.0f} of those minutes."
    )
    if (pilot / "costs.json").exists():
        pilot_cost = read_json(pilot / "costs.json")["actual_complete_usd"]
        sentence += f" The pilot run cost ${pilot_cost:,.2f}."
    return sentence


def wallclock_seconds(run_dir, manifest):
    """Time from run creation to the last saved response."""
    responses = (run_dir / "requests").glob("*/attempt_*.json")
    finished = datetime.fromisoformat(max(read_json(path)["timestamp"] for path in responses))
    return (finished - datetime.fromisoformat(manifest["created_at"])).total_seconds()


def optimizer_time(run_dir):
    """Number of optimizer requests and the seconds spent waiting for their responses."""
    calls, seconds = 0, 0.0
    for request in (run_dir / "requests").glob("*/request.json"):
        if read_json(request)["role"] == "optimizer":
            calls += 1
            seconds += sum(
                read_json(path)["duration_seconds"] for path in request.parent.glob("attempt_*.json")
            )
    return calls, seconds


def length_limitation(run, selected):
    """Describe the sections that missed their length target and the gap without them."""
    candidates = run["candidates"]
    missed = sum(not c["length_compliant"] for c in candidates.values())
    low, high = length_window(100)  # a 100-word target gives the window in percent
    target = f"{low:.0f}-{high:.0f}% of their target length"
    if not missed:
        return f"All {len(candidates)} generated sections came within {target}."
    # A section that misses the target has used every writer attempt.
    text = (
        f"{missed} of {len(candidates)} generated sections were still outside {target} after "
        f"{WRITER_ATTEMPTS - 1} rewrites each; they stay in the analysis."
    )
    before = [r["gap"] for r in run["rows"][(0, "validation")] if r["length_compliant"]]
    after = [r["gap"] for r in run["rows"][(selected, "validation")] if r["length_compliant"]]
    if not before or not selected:
        return text
    # Both lists hold the same sections, so the change in the mean is the mean paired improvement.
    start, end = statistics.mean(before), statistics.mean(after)
    return text + (
        f" On the {len(after)} validation sections that met the target, the gap went from "
        f"{start:+.2f} at P0 to {end:+.2f} at P{selected}, an improvement of {end - start:+.2f}."
    )


def copy_limitation(candidates):
    flagged = sum(c["contamination"]["flagged"] for c in candidates.values())
    found = f"{flagged} of the {len(candidates)}" if flagged else "none of them"
    return (
        f"The generated sections were checked for text copied from the author's; the check flagged "
        f"{found}. It compares wording only, so it cannot tell whether Muse saw the papers in training."
    )


def serving_limitation(manifest):
    """State each model's provider and weight precision, as OpenRouter reported them."""
    notes = {}
    for role in ROLES:
        endpoint = manifest["endpoints"][role]["endpoint"]
        name, provider = display_name(manifest["endpoints"][role]), endpoint["provider_name"]
        if endpoint["quantization"] == "unknown":
            notes[name] = f"OpenRouter does not report the precision {provider} serves {name} at"
        else:
            notes[name] = f"{name} was served by {provider} at {endpoint['quantization']} precision"
    return (
        "; ".join(notes.values())
        + ". The blog does not say how its models were served, so these may not be the same model "
        "weights Meta ran."
    )

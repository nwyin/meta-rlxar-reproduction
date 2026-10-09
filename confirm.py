"""Evaluate three frozen training-selected prompts on the untouched confirmation split.

Paid command, with no optimization or prompt selection on confirmation scores:
uv run python confirm.py DOMAIN OUTPUT SOURCE_RUN_1 SOURCE_RUN_2 SOURCE_RUN_3 [--resume]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
from pathlib import Path

from replicate import interval
from run import (
    ROOT,
    SCORE_EPSILON,
    OpenRouter,
    RunError,
    best_checkpoint,
    digest,
    evaluate_checkpoint,
    summarize,
    write_json,
    writer_candidates,
)


def load_frozen(sources):
    records, prompts, settings, snapshots = [], [], None, None
    ignored = {"output", "seed", "sample_seed", "reuse_candidates", "validation_policy"}
    for source in sources:
        source = source.resolve()
        manifest = json.loads((source / "manifest.json").read_text())
        config = manifest["config"]
        method = {k: v for k, v in config.items() if k not in ignored}
        if settings is not None and method != settings:
            raise RunError("Confirmation sources must use the same frozen method")
        settings = method
        if json.loads((source / "status.json").read_text())["state"] != "complete":
            raise RunError("Confirmation requires completed source runs")
        frozen = json.loads((source / "freeze.json").read_text())
        selected = json.loads((source / "summary.json").read_text())["selected_iteration"]
        if selected != frozen["selected"] or selected != best_checkpoint(frozen["training_gaps"]):
            raise RunError("Source selection differs from its training freeze")
        initial = (source / "prompts/iter_00.md").read_text()
        learned = (source / f"prompts/iter_{selected:02d}.md").read_text()
        if digest(initial) != frozen["prompt_hashes"][0] or digest(learned) != frozen["prompt_hashes"][selected]:
            raise RunError("Source prompt changed after training selection")
        current = {name: (source / "input_prompts" / f"{name}.md").read_text()
                   for name in ("writer", "rubric_wrapper", "judge")}
        if snapshots is not None and (current != snapshots or initial != prompts[0]):
            raise RunError("Source writer, judge, wrapper, or initial prompt differs")
        snapshots = current
        if not prompts:
            prompts.append(initial)
        prompts.append(learned)
        records.append({"run": str(source), "selected": selected, "prompt_hash": digest(learned),
                        "dataset_hash": manifest["dataset_hash"], "training_freeze": frozen})
    return config, snapshots, prompts, records


def assess_confirmation(rows, domain):
    if len(rows) != 4:
        raise RunError("Confirmation requires an initial checkpoint and three frozen prompts")
    expected = {r["example_id"] for r in rows[0]}
    if any({r["example_id"] for r in values} != expected or len(values) != len(expected) for values in rows):
        raise RunError("Confirmation checkpoints must cover identical unique examples")
    summaries = [summarize(values) for values in rows]
    baseline = {r["example_id"]: r for r in rows[0]}
    gaps, gains = {}, {}
    for values in rows[1:]:
        for row in values:
            before = baseline[row["example_id"]]["gap"]
            if row["gap"] is not None and before is not None:
                gaps.setdefault(row["source_id"], []).append(row["gap"])
                gains.setdefault(row["source_id"], []).append(row["gap"] - before)
    source_gaps = [statistics.mean(v) for v in gaps.values()]
    source_gains = [statistics.mean(v) for v in gains.values()]
    gap_ci, gain_ci = interval(source_gaps), interval(source_gains)
    complete = all(s["complete"] and s["examples"] > 0 and s["length_compliant"] == s["examples"] for s in summaries)
    improvements = [s["gap"] - summaries[0]["gap"] if s["gap"] is not None and summaries[0]["gap"] is not None else None
                    for s in summaries[1:]]
    passed = (complete and all(s["gap"] > SCORE_EPSILON for s in summaries[1:])
              and all(v > SCORE_EPSILON for v in improvements)
              and gap_ci is not None and gain_ci is not None and gap_ci[0] > SCORE_EPSILON and gain_ci[0] > SCORE_EPSILON
              and (domain != "arxiv" or summaries[0]["gap"] < -SCORE_EPSILON))
    return {"domain": domain, "complete": complete, "passed": bool(passed), "summaries": summaries,
            "improvements": improvements, "sources": len(gaps),
            "source_mean_gap": statistics.mean(source_gaps) if source_gaps else None,
            "source_mean_improvement": statistics.mean(source_gains) if source_gains else None,
            "gap_95pct_source_bootstrap": gap_ci, "improvement_95pct_source_bootstrap": gain_ci,
            "limitation": "One frozen confirmation evaluation. Source-bootstrap intervals are descriptive with few sources. "
                          "Rubric preference is not an independent expert judgment of writing quality."}


def confirm(domain, output, sources, *, resume=False):
    if len(sources) != 3 or len({p.resolve() for p in sources}) != 3:
        raise RunError("Provide three distinct source runs from the frozen batch")
    config, snapshots, prompts, records = load_frozen(sources)
    dataset = Path(config["dataset"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if any(r["dataset_hash"] != dataset_hash for r in records):
        raise RunError("Dataset changed since the source runs")
    all_examples = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    examples = [e for e in all_examples if e["split"] == "confirmation"]
    held_sources = {e["source_id"] for e in examples}
    seen_sources = {e["source_id"] for e in all_examples if e["split"] != "confirmation"}
    if not examples or held_sources & seen_sources or len({e["example_id"] for e in examples}) != len(examples):
        raise RunError("Confirmation must be nonempty, unique, and source-disjoint")
    output = output.resolve()
    config = {**config, "roles": {r: config["roles"][r] for r in ("writer", "rubric", "judge")},
              "dataset": str(dataset), "output": str(output), "prompts": str(output / "input_prompts"),
              "sample_fraction": 1.0, "split": "confirmation", "iterations": 0, "seed": 0,
              "contrastive_feedback": False, "strict_length": True}
    config.pop("reuse_candidates", None)
    manifest = {"domain": domain, "config": config, "sources": records, "dataset_hash": dataset_hash,
                "example_ids": [e["example_id"] for e in examples], "prompt_hashes": [digest(p) for p in prompts],
                "input_prompt_hashes": {k: digest(v) for k, v in snapshots.items()},
                "criterion": "All three frozen prompts have positive gaps and improvements; pooled source-bootstrap lower bounds "
                             "are positive; all pairs complete and length compliant; paper baseline negative. No winner selection."}
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise RunError("Confirmation resume inputs changed")
        if json.loads((output / "status.json").read_text())["state"] not in ("failed", "incomplete"):
            raise RunError("Resume requires a failed or incomplete confirmation")
    lock = output / "active.lock"
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    api = None
    status = {"state": "failed"}
    try:
        write_json(output / "manifest.json", manifest)
        write_json(output / "execution.json", {"started_at": dt.datetime.now(dt.UTC).isoformat(),
                   "code_hash": digest(Path(__file__).read_text()), "resume": resume})
        write_json(output / "status.json", {"state": "running", "pid": os.getpid()})
        (output / "input_prompts").mkdir(exist_ok=True)
        for name, value in snapshots.items():
            (output / "input_prompts" / f"{name}.md").write_text(value)
        api = OpenRouter(output, config)
        api.resume = resume
        candidates = writer_candidates(api, examples, output)
        rows, cache, checkpoint_indices = [], {}, []
        for index, prompt in enumerate(prompts):
            key = digest(prompt)
            if key not in cache:
                cache[key] = (index, evaluate_checkpoint(api, examples, candidates, prompt, index, output))
            checkpoint, values = cache[key]
            checkpoint_indices.append(checkpoint)
            rows.append(values)
        result = {**assess_confirmation(rows, domain), "evaluation_checkpoints": checkpoint_indices, "costs": api.costs()}
        write_json(output / "summary.json", result)
        status = {"state": "complete" if result["complete"] else "incomplete", "passed": result["passed"]}
        return result
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "status.json", status)
        lock.unlink()
        if api is not None:
            api.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("arxiv", "fiction"))
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", nargs=3, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = confirm(args.domain, args.output, args.sources, resume=args.resume)
    print(json.dumps(result, indent=2))

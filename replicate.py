"""Freeze training-selected methods, then run three fresh reduced replications per domain.

Usage: uv run python replicate.py OUTPUT_ROOT ARXIV_CONFIG FICTION_CONFIG
This is a paid command. It never selects a method using validation results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from run import SCORE_EPSILON, digest, write_json


def interval(values):
    """Percentile bootstrap over independent source-level means, not individual grades."""
    if len(values) < 2:
        return None
    rng = random.Random(20261007)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def assess(paths, domain):
    trials, source_gaps, source_deltas = [], {}, {}
    for path in paths:
        result = json.loads((path / "summary.json").read_text())
        selected = result["selected_iteration"]
        baseline = result["summaries"][0]["validation"]
        final = result["summaries"][selected]["validation"]
        improvement = final["gap"] - baseline["gap"] if final["gap"] is not None and baseline["gap"] is not None else None
        trials.append({"run": str(path), "selected": selected, "baseline_gap": baseline["gap"],
                       "selected_gap": final["gap"], "improvement": improvement,
                       "length_compliant": final["length_compliant"] == final["examples"],
                       "complete": baseline["complete"] and final["complete"]})
        initial_rows = json.loads((path / "scores/main/0/validation_rows.json").read_text())
        final_rows = json.loads((path / f"scores/main/{selected}/validation_rows.json").read_text())
        initial = {row["example_id"]: row for row in initial_rows}
        for row in final_rows:
            before = initial[row["example_id"]]
            if row["gap"] is None or before["gap"] is None:
                continue
            source_gaps.setdefault(row["source_id"], []).append(row["gap"])
            source_deltas.setdefault(row["source_id"], []).append(row["gap"] - before["gap"])
    gaps = [statistics.mean(v) for v in source_gaps.values()]
    deltas = [statistics.mean(v) for v in source_deltas.values()]
    gap_ci, delta_ci = interval(gaps), interval(deltas)
    passed = (len(trials) == 3 and all(t["complete"] and t["length_compliant"] and t["selected_gap"] > SCORE_EPSILON
                                     and t["improvement"] > SCORE_EPSILON for t in trials)
              and gap_ci is not None and gap_ci[0] > SCORE_EPSILON and delta_ci is not None and delta_ci[0] > SCORE_EPSILON)
    reversal = passed and all(t["baseline_gap"] < -SCORE_EPSILON for t in trials)
    return {"domain": domain, "trials": trials, "sources": len(gaps), "source_mean_gap": statistics.mean(gaps) if gaps else None,
            "source_mean_improvement": statistics.mean(deltas) if deltas else None,
            "gap_95pct_source_bootstrap": gap_ci, "improvement_95pct_source_bootstrap": delta_ci,
            "consistent_preference_and_improvement": bool(passed), "consistent_reversal": bool(reversal),
            "score_tie_tolerance": SCORE_EPSILON,
            "limitation": "Small reduced samples; percentile intervals are descriptive, especially with few sources. "
                          "This checks the learned scoring preference, not independent expert judgments of quality."}


def replicate(root, configurations):
    root.mkdir(parents=True, exist_ok=False)
    jobs, frozen = [], {}
    for domain, config_path in zip(("arxiv", "fiction"), configurations, strict=True):
        config = yaml.safe_load(Path(config_path).read_text())
        frozen[domain] = {"source_config": str(config_path), "config": config}
        for repeat in range(3):
            cfg = dict(config)
            cfg.pop("reuse_candidates", None)
            cfg.update(train_only=False, sample_seed=20261008 + repeat, seed=repeat,
                       output=str(root / f"{domain}-repeat{repeat}"))
            path = root / f"{domain}-repeat{repeat}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append((domain, path, Path(cfg["output"])))
    write_json(root / "freeze.json", {"methods": frozen, "config_hash": digest(frozen),
               "selection": "Training results only, before fresh validation calls",
               "criterion": "All 3 trials have positive selected validation gap and positive improvement; "
                            "source-cluster bootstrap lower bounds exceed zero for both. "
                            "Paper reversal additionally requires all baseline gaps to be negative."})

    def launch(job):
        _domain, config_path, output = job
        with output.with_suffix(".log").open("w") as log:
            command = [sys.executable, "-u", "run.py", str(config_path)]
            for attempt in range(3):
                result = subprocess.run(command + (["--resume"] if attempt else []), stdout=log, stderr=subprocess.STDOUT, check=False)
                status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
                if status.get("state") != "incomplete" and "Training grades missing" not in status.get("error", ""):
                    break
        write_json(output / "process_exit.json", {"returncode": result.returncode})
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(launch, jobs))
    if any(codes):
        write_json(root / "status.json", {"state": "failed", "exit_codes": codes})
        return
    results = [assess([out for dom, _, out in jobs if dom == domain], domain) for domain in ("arxiv", "fiction")]
    write_json(root / "assessment.json", results)
    success = results[0]["consistent_reversal"] and results[1]["consistent_preference_and_improvement"]
    write_json(root / "status.json", {"state": "complete", "success": success})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("arxiv_config", type=Path)
    parser.add_argument("fiction_config", type=Path)
    args = parser.parse_args()
    replicate(args.output, [args.arxiv_config, args.fiction_config])

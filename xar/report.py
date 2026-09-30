"""Render the blog-comparison report and checkpoint plots from the audited research run."""

from __future__ import annotations

from pathlib import Path

import jsonschema
import yaml

from xar.audit import audit_xar_run, validate_primary_manifest
from xar.stats import paired_improvement
from xar.util import ROOT, RunError, read_json, write_json, write_table


def render_report(runs_root, output):
    """Report only the declared same-model trajectory; old pilots are historical evidence."""
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    if design.get("scope") != "meta_blog_initial_empirical_investigation":
        raise RunError("Reporting requires the active blog reproduction design")
    output, source = Path(output), Path(runs_root) / design["research_run"]
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "scope": design["scope"],
        "expected_trajectories": 1,
        "raw_verified_trajectories": 0,
        "complete": False,
        "source_run": str(source),
        "failures": [],
    }
    lines = [
        "# Meta blog same-model reproduction",
        "",
        "Muse Spark 1.1: writer, rubric generator, judge. Kimi K2.6: optimizer.",
        "",
        "One trajectory; 52 sections from 8 training and 5 validation papers; seven updates.",
        "",
        "The paper IDs, original prompts, and decoding settings were not published in the blog.",
        "This reconstruction uses frozen arXiv author sections and declared prompts/settings.",
        "A reversal measures optimized judging preferences, not objective writing quality.",
        "",
    ]
    comparison = {
        "source_url": design["source"],
        "blog_reported": design["reported_validation"],
        "reproduction": None,
        "exact_source_data_and_prompts_available": False,
    }
    if not (source / "manifest.json").exists():
        audit["state"] = "not_started"
        lines += ["Research has not started. Historical alternative-model pilots are excluded.", ""]
    else:
        manifest = read_json(source / "manifest.json")
        try:
            validate_primary_manifest(manifest, design)
            if (
                manifest["arguments"]["split"] != "research"
                or manifest["arguments"]["seed"] != design["seed"]
            ):
                raise RunError("Declared research split and trajectory required")
            run = audit_xar_run(source)
        except (RunError, FileNotFoundError, jsonschema.ValidationError, ValueError) as error:
            audit.update(state="incomplete", failures=[str(error)])
            lines += [
                "Research is incomplete; no verified comparison is available.",
                "",
                f"Audit: {error}",
                "",
            ]
        else:
            table, selected = run["table"], run["freeze"]["selected"]
            initial, final = table[0], table[selected]
            positives = [r["iteration"] for r in table if r["val_gap"] > 0]
            peak = max(table, key=lambda r: (r["val_gap"], -r["iteration"]))
            comparison["reproduction"] = {
                "initial_gap": initial["val_gap"],
                "selected_gap": final["val_gap"],
                "selected_iteration": selected,
                "terminal_gap": table[-1]["val_gap"],
                "first_positive_iteration": positives[0] if positives else None,
                "descriptive_validation_peak_gap": peak["val_gap"],
                "descriptive_validation_peak_iteration": peak["iteration"],
                "validation_peak_used_for_selection": False,
                "initial_human": initial["val_human"],
                "selected_human": final["val_human"],
                "initial_model": initial["val_model"],
                "selected_model": final["val_model"],
                "paired_improvement": paired_improvement(
                    run["rows"][(0, "validation")], run["rows"][(selected, "validation")], design["seed"]
                ),
                "descriptive_reversal": initial["val_gap"] < 0 < final["val_gap"],
            }
            audit.update(state="raw_verified", complete=True, raw_verified_trajectories=1)
            write_table(output / "checkpoints.csv", table)
            lines += [
                f"Raw-verified trajectory: 1/1. Training selected checkpoint {selected}.",
                "",
                f"Validation gap: {initial['val_gap']:.3f} → {final['val_gap']:.3f}.",
                "",
                "Blog reports −4.2 → +2.76; validation crossing at 4 and peak at 5.",
                "",
                "The entire checkpoint curve is reported; the validation maximum is descriptive only.",
                "",
            ]
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            xs = [r["iteration"] for r in table]
            axes[0].plot(xs, [r["train_gap"] for r in table], "--o", label="Train")
            axes[0].plot(xs, [r["val_gap"] for r in table], "-o", label="Validation")
            axes[0].axhline(0, color="black", linewidth=0.7)
            axes[0].set_title("Human minus model score")
            axes[1].plot(xs, [r["val_human"] for r in table], "-o", label="Author")
            axes[1].plot(xs, [r["val_model"] for r in table], "-o", label="Muse Spark 1.1")
            axes[1].set_title("Validation mean scores")
            for axis in axes:
                axis.set_xlabel("Prompt update")
                axis.legend()
            fig.tight_layout()
            fig.savefig(output / "gap_curves.png", dpi=160)
            fig.savefig(output / "gap_curves.svg")
            plt.close(fig)
    if not audit["complete"]:
        for name in ("checkpoints.csv", "gap_curves.png", "gap_curves.svg"):
            (output / name).unlink(missing_ok=True)
    write_json(output / "audit.json", audit)
    write_json(output / "blog_comparison.json", comparison)
    (output / "results.md").write_text("\n".join(lines))
    print(f"Report saved: {output}; blog trajectory verified {audit['raw_verified_trajectories']}/1")

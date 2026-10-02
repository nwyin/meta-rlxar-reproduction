"""Post hoc probe: rubrics generated with the expert's section visible (the blog's GT variant).

Reuses the repo's client, prompts and schemas. Writes to runs/probe-gt-rubrics only; never touches
the main runs. Compares the GT-rubric grades with the main run's P3 grades on the same sections.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xar.data import task_data
from xar.openrouter import OpenRouter, role_config
from xar.pipeline import RUBRIC_SCHEMA, grade_candidate, validate_rubric
from xar.util import bounded_map, digest, prompt, read_json, write_json

MAIN = ROOT / "runs/meta-blog-v2-seed0"
OUT = ROOT / "runs/probe-gt-rubrics"
PAPERS = ("1605.05804", "1604.04494")
CHECKPOINT = 3

GT_NOTE = (
    "\n\nFor this request only, the finished section the paper's authors published is attached as "
    "`hidden_section`. Use it to decide what this section must do for this paper and what an "
    "expert chose to include and leave out. The rubric will be used to grade other candidates for "
    "the same section, so write criteria and anchors about the section's qualities, not about "
    "matching that text word for word."
)

examples = [json.loads(line) for line in (ROOT / "data/examples.jsonl").open()]
examples = [e for e in examples if e["paper_id"] in PAPERS]
candidates = read_json(MAIN / "generations/candidates.json")["candidates"]
meta_prompt = (MAIN / f"prompts/iter_{CHECKPOINT:02d}.md").read_text()
main_rows = {r["example_id"]: r for r in read_json(MAIN / f"scores/main/{CHECKPOINT}/validation_rows.json")}

roles = {role: role_config(role) for role in ("rubric", "judge")}
api = OpenRouter(OUT, roles, seed=0)


def run_example(example):
    eid = example["example_id"]
    result = api.structured(
        "rubric",
        prompt("rubric_wrapper") + GT_NOTE,
        {**task_data(example), "meta_prompt": meta_prompt, "hidden_section": example["reference"]},
        RUBRIC_SCHEMA,
        {"gt_rubric": eid, "checkpoint": CHECKPOINT},
        validate_rubric,
    )
    rubric = {"example_id": eid, "meta_prompt_hash": digest(meta_prompt), **result}
    rubric["rubric_hash"] = digest(result["value"]) if result["value"] else None
    write_json(OUT / "rubrics" / eid / "rubric.json", rubric)
    origins = ["human", "model"]
    random.Random(int(digest({"seed": 0, "label": f"gt/{eid}"})[:16], 16)).shuffle(origins)
    totals = {}
    for slot, origin in enumerate(origins):
        text = example["reference"] if origin == "human" else candidates[eid]["text"]
        grade = grade_candidate(
            api,
            example,
            rubric,
            text,
            {"gt_grade": eid, "slot": slot},
            OUT / "scores" / eid / f"{origin}.json",
        )
        totals[origin] = grade["total"]
    return {
        "example_id": eid,
        "paper_id": example["paper_id"],
        "section_type": example["section_type"],
        **totals,
    }


rows = bounded_map(run_example, examples, 7, api.dispatch_stopped)
write_json(OUT / "rows.json", rows)
print(
    f"{'paper':12s} {'section':13s} {'P3 human':>9s} {'P3 model':>9s} {'P3 gap':>7s} | {'GT human':>9s} {'GT model':>9s} {'GT gap':>7s}"
)
for r in rows:
    m = main_rows[r["example_id"]]
    gt_gap = r["human"] - r["model"] if r["human"] is not None and r["model"] is not None else None
    print(
        f"{r['paper_id']:12s} {r['section_type']:13s} {m['human']:9.2f} {m['model']:9.2f} {m['gap']:7.2f} | "
        f"{r['human']:9.2f} {r['model']:9.2f} {gt_gap:7.2f}"
    )
ok = [r for r in rows if r["human"] is not None and r["model"] is not None]
print(
    f"mean P3 gap {sum(main_rows[r['example_id']]['gap'] for r in ok) / len(ok):.2f} | "
    f"mean GT gap {sum(r['human'] - r['model'] for r in ok) / len(ok):.2f} over {len(ok)} sections"
)
print("costs", api.costs())

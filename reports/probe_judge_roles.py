"""Check preregistered judges against an exact saved training-pilot rubric."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import (
    RUBRIC_SCHEMA,
    ContractError,
    audit_saved_output,
    common_parser,
    digest,
    evaluate_checkpoint,
    file_hash,
    initialize_run,
    main_guard,
    operational_summary,
    parse_args,
    read_json,
    role_config,
    run_lock,
    selected_examples,
    task_data,
    validate_rubric,
    write_json,
)


def main():
    parser = common_parser(__doc__, ["judge"])
    parser.set_defaults(split="pilot")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--evaluation-judge-models", nargs="+", default=["z-ai/glm-5.2", "qwen/qwen3.5-9b"])
    parser.add_argument("--evaluation-provider-map", default="{}")
    args = parse_args(parser)
    source = Path(args.source_run)
    manifest = read_json(source / "manifest.json")
    if args.split != "pilot" or manifest["arguments"]["split"] != "pilot":
        raise ContractError("Judge capability checks use only separate pilot papers")
    for name in ("dataset", "splits"):
        if file_hash(getattr(args, name)) != manifest[name + "_hash"]:
            raise ContractError("Judge probe inputs differ from frozen pilot source")
    examples = selected_examples(args)
    paper = min(e["paper_id"] for e in examples)
    example = min((e for e in examples if e["paper_id"] == paper), key=lambda e: e["example_id"])
    example = {**example, "split": "train"}
    eid = example["example_id"]
    candidates = read_json(source / "generations/candidates.json")["candidates"]
    candidate = candidates[eid]
    if candidate["context_hash"] != example["context_hash"] or candidate["text_hash"] != digest(candidate["text"]):
        raise ContractError("Judge probe writer artifact differs")
    rubric = read_json(source / f"rubrics/main/0/train/{eid}/rubric.json")
    payload = audit_saved_output(rubric, RUBRIC_SCHEMA, manifest["roles"]["rubric"], manifest["endpoints"]["rubric"],
                                 manifest["software_hashes"]["prompts/rubric_wrapper.md"])
    validate_rubric(rubric["value"])
    initial = (source / "prompts/iter_00.md").read_text()
    if (rubric["context_hash"] != example["context_hash"] or rubric["rubric_hash"] != digest(rubric["value"])
            or json.loads(payload["messages"][1]["content"]) != {**task_data(example), "meta_prompt": initial}):
        raise ContractError("Judge probe rubric/input identity differs")
    providers = json.loads(args.evaluation_provider_map)
    roles = {"evaluation_" + str(i): role_config(args, "judge", model=model, provider=providers.get(model))
             for i, model in enumerate(args.evaluation_judge_models)}
    with run_lock(args.output_dir):
        api = initialize_run(args, roles, "judge_capability_probe", {
            "source_hash": manifest["substantive_hash"], "rubric_hash": rubric["rubric_hash"],
            "probe_script_hash": file_hash(__file__), "gate_uses_score_gap": False,
            "example_selection": "first_sorted_pilot_training_example", "example_id": eid})
        coverage = {}
        for role in roles:
            rows = evaluate_checkpoint(api, [example], candidates, None, 0, args.output_dir, args.concurrency,
                                       frozen_rubrics={eid: rubric}, judge_role=role, namespace=role)
            coverage[roles[role]["model"]] = {"human_valid": rows[0]["human"] is not None,
                                             "model_valid": rows[0]["model"] is not None}
        complete = all(all(v.values()) for v in coverage.values())
        write_json(Path(args.output_dir) / "status.json", {"state": "complete" if complete else "format_incomplete",
                   "coverage": coverage, "gate_uses_score_gap": False, "rubric_regeneration": False, "paid": True})
        operational_summary(args.output_dir)
        write_json(Path(args.output_dir) / "costs.json", api.ledger.summary())
        print("Judge capability check complete; formats_valid:", complete, flush=True)


if __name__ == "__main__":
    main_guard(main)

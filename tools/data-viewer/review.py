"""A fixed, blind review of four training pairs, saved separately from experiment evidence."""

import hashlib
import json
import random
from datetime import UTC, datetime
from threading import Lock

RUN = "meta-blog-v2-seed0"
SEED = 20261003
SESSION = "blind-review-v2"
LOCK = Lock()


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def timestamp():
    return datetime.now(UTC).isoformat()


def save(root, session):
    path = root / "reports" / f"{SESSION}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def create(root):
    run = root / "runs" / RUN
    manifest = read(run / "manifest.json")
    dataset = root / "data/examples.jsonl"
    splits = root / "data/splits.json"
    if file_hash(dataset) != manifest["dataset_hash"] or file_hash(splits) != manifest["splits_hash"]:
        raise ValueError("The current dataset does not match the saved v2 run.")
    training = set(read(splits)["papers"]["train"])
    examples = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    examples = [e for e in examples if e["paper_id"] in training and e["split"] == "train"]
    rng = random.Random(SEED)
    selected, used = [], set()
    # Pick the rare section first. Sampling never reads rubrics, grades, or feedback.
    for kind in ("related_work", "abstract", "conclusion", "introduction"):
        pool = sorted(
            (e for e in examples if e["section_type"] == kind and e["paper_id"] not in used),
            key=lambda e: e["example_id"],
        )
        if not pool:
            raise ValueError(f"No unused training paper has a {kind} section.")
        example = rng.choice(pool)
        selected.append(example)
        used.add(example["paper_id"])
    selected.sort(key=lambda e: ("abstract", "conclusion", "related_work", "introduction").index(
        e["section_type"]
    ))
    sides = ["A", "A", "B", "B"]
    rng.shuffle(sides)
    pairs = []
    for index, (example, author_side) in enumerate(zip(selected, sides), 1):
        generation = read(run / "generations" / f"{example['example_id']}.json")
        if generation["context_hash"] != example["context_hash"] or not generation["complete"]:
            raise ValueError("A selected draft is incomplete or has a different paper context.")
        reference, model = example["reference"], generation["text"]
        pairs.append({
            "id": f"pair-{index}", "example_id": example["example_id"],
            "paper_id": example["paper_id"], "section_type": example["section_type"],
            "field": example["provenance"].get("primary_category", ""),
            "target_words": example["target_words"], "context": example["context"],
            "A": reference if author_side == "A" else model,
            "B": model if author_side == "A" else reference,
            "author_side": author_side, "title": example["provenance"]["title"],
            "source_url": example["provenance"]["source_url"],
            "reference_sha256": text_hash(reference), "model_sha256": text_hash(model),
        })
    session = {
        "id": SESSION, "created_at": timestamp(), "run": RUN, "seed": SEED,
        "sampling": "One section per type from distinct training papers; independent of scores.",
        "dataset_hash": manifest["dataset_hash"], "splits_hash": manifest["splits_hash"],
        "pairs": pairs, "answers": {}, "extraction": {},
    }
    save(root, session)
    return session


def load(root):
    path = root / "reports" / f"{SESSION}.json"
    return read(path) if path.exists() else create(root)


def public(session):
    complete = len(session["answers"]) == len(session["pairs"])
    fields = ("id", "section_type", "field", "target_words", "context", "A", "B")
    if complete:
        fields += ("author_side", "title", "source_url", "example_id")
    return {
        "id": session["id"], "complete": complete,
        "pairs": [{key: pair[key] for key in fields} for pair in session["pairs"]],
        "answers": session["answers"], "extraction": session["extraction"],
    }


def get_session(root):
    with LOCK:
        return public(load(root))


def record(root, payload, *, extraction=False):
    if not isinstance(payload, dict):
        raise TypeError("Expected a review answer.")
    with LOCK:
        session = load(root)
        pair_id = payload.get("pair_id")
        if pair_id not in {pair["id"] for pair in session["pairs"]}:
            raise ValueError("Unknown pair.")
        notes = payload.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 10000:
            raise ValueError("Notes must contain at most 10,000 characters.")
        if extraction:
            if len(session["answers"]) != len(session["pairs"]):
                raise ValueError("Finish the blind judgments before checking the source.")
            if payload.get("status") not in ("clean", "problem", "unsure"):
                raise ValueError("Choose an extraction status.")
            session["extraction"][pair_id] = {
                "status": payload["status"], "notes": notes, "saved_at": timestamp(),
            }
        else:
            choice, confidence = payload.get("choice"), payload.get("confidence")
            familiar = payload.get("familiar", False)
            if choice not in ("A", "B", "tie", "unsure"):
                raise ValueError("Choose A, B, a tie, or cannot judge.")
            if confidence not in ("low", "medium", "high") or not isinstance(familiar, bool):
                raise ValueError("Choose a confidence level and valid familiarity flag.")
            answer = {"choice": choice, "confidence": confidence, "notes": notes, "familiar": familiar}
            previous = session["answers"].get(pair_id)
            if previous:
                if any(previous[key] != value for key, value in answer.items()):
                    raise ValueError("This blind judgment is already saved and locked.")
                return public(session)  # Retrying a successful save is harmless.
            session["answers"][pair_id] = {**answer, "saved_at": timestamp()}
        save(root, session)
        return public(session)

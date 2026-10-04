"""Run a fresh Meta XAR reproduction with OpenRouter.

Prepare data with fetch_data.py, then choose a new --output directory.
Settings and model roles come from configs/; prompts stay in prompts/. No inputs are truncated.
"""

from __future__ import annotations; import argparse, collections, concurrent.futures, csv, datetime as dt, hashlib, json, math, os, random, re, statistics, threading, time; from dataclasses import asdict, dataclass; from pathlib import Path; import httpx, jsonschema, yaml; from dotenv import load_dotenv  # noqa: I001  # fmt: skip

ROOT = Path(__file__).resolve().parent
ROLES = ("writer", "rubric", "optimizer", "judge")
SECTIONS = ("abstract", "introduction", "related_work", "conclusion")
FAILING_FEEDBACK = "failing_gap_without_paper"
MAX_CONCURRENCY = 32
SCHEMA_VERSION = 1


class RunError(RuntimeError):
    """The run cannot continue safely."""


class InvalidOutput(RunError):
    """A model reply failed validation."""


class Skipped(RunError):
    """Another parallel task failed before this task started."""


def canonical(value):
    """Deterministic JSON (sorted keys, no whitespace) used for hashing and request bodies."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    """SHA-256 hex of a string, or of any other value's canonical JSON."""
    return hashlib.sha256((value if isinstance(value, str) else canonical(value)).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write a complete JSON record atomically, including from parallel tasks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bounded_map(function, items, concurrency, stopped):
    """Run function over items on up to `concurrency` threads; return the results in input order.

    A failure sets `stopped`, and items that start after that raise Skipped instead of running.
    When every item has finished or been skipped, the first failure other than Skipped is raised.
    """

    def invoke(item):
        if stopped.is_set():
            raise Skipped("Another task failed")
        try:
            return function(item)
        except BaseException:
            stopped.set()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke, item) for item in items]
        try:
            concurrent.futures.wait(futures)
        except BaseException:
            stopped.set()  # interrupted while waiting: items that have not started will skip themselves
            raise
    failures = [future.exception() for future in futures if future.exception()]
    if failures:
        # Raise the error that caused the stop, not one of the Skipped errors that followed it.
        original = [error for error in failures if not isinstance(error, Skipped)]
        raise (original or failures)[0]
    return [future.result() for future in futures]


def write_table(path, rows):
    """Write a non-empty list of dicts with the same keys as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Read only the prepared JSON files; fetching and parsing papers live in fetch_data.py.
PILOT_PAPERS = 2


def task_data(example):
    """The only example fields that go into writer, rubric and judge prompts.

    Everything else (the author's section, split labels, IDs, earlier feedback) stays out.
    """
    return {
        "visible_paper": example["context"],
        "section_type": example["section_type"],
        "target_words": example["target_words"],
    }


def run_examples(examples, split):
    """The examples a run uses, labelled train or validation.

    A research run uses the train and validation papers as saved. A pilot run uses the papers
    labelled pilot in an older dataset, or else the first PILOT_PAPERS training papers; the first
    (sorted by ID) becomes training and the rest become validation.
    """
    if split == "pilot":
        papers = list(dict.fromkeys(e["paper_id"] for e in examples if e["split"] == "pilot"))
        papers = papers or list(dict.fromkeys(e["paper_id"] for e in examples if e["split"] == "train"))
        papers = sorted(papers[:PILOT_PAPERS])
        pilot = [e for e in examples if e["paper_id"] in papers]
        if len(papers) < PILOT_PAPERS:
            raise RunError(f"Pilot needs {PILOT_PAPERS} papers; found {len(papers)}")
        return [{**e, "split": "train" if e["paper_id"] == papers[0] else "validation"} for e in pilot]
    return [e for e in examples if e["split"] in ("train", "validation")]


# OpenRouter: configured routing, bounded retries, and raw requests and responses.
API_BASE = "https://openrouter.ai/api/v1"
MAX_SENDS = 6
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
BACKOFF_SECONDS = (2, 5, 15, 30, 60)
MAX_RETRY_AFTER_SECONDS = 120
FORMAT_REPAIR = "\nFORMAT REPAIR: Return complete valid JSON matching the schema. "
MAX_REPAIR_ERROR_CHARS = 500


def role_configs():
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    return {
        role: {"model": selected["model"], **config["models"][selected["model"]], "temperature": selected["temperature"]}
        for role in ROLES
        for selected in [config["roles"][role]]
    }


def routing_fields(cfg):
    """Request fields that pin one role's model, provider and decoding settings."""
    return {
        "model": cfg["model"],
        "stream": False,
        "plugins": [],
        "transforms": [],
        "temperature": cfg["temperature"],
        "reasoning": cfg["reasoning"],
        "max_tokens": cfg["max_tokens"],
        "provider": {
            "only": [cfg["provider"]],
            "order": [cfg["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }


class OpenRouter:
    """Send each fresh-run request and keep every attempt and raw response."""

    def __init__(self, output, roles, seed, client=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key and client is None:
            raise RunError("Set OPENROUTER_API_KEY in .env")
        self.headers = {"X-OpenRouter-Title": "Independent XAR reproduction", **({"Authorization": "Bearer " + self.key} if self.key else {})}
        self.output, self.roles, self.seed = Path(output), roles, seed
        self.client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30))
        self.dispatch_stopped = threading.Event()

    def costs(self):
        """Look up OpenRouter's recorded charges for this run's unique generation IDs."""
        generation_ids, missing_ids = set(), []
        for path in sorted(self.output.glob("requests/*/attempt_*.json")):
            record = json.loads(path.read_text())
            generation_id = record.get("response_id") or record.get("response", {}).get("id")
            if isinstance(generation_id, str) and generation_id:
                generation_ids.add(generation_id)
            elif record["status"] != "connect_error":
                missing_ids.append(str(path.relative_to(self.output)))
        charges, errors = {}, {}
        for generation_id in sorted(generation_ids):
            try:
                response = self.client.get(API_BASE + "/generation", params={"id": generation_id}, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.json()["data"]
                cost = data["total_cost"]
                if data["id"] != generation_id or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                    raise ValueError("Invalid generation ID or total_cost")
                charges[generation_id] = cost
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                errors[generation_id] = f"{type(error).__name__}: {error}"
        result = {
            "source": API_BASE + "/generation",
            "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
            "actual_complete_usd": math.fsum(charges.values()),
            "requests": len(charges),
            "unresolved": len(errors) + len(missing_ids),
            "generation_costs": charges,
            "lookup_errors": errors,
            "missing_generation_ids": missing_ids,
        }
        write_json(self.output / "costs.json", result)
        return result

    def payload(self, role, system, data, schema):
        result = {
            **routing_fields(self.roles[role]),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": canonical(data)}],
        }
        if schema is not None:
            result["response_format"] = {"type": "json_schema", "json_schema": {"name": "xar_output", "strict": True, "schema": schema}}
        return result

    def call(self, role, system, data, schema, identity):
        try:
            return self._call(role, system, data, schema, identity)
        except BaseException:
            self.dispatch_stopped.set()
            raise

    def _call(self, role, system, data, schema, identity):
        payload = self.payload(role, system, data, schema)
        key = digest({"payload": payload, "identity": identity, "schema_version": SCHEMA_VERSION})
        directory = self.output / "requests" / key
        directory.mkdir(parents=True, exist_ok=False)
        write_json(
            directory / "request.json",
            {
                "payload": payload,
                "identity": identity,
                "role": role,
                "key": key,
                "schema_version": SCHEMA_VERSION,
            },
        )
        for attempt in range(MAX_SENDS):
            if self.dispatch_stopped.is_set():
                raise Skipped("Another request failed")
            path = directory / f"attempt_{attempt}.json"
            record = {"status": "uncertain", "sent_at": dt.datetime.now(dt.UTC).isoformat()}
            write_json(path, record)
            started, retry_after = time.monotonic(), None
            try:
                response = self.client.post(API_BASE + "/chat/completions", json=payload, headers=self.headers)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as error:
                record.update(status="connect_error", error=str(error))
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            else:
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"raw_body": response.text}
                if not isinstance(raw, dict):
                    raw = {"raw_body": raw}
                choices = raw.get("choices")
                valid_choice = (
                    isinstance(choices, list)
                    and bool(choices)
                    and isinstance(choices[0], dict)
                    and isinstance(choices[0].get("message"), dict)
                    and isinstance(choices[0]["message"].get("content", ""), (str, type(None)))
                )
                success = response.status_code == 200 and valid_choice and not raw.get("error")
                record.update(
                    response=raw,
                    response_id=raw.get("id") or response.headers.get("X-Generation-Id"),
                    http_status=response.status_code,
                    status="success" if success else "http_error",
                )
                record["duration_seconds"] = time.monotonic() - started
                write_json(path, record)
                if success:
                    choice = raw["choices"][0]
                    return {
                        "content": choice["message"].get("content") or "",
                        "finish_reason": choice.get("finish_reason"),
                        "request_key": key,
                        "response_id": record["response_id"],
                        "usage": raw.get("usage") or {},
                        "model": raw.get("model"),
                        "provider": raw.get("provider"),
                        "raw_response": str(path),
                    }
                if response.status_code not in RETRYABLE_STATUS:
                    raise RunError(f"Unusable HTTP {response.status_code}: {path}")
                retry_after = response.headers.get("Retry-After")
            record["duration_seconds"] = time.monotonic() - started
            write_json(path, record)
            if attempt < MAX_SENDS - 1:
                wait = BACKOFF_SECONDS[attempt]
                if retry_after is not None and retry_after.strip().isdigit():
                    wait = max(wait, min(int(retry_after), MAX_RETRY_AFTER_SECONDS))
                print(f"Retryable failure for {role}; waiting {wait} s (see {path})", flush=True)
                time.sleep(wait)
        raise RunError(f"{role}: failed after {MAX_SENDS} sends; see {directory}")

    def structured(self, role, system, data, schema, identity, validator=None, repair=True):
        """Ask role for JSON matching schema and check it with validator.

        If the reply is incomplete, not valid JSON or fails a check, and repair is on, the request is
        sent once more with the error appended to the system prompt. Returns the status ("valid" or
        "missing"), the parsed value and every attempt.
        """
        attempts = []
        for format_attempt in range(2 if repair else 1):
            instructions = system
            if format_attempt:
                instructions += FORMAT_REPAIR
                instructions += "Keep the substantive task inputs unchanged. Previous validation error: " + attempts[-1]["error"]
            response = self.call(role, instructions, data, schema, {"task": identity, "format_attempt": format_attempt})
            try:
                if response["finish_reason"] != "stop":
                    raise InvalidOutput(f"Incomplete output: {response['finish_reason']}")
                value = json.loads(response["content"])
                jsonschema.validate(value, schema)
                if validator:
                    validator(value)
                attempts.append({"response": response, "status": "valid"})
                return {"status": "valid", "value": value, "attempts": attempts}
            except (json.JSONDecodeError, jsonschema.ValidationError, InvalidOutput) as e:
                attempts.append({"response": response, "status": "invalid", "error": str(e)[:MAX_REPAIR_ERROR_CHARS]})
        return {"status": "missing", "value": None, "attempts": attempts}


# Rubrics, anonymous grading, and training-only prompt optimization.
TEXT = {"type": "string", "minLength": 1}
SCORE = {"type": "number", "minimum": 0, "maximum": 10}


def object_schema(properties):
    """A JSON object with exactly these properties, all required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def criteria_list(item):
    """A rubric has 4 to 8 criteria, and a grade has one score per criterion."""
    return {"type": "array", "minItems": 4, "maxItems": 8, "items": item}


CRITERION_SCHEMA = object_schema({"id": TEXT, "description": TEXT, "low": TEXT, "middle": TEXT, "high": TEXT})
CRITERION_SCORE_SCHEMA = object_schema({"id": TEXT, "score": SCORE, "evidence": TEXT})
RUBRIC_SCHEMA = object_schema({"criteria": criteria_list(CRITERION_SCHEMA)})
GRADE_SCHEMA = object_schema({"scores": criteria_list(CRITERION_SCORE_SCHEMA)})
PROPOSAL_SCHEMA = object_schema({"prompt": TEXT, "rationale": TEXT})
MAX_RUBRIC_WORDS = 1000
QUOTED_TEXT = re.compile(r'["“]([^"”]+)["”]')
WRITER_ATTEMPTS = 3
PROPOSAL_ATTEMPTS = 2


def validate_rubric(rubric):
    """Check a schema-valid rubric for unique criterion IDs and total length.

    Raises with a short message; structured() sends that message back in its format repair."""
    ids = [c["id"] for c in rubric["criteria"]]
    if len(ids) != len(set(ids)):
        raise InvalidOutput("Duplicate criterion IDs")
    total = sum(len(text.split()) for criterion in rubric["criteria"] for text in criterion.values())
    if total > MAX_RUBRIC_WORDS:
        raise InvalidOutput("Rubric exceeds 1000 words")


def validate_grade(grade, rubric, supplied_text):
    """Check a schema-valid grade against its rubric and supplied text.

    Every criterion must be scored exactly once, and anything the judge quotes as evidence
    must occur in supplied_text (the graded section plus the visible paper)."""
    ids = [s["id"] for s in grade["scores"]]
    if len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in rubric["criteria"]}:
        raise InvalidOutput("Criterion IDs mismatch")
    # json.loads accepts NaN, and NaN passes the schema's minimum and maximum.
    if not all(math.isfinite(s["score"]) for s in grade["scores"]):
        raise InvalidOutput("Score must be finite")
    normalized_text = " ".join(supplied_text.split()).casefold()
    for score in grade["scores"]:
        for quoted in QUOTED_TEXT.findall(score["evidence"]):
            if " ".join(quoted.split()).casefold() not in normalized_text:
                raise InvalidOutput("Quote absent from supplied text")


def grade_total(grade):
    """The mean criterion score of a grade."""
    return statistics.mean(s["score"] for s in grade["scores"])


def writer_candidates(api, examples, output, concurrency=1):
    """Write one model section per example, `concurrency` examples at a time.

    Each section gets up to WRITER_ATTEMPTS drafts: a draft that is cut off or outside the length
    window goes back to the writer with a revision note, and the last draft is kept either way.
    Each section is saved as it finishes. Returns the records keyed by example ID."""
    prompt = (ROOT / "prompts/writer.md").read_text()
    config_hash = digest({"writer": api.roles["writer"], "prompt": prompt, "schema_version": SCHEMA_VERSION})
    stopped = api.dispatch_stopped

    def write_section(example):
        eid = example["example_id"]
        path = output / "generations" / (eid + ".json")
        low, high = 0.85 * example["target_words"], 1.15 * example["target_words"]
        attempts = []
        for attempt in range(WRITER_ATTEMPTS):
            data = task_data(example)
            if attempt:
                data.update(
                    previous_section=attempts[-1]["text"],
                    length_revision=f"Revise only to fit {math.ceil(low)}–{math.floor(high)} words. Preserve claims.",
                )
            if stopped.is_set():
                raise Skipped(f"{eid}: another section failed")
            identity = {"writer": config_hash, "example": eid, "attempt": attempt}
            response = api.call("writer", prompt, data, None, identity)
            text = response["content"].strip()
            count = len(text.split())
            compliant = low <= count <= high
            attempts.append(
                {
                    "attempt_index": attempt,
                    "text": text,
                    "text_hash": digest(text),
                    "words": count,
                    "length_compliant": compliant,
                    "response": response,
                }
            )
            if compliant and response["finish_reason"] == "stop":
                break
        accepted = attempts[-1]
        record = {
            "example_id": eid,
            "context_hash": example["context_hash"],
            "writer_configuration_hash": config_hash,
            "writer_configuration": api.roles["writer"],
            "attempts": attempts,
            "accepted_attempt": accepted["attempt_index"],
            "text": accepted["text"],
            "text_hash": accepted["text_hash"],
            "words": accepted["words"],
            "length_compliant": accepted["length_compliant"],
            "complete": accepted["response"]["finish_reason"] == "stop",
            "length_ratio": accepted["words"] / example["target_words"],
        }
        write_json(path, record)
        return record

    records = bounded_map(write_section, examples, concurrency, stopped)
    # bounded_map returns results in input order, so candidates.json keeps the dataset order.
    candidates = {record["example_id"]: record for record in records}
    artifact = {
        "writer_configuration_hash": config_hash,
        "writer_configuration": api.roles["writer"],
        "candidates": candidates,
    }
    write_json(output / "generations/candidates.json", artifact)
    return candidates


def generate_rubric(api, example, meta_prompt, identity, output):
    result = api.structured(
        "rubric",
        (ROOT / "prompts/rubric_wrapper.md").read_text(),
        {**task_data(example), "meta_prompt": meta_prompt},
        RUBRIC_SCHEMA,
        identity,
        validate_rubric,
    )
    record = {
        "example_id": example["example_id"],
        "context_hash": example["context_hash"],
        "meta_prompt_hash": digest(meta_prompt),
        "generator_configuration": api.roles["rubric"],
        "identity": identity,
        **result,
    }
    record["rubric_hash"] = digest(result["value"]) if result["value"] else None
    write_json(output, record)
    return record


def grade_candidate(api, example, rubric_record, text, identity, output):
    """Grade one anonymous section against a rubric."""
    if rubric_record["status"] != "valid":
        result = {"status": "missing", "value": None, "attempts": [], "reason": "invalid_rubric"}
    else:
        rubric = rubric_record["value"]
        result = api.structured(
            "judge",
            (ROOT / "prompts/judge.md").read_text(),
            {**task_data(example), "rubric": rubric, "candidate": text},
            GRADE_SCHEMA,
            identity,
            lambda value: validate_grade(value, rubric, text + " " + example["context"]),
        )
    record = {
        "example_id": example["example_id"],
        "candidate_hash": digest(text),
        "rubric_hash": rubric_record["rubric_hash"],
        "judge_configuration": api.roles["judge"],
        "identity": identity,
        **result,
    }
    record["total"] = grade_total(result["value"]) if result["value"] else None
    write_json(output, record)
    return record


def evaluate_checkpoint(api, examples, candidates, meta_prompt, checkpoint, output, concurrency):
    """Score one meta prompt on examples: generate a rubric per example, then grade both sections.

    The judge grades the human and the model section separately, in a seeded random order, and
    never learns which is which. Returns one row per example with the human-minus-model gap
    (None if the rubric or a grade is missing), and saves the rows next to the grades."""
    root = output

    def evaluate(example):
        eid = example["example_id"]
        candidate = candidates[eid]
        # The label is part of the saved paths and request identities, "main/" prefix included.
        label = f"main/{checkpoint}/{example['split']}/{eid}"
        rubric_path = root / "rubrics" / label / "rubric.json"
        rubric = generate_rubric(api, example, meta_prompt, {"rubric": label}, rubric_path)
        origins = ["human", "model"]
        order_seed = int(digest({"seed": api.seed, "label": label})[:16], 16)
        random.Random(order_seed).shuffle(origins)
        grade_paths = {origin: root / "scores" / label / (origin + ".json") for origin in origins}
        # The origin appears only in the file path, never in the request or its identity.
        totals = {
            origin: grade_candidate(
                api, example, rubric, example["reference"] if origin == "human" else candidate["text"], {"grade": label, "slot": slot}, grade_paths[origin]
            )["total"]
            for slot, origin in enumerate(origins)
        }
        human, model = totals["human"], totals["model"]
        return {
            "example_id": eid,
            "paper_id": example["paper_id"],
            "section_type": example["section_type"],
            "split": example["split"],
            "checkpoint": checkpoint,
            "human": human,
            "model": model,
            "gap": human - model if human is not None and model is not None else None,
            "length_ratio": candidate["length_ratio"],
            "length_compliant": candidate["length_compliant"],
            "rubric_path": str(rubric_path),
            "grade_paths": {origin: str(path) for origin, path in grade_paths.items()},
        }

    rows = bounded_map(evaluate, examples, concurrency, api.dispatch_stopped)
    write_json(root / "scores" / "main" / str(checkpoint) / f"{examples[0]['split']}_rows.json", rows)
    return rows


PROPOSAL_RED_FLAGS = {
    # Tells the rubric to favour the human section or to penalize the model's.
    "origin_preference": r"(?:prefer|reward|favor|favour|boost)\s+(?:the\s+)?(?:human|expert|original)|"
    r"(?:penaliz|penalis|punish|downscore)\w*\s+(?:the\s+)?(?:model|ai|generated)",
    # Tells the rubric to work out who wrote the section.
    "origin_detection": r"(?:detect|guess|infer|identify|determine)\s+(?:the\s+)?(?:authorship|origin|provenance)|"
    r"(?:human.written|ai.generated|model.generated)\s+(?:tells|markers|signals)",
    # Overrides the wrapper, or weights or rescales the 0-10 criterion scores.
    "wrapper_override": r"(?:ignore|override|replace)\s+(?:the\s+)?(?:wrapper|system|grading|schema)|"
    r"(?:weighted\s+(?:mean|average)|unequal\s+weight|score\s+(?:from\s+)?0\s*(?:to|[-–])\s*100)",
    # Model-side markup leaked into the prompt text, or a prompt cut off inside it. Seen from
    # MiMo-V2.6-Flash: a prompt ending "...<tool_call><function=json>{" inside valid JSON.
    "leaked_markup": r"<tool_call|<function=|</function|<\|im_|\{\s*$",
}
SPAN = 12  # a run of this many words copied from a training paper counts as leakage
MIN_IDENTIFIER_LENGTH = 5  # shorter paper IDs, titles or author names match by chance


def audit_proposal(text, examples, initial_prompt, max_words):
    """Statically check an optimizer proposal before it can become the next meta prompt.

    Rejects a prompt that is longer than max_words, matches PROPOSAL_RED_FLAGS (including leaked
    tool-call markup), names a training paper or its authors, or copies SPAN consecutive words
    from a training paper (unless the initial prompt already contains them)."""
    word_count = len(text.split())
    reasons, flags = [], []
    if word_count > max_words:
        reasons.append("meta_prompt_word_bound")
    lower = text.casefold()
    for name, pattern in PROPOSAL_RED_FLAGS.items():
        if re.search(pattern, lower):
            reasons.append(name)
    proposal_words = lower.split()
    initial_lower = initial_prompt.casefold()
    for example in examples:
        paper = example["paper_id"]
        metadata = example["provenance"]
        for identifier in [paper, metadata["title"], *metadata.get("authors", [])]:
            if len(identifier) >= MIN_IDENTIFIER_LENGTH and identifier.casefold() in lower:
                flags.append({"type": "training_identifier", "paper_id": paper, "match": identifier})
        for field in ("reference", "context"):
            source_words = example[field].casefold().split()
            source_spans = {tuple(source_words[i : i + SPAN]) for i in range(len(source_words) - SPAN + 1)}
            for i in range(len(proposal_words) - SPAN + 1):
                span = proposal_words[i : i + SPAN]
                span_text = " ".join(span)
                if tuple(span) in source_spans and span_text not in initial_lower:
                    flags.append({"type": "copied_training_span", "paper_id": paper, "match": span_text})
                    break
    if flags:
        reasons.append("training_leakage")
    return {
        "accepted": not reasons,
        "reasons": sorted(reasons),
        "flags": flags,
        "words": word_count,
        "policy": "fixed_static_scan_v1",
        "limitation": "Static scan cannot prove semantic absence of provenance heuristics",
    }


def propose_prompt(api, current, feedback, examples, initial, iteration, *, output, max_words):
    """Ask the optimizer for the next meta prompt, given the training feedback on the current one.

    A proposal that is malformed or fails audit_proposal is sent back once with the reasons. If
    the repair fails too, the update is used up and the current prompt carries over unchanged.
    Saves the feedback and proposal under feedback/iter_XX/ and returns the next prompt."""
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}"
    write_json(directory / "training.json", feedback)
    data = {"feedback": feedback, "max_meta_prompt_words": max_words}
    attempts = []
    for attempt in range(PROPOSAL_ATTEMPTS):
        instruction = (ROOT / "prompts/optimizer.md").read_text()
        if attempt:
            rejected = attempts[-1]
            instruction += "\nBOUNDED REPAIR: Fix these proposal violations: " + ", ".join(rejected["audit"]["reasons"])
            # Show the rejected proposal, or the raw reply if it was not valid JSON.
            previous = rejected["value"] or rejected["attempts"][-1]["response"]["content"]
            data = {**data, "previous_proposal": previous}
        result = api.structured(
            "optimizer",
            instruction,
            data,
            PROPOSAL_SCHEMA,
            {"proposal": iteration, "bounded_attempt": attempt},
            repair=False,
        )
        if result["value"]:
            audit = audit_proposal(result["value"]["prompt"], examples, initial, max_words)
        else:
            audit = {"accepted": False, "reasons": ["invalid_format"], "flags": []}
        attempts.append({"attempt": attempt, "audit": audit, **result})
        if audit["accepted"]:
            break
    accepted = attempts[-1]["audit"]["accepted"]
    proposal = attempts[-1]["value"]["prompt"] if accepted else current
    record = {
        "iteration": iteration,
        "parent_prompt_hash": digest(current),
        "feedback_hash": digest(feedback),
        "optimizer_configuration": api.roles["optimizer"],
        "attempts": attempts,
        "accepted": accepted,
        "update_consumed": True,
        "prompt": proposal,
        "prompt_hash": digest(proposal),
    }
    write_json(directory / "proposal.json", record)
    return proposal


def build_feedback(examples, candidates, rows, current_prompt, failure_count):
    """The active feedback policy: nonpositive training gaps, without full papers.

    Keep the recorded all_training_gaps field's existing meaning: all nonpositive gaps, before
    limiting detailed failure examples. Validation and confirmation never enter this payload.
    """
    if any(item["split"] != "train" for item in [*examples, *rows]):
        raise RunError("Feedback requires training sections")
    if any(row["gap"] is None for row in rows):
        raise RunError("Training grades missing")
    lookup = {example["example_id"]: example for example in examples}
    by_gap = sorted((row for row in rows if row["gap"] <= 0), key=lambda row: (row["gap"], row["example_id"]))
    failures = []
    for row in by_gap[:failure_count]:
        example = lookup[row["example_id"]]
        failures.append(
            {
                "example_id": example["example_id"],
                "section_type": example["section_type"],
                "target_words": example["target_words"],
                "human_candidate": example["reference"],
                "model_candidate": candidates[example["example_id"]]["text"],
                "rubric": json.loads(Path(row["rubric_path"]).read_text())["value"],
                "grades": {origin: json.loads(Path(path).read_text())["value"] for origin, path in row["grade_paths"].items()},
                "gap": row["gap"],
            }
        )
    return {
        "current_prompt": current_prompt,
        "summary": summarize(rows),
        "failures": failures,
        "example_ids_used_for_aggregate": sorted(lookup),
        "selection": FAILING_FEEDBACK,
        "all_training_gaps": [{"example_id": row["example_id"], "section_type": row["section_type"], "gap": row["gap"]} for row in by_gap],
    }


# Preserve whole-paper bootstrap statistics, including the fields sent to the optimizer.
BOOTSTRAP_REPLICATES = 2000
INTERVAL = (0.025, 0.975)


def bootstrap(rows, seed=0):
    """Percentile bootstrap of the mean gap that resamples whole papers.

    Sections from one paper are not independent, so each replicate draws papers with
    replacement and takes all of a drawn paper's gaps. Rows without a gap are ignored;
    returns None if no row has one."""
    gaps_by_paper = collections.defaultdict(list)
    for row in rows:
        if row.get("gap") is not None:
            gaps_by_paper[row["paper_id"]].append(row["gap"])
    if not gaps_by_paper:
        return None
    rng = random.Random(seed)
    papers = sorted(gaps_by_paper)
    means = sorted(
        statistics.mean(gap for paper in rng.choices(papers, k=len(papers)) for gap in gaps_by_paper[paper]) for _ in range(BOOTSTRAP_REPLICATES)
    )
    return {
        "method": "paired_whole_paper_percentile_bootstrap",
        "paper_clusters": len(papers),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "estimate": statistics.mean(gap for gaps in gaps_by_paper.values() for gap in gaps),
        "bootstrap_median": means[BOOTSTRAP_REPLICATES // 2],
        "low": means[int(INTERVAL[0] * BOOTSTRAP_REPLICATES)],
        "high": means[int(INTERVAL[1] * BOOTSTRAP_REPLICATES)],
    }


def section_summary(graded, section):
    gaps = [row["gap"] for row in graded if row["section_type"] == section]
    return {"coverage": len(gaps), "gap": statistics.mean(gaps) if gaps else None}


def summarize(rows, seed=0):
    """Summarize one checkpoint's rows: mean human, model and gap scores over the sections with
    both grades, per-section gaps, and paper-bootstrap intervals for all sections, the
    length-compliant ones."""
    graded = [row for row in rows if row["gap"] is not None]

    def mean(key):
        return statistics.mean(row[key] for row in graded) if graded else None

    return {
        "examples": len(rows),
        "paired_coverage": len(graded),
        "complete": len(graded) == len(rows),
        "human": mean("human"),
        "model": mean("model"),
        "gap": mean("gap"),
        "positive_gap_fraction": sum(row["gap"] > 0 for row in graded) / len(graded) if graded else None,
        "ties": sum(row["gap"] == 0 for row in graded),
        "paper_interval": bootstrap(graded, seed),
        "length_compliant": sum(row["length_compliant"] for row in rows),
        "sections": {section: section_summary(graded, section) for section in SECTIONS},
        "compliant_sensitivity": bootstrap([row for row in graded if row["length_compliant"]], seed),
    }


def paired_improvement(initial, selected, seed=0):
    """Per-section change in gap from the initial to the selected checkpoint, with a
    paper-bootstrap interval. Sections missing a gap at either checkpoint are left out."""
    initial_by_id = {row["example_id"]: row for row in initial}
    changes = [
        {"paper_id": row["paper_id"], "gap": row["gap"] - before}
        for row in selected
        if row["gap"] is not None and (before := initial_by_id[row["example_id"]]["gap"]) is not None
    ]
    return {
        "paired_coverage": len(changes),
        "mean": statistics.mean(change["gap"] for change in changes) if changes else None,
        "interval": bootstrap(changes, seed),
    }


@dataclass
class RunSettings:
    output_dir: str
    split: str
    seed: int
    iterations: int
    max_meta_prompt_words: int
    failure_examples: int
    concurrency: int
    dataset: str = "data/examples.jsonl"


def write_report(output, summaries, rows, selected, costs, seed):
    table = [
        {
            "iteration": index,
            "train_human": train["human"],
            "train_model": train["model"],
            "train_gap": train["gap"],
            "val_human": validation["human"],
            "val_model": validation["model"],
            "val_gap": validation["gap"],
            "val_coverage": validation["paired_coverage"],
            "selected_by_train": index == selected,
        }
        for index, (train, validation) in enumerate(summaries)
    ]
    improvement = paired_improvement(rows[0], rows[selected], seed)
    result = {
        "selected_iteration": selected,
        "checkpoints": table,
        "summaries": [{"train": train, "validation": validation} for train, validation in summaries],
        "paired_improvement": improvement,
        "costs": costs,
    }
    write_table(output / "checkpoints.csv", table)
    write_json(output / "summary.json", result)
    lines = [
        "# Reproduction results",
        "",
        f"Training selected P{selected} before validation began.",
        "",
        "The gap is the author's score minus the model's score. Each criterion has equal weight.",
        "",
        "| Prompt | Training gap | Validation gap | Validation coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in table:
        gap = "missing" if row["val_gap"] is None else f"{row['val_gap']:+.3f}"
        lines.append(f"| P{row['iteration']} | {row['train_gap']:+.3f} | {gap} | {row['val_coverage']} |")
    interval = improvement["interval"]
    if interval:
        lines += [
            "",
            (f"The selected prompt changed the paired validation gap by {improvement['mean']:+.3f} "
            f"(95% whole-paper bootstrap interval: {interval['low']:+.3f} to {interval['high']:+.3f})."),
        ]
    lines += [
        "",
        (f"OpenRouter-reported cost{' (partial)' if costs['unresolved'] else ''}: "
        f"${costs['actual_complete_usd']:.4f} across {costs['requests']} generations. "
        f"Unresolved lookups or sends: {costs['unresolved']}. Details are in costs.json."),
        "",
        ("This is an independent reproduction with repository-specific papers and prompts. "
        "The confirmation split remains unused. Bootstrap intervals resample whole papers."),
        "",
        "Full statistics and coverage are in summary.json; every request and response is saved.",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return result


def run_xar(settings):
    initial = (ROOT / "prompts/rubric_initial.md").read_text()
    if len(initial.split()) > settings.max_meta_prompt_words:
        raise RunError("Initial prompt exceeds max_meta_prompt_words")
    examples = [json.loads(line) for line in Path(settings.dataset).read_text().splitlines() if line.strip()]
    examples = run_examples(examples, settings.split)
    train = [example for example in examples if example["split"] == "train"]
    validation = [example for example in examples if example["split"] == "validation"]
    if not train or not validation:
        raise RunError("Need train and validation papers")
    roles = role_configs()
    output = Path(settings.output_dir)
    # Atomic directory creation also prevents two processes from starting in the same directory.
    api = OpenRouter(output, roles, settings.seed)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunError(f"Output exists: {output}") from None
    status = {"state": "failed"}
    try:
        files = [
            ROOT / "run.py",
            ROOT / "fetch_data.py",
            ROOT / "uv.lock",
            ROOT / "configs/experiments.yaml",
            ROOT / "configs/models.yaml",
            *sorted((ROOT / "prompts").glob("*.md")),
        ]
        write_json(
            output / "manifest.json",
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "arguments": asdict(settings),
                "roles": roles,
                "dataset_hash": file_hash(settings.dataset),
                "software_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in files},
                "initial_meta_prompt_hash": digest(initial),
                "feedback_policy": FAILING_FEEDBACK,
                "context_handling": "Providers enforce context limits; full inputs are sent without truncation",
            },
        )
        candidates = writer_candidates(api, examples, output, settings.concurrency)
        prompts, training_summaries, training_rows = [initial], [], None
        for iteration in range(settings.iterations + 1):
            if iteration:
                feedback = build_feedback(train, candidates, training_rows, prompts[-1], settings.failure_examples)
                prompts.append(
                    propose_prompt(
                        api, prompts[-1], feedback, train, initial, iteration, output=output, max_words=settings.max_meta_prompt_words,
                    )
                )
            (output / "prompts").mkdir(exist_ok=True)
            (output / "prompts" / f"iter_{iteration:02d}.md").write_text(prompts[-1])
            training_rows = evaluate_checkpoint(api, train, candidates, prompts[-1], iteration, output, settings.concurrency)
            summary = summarize(training_rows, settings.seed)
            training_summaries.append(summary)
            print(
                f"P{iteration}: training gap {summary['gap']}, coverage {summary['paired_coverage']}/{len(train)}",
                flush=True,
            )
            if not summary["complete"]:
                raise RunError("Training grades missing")
        gaps = [summary["gap"] for summary in training_summaries]
        selected = gaps.index(max(gaps))  # Earliest checkpoint wins a tie.
        write_json(
            output / "freeze.json",
            {
                "selected": selected,
                "training_gaps": gaps,
                "prompt_hashes": [digest(value) for value in prompts],
                "frozen_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        )
        print(f"Training selected P{selected}; scoring validation", flush=True)
        validation_rows = [
            evaluate_checkpoint(api, validation, candidates, value, index, output, settings.concurrency) for index, value in enumerate(prompts)
        ]
        validation_summaries = [summarize(rows, settings.seed) for rows in validation_rows]
        result = write_report(
            output, list(zip(training_summaries, validation_summaries, strict=True)), validation_rows, selected, api.costs(), settings.seed,
        )
        status = {"state": "complete" if all(summary["complete"] for summary in validation_summaries) else "incomplete"}
        return result
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "status.json", status)
        api.client.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="smoke test on 2 training papers")
    parser.add_argument("--output", help="new run directory (default: runs/<configured run name>)")
    parser.add_argument("--seed", type=int, help="grading-order and bootstrap seed, not a model decoding seed")
    parser.add_argument("--concurrency", type=int, choices=range(1, MAX_CONCURRENCY + 1), metavar="N", help="parallel examples, 1–32")
    parser.add_argument("--dataset", default="data/examples.jsonl")
    return parser.parse_args(argv)


def settings_for(options):
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    if design["feedback_policy"] != FAILING_FEEDBACK:
        raise RunError(f"Use feedback_policy={FAILING_FEEDBACK!r}")
    return RunSettings(
        output_dir=options.output or str(Path("runs") / design["pilot_run" if options.pilot else "research_run"]),
        split="pilot" if options.pilot else "research",
        seed=design["seed"] if options.seed is None else options.seed,
        iterations=design["pilot_iterations" if options.pilot else "iterations"],
        max_meta_prompt_words=design["max_meta_prompt_words"],
        failure_examples=design["failure_examples"],
        concurrency=options.concurrency or design["concurrency"],
        dataset=options.dataset,
    )


if __name__ == "__main__":
    try:
        run_xar(settings_for(parse_args()))
    except (RunError, httpx.HTTPError) as error:
        raise SystemExit(f"STOP: {error}") from None

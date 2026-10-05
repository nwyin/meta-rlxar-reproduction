"""Run a fresh Meta XAR reproduction with OpenRouter.

Prepare data with fetch_data.py (papers) or fetch_fiction.py (fiction), then set a new `output`
directory in the config file (configs/arxiv.yaml, the default, or configs/fiction.yaml). The config defines the whole run,
including its dataset, prompt directory and the model for each role, and is copied into the run
directory. No inputs are truncated.
"""

from __future__ import annotations; import argparse, concurrent.futures, csv, datetime as dt, hashlib, json, math, os, random, re, statistics, threading, time, unicodedata; from pathlib import Path; import httpx, jsonschema, yaml; from dotenv import load_dotenv  # noqa: I001  # fmt: skip

ROOT = Path(__file__).resolve().parent
PROMPTS = ("writer", "rubric_initial", "rubric_wrapper", "judge", "optimizer")
FAILING_FEEDBACK = "failing_gap_without_context"
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


TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"', "…": "...", " ": " "})


def normalize_text(text, paragraphs):
    """The one plain-text form for references, contexts and model sections, so typography cannot
    tell them apart: NFC, straight quotes, "..." for an ellipsis and single spaces. With
    `paragraphs`, each line is a paragraph and paragraphs are separated by one blank line;
    without, the text is one line."""
    text = unicodedata.normalize("NFC", text).translate(TYPOGRAPHY)
    if not paragraphs:
        return " ".join(text.split())
    return "\n\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


# Read only the prepared JSON files; fetching and parsing sources live in fetch_data.py and fetch_fiction.py.
def task_data(example):
    """The only example fields that go into writer, rubric and judge prompts.

    Everything else (the reference, split labels, IDs, earlier feedback) stays out.
    """
    return {
        "context": example["context"],
        "kind": example["kind"],
        "target_words": example["target_words"],
    }


def read_prompt(config, name):
    """One prompt from the config's prompt directory (prompts/papers or prompts/fiction)."""
    return (ROOT / config["prompts"] / f"{name}.md").read_text()


def run_examples(examples, config):
    """The examples a run uses, labelled train or validation.

    A research run uses the train and validation sources (papers or books) as saved. A pilot run,
    for a dry run, uses the first `pilot_sources` training sources; the first (sorted by ID) becomes training and
    the rest become validation.
    """
    if config["split"] not in ("pilot", "research"):
        raise RunError(f"Invalid split {config['split']!r}; use pilot or research")
    if config["split"] == "pilot":
        sources = sorted(list(dict.fromkeys(e["source_id"] for e in examples if e["split"] == "train"))[: config["pilot_sources"]])
        if len(sources) < config["pilot_sources"]:
            raise RunError(f"Pilot needs {config['pilot_sources']} sources; found {len(sources)}")
        return [{**e, "split": "train" if e["source_id"] == sources[0] else "validation"} for e in examples if e["source_id"] in sources]
    return [e for e in examples if e["split"] in ("train", "validation")]


# OpenRouter: configured routing, bounded retries, and raw requests and responses.


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

    def __init__(self, output, config, client=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key and client is None:
            raise RunError("Set OPENROUTER_API_KEY in .env")
        self.headers = {"X-OpenRouter-Title": "Independent XAR reproduction", **({"Authorization": "Bearer " + self.key} if self.key else {})}
        self.output, self.roles, self.config = Path(output), config["roles"], config
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(config["request_timeout_seconds"], connect=config["connect_timeout_seconds"])
        )
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
                response = self.client.get(
                    self.config["api_base"] + "/generation",
                    params={"id": generation_id},
                    headers=self.headers,
                    timeout=self.config["lookup_timeout_seconds"],
                )
                response.raise_for_status()
                data = response.json()["data"]
                cost = data["total_cost"]
                if data["id"] != generation_id or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                    raise ValueError("Invalid generation ID or total_cost")
                charges[generation_id] = cost
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                errors[generation_id] = f"{type(error).__name__}: {error}"
        result = {
            "source": self.config["api_base"] + "/generation",
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
        for attempt in range(self.config["max_sends"]):
            if self.dispatch_stopped.is_set():
                raise Skipped("Another request failed")
            path = directory / f"attempt_{attempt}.json"
            record = {"status": "uncertain", "sent_at": dt.datetime.now(dt.UTC).isoformat()}
            write_json(path, record)
            started, retry_after = time.monotonic(), None
            try:
                response = self.client.post(self.config["api_base"] + "/chat/completions", json=payload, headers=self.headers)
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
                if response.status_code not in self.config["retryable_status"]:
                    raise RunError(f"Unusable HTTP {response.status_code}: {path}")
                retry_after = response.headers.get("Retry-After")
            finally:
                record["duration_seconds"] = time.monotonic() - started
                write_json(path, record)
            if attempt < self.config["max_sends"] - 1:
                wait = self.config["backoff_seconds"][attempt]
                if retry_after is not None and retry_after.strip().isdigit():
                    wait = max(wait, min(int(retry_after), self.config["max_retry_after_seconds"]))
                print(f"Retryable failure for {role}; waiting {wait} s (see {path})", flush=True)
                time.sleep(wait)
        raise RunError(f"{role}: failed after {self.config['max_sends']} sends; see {directory}")

    def structured(self, role, system, data, schema, identity, validator=None, repair=True):
        """Ask role for JSON matching schema and check it with validator.

        If the reply is incomplete, not valid JSON or fails a check, and repair is on, the request is
        sent once more with the error appended to the system prompt. Returns the status ("valid" or
        "missing"), the parsed value and every attempt.
        """
        attempts = []
        for format_attempt in range(self.config["format_repairs"] + 1 if repair else 1):
            instructions = system
            if format_attempt:
                instructions += (
                    "\nFORMAT REPAIR: Return complete valid JSON matching the schema. "
                    "Keep the substantive task inputs unchanged. Previous validation error: " + attempts[-1]["error"]
                )
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
                attempts.append({"response": response, "status": "invalid", "error": str(e)[: self.config["max_repair_error_chars"]]})
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
    """A rubric has at least 1 criterion, and a grade has one score per criterion."""
    return {"type": "array", "minItems": 1, "items": item}


CRITERION_SCHEMA = object_schema({"id": TEXT, "description": TEXT, "low": TEXT, "middle": TEXT, "high": TEXT})
CRITERION_SCORE_SCHEMA = object_schema({"id": TEXT, "score": SCORE, "evidence": TEXT})
RUBRIC_SCHEMA = object_schema({"criteria": criteria_list(CRITERION_SCHEMA)})
GRADE_SCHEMA = object_schema({"scores": criteria_list(CRITERION_SCORE_SCHEMA)})
PROPOSAL_SCHEMA = object_schema({"prompt": TEXT, "rationale": TEXT})
QUOTED_TEXT = re.compile(r'["“]([^"”]+)["”]')


def validate_rubric(rubric):
    """Check a schema-valid rubric for unique criterion IDs.

    Raises with a short message; structured() sends that message back in its format repair."""
    ids = [c["id"] for c in rubric["criteria"]]
    if len(ids) != len(set(ids)):
        raise InvalidOutput("Duplicate criterion IDs")


def quote_words(text):
    """The text's words, casefolded and space-separated, with a space at each end."""
    return " " + " ".join(re.findall(r"\w+", text.casefold())) + " "


def validate_grade(grade, rubric, supplied_text):
    """Check a schema-valid grade against its rubric and supplied text.

    Every criterion must be scored exactly once, and the words of anything the judge quotes as
    evidence must occur in order in supplied_text (the graded section plus the visible context).
    Punctuation and case are ignored, because judges move a comma inside a quotation or swap
    double quotes for single ones around dialogue. An ellipsis in a quote marks omitted words, so
    the parts around it must occur in order but need not be adjacent."""
    ids = [s["id"] for s in grade["scores"]]
    if len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in rubric["criteria"]}:
        raise InvalidOutput("Criterion IDs mismatch")
    # json.loads accepts NaN, and NaN passes the schema's minimum and maximum.
    if not all(math.isfinite(s["score"]) for s in grade["scores"]):
        raise InvalidOutput("Score must be finite")
    supplied_words = quote_words(supplied_text)
    for score in grade["scores"]:
        for quoted in QUOTED_TEXT.findall(score["evidence"].replace('\\"', '"')):
            start = 0
            for part in re.split(r"\.\.\.|…", quoted):
                if not quote_words(part).strip():
                    continue
                start = supplied_words.find(quote_words(part), start)
                if start < 0:
                    raise InvalidOutput("Quote absent from supplied text")
                start += len(quote_words(part)) - 1


def writer_candidates(api, examples, output):
    """Write one model section per example, `concurrency` examples at a time.

    Each section gets up to `writer_attempts` drafts: a draft that is cut off or outside the length
    window goes back to the writer with a revision note, and the last draft is kept either way.
    Each section is saved as it finishes, in the dataset's text form (one line, or one paragraph per
    line when the references keep paragraphs). Returns the records keyed by example ID."""
    prompt = read_prompt(api.config, "writer")
    paragraphs = any("\n" in example["reference"] for example in examples)
    config_hash = digest({"writer": api.roles["writer"], "prompt": prompt, "schema_version": SCHEMA_VERSION})
    stopped = api.dispatch_stopped

    def write_section(example):
        eid = example["example_id"]
        path = output / "generations" / (eid + ".json")
        low, high = 0.85 * example["target_words"], 1.15 * example["target_words"]
        attempts = []
        for attempt in range(api.config["writer_attempts"]):
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
            text = normalize_text(response["content"], paragraphs)
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

    records = bounded_map(write_section, examples, api.config["concurrency"], stopped)
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
        read_prompt(api.config, "rubric_wrapper"),
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
            read_prompt(api.config, "judge"),
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
    record["total"] = statistics.mean(s["score"] for s in result["value"]["scores"]) if result["value"] else None  # mean criterion score
    write_json(output, record)
    return record


def evaluate_checkpoint(api, examples, candidates, meta_prompt, checkpoint, output):
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
        order_seed = int(digest({"seed": api.config["seed"], "label": label})[:16], 16)
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
            "source_id": example["source_id"],
            "kind": example["kind"],
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

    rows = bounded_map(evaluate, examples, api.config["concurrency"], api.dispatch_stopped)
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


def audit_proposal(text, examples, initial_prompt, config):
    """Statically check an optimizer proposal before it can become the next meta prompt.

    Rejects a prompt that is longer than `max_meta_prompt_words`, matches PROPOSAL_RED_FLAGS (including leaked
    tool-call markup), names a training source or its authors, or copies `copy_span_words` consecutive words
    from a training source (unless the initial prompt already contains them)."""
    word_count = len(text.split())
    reasons, flags = [], []
    if word_count > config["max_meta_prompt_words"]:
        reasons.append("meta_prompt_word_bound")
    lower = text.casefold()
    for name, pattern in PROPOSAL_RED_FLAGS.items():
        if re.search(pattern, lower):
            reasons.append(name)
    proposal_words, span_words = lower.split(), config["copy_span_words"]
    initial_lower = initial_prompt.casefold()
    for example in examples:
        source = example["source_id"]
        metadata = example["provenance"]
        for identifier in [source, metadata["title"], *metadata.get("authors", [])]:
            if len(identifier) >= config["min_identifier_length"] and identifier.casefold() in lower:
                flags.append({"type": "training_identifier", "source_id": source, "match": identifier})
        for field in ("reference", "context"):
            source_words = example[field].casefold().split()
            source_spans = {tuple(source_words[i : i + span_words]) for i in range(len(source_words) - span_words + 1)}
            for i in range(len(proposal_words) - span_words + 1):
                span = proposal_words[i : i + span_words]
                span_text = " ".join(span)
                if tuple(span) in source_spans and span_text not in initial_lower:
                    flags.append({"type": "copied_training_span", "source_id": source, "match": span_text})
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


def propose_prompt(api, current, feedback, examples, initial, iteration, *, output):
    """Ask the optimizer for the next meta prompt, given the training feedback on the current one.

    A proposal that is malformed or fails audit_proposal is sent back once with the reasons. If
    the repair fails too, the update is used up and the current prompt carries over unchanged.
    Saves the feedback and proposal under feedback/iter_XX/ and returns the next prompt."""
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}"
    write_json(directory / "training.json", feedback)
    data = {"feedback": feedback, "max_meta_prompt_words": api.config["max_meta_prompt_words"]}
    attempts = []
    for attempt in range(api.config["proposal_attempts"]):
        instruction = read_prompt(api.config, "optimizer")
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
            audit = audit_proposal(result["value"]["prompt"], examples, initial, api.config)
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


def build_feedback(examples, candidates, rows, current_prompt, config):
    """The active feedback policy: nonpositive training gaps, without the visible contexts.

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
    for row in by_gap[: config["failure_examples"]]:
        example = lookup[row["example_id"]]
        failures.append(
            {
                "example_id": example["example_id"],
                "kind": example["kind"],
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
        "all_training_gaps": [{"example_id": row["example_id"], "kind": row["kind"], "gap": row["gap"]} for row in by_gap],
    }


def kind_summary(graded, kind):
    gaps = [row["gap"] for row in graded if row["kind"] == kind]
    return {"coverage": len(gaps), "gap": statistics.mean(gaps) if gaps else None}


def summarize(rows):
    """Summarize one checkpoint's rows: mean human, model and gap scores over the examples with
    both grades, and the gap for each kind."""
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
        "length_compliant": sum(row["length_compliant"] for row in rows),
        "kinds": {kind: kind_summary(graded, kind) for kind in sorted({row["kind"] for row in rows})},
    }


def write_report(output, summaries, selected, costs):
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
    result = {
        "selected_iteration": selected,
        "checkpoints": table,
        "summaries": [{"train": train, "validation": validation} for train, validation in summaries],
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
    lines += [
        "",
        (f"OpenRouter-reported cost{' (partial)' if costs['unresolved'] else ''}: "
        f"${costs['actual_complete_usd']:.4f} across {costs['requests']} generations. "
        f"Unresolved lookups or sends: {costs['unresolved']}. Details are in costs.json."),
        "",
        ("This is an independent reproduction with repository-specific sources and prompts. "
        "The confirmation split remains unused."),
        "",
        "Full statistics and coverage are in summary.json; every request and response is saved.",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return result


def run_xar(config_path):
    config_text = Path(config_path).read_text()
    config = yaml.safe_load(config_text)
    initial = read_prompt(config, "rubric_initial")
    if len(initial.split()) > config["max_meta_prompt_words"]:
        raise RunError("Initial prompt exceeds max_meta_prompt_words")
    examples = [json.loads(line) for line in Path(config["dataset"]).read_text().splitlines() if line.strip()]
    examples = run_examples(examples, config)
    train = [example for example in examples if example["split"] == "train"]
    validation = [example for example in examples if example["split"] == "validation"]
    output = Path(config["output"])
    # Atomic directory creation also prevents two processes from starting in the same directory.
    api = OpenRouter(output, config)
    output.mkdir(parents=True, exist_ok=False)
    status = {"state": "failed"}
    try:
        (output / "config.yaml").write_text(config_text)
        write_json(
            output / "manifest.json",
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "config": config,
                "roles": config["roles"],
                "dataset_hash": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest(),
                "initial_meta_prompt_hash": digest(initial),
                "prompt_hashes": {name: digest(read_prompt(config, name)) for name in PROMPTS},
                "feedback_policy": FAILING_FEEDBACK,
                "context_handling": "Providers enforce context limits; full inputs are sent without truncation",
            },
        )
        candidates = writer_candidates(api, examples, output)
        prompts, training_summaries, training_rows = [initial], [], None
        for iteration in range(config["iterations"] + 1):
            if iteration:
                feedback = build_feedback(train, candidates, training_rows, prompts[-1], config)
                prompts.append(propose_prompt(api, prompts[-1], feedback, train, initial, iteration, output=output))
            (output / "prompts").mkdir(exist_ok=True)
            (output / "prompts" / f"iter_{iteration:02d}.md").write_text(prompts[-1])
            training_rows = evaluate_checkpoint(api, train, candidates, prompts[-1], iteration, output)
            summary = summarize(training_rows)
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
        validation_rows = [evaluate_checkpoint(api, validation, candidates, value, index, output) for index, value in enumerate(prompts)]
        validation_summaries = [summarize(rows) for rows in validation_rows]
        result = write_report(
            output, list(zip(training_summaries, validation_summaries, strict=True)), selected, api.costs(),
        )
        status = {"state": "complete" if all(summary["complete"] for summary in validation_summaries) else "incomplete"}
        return result
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "status.json", status)
        api.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(ROOT / "configs/arxiv.yaml"), help="run configuration (default: configs/arxiv.yaml)")
    try:
        run_xar(parser.parse_args().config)
    except (RunError, httpx.HTTPError, FileExistsError) as error:
        raise SystemExit(f"STOP: {error}") from None

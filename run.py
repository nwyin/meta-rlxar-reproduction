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
SCORE_EPSILON = 1e-9


def best_checkpoint(gaps):
    """Choose the earliest maximum, treating floating-point roundoff as a tie."""
    maximum = max(gaps)
    return next(i for i, gap in enumerate(gaps) if maximum - gap <= SCORE_EPSILON)


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


def prompt_names(config):
    return PROMPTS + (("critic",) if config.get("contrastive_feedback") else ())


def run_examples(examples, config):
    """The examples a run uses, labelled train or validation.

    A research run samples within the saved train and validation splits. A pilot run,
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
    fraction = config.get("sample_fraction", 1.0)
    if not 0 < fraction <= 1:
        raise RunError("sample_fraction must be in (0, 1]")
    selected = []
    for split in ("train", "validation"):
        pool = [e for e in examples if e["split"] == split]
        # Round-robin sources before taking sections; preserve the frozen split assignment.
        rng = random.Random(f"{config.get('sample_seed', 0)}:{split}")
        sources = sorted({e["source_id"] for e in pool})
        rng.shuffle(sources)
        groups = {s: [e for e in pool if e["source_id"] == s] for s in sources}
        for group in groups.values():
            rng.shuffle(group)
        ordered = [groups[s][i] for i in range(max(map(len, groups.values()), default=0)) for s in sources if i < len(groups[s])]
        selected.extend(ordered[:max(1, math.floor(len(pool) * fraction + 0.5))])
    if not selected or not any(e["split"] == "train" for e in selected):
        raise RunError("No training examples")
    return selected


# OpenRouter: configured routing, bounded retries, and raw requests and responses.


def routing_fields(cfg):
    """Request fields that pin one role's model, provider and decoding settings."""
    return {
        "model": cfg["model"],
        "stream": False,
        "plugins": [],
        "transforms": [],
        **({"temperature": cfg["temperature"]} if "temperature" in cfg else {}),
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
        self.resume = False

    def costs(self):
        """Read recorded charges, using response usage when a generation lookup is unavailable."""
        generation_ids, missing_ids, response_costs = set(), [], {}
        for path in sorted(self.output.glob("requests/*/attempt_*.json")):
            record = json.loads(path.read_text())
            generation_id = record.get("response_id") or record.get("response", {}).get("id")
            if isinstance(generation_id, str) and generation_id:
                generation_ids.add(generation_id)
                cost = record.get("response", {}).get("usage", {}).get("cost")
                if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
                    response_costs[generation_id] = cost
            elif record["status"] != "connect_error":
                missing_ids.append(str(path.relative_to(self.output)))
        charges, errors, sources = {}, {}, {}
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
                sources[generation_id] = "generation.total_cost"
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                errors[generation_id] = f"{type(error).__name__}: {error}"
                if generation_id in response_costs:
                    charges[generation_id] = response_costs[generation_id]
                    sources[generation_id] = "response.usage.cost"
        result = {
            "source": self.config["api_base"] + "/generation",
            "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
            "actual_complete_usd": math.fsum(charges.values()),
            "requests": len(charges),
            "unresolved": len(generation_ids - charges.keys()) + len(missing_ids),
            "generation_costs": charges,
            "generation_cost_sources": sources,
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

    @staticmethod
    def saved_response(record, key, path):
        raw = record["response"]
        choice = raw["choices"][0]
        return {
            "content": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
            "request_key": key, "response_id": record["response_id"], "usage": raw.get("usage") or {},
            "model": raw.get("model"), "provider": raw.get("provider"), "raw_response": str(path),
        }

    def _call(self, role, system, data, schema, identity):
        payload = self.payload(role, system, data, schema)
        key = digest({"payload": payload, "identity": identity, "schema_version": SCHEMA_VERSION})
        directory = self.output / "requests" / key
        previous = []
        if self.resume and directory.exists():
            saved = json.loads((directory / "request.json").read_text())
            if saved["payload"] != payload or saved["identity"] != identity:
                raise RunError("Cached request does not match resume inputs")
            previous = sorted(directory.glob("attempt_*.json"), key=lambda p: int(p.stem.split("_")[1]))
            for path in previous:
                record = json.loads(path.read_text())
                if record["status"] == "success" and not (path.parent / ("invalid_" + path.name)).exists():
                    return self.saved_response(record, key, path)
        directory.mkdir(parents=True, exist_ok=self.resume)
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
            offset = max((int(p.stem.split("_")[1]) + 1 for p in previous), default=0)
            path = directory / f"attempt_{offset + attempt}.json"
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
                    return self.saved_response(record, key, path)
                error = raw.get("error") or {}
                metadata = error.get("metadata") or {} if isinstance(error, dict) else {}
                transient_budget = response.status_code == 402 and metadata.get("reason") == "in_flight_budget_exhausted"
                if response.status_code not in self.config["retryable_status"] and not transient_budget:
                    raise RunError(f"Unusable HTTP {response.status_code}: {path}")
                retry_after = response.headers.get("Retry-After") or str(metadata.get("headers", {}).get("Retry-After", ""))
                if transient_budget and not retry_after:
                    retry_after = "120"
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
                raw_path = Path(response["raw_response"])
                write_json(raw_path.parent / ("invalid_" + raw_path.name), {"error": str(e)[: self.config["max_repair_error_chars"]]})
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
                    raise InvalidOutput(f"Quote absent from supplied text: {quoted[:180]!r}. Paraphrase without quotation marks or quote exact text.")
                start += len(quote_words(part)) - 1


def writer_candidates(api, examples, output):
    """Write one model section per example, `concurrency` examples at a time.

    Each section gets up to `writer_attempts` drafts: a draft that is cut off or outside the length
    window goes back to the writer with a revision note, and the last draft is kept either way.
    Each section is saved as it finishes, in the dataset's text form (one line, or one paragraph per
    line when the references keep paragraphs). Returns the records keyed by example ID."""
    prompt = read_prompt(api.config, "writer")
    paragraphs = any("\n" in example["reference"] for example in examples)
    writer_identity = {"writer": api.roles["writer"], "prompt": prompt, "schema_version": SCHEMA_VERSION}
    if api.config.get("length_revision_mode"):
        writer_identity["length_revision_mode"] = api.config["length_revision_mode"]
    config_hash = digest(writer_identity)
    stopped = api.dispatch_stopped

    def write_section(example):
        eid = example["example_id"]
        path = output / "generations" / (eid + ".json")
        if api.config.get("reuse_candidates"):
            record = json.loads((Path(api.config["reuse_candidates"]) / "generations" / (eid + ".json")).read_text())
            if record["context_hash"] != example["context_hash"] or record["writer_configuration_hash"] != config_hash:
                raise RunError(f"{eid}: reused draft has different context or writer settings")
            if digest(record["text"]) != record["text_hash"]:
                raise RunError(f"{eid}: reused draft hash mismatch")
            if api.config.get("strict_length") and (not record["length_compliant"] or not record["complete"]):
                raise RunError(f"{eid}: reused draft violates strict length/completion requirements")
            write_json(path, record)
            return record
        low, high = 0.85 * example["target_words"], 1.15 * example["target_words"]
        attempts = []
        for attempt in range(api.config["writer_attempts"]):
            data = task_data(example)
            instructions = prompt
            if attempt:
                previous_words = attempts[-1]["words"]
                data.update(
                    previous_section=attempts[-1]["text"],
                    length_revision=(f"The previous draft contains {previous_words} words by the evaluator's count. "
                                     f"Rewrite it to fit {math.ceil(low)}–{math.floor(high)} words. "
                                     f"Aim for {example['target_words']} words: change the length by approximately "
                                     f"{example['target_words'] - previous_words:+d} words. Preserve the important claims or narrative work. "
                                     "Return only the revised text, without a word-count note."),
                )
                if api.config.get("length_revision_mode") == "edit_only":
                    instructions = (
                        "Edit the supplied draft to satisfy its measured word limit. This is a focused length edit, "
                        "not a fresh writing task. Preserve its central meaning, voice, and useful specific details. "
                        "Do not add new facts, scenes, claims, or explanations. Cut redundancy and secondary material "
                        "when shortening; clarify existing material when expanding. The supplied measured word count "
                        "is authoritative. Words are counted by whitespace. Return ONLY the revised prose, without "
                        "a heading, word-count note, quotation wrapper, or editing commentary."
                    )
                    data = {"draft": attempts[-1]["text"], "measured_words": previous_words,
                            "minimum_words": math.ceil(low), "maximum_words": math.floor(high),
                            "aim_words": round(example["target_words"] * 0.9)}
            if stopped.is_set():
                raise Skipped(f"{eid}: another section failed")
            identity = {"writer": config_hash, "example": eid, "attempt": attempt}
            response = api.call("writer", instructions, data, None, identity)
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
        if api.config.get("strict_length") and (not record["length_compliant"] or not record["complete"]):
            for item in attempts:
                raw_path = Path(item["response"]["raw_response"])
                write_json(raw_path.parent / ("invalid_" + raw_path.name), {"error": "Writer did not satisfy the fixed length window"})
            raise RunError(f"{eid}: writer length/completion failure after {len(attempts)} drafts")
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
    if api.resume and Path(output).exists():
        saved = json.loads(Path(output).read_text())
        if saved["status"] == "valid":
            if (saved["context_hash"] != example["context_hash"] or saved["meta_prompt_hash"] != digest(meta_prompt)
                    or saved["generator_configuration"] != api.roles["rubric"]):
                raise RunError("Saved rubric differs from resume inputs")
            return saved
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
    if api.resume and Path(output).exists():
        saved = json.loads(Path(output).read_text())
        if saved["status"] == "valid":
            if (saved["candidate_hash"] != digest(text) or saved["rubric_hash"] != rubric_record["rubric_hash"]
                    or saved["judge_configuration"] != api.roles["judge"]):
                raise RunError("Saved grade differs from resume inputs")
            return saved
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


def diagnose_feedback(api, feedback, iteration, output):
    """Read each selected training pair in full, then pass compact diagnoses to the optimizer."""
    if any("context" not in pair for pair in feedback["failures"]):
        raise RunError("Contrastive diagnoses require full training contexts")
    instruction = read_prompt(api.config, "critic")
    schema = object_schema({name: TEXT for name in ("quality_differences", "rubric_errors", "proposed_revision", "counterevidence")})
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}" / "diagnoses"
    write_json(directory / "full_training_feedback.json", feedback)

    def diagnose(pair):
        data = {"current_prompt": feedback["current_prompt"], "training_pair": pair}
        identity = {"diagnosis": iteration, "example_id": pair["example_id"]}
        path = directory / f"{pair['example_id']}.json"
        signature = {"input_hash": digest(data), "instruction_hash": digest(instruction), "critic": api.roles["critic"]}
        if api.resume and path.exists():
            saved = json.loads(path.read_text())
            if saved["signature"] != signature:
                raise RunError("Saved diagnosis differs from resume inputs")
            if saved["result"]["status"] == "valid":
                return {"example_id": pair["example_id"], "kind": pair["kind"], "gap": pair["gap"],
                        "diagnosis": saved["result"]["value"]}

        def validate(value):
            if sum(len(text.split()) for text in value.values()) > 600:
                raise InvalidOutput("Keep the entire diagnosis within 600 words")

        result = api.structured("critic", instruction, data, schema, identity, validator=validate)
        write_json(path, {"signature": signature, "result": result})
        if result["status"] != "valid":
            raise RunError("Training diagnoses missing")
        return {"example_id": pair["example_id"], "kind": pair["kind"], "gap": pair["gap"], "diagnosis": result["value"]}

    diagnoses = bounded_map(diagnose, feedback["failures"], api.config["concurrency"], api.dispatch_stopped)
    return {**{key: value for key, value in feedback.items() if key != "failures"},
            "diagnoses": diagnoses, "feedback_format": "independent_full_context_diagnoses_v1"}


def propose_prompt(api, current, feedback, examples, initial, iteration, *, output):
    """Ask the optimizer for the next meta prompt, given the training feedback on the current one.

    A proposal that is malformed or fails audit_proposal is sent back once with the reasons. If
    the repair fails too, the update is used up and the current prompt carries over unchanged.
    Saves the feedback and proposal under feedback/iter_XX/ and returns the next prompt."""
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}"
    if api.resume and (directory / "proposal.json").exists():
        saved = json.loads((directory / "proposal.json").read_text())
        if (saved["parent_prompt_hash"] != digest(current) or saved["feedback_hash"] != digest(feedback)
                or saved["optimizer_configuration"] != api.roles["optimizer"]):
            raise RunError("Saved proposal differs from resume inputs")
        return saved["prompt"]
    write_json(directory / "training.json", feedback)
    data = {"feedback": feedback, "max_meta_prompt_words": api.config["max_meta_prompt_words"]}
    attempts = []
    for attempt in range(api.config["proposal_attempts"]):
        instruction = read_prompt(api.config, "optimizer")
        schema = PROPOSAL_SCHEMA
        if api.config.get("proposal_rationale_first"):
            schema = object_schema({"rationale": TEXT, "prompt": TEXT})
            instruction += (
                "\nFirst write a brief diagnosis in rationale, then write the COMPLETE revised meta-prompt in prompt. "
                "Downstream models receive ONLY prompt; they never receive rationale. Every change you recommend "
                "must be explicitly implemented in prompt. Before returning, check that prompt actually expresses "
                "the distinctions you identified. Do not put a plan for future changes only in rationale."
            )
        if attempt:
            rejected = attempts[-1]
            instruction += "\nBOUNDED REPAIR: Fix these proposal violations: " + ", ".join(rejected["audit"]["reasons"])
            # Show the rejected proposal, or the raw reply if it was not valid JSON.
            previous = rejected["value"] or rejected["attempts"][-1]["response"]["content"]
            data = {**data, "previous_proposal": previous}
        identity = {"proposal": iteration, "bounded_attempt": attempt}
        if api.config.get("proposal_plaintext"):
            instruction += (
                "\nOUTPUT FORMAT FOR THIS RUN: Return ONLY the complete revised meta-prompt as plain text. "
                "This replaces the JSON output instruction above. Do not include a rationale, analysis, JSON, "
                "or code fences. Implement your diagnosis directly in the returned meta-prompt."
            )
            response = api.call("optimizer", instruction, data, None, identity)
            valid = response["finish_reason"] == "stop" and bool(response["content"].strip())
            result = {"status": "valid" if valid else "missing",
                      "value": {"prompt": response["content"].strip(), "rationale": "Plain-text proposal; see raw response."} if valid else None,
                      "attempts": [{"response": response, "status": "valid" if valid else "invalid"}]}
        else:
            result = api.structured("optimizer", instruction, data, schema, identity, repair=False)
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
    """Build training-only feedback under the configured context and example-selection policy.

    all_training_gaps includes the policy's eligible pairs before limiting detailed examples.
    Validation and confirmation never enter this payload.
    """
    if any(item["split"] != "train" for item in [*examples, *rows]):
        raise RunError("Feedback requires training sections")
    if any(row["gap"] is None for row in rows):
        raise RunError("Training grades missing")
    lookup = {example["example_id"]: example for example in examples}
    policy = config.get("feedback_policy", FAILING_FEEDBACK)
    if policy not in (FAILING_FEEDBACK, "full_context", "all_pairs_context"):
        raise RunError(f"Unknown feedback policy: {policy}")
    by_gap = sorted((row for row in rows if row["gap"] <= 0 or policy == "all_pairs_context"),
                    key=lambda row: (row["gap"], row["example_id"]))
    detailed = by_gap
    if config.get("balanced_feedback"):
        groups = [[r for r in by_gap if r["kind"] == kind] for kind in sorted({r["kind"] for r in by_gap})]
        detailed = [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]
    failures = []
    for row in detailed[: config["failure_examples"]]:
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
                **({"context": example["context"]} if policy != FAILING_FEEDBACK else {}),
            }
        )
    return {
        "current_prompt": current_prompt,
        "summary": summarize(rows),
        "failures": failures,
        "example_ids_used_for_aggregate": sorted(lookup),
        "selection": policy,
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


def recheck_selection(api, examples, candidates, prompts, row_history, output):
    """Choose between the two original training leaders using one independent recheck each."""
    if any(e["split"] != "train" for e in examples) or len(prompts) != len(row_history):
        raise RunError("Selection rechecks require matching training inputs")
    if any(r["split"] != "train" or r["gap"] is None for rows in row_history for r in rows):
        raise RunError("Selection rechecks require complete training grades")
    original = [summarize(rows)["gap"] for rows in row_history]
    ranked, shortlist = original.copy(), []
    for _ in range(min(2, len(prompts))):
        index = best_checkpoint(ranked)
        shortlist.append(index)
        ranked[index] = float("-inf")
    entries = []
    for index in sorted(shortlist):
        rows = evaluate_checkpoint(api, examples, candidates, prompts[index], f"selection_{index}", output)
        summary = summarize(rows)
        if not summary["complete"] or summary["examples"] != len(examples):
            raise RunError("Training recheck grades missing")
        entries.append({"checkpoint": index, "original_gap": original[index], "recheck_gap": summary["gap"],
                        "mean_gap": (original[index] + summary["gap"]) / 2})
    selected = entries[best_checkpoint([e["mean_gap"] for e in entries])]["checkpoint"]
    record = {"selected": selected, "candidates": entries, "selection_data": "train"}
    write_json(output / "selection_recheck.json", record)
    return selected, record


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
            "val_evaluated": validation.get("evaluated", True),
            "selected_by_train": index == selected,
        }
        for index, (train, validation) in enumerate(summaries)
    ]
    result = {
        "selected_iteration": selected,
        "validation_checkpoints": [row["iteration"] for row in table if row["val_evaluated"]],
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
        if not row["val_evaluated"]:
            gap = "not evaluated"
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


def evaluate_validation(api, examples, candidates, prompts, selected, output):
    """Score the requested checkpoints after training has frozen the selection."""
    requested = {0, selected} if api.config.get("validation_policy", "all") == "selected" else set(range(len(prompts)))
    summaries = []
    for index, prompt in enumerate(prompts):
        if index in requested:
            summary = summarize(evaluate_checkpoint(api, examples, candidates, prompt, index, output))
            summary["evaluated"] = True
        else:
            summary = {**summarize([]), "evaluated": False, "complete": False, "examples": len(examples)}
        summaries.append(summary)
    return summaries


def run_xar(config_path, *, resume=False):
    config_text = Path(config_path).read_text()
    config = yaml.safe_load(config_text)
    if config.get("validation_policy", "all") not in ("all", "selected"):
        raise RunError("validation_policy must be all or selected")
    if config.get("contrastive_feedback") and ("critic" not in config["roles"]
            or config.get("feedback_policy", FAILING_FEEDBACK) == FAILING_FEEDBACK):
        raise RunError("Contrastive feedback requires a critic role and a full-context feedback policy")
    initial = read_prompt(config, "rubric_initial")
    if len(initial.split()) > config["max_meta_prompt_words"]:
        raise RunError("Initial prompt exceeds max_meta_prompt_words")
    examples = [json.loads(line) for line in Path(config["dataset"]).read_text().splitlines() if line.strip()]
    examples = run_examples(examples, config)
    if config.get("train_only", False):
        examples = [e for e in examples if e["split"] == "train"]
    train = [example for example in examples if example["split"] == "train"]
    validation = [example for example in examples if example["split"] == "validation"]
    output = Path(config["output"])
    # Atomic directory creation also prevents two processes from starting in the same directory.
    api = OpenRouter(output, config)
    api.resume = resume
    if resume:
        old_status = json.loads((output / "status.json").read_text())
        if old_status["state"] not in ("failed", "incomplete"):
            raise RunError("Resume requires a failed or incomplete run")
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["config"] != config or manifest["dataset_hash"] != hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest():
            raise RunError("Resume config or dataset changed")
        if manifest["prompt_hashes"] != {name: digest(read_prompt(config, name)) for name in prompt_names(config)}:
            raise RunError("Resume prompts changed")
    output.mkdir(parents=True, exist_ok=resume)
    lock_path = output / "active.lock"
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock)
    status = {"state": "failed"}
    try:
        if resume:
            missing = {}
            for folder in ("scores", "rubrics"):
                for path in output.glob(f"{folder}/**/*.json"):
                    record = json.loads(path.read_text())
                    if isinstance(record, dict) and record.get("status") == "missing":
                        missing[str(path.relative_to(output))] = record
                        for attempt in record.get("attempts", []):
                            if attempt.get("status") == "invalid":
                                raw_path = Path(attempt["response"]["raw_response"])
                                write_json(raw_path.parent / ("invalid_" + raw_path.name), {"error": attempt.get("error")})
            write_json(output / "resumes" / f"{time.time_ns()}.json", {
                "previous_status": old_status, "code_hash": digest(Path(__file__).read_text()),
                "missing_outputs": missing,
            })
        write_json(output / "status.json", {"state": "running", "pid": os.getpid()})
        (output / "config.yaml").write_text(config_text)
        if not resume:
            write_json(
                output / "manifest.json",
                {
                    "created_at": dt.datetime.now(dt.UTC).isoformat(),
                    "config": config,
                    "roles": config["roles"],
                    "dataset_hash": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest(),
                    "initial_meta_prompt_hash": digest(initial),
                    "prompt_hashes": {name: digest(read_prompt(config, name)) for name in prompt_names(config)},
                    "feedback_policy": config.get("feedback_policy", FAILING_FEEDBACK),
                    "example_ids": [e["example_id"] for e in examples],
                    "code_hash": digest(Path(__file__).read_text()),
                    "context_handling": "Providers enforce context limits; full inputs are sent without truncation",
                },
            )
        for name in prompt_names(config):
            snapshot = output / "input_prompts" / f"{name}.md"
            snapshot.parent.mkdir(exist_ok=True)
            snapshot.write_text(read_prompt(config, name))
        candidates = writer_candidates(api, examples, output)
        prompts, training_summaries, row_history = [initial], [], []
        for iteration in range(config["iterations"] + 1):
            if iteration:
                parent = best_checkpoint([s["gap"] for s in training_summaries]) if config.get("best_parent") else iteration - 1
                feedback = build_feedback(train, candidates, row_history[parent], prompts[parent], config)
                if config.get("contrastive_feedback"):
                    feedback = diagnose_feedback(api, feedback, iteration, output)
                if config.get("optimizer_history"):
                    feedback["history"] = [{"checkpoint": i, "prompt": prompts[i], "summary": training_summaries[i]}
                                           for i in range(iteration)]
                prompts.append(propose_prompt(api, prompts[parent], feedback, train, initial, iteration, output=output))
            (output / "prompts").mkdir(exist_ok=True)
            (output / "prompts" / f"iter_{iteration:02d}.md").write_text(prompts[-1])
            training_rows = evaluate_checkpoint(api, train, candidates, prompts[-1], iteration, output)
            row_history.append(training_rows)
            summary = summarize(training_rows)
            training_summaries.append(summary)
            print(
                f"P{iteration}: training gap {summary['gap']}, coverage {summary['paired_coverage']}/{len(train)}",
                flush=True,
            )
            if not summary["complete"]:
                raise RunError("Training grades missing")
        gaps = [summary["gap"] for summary in training_summaries]
        selected = best_checkpoint(gaps)
        selection_recheck = None
        if config.get("selection_recheck"):
            selected, selection_recheck = recheck_selection(api, train, candidates, prompts, row_history, output)
        write_json(
            output / "freeze.json",
            {
                "selected": selected,
                "validation_policy": config.get("validation_policy", "all"),
                "training_gaps": gaps,
                "prompt_hashes": [digest(value) for value in prompts],
                "frozen_at": dt.datetime.now(dt.UTC).isoformat(),
                **({"selection_recheck": selection_recheck} if selection_recheck is not None else {}),
            },
        )
        if config.get("train_only", False):
            result = {"selected": selected, "training_summaries": training_summaries, "costs": api.costs(), "train_only": True}
            write_json(output / "summary.json", result)
            status = {"state": "complete"}
            return result
        print(f"Training selected P{selected}; scoring validation", flush=True)
        validation_summaries = evaluate_validation(api, validation, candidates, prompts, selected, output)
        result = write_report(
            output, list(zip(training_summaries, validation_summaries, strict=True)), selected, api.costs(),
        )
        status = {"state": "complete" if all(s["complete"] for s in validation_summaries if s["evaluated"]) else "incomplete"}
        return result
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "status.json", status)
        lock_path.unlink()
        api.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(ROOT / "configs/arxiv.yaml"), help="run configuration (default: configs/arxiv.yaml)")
    parser.add_argument("--resume", action="store_true", help="resume a failed run with unchanged config, dataset and prompts")
    try:
        args = parser.parse_args()
        run_xar(args.config, resume=args.resume)
    except (RunError, httpx.HTTPError, FileExistsError) as error:
        raise SystemExit(f"STOP: {error}") from None

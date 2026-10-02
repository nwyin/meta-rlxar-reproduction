"""Shared test helpers: example builders, an on-disk dataset and a fake OpenRouter provider."""

import json

import httpx
import pytest

from xar import runs
from xar.data import SECTIONS
from xar.openrouter import MODEL_CATALOG, SNAPSHOTS, endpoints_filename
from xar.util import canonical, digest, load_design, read_json, words, write_json


@pytest.fixture
def design():
    return load_design()


def dummy_example(paper="paper1", section="abstract", split="train"):
    reference = "Original section " + " ".join(f"fact{i}" for i in range(60))
    context = "Visible paper context. Bibliography: source one, source two. [Missing section]"
    return {
        "example_id": paper + "_" + section,
        "paper_id": paper,
        "section_type": section,
        "split": split,
        "context": context,
        "reference": reference,
        "context_hash": digest(context),
        "reference_hash": digest(reference),
        "target_words": words(reference),
        "provenance": {"title": "A distinctive research title", "authors": ["Unique Author"]},
    }


def rubric():
    return {
        "criteria": [
            {
                "id": str(i),
                "description": "Supported claims",
                "low": "unsupported",
                "middle": "partially supported",
                "high": "fully supported",
            }
            for i in range(4)
        ]
    }


def write_dataset(tmp_path, papers_by_split):
    """Write a dataset with all four sections of each paper, and its splits.

    Returns the dataset and splits paths."""
    examples = [
        dummy_example(paper, section, split)
        for split, papers in papers_by_split.items()
        for paper in papers
        for section in SECTIONS
    ]
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    dataset.write_text("".join(canonical(e) + "\n" for e in examples))
    write_json(splits, {"papers": papers_by_split})
    return dataset, splits


def call_kind(payload):
    """Which step a request is for, going by the fields its response schema asks for."""
    schema = payload.get("response_format", {}).get("json_schema", {}).get("schema", {})
    properties = schema.get("properties", {})
    for kind, field in (("rubric", "criteria"), ("grade", "scores"), ("optimizer", "prompt")):
        if field in properties:
            return kind
    return "writer"


def snapshot_response(request):
    """Serve the saved OpenRouter catalog, or a model's saved endpoint list."""
    if request.url.path.endswith("/endpoints"):
        model = request.url.path.split("/models/")[-1].removesuffix("/endpoints")
        return read_json(SNAPSHOTS / endpoints_filename(model))
    return read_json(MODEL_CATALOG)


# Provider tags from configs/models.yaml and the names OpenRouter reports for them.
PROVIDER_NAMES = {
    "siliconflow/fp8": "SiliconFlow",
    "meta": "Meta",
    "alibaba": "Alibaba",
    "xiaomi/fp8": "Xiaomi",
}


class FakeProvider:
    """Answers OpenRouter requests with valid outputs. Records every chat payload it receives.

    The judge gives the human section 7 and the model section 6 on every criterion, so every
    gap is 1 and every checkpoint ties."""

    def __init__(self):
        self.payloads = []
        self.drop_responses = 0  # chat requests whose response is lost after the send

    def handle(self, request):
        if request.method == "GET":
            return httpx.Response(200, json=snapshot_response(request))
        payload = json.loads(request.content)
        self.payloads.append(payload)
        if self.drop_responses:
            self.drop_responses -= 1
            raise httpx.ReadError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["require_parameters"] is True
        assert len(payload["provider"]["only"]) == 1
        data = json.loads(payload["messages"][1]["content"])
        kind = call_kind(payload)
        if kind == "rubric":
            assert "candidate" not in data and "reference" not in data and "feedback" not in data
            value = rubric()
        elif kind == "grade":
            assert set(data) == {"visible_paper", "section_type", "target_words", "rubric", "candidate"}
            score = 7 if data["candidate"].startswith("Original") else 6
            evidence = "The section gives supported claims."
            value = {"scores": [{"id": str(i), "score": score, "evidence": evidence} for i in range(4)]}
        elif kind == "optimizer":
            value = {
                "prompt": "Assess clear organization and support for specific claims.",
                "rationale": "Clarify criteria.",
            }
        else:
            assert "Write the missing section" in payload["messages"][0]["content"]
            value = " ".join("generated" for _ in range(data["target_words"]))
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": PROVIDER_NAMES[payload["provider"]["only"][0]],
                "id": f"fake-{len(self.payloads)}",
                "usage": {"cost": 0.0001, "prompt_tokens": 100, "completion_tokens": 100},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(value) if isinstance(value, dict) else value,
                            "reasoning": "Not included in candidate text",
                        },
                    }
                ],
            },
        )


def install_fake(monkeypatch):
    """Send the requests of every OpenRouter client that initialize_run creates to a FakeProvider."""
    fake = FakeProvider()
    client = httpx.Client(transport=httpx.MockTransport(fake.handle))
    real = runs.OpenRouter
    # initialize_run looks OpenRouter up in xar.runs, so patch it there.
    monkeypatch.setattr(runs, "OpenRouter", lambda *args, **kwargs: real(*args, **kwargs, client=client))
    return fake

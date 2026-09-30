"""Shared test helpers: example builders, a fake OpenRouter provider and a CLI runner."""

import json

import httpx

from xar import pipeline, runs
from xar.util import ROOT, digest, read_json, words


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


class FakeProvider:
    def __init__(self):
        self.payloads = []

    def handle(self, request):
        if request.method == "GET":
            if request.url.path.endswith("/endpoints"):
                model = request.url.path.split("/models/")[-1].removesuffix("/endpoints")
                return httpx.Response(
                    200,
                    json=read_json(
                        ROOT / "configs/snapshots" / (model.replace("/", "_") + "-endpoints.json")
                    ),
                )
            return httpx.Response(
                200,
                json=read_json(ROOT / "configs/snapshots/openrouter-models-2026-09-29.json"),
            )
        payload = json.loads(request.content)
        self.payloads.append(payload)
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["require_parameters"] is True
        assert len(payload["provider"]["only"]) == 1
        data = json.loads(payload["messages"][1]["content"])
        system = payload["messages"][0]["content"]
        properties = (
            payload.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        )
        if "criteria" in properties:
            assert "candidate" not in data and "reference" not in data and "feedback" not in data
            value = rubric()
        elif "scores" in properties:
            assert set(data) == {"visible_paper", "section_type", "target_words", "rubric", "candidate"}
            value = {
                "scores": [
                    {
                        "id": str(i),
                        "score": 7 if data["candidate"].startswith("Original") else 6,
                        "evidence": "The section gives supported claims.",
                    }
                    for i in range(4)
                ]
            }
        elif "prompt" in properties:
            assert all(f["paper_id"].startswith("pilot0") for f in data["feedback"]["failures"])
            value = {
                "prompt": "Assess clear organization and support for specific claims.",
                "rationale": "Clarify criteria.",
            }
        else:
            assert "Write the missing section" in system
            value = " ".join("generated" for _ in range(data["target_words"]))
        provider = payload["provider"]["only"][0]
        names = {"siliconflow/fp8": "SiliconFlow", "meta": "Meta"}
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": names[provider],
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
    fake = FakeProvider()
    original = runs.OpenRouter
    client = httpx.Client(transport=httpx.MockTransport(fake.handle))

    def factory(*args, **kwargs):
        kwargs["client"] = client
        return original(*args, **kwargs)

    monkeypatch.setattr(runs, "OpenRouter", factory)
    monkeypatch.setattr(pipeline, "initialize_run", runs.initialize_run)
    return fake


def invoke(monkeypatch, module, argv):
    monkeypatch.setattr("sys.argv", [module.__name__ + ".py", *argv])
    module.main()

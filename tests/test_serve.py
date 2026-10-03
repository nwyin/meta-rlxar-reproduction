"""HTTP handling for the viewer's local review endpoints, without opening a port."""

import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def server_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools/data-viewer"))
    return importlib.import_module("serve")


class Socket:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, *args, **kwargs):
        return self.input

    def sendall(self, data):
        self.output.extend(data)


def send(module, method, path, payload=None, origin="http://127.0.0.1:8432", content_type="application/json"):
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = (
        f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:8432\r\nOrigin: {origin}\r\n"
        f"Content-Type: {content_type}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    )
    socket = Socket(headers.encode() + body)
    module.Handler(socket, ("127.0.0.1", 1234), SimpleNamespace())
    header, response = bytes(socket.output).split(b"\r\n\r\n", 1)
    return int(header.split()[1]), response, header


def test_review_get_returns_blind_projection_without_caching(server_module, monkeypatch):
    view = {"complete": False, "pairs": [{"id": "pair-1", "A": "First", "B": "Second"}]}
    monkeypatch.setattr(server_module.review, "get_session", lambda root: view)
    status, response, headers = send(server_module, "GET", "/api/review")
    assert status == 200 and json.loads(response) == view
    assert b"Cache-Control: no-store" in headers


def test_post_routes_answers_and_extraction(server_module, monkeypatch):
    calls = []

    def record(root, payload, *, extraction):
        calls.append((payload, extraction))
        return {"saved": True}

    monkeypatch.setattr(server_module.review, "record", record)
    for path, extraction in [("answer", False), ("extraction", True)]:
        status, response, _ = send(server_module, "POST", f"/api/review/{path}", {"pair_id": "pair-1"})
        assert status == 200 and json.loads(response) == {"saved": True}
        assert calls[-1] == ({"pair_id": "pair-1"}, extraction)


def test_post_rejects_foreign_origin_and_form_submission(server_module, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid request reached the save function")

    monkeypatch.setattr(server_module.review, "record", unexpected)
    assert send(server_module, "POST", "/api/review/answer", {}, origin="https://elsewhere.test")[0] == 403
    assert send(server_module, "POST", "/api/review/answer", {}, content_type="text/plain")[0] == 415


def test_invalid_answer_returns_json_error(server_module, monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("This judgment is locked.")

    monkeypatch.setattr(server_module.review, "record", invalid)
    status, response, _ = send(server_module, "POST", "/api/review/answer", {})
    assert status == 400 and json.loads(response)["error"] == "This judgment is locked."

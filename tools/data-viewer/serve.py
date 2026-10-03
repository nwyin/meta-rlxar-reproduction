"""Serve the data viewer and the run data it reads, on localhost only.

Only the viewer, runs/ and the two dataset files are served, so .env and the rest of the
repository stay private. Usage: uv run python tools/data-viewer/serve.py [--port 8431]
"""

import argparse
import functools
import http.server
import json
from pathlib import Path
from urllib.parse import urlsplit

import review

ROOT = Path(__file__).resolve().parents[2]
VIEWER = "/tools/data-viewer/"
ALLOWED_FOLDERS = (ROOT / "tools" / "data-viewer", ROOT / "runs")
ALLOWED_FILES = {ROOT / "data" / "examples.jsonl", ROOT / "data" / "splits.json"}


def list_runs():
    """Run directories that have scored checkpoints, newest first."""
    runs = []
    for manifest_path in (ROOT / "runs").glob("*/manifest.json"):
        run = manifest_path.parent
        if not (run / "scores" / "main").is_dir():
            continue
        manifest = json.loads(manifest_path.read_text())
        runs.append(
            {
                "name": run.name,
                "complete": (run / "freeze.json").exists(),
                "created_at": manifest.get("created_at", ""),
                "split": manifest.get("arguments", {}).get("split", ""),
            }
        )
    return sorted(runs, key=lambda run: run["created_at"], reverse=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(302)
            self.send_header("Location", VIEWER)
            self.end_headers()
        elif path == "/api/runs":
            self.send_json(list_runs())
        elif path == "/api/review":
            try:
                self.send_json(review.get_session(ROOT))
            except (ValueError, OSError, KeyError) as error:
                self.send_json({"error": str(error)}, 400)
        elif self.allowed(Path(self.translate_path(self.path)).resolve()):
            super().do_GET()
        else:
            self.send_error(404)

    def send_json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/api/review/answer", "/api/review/extraction"):
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_error(403)
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send_error(415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid answer size.")
            payload = json.loads(self.rfile.read(length))
            self.send_json(review.record(ROOT, payload, extraction=self.path.endswith("/extraction")))
        except (TypeError, ValueError, OSError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)

    @staticmethod
    def allowed(target):
        """Check the decoded, resolved file path, so encoded "../" tricks can't reach other files."""
        if target in ALLOWED_FILES:
            return True
        return any(target == folder or folder in target.parents for folder in ALLOWED_FOLDERS)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8431)
    port = parser.parse_args().port
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Data viewer: http://127.0.0.1:{port}{VIEWER}")
        server.serve_forever()


if __name__ == "__main__":
    main()

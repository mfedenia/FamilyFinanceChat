"""Deterministic OpenAI-compatible provider for the CI test stack."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ok"})
        elif self.path in ("/models", "/v1/models"):
            self.respond(
                200,
                {"object": "list", "data": [{"id": CHAT_MODEL, "object": "model", "owned_by": "ci"}]},
            )
        else:
            self.respond(404, {"error": {"message": "Unknown endpoint"}})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
        except ValueError:
            self.respond(400, {"error": {"message": "Invalid JSON"}})
            return

        if self.path in ("/embeddings", "/v1/embeddings"):
            if payload.get("model") != EMBEDDING_MODEL:
                self.respond(404, {"error": {"message": "Unknown embedding model"}})
                return
            inputs = payload.get("input", [])
            if not isinstance(inputs, list):
                inputs = [inputs]
            self.respond(
                200,
                {
                    "object": "list",
                    "model": EMBEDDING_MODEL,
                    "data": [
                        {"object": "embedding", "index": index, "embedding": [1.0, 0.0, 0.0]}
                        for index, _ in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
                },
            )
        elif self.path in ("/chat/completions", "/v1/chat/completions"):
            if payload.get("model") != CHAT_MODEL:
                self.respond(
                    404,
                    {"error": {"message": f"Model {payload.get('model')} does not exist", "code": "model_not_found"}},
                )
                return
            self.respond(
                200,
                {
                    "id": "chatcmpl-ci-smoke",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": CHAT_MODEL,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Hello from CI."}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
                },
            )
        else:
            self.respond(404, {"error": {"message": "Unknown endpoint"}})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()

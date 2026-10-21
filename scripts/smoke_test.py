"""Check Open WebUI chat, document upload, and knowledge retrieval.

By default, requires a fresh stack whose first signup becomes admin.
Use --existing-token-env for an existing instance. Both modes create test resources.
"""

import argparse
import json
import os
import secrets
import sys
import time

import requests


def fail(message):
    raise RuntimeError(message)


def api(session, method, base_url, path, **kwargs):
    try:
        response = session.request(method, base_url + path, timeout=120, **kwargs)
    except requests.RequestException as exc:
        fail(f"{method} {path} request failed ({type(exc).__name__})")
    if not response.ok:
        # An upstream error body can echo credentials or private data.
        fail(f"{method} {path} returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"{method} {path} did not return JSON") from exc


def wait_for_endpoint(session, base_url, path, deadline):
    print(f"Waiting for {path}...", flush=True)
    while time.monotonic() < deadline:
        try:
            response = session.get(base_url + path, timeout=5)
            if response.ok:
                print(f"{path}: HTTP {response.status_code}", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    fail(f"{path} did not become ready before the deadline")


def record_resource(kind, resource_id):
    print("Smoke resource: " + json.dumps({"kind": kind, "id": resource_id}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:3003")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--existing-token-env",
        metavar="ENV_NAME",
        help="Read an existing user's bearer token from this environment variable; skip signup",
    )
    args = parser.parse_args(argv)
    token = None
    if args.existing_token_env is not None:
        token = os.environ.get(args.existing_token_env, "").strip()
        if not token:
            fail("The environment variable selected by --existing-token-env is missing or empty")
    base_url = args.base_url.rstrip("/")
    session = requests.Session()

    deadline = time.monotonic() + args.startup_timeout
    wait_for_endpoint(session, base_url, "/health", deadline)
    wait_for_endpoint(session, base_url, "/ready", deadline)

    run_id = secrets.token_hex(8)
    print(f"Smoke run: {run_id}", flush=True)
    if args.existing_token_env is None:
        signup = api(
            session,
            "POST",
            base_url,
            "/api/v1/auths/signup",
            json={
                "name": "CI Smoke Test",
                "email": f"smoke-{run_id}@example.invalid",
                "password": secrets.token_urlsafe(32),
            },
        )
        if signup.get("id"):
            record_resource("user", signup["id"])
        token = signup.get("token")
        if not token or signup.get("role") != "admin":
            fail("First-user signup did not return an admin bearer token")
        print("Admin signup: passed", flush=True)
    else:
        print("Using existing bearer token; signup skipped", flush=True)
    session.headers.update({"Authorization": f"Bearer {token}"})

    completion = api(
        session,
        "POST",
        base_url,
        "/api/chat/completions",
        json={
            "model": args.model,
            "chat_id": f"smoke-test-{run_id}",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with one short greeting."
                }
            ],
        },
    )

    choices = completion.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        fail("Chat completion contained no assistant text")
    print("Chat completion: non-empty response", flush=True)

    marker = f"CI-RETRIEVAL-{run_id}"
    document = (
        "FamilyFinanceChat smoke test document. "
        f"The unique verification marker for this run is {marker}. "
        "A successful knowledge base query must retrieve this exact sentence.\n"
    )
    knowledge = api(
        session,
        "POST",
        base_url,
        "/api/v1/knowledge/create",
        json={"name": f"CI Smoke {run_id}", "description": "Temporary CI retrieval check"},
    )
    kb_id = knowledge.get("id")
    if not kb_id:
        fail("Knowledge base creation returned no id")
    record_resource("knowledge", kb_id)

    uploaded = api(
        session,
        "POST",
        base_url,
        "/api/v1/files/?process=true&process_in_background=false",
        files={"file": (f"smoke-{run_id}.txt", document.encode("utf-8"), "text/plain")},
    )
    file_id = uploaded.get("id")
    if not file_id:
        fail("File upload returned no id")
    record_resource("file", file_id)
    status = api(session, "GET", base_url, f"/api/v1/files/{file_id}/process/status")
    if status.get("status") != "completed":
        fail("File processing did not complete")
    print("Document upload and processing: passed", flush=True)

    attached = api(
        session,
        "POST",
        base_url,
        f"/api/v1/knowledge/{kb_id}/file/add",
        json={"file_id": file_id},
    )
    if not any(item.get("id") == file_id for item in attached.get("files") or []):
        fail("Uploaded file was not attached to the knowledge base")

    retrieved = api(
        session,
        "POST",
        base_url,
        "/api/v1/retrieval/query/collection",
        json={"collection_names": [kb_id], "query": f"verification marker {marker}", "k": 5},
    )
    documents = retrieved.get("documents") or []
    if not any(marker in document for group in documents for document in group if isinstance(document, str)):
        fail("Knowledge base retrieval did not return the uploaded document marker")
    print("Knowledge base retrieval: uploaded marker found", flush=True)
    print("Smoke test passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, requests.RequestException) as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)

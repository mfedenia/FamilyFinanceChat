"""Inspect and optionally remove one smoke run's file and knowledge base.

Use the resource IDs printed by smoke_test.py. Deletion requires --execute and
exact resource-name and owner checks; this script never deletes users or chats.
"""

import argparse
import os
import re
import sys

import requests


def api(session, method, base_url, path):
    try:
        response = session.request(method, base_url + path, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"{method} {path} request failed ({type(exc).__name__})") from None
    if not response.ok:
        raise RuntimeError(f"{method} {path} returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise RuntimeError(f"{method} {path} returned invalid JSON") from None


def inspect_resources(session, base_url, run_id, knowledge_id, file_id):
    knowledge = api(session, "GET", base_url, f"/api/v1/knowledge/{knowledge_id}")
    uploaded = api(session, "GET", base_url, f"/api/v1/files/{file_id}")
    if knowledge.get("id") != knowledge_id or knowledge.get("name") != f"CI Smoke {run_id}":
        raise RuntimeError("Knowledge base ID or name does not match this smoke run")
    if uploaded.get("id") != file_id or uploaded.get("filename") != f"smoke-{run_id}.txt":
        raise RuntimeError("File ID or name does not match this smoke run")
    owner = knowledge.get("user_id")
    if not owner or owner != uploaded.get("user_id"):
        raise RuntimeError("Knowledge base and file owners do not match")
    files = knowledge.get("files") or []
    if any(not isinstance(item, dict) or item.get("id") != file_id for item in files):
        raise RuntimeError("Knowledge base contains a file from another run")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--knowledge-id", required=True)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--token-env", required=True, help="Environment variable containing a bearer token")
    parser.add_argument("--execute", action="store_true", help="Actually delete the verified resources")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{16}", args.run_id):
        raise RuntimeError("Run ID must be the 16-character hex value printed by smoke_test.py")
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise RuntimeError("Selected token environment variable is missing or empty")
    base_url = args.base_url.rstrip("/")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    inspect_resources(session, base_url, args.run_id, args.knowledge_id, args.file_id)
    print("Verified smoke knowledge base and file IDs, names and owner")
    if not args.execute:
        print("Dry run only; pass --execute to remove these resources")
        return
    api(session, "DELETE", base_url, f"/api/v1/files/{args.file_id}")
    api(session, "DELETE", base_url, f"/api/v1/knowledge/{args.knowledge_id}/delete")
    print("Smoke file and knowledge base removed")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Smoke cleanup failed: {exc}", file=sys.stderr)
        sys.exit(1)

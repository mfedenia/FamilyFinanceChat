#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
ENV_FILE="$SCRIPT_DIR/.env"
REFRESH_PORT="9550"
RUN_API_TEST="${RUN_API_TEST:-1}"

make_temp_dir() {
  mktemp -d "${TMPDIR:-/tmp}/grading-refresh.XXXXXX"
}

find_free_port() {
  "$PYTHON_EXECUTABLE" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

detect_python() {
  if [[ -x "$BACKEND_DIR/venv/bin/python" ]]; then
    echo "$BACKEND_DIR/venv/bin/python"
    return
  fi
  if command -v py >/dev/null 2>&1; then
    echo "py -3"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi

  echo ""
}

PYTHON_CMD="$(detect_python)"
if [[ -z "$PYTHON_CMD" ]]; then
  echo "ERROR: Python was not found (tried: py -3, python3, python)."
  echo "Install Python or add it to PATH, then re-run this script."
  exit 1
fi

PYTHON_EXECUTABLE="${PYTHON_CMD%% *}"

if [[ -f "$ENV_FILE" ]]; then
  INVALID_ASSIGNMENTS="$(grep -nE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]+=' "$ENV_FILE" || true)"
  if [[ -n "$INVALID_ASSIGNMENTS" ]]; then
    echo "ERROR: Invalid .env assignment syntax detected in $ENV_FILE"
    echo "Use VAR=value (no spaces around =)."
    echo "$INVALID_ASSIGNMENTS"
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${OPENWEBUI_BASE_URL:-}" ]]; then
  echo "ERROR: OPENWEBUI_BASE_URL is not set. Put OPENWEBUI_BASE_URL in grading_feature/.env"
  exit 1
fi

if [[ -z "${OUTPUT_PATH:-}" ]]; then
  echo "ERROR: OUTPUT_PATH is not set. Put OUTPUT_PATH in grading_feature/.env"
  exit 1
fi

export DATA_PATH="${DATA_PATH:-$OUTPUT_PATH}"

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
if [[ -e "$OUTPUT_PATH" && ! -w "$OUTPUT_PATH" ]]; then
  echo "ERROR: OUTPUT_PATH exists but is not writable: $OUTPUT_PATH"
  echo "Fix permissions or choose a writable OUTPUT_PATH in $ENV_FILE"
  exit 1
fi
if [[ ! -e "$OUTPUT_PATH" && ! -w "$OUTPUT_DIR" ]]; then
  echo "ERROR: OUTPUT_PATH directory is not writable: $OUTPUT_DIR"
  echo "Fix permissions or choose a writable OUTPUT_PATH in $ENV_FILE"
  exit 1
fi

PRE_MTIME=""
if [[ -e "$OUTPUT_PATH" ]]; then
  PRE_MTIME="$(stat -c %Y "$OUTPUT_PATH" 2>/dev/null || stat -f %m "$OUTPUT_PATH" 2>/dev/null || true)"
fi

echo "==> Syntax check"
$PYTHON_CMD -m py_compile "$BACKEND_DIR/main.py" "$BACKEND_DIR/extract_chats.py"

echo "==> Running extractor directly"
(
  cd "$BACKEND_DIR"
  $PYTHON_CMD extract_chats.py
)

echo "==> Verifying extractor metadata return"
(
  cd "$BACKEND_DIR"
  $PYTHON_CMD - <<'PY'
import json
import extract_chats

print(f"extract_chats module path: {extract_chats.__file__}")
metadata = extract_chats.main()
if not isinstance(metadata, dict):
    raise SystemExit(
        f"ERROR: extract_chats.main() returned {type(metadata).__name__}, expected dict metadata"
    )

required = [
    "users_processed",
    "chat_entries_processed",
    "message_pairs_processed",
    "latest_message_timestamp_found",
    "output_file_path",
]
missing = [k for k in required if k not in metadata]
if missing:
    raise SystemExit(f"ERROR: extract_chats.main() metadata is missing keys: {missing}")

print("extract_chats.main() metadata validated:")
print(json.dumps({k: metadata.get(k) for k in required}, indent=2))
PY
)

if [[ ! -s "$OUTPUT_PATH" ]]; then
  echo "ERROR: OUTPUT_PATH file was not created or is empty: $OUTPUT_PATH"
  exit 1
fi

POST_MTIME="$(stat -c %Y "$OUTPUT_PATH" 2>/dev/null || stat -f %m "$OUTPUT_PATH" 2>/dev/null || true)"
if [[ -n "$PRE_MTIME" && -n "$POST_MTIME" && "$PRE_MTIME" == "$POST_MTIME" ]]; then
  echo "ERROR: OUTPUT_PATH timestamp did not change after extractor run: $OUTPUT_PATH"
  echo "This usually means extraction did not write new data (often permission or extractor failure)."
  exit 1
fi

echo "==> Validating output JSON"
$PYTHON_CMD -m json.tool "$OUTPUT_PATH" >/dev/null

echo "Extractor output looks valid."

echo "==> OUTPUT_PATH: $OUTPUT_PATH"
ls -l "$OUTPUT_PATH"

if [[ "$RUN_API_TEST" != "1" ]]; then
  echo "Skipping /refresh API test (RUN_API_TEST=$RUN_API_TEST)."
  echo "PASS"
  exit 0
fi

TEST_PORT="$(find_free_port)"
echo "==> Using free refresh test port: $TEST_PORT"

echo "==> Starting API and testing /refresh"
TEST_TMPDIR="$(make_temp_dir)"
API_LOG="$TEST_TMPDIR/grading_refresh_api.log"
REFRESH_BODY="$TEST_TMPDIR/grading_refresh_response.json"

(
  cd "$BACKEND_DIR"
  $PYTHON_CMD -m uvicorn main:app --host 127.0.0.1 --port "$TEST_PORT" >"$API_LOG" 2>&1
) &
API_PID=$!

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
  wait "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2

if ! kill -0 "$API_PID" >/dev/null 2>&1; then
  echo "ERROR: test API process failed to start (pid $API_PID)."
  echo "API log: $API_LOG"
  if [[ -f "$API_LOG" ]]; then
    tail -n 100 "$API_LOG"
  fi
  exit 1
fi

echo "==> Testing asynchronous /refresh and /refresh/status"
STATUS_BODY="$TEST_TMPDIR/grading_status_response.json"

HTTP_CODE="$(curl -sS -o "$REFRESH_BODY" -w "%{http_code}" -X POST "http://127.0.0.1:${TEST_PORT}/refresh")"

if [[ "$HTTP_CODE" != "202" ]]; then
  echo "ERROR: POST /refresh returned HTTP $HTTP_CODE (expected 202)"
  echo "Response:"
  cat "$REFRESH_BODY"
  echo
  echo "API log: $API_LOG"
  exit 1
fi

CONFLICT_CODE="$(curl -sS -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${TEST_PORT}/refresh" || true)"
if [[ "$CONFLICT_CODE" == "409" ]]; then
  echo "Verified re-run lock: concurrent POST /refresh returned 409 Conflict."
fi

echo "==> Polling /refresh/status until completion"
$PYTHON_CMD - "$TEST_PORT" <<'PY'
import json
import sys
import time
import urllib.request

port = sys.argv[1]
status_url = f"http://127.0.0.1:{port}/refresh/status"

start = time.time()
while time.time() - start < 30:
    req = urllib.request.Request(status_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status = data.get("status")
            progress = data.get("progress")
            stage = data.get("stage")
            print(f"Status poll: {status} ({progress}%, stage={stage})")
            if status == "completed":
                meta = data.get("refresh_metadata")
                if not meta or not isinstance(meta, dict):
                    raise SystemExit("ERROR: Completed refresh missing refresh_metadata")
                required = [
                    "users_processed",
                    "chat_entries_processed",
                    "message_pairs_processed",
                    "latest_message_timestamp_found",
                    "output_file_path",
                ]
                missing = [k for k in required if k not in meta]
                if missing:
                    raise SystemExit(f"ERROR: Missing refresh_metadata keys: {missing}")
                print("Asynchronous refresh completed and validated metadata successfully.")
                sys.exit(0)
            elif status == "failed":
                raise SystemExit(f"ERROR: Refresh job failed: {data.get('error')}")
    except Exception as e:
        print(f"Polling error: {e}")
    time.sleep(0.5)

raise SystemExit("ERROR: Timed out waiting for refresh job to complete.")
PY

echo "==> Testing legacy/sync /refresh?sync=true"
SYNC_HTTP_CODE="$(curl -sS -o "$REFRESH_BODY" -w "%{http_code}" "http://127.0.0.1:${TEST_PORT}/refresh?sync=true")"
if [[ "$SYNC_HTTP_CODE" != "200" ]]; then
  echo "ERROR: /refresh?sync=true returned HTTP $SYNC_HTTP_CODE"
  cat "$REFRESH_BODY"
  exit 1
fi


echo "PASS"

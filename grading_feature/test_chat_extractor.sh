#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
ENV_FILE="$SCRIPT_DIR/.env"
REFRESH_PORT="9550"
RUN_API_TEST="${RUN_API_TEST:-1}"

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

if [[ -z "${DB_PATH:-}" ]]; then
  echo "ERROR: DB_PATH is not set. Put DB_PATH in grading_feature/.env"
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

TEST_PORT="$REFRESH_PORT"

echo "==> Starting API and testing /refresh"
API_LOG="${TMPDIR:-/tmp}/grading_refresh_api.log"
REFRESH_BODY="${TMPDIR:-/tmp}/grading_refresh_response.json"

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

HTTP_CODE="$(curl -sS -o "$REFRESH_BODY" -w "%{http_code}" "http://127.0.0.1:${TEST_PORT}/refresh")"

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "ERROR: /refresh returned HTTP $HTTP_CODE"
  echo "Response:"
  cat "$REFRESH_BODY"
  echo
  echo "API log: $API_LOG"
  exit 1
fi

$PYTHON_CMD - "$REFRESH_BODY" <<'PY'
import json
import sys

body_path = sys.argv[1]
with open(body_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if payload is None:
  raise SystemExit("ERROR: /refresh returned JSON null. Check API logs.")

if not isinstance(payload, dict):
  raise SystemExit(
    "ERROR: /refresh returned non-object JSON. Full payload: "
    + json.dumps(payload, indent=2)
  )

if payload.get("status") != "success":
  raise SystemExit(
    "ERROR: /refresh did not return status=success. Full payload: "
    + json.dumps(payload, indent=2)
  )

meta = payload.get("refresh_metadata")
if meta is None or not isinstance(meta, dict):
  raise SystemExit(
    "ERROR: /refresh payload is missing refresh_metadata. Full payload: "
    + json.dumps(payload, indent=2)
  )

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

print("/refresh success payload validated.")
print(json.dumps({k: meta.get(k) for k in required}, indent=2))
PY

echo "PASS"

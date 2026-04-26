#!/usr/bin/env bash
# Backfill historical Prometheus metrics from openwebui_history.json.
#
# What this does:
#   1. Converts openwebui_history.json → OpenMetrics text format (Python)
#   2. Stops the Prometheus container
#   3. Runs a temporary promtool container (same image) that writes TSDB block
#      files directly into the Prometheus data volume
#   4. Restarts Prometheus — it loads all blocks including the new ones on boot
#   5. Verifies Prometheus is healthy
#
# Prerequisites:
#   - Docker running on this machine with the stack already deployed
#   - Python 3 in PATH
#   - openwebui_history.json present in the repo root
#
# Configurable via environment variables:
#   PROMETHEUS_DATA_DIR   host path of the Prometheus data volume
#                         (default: /opt/prometheus/data)
#   PROMETHEUS_URL        health-check URL  (default: http://localhost:9090)
#   PROMETHEUS_CONTAINER  container name    (default: prometheus)
#   PROMETHEUS_IMAGE      Docker image      (default: prom/prometheus:v2.51.2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROMETHEUS_DATA_DIR="${PROMETHEUS_DATA_DIR:-/opt/prometheus/data}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
PROMETHEUS_CONTAINER="${PROMETHEUS_CONTAINER:-prometheus}"
PROMETHEUS_IMAGE="${PROMETHEUS_IMAGE:-prom/prometheus:v2.51.2}"
OPENMETRICS_FILE="$REPO_ROOT/backfill_history.openmetrics"

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

wait_ready() {
    log "Waiting for Prometheus to be ready..."
    for i in $(seq 1 30); do
        if curl -sf "$PROMETHEUS_URL/-/ready" > /dev/null 2>&1; then
            log "Prometheus is ready."
            return 0
        fi
        echo "  attempt $i/30 — retrying in 2 s"
        sleep 2
    done
    die "Prometheus did not become ready within 60 s"
}

# ── preflight checks ──────────────────────────────────────────────────────────

cd "$REPO_ROOT"

[[ -f openwebui_history.json ]] || die "openwebui_history.json not found in $REPO_ROOT"
[[ -d "$PROMETHEUS_DATA_DIR" ]] || die "Prometheus data dir not found: $PROMETHEUS_DATA_DIR"
command -v python3 > /dev/null || die "python3 not found"
command -v docker  > /dev/null || die "docker not found"

log "Prometheus data dir : $PROMETHEUS_DATA_DIR"
log "Prometheus URL      : $PROMETHEUS_URL"
log "Prometheus container: $PROMETHEUS_CONTAINER"
log "Prometheus image    : $PROMETHEUS_IMAGE"

# ── step 1: convert JSON → OpenMetrics ───────────────────────────────────────

log "=== Step 1: Convert openwebui_history.json → OpenMetrics ==="
python3 scripts/backfill_history.py
[[ -f "$OPENMETRICS_FILE" ]] || die "Conversion produced no output file"

# ── step 2: stop Prometheus ───────────────────────────────────────────────────

log "=== Step 2: Stop Prometheus ==="
docker stop "$PROMETHEUS_CONTAINER"
log "Prometheus stopped."

# ── step 3: create TSDB blocks ────────────────────────────────────────────────

log "=== Step 3: Create TSDB blocks via promtool ==="
log "This may take a minute for 5+ million samples..."

docker run --rm \
    --entrypoint /bin/promtool \
    --user 65534:65534 \
    -v "$PROMETHEUS_DATA_DIR":/prometheus \
    -v "$OPENMETRICS_FILE":/tmp/history.openmetrics:ro \
    "$PROMETHEUS_IMAGE" \
    tsdb create-blocks-from openmetrics \
        /tmp/history.openmetrics \
        /prometheus

log "TSDB blocks written to $PROMETHEUS_DATA_DIR"

# ── step 4: restart Prometheus ────────────────────────────────────────────────

log "=== Step 4: Start Prometheus ==="
docker start "$PROMETHEUS_CONTAINER"
wait_ready

# ── step 5: verify ────────────────────────────────────────────────────────────

log "=== Step 5: Verify import ==="

# Query for the earliest timestamp of any openwebui metric to confirm history loaded
QUERY='min(min_over_time(openwebui_api_errors_total[35d]))'
RESULT=$(curl -sf \
    "$PROMETHEUS_URL/api/v1/query?query=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")" \
    | python3 -c "
import json, sys, datetime
d = json.load(sys.stdin)
r = d.get('data',{}).get('result',[])
if r:
    ts = float(r[0]['value'][0])
    print(datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime('%Y-%m-%d %H:%M UTC'))
else:
    print('no openwebui metrics found — check metric names')
" 2>/dev/null || echo "query failed — check Prometheus logs")

log "Earliest openwebui data visible in Prometheus: $RESULT"
log ""
log "Import complete."
log "  Grafana  → http://localhost:3001  (add/confirm Prometheus data source at http://prometheus:9090)"
log "  Prometheus → $PROMETHEUS_URL"
log ""
log "NOTE: Prometheus retention is 30 days. Samples older than that will be"
log "  purged by Prometheus on the next compaction cycle."

#!/usr/bin/env bash
# A20: external uptime check on Open WebUI's /ready, alerting a named person.
#
# /ready returns 200 only when both the database and Redis are up. This check runs
# from Google's checkers outside the VM, so it is the alert that still fires when
# the whole VM (and therefore Grafana) is down.
#
# Creates, or reuses by display name, three Cloud Monitoring objects:
#   1. an email notification channel for ALERT_EMAIL
#   2. an uptime check: GET <protocol>://<host>:<port>/ready every 60s, expect 200
#   3. an alert policy: fires when checkers in 2+ regions fail for 60s
# Safe to re-run: existing objects are reported and left unchanged.
#
# Usage:
#   GCP_PROJECT=my-project UPTIME_HOST=34.1.2.3 ALERT_EMAIL=person@example.edu \
#     bash scripts/gcp_uptime_alert.sh
# Optional: UPTIME_PORT (default 3000), UPTIME_PROTOCOL (http|https, default http).
# After A16 puts TLS in front: UPTIME_HOST=<hostname> UPTIME_PORT=443 UPTIME_PROTOCOL=https
#
# Needs: gcloud (authenticated, with roles/monitoring.editor), curl, python3.
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${UPTIME_HOST:?Set UPTIME_HOST to the VM IP or hostname}"
: "${ALERT_EMAIL:?Set ALERT_EMAIL to the person who gets paged}"
UPTIME_PORT="${UPTIME_PORT:-3000}"
UPTIME_PROTOCOL="${UPTIME_PROTOCOL:-http}"

NAME="FamilyFinanceChat /ready"
API="https://monitoring.googleapis.com/v3/projects/${GCP_PROJECT}"
TOKEN="$(gcloud auth print-access-token)"

api() { # api METHOD PATH [JSON-BODY]
  curl -sS --fail-with-body -X "$1" "${API}/$2" \
    -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
    ${3:+--data "$3"}
}

# Print the "name" of the first item in list field $1 whose displayName is $2.
find_by_display_name() {
  python3 -c '
import json, sys
field, wanted = sys.argv[1], sys.argv[2]
for item in json.load(sys.stdin).get(field, []):
    if item.get("displayName") == wanted:
        print(item["name"]); break
' "$1" "$2"
}

field() { python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

echo "Preflight: ${UPTIME_PROTOCOL}://${UPTIME_HOST}:${UPTIME_PORT}/ready from this machine"
if ! curl -fsS -m 10 "${UPTIME_PROTOCOL}://${UPTIME_HOST}:${UPTIME_PORT}/ready" >/dev/null; then
  echo "WARNING: /ready is not reachable from here. The check will fail until it is." >&2
fi

# 1. Notification channel
CHANNEL_NAME="Alerting: ${ALERT_EMAIL}"
CHANNEL=$(api GET notificationChannels | find_by_display_name notificationChannels "$CHANNEL_NAME")
if [ -z "$CHANNEL" ]; then
  CHANNEL=$(api POST notificationChannels "$(python3 -c '
import json, sys
print(json.dumps({"type": "email", "displayName": sys.argv[1],
                  "labels": {"email_address": sys.argv[2]}}))' "$CHANNEL_NAME" "$ALERT_EMAIL")" | field name)
  echo "Created notification channel ${CHANNEL}"
else
  echo "Reusing notification channel ${CHANNEL}"
fi

# 2. Uptime check
CHECK=$(api GET uptimeCheckConfigs | find_by_display_name uptimeCheckConfigs "$NAME")
if [ -z "$CHECK" ]; then
  CHECK=$(api POST uptimeCheckConfigs "$(python3 -c '
import json, sys
name, project, host, port, proto = sys.argv[1:6]
print(json.dumps({
    "displayName": name,
    "monitoredResource": {"type": "uptime_url",
                          "labels": {"project_id": project, "host": host}},
    "httpCheck": {"requestMethod": "GET", "path": "/ready", "port": int(port),
                  "useSsl": proto == "https", "validateSsl": proto == "https",
                  "acceptedResponseStatusCodes": [{"statusValue": 200}]},
    "period": "60s",
    "timeout": "10s",
}))' "$NAME" "$GCP_PROJECT" "$UPTIME_HOST" "$UPTIME_PORT" "$UPTIME_PROTOCOL")" | field name)
  echo "Created uptime check ${CHECK}"
else
  echo "Reusing uptime check ${CHECK} (edit or delete it in the console to change host/port)"
fi
CHECK_ID="${CHECK##*/}"

# 3. Alert policy
POLICY=$(api GET alertPolicies | find_by_display_name alertPolicies "$NAME down")
if [ -z "$POLICY" ]; then
  POLICY=$(api POST alertPolicies "$(python3 -c '
import json, sys
name, check_id, channel = sys.argv[1:4]
print(json.dumps({
    "displayName": name + " down",
    "combiner": "OR",
    "conditions": [{
        "displayName": "/ready failing from 2+ regions",
        "conditionThreshold": {
            "filter": ("metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" "
                       "AND resource.type=\"uptime_url\" "
                       f"AND metric.label.check_id=\"{check_id}\""),
            "aggregations": [{
                "alignmentPeriod": "60s",
                "perSeriesAligner": "ALIGN_NEXT_OLDER",
                "crossSeriesReducer": "REDUCE_COUNT_FALSE",
                "groupByFields": ["resource.label.project_id", "resource.label.host"],
            }],
            "comparison": "COMPARISON_GT",
            "thresholdValue": 1,
            "duration": "60s",
            "trigger": {"count": 1},
        },
    }],
    "notificationChannels": [channel],
    "alertStrategy": {"autoClose": "1800s"},
    "documentation": {
        "mimeType": "text/markdown",
        "content": ("FamilyFinanceChat is down for students: Open WebUI /ready is not "
                    "returning 200 (database or Redis down, or the VM is unreachable).\n\n"
                    "On the VM: `docker compose ps`, then `docker logs --tail 200 open-webui` "
                    "and `docker logs --tail 50 redis`."),
    },
}))' "$NAME" "$CHECK_ID" "$CHANNEL")" | field name)
  echo "Created alert policy ${POLICY}"
else
  echo "Reusing alert policy ${POLICY}"
fi

cat <<EOF

Done. Google's checkers must be able to reach port ${UPTIME_PORT} on the VM.
If the VM firewall restricts that port, allow the checker ranges listed by:
  gcloud monitoring uptime list-ips
Then run the A20 acceptance test in SETUP.md (stop redis, expect an email within 5 minutes).
EOF

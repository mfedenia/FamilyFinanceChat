#!/usr/bin/env bash
# Back up the production FamilyFinanceChat stack without stopping it (task A2).
#
# What this does:
#   1. Copies Open WebUI's SQLite database with SQLite's online .backup and
#      runs an integrity check on the copy
#   2. Archives the rest of the Open WebUI data directory, minus the model
#      cache and old hand-made webui.db.* copies
#   3. Archives the uploaded course files
#   4. Records Qdrant point counts, takes a Qdrant full snapshot through its
#      API, and copies the snapshot out of the container
#   5. Copies Grafana's database (online .backup) and archives the rest
#   6. Copies the production .env
#   7. Writes MANIFEST.txt (versions, point counts, exclusions) and SHA256SUMS
#
# Read-only against production: nothing is stopped, restarted or modified.
# Every existing file is only read; databases are opened read-only. The one
# write outside the backup folder is Qdrant's own temporary snapshot file inside
# its container, which this script deletes again, even when it fails.
# Output goes to $BACKUP_ROOT/<timestamp>.partial and is renamed to
# $BACKUP_ROOT/<timestamp> only when every step succeeded. A directory still
# ending in .partial is a failed run — delete it.
#
# Procedure, scope decisions and restore notes: docs/backup-restore.md
#
# Usage (on the VM):  sudo bash backup.sh
#
# Configurable via environment variables (defaults match qasap-vm01 as of 2026-10-17):
#   BACKUP_ROOT          where backups are written        (default: /opt/backup)
#   OPENWEBUI_DATA       Open WebUI data directory        (default: /opt/openwebui/data)
#   UPLOADS_DIR          uploaded course files            (default: /mnt/gcs/fin602)
#   GRAFANA_DATA         Grafana data directory           (default: /opt/grafana/data)
#   QDRANT_STORAGE       Qdrant storage, for the disk-space estimate only
#                                                         (default: /opt/qdrant/storage)
#   ENV_FILE             production .env                  (default: /opt/openwebui/FamilyFinanceChat/.env)
#   OPENWEBUI_CONTAINER  Open WebUI container name        (default: open-webui)
#   QDRANT_CONTAINER     Qdrant container name            (default: qdrant)
#   QDRANT_URL           Qdrant URL as seen from inside the Open WebUI container
#                                                         (default: http://qdrant:6333)
#   QDRANT_SNAPSHOT_DIR  snapshot directory inside the Qdrant container
#                                                         (default: /qdrant/snapshots)
#   RESERVE_GB           free space that must remain on the backup disk after
#                        the run, so production can keep writing   (default: 5)

set -euo pipefail
umask 077   # everything written here is student data or secrets

BACKUP_ROOT="${BACKUP_ROOT:-/opt/backup}"
OPENWEBUI_DATA="${OPENWEBUI_DATA:-/opt/openwebui/data}"
UPLOADS_DIR="${UPLOADS_DIR:-/mnt/gcs/fin602}"
GRAFANA_DATA="${GRAFANA_DATA:-/opt/grafana/data}"
QDRANT_STORAGE="${QDRANT_STORAGE:-/opt/qdrant/storage}"
ENV_FILE="${ENV_FILE:-/opt/openwebui/FamilyFinanceChat/.env}"
OPENWEBUI_CONTAINER="${OPENWEBUI_CONTAINER:-open-webui}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
QDRANT_SNAPSHOT_DIR="${QDRANT_SNAPSHOT_DIR:-/qdrant/snapshots}"
RESERVE_GB="${RESERVE_GB:-5}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Qdrant's port is not published on the VM and the Qdrant image has no curl,
# so talk to it from inside the Open WebUI container, over the Docker network.
qcurl() { docker exec "$OPENWEBUI_CONTAINER" curl -sf "$@"; }

# Online copy of a live SQLite database. Safe while the app is writing; waits
# up to 10 s for a lock instead of failing immediately. -readonly matters: a
# read-write open would roll back a leftover journal and so modify the source.
sqlite_backup() {  # sqlite_backup <source.db> <dest.db>
  sqlite3 -readonly -cmd ".timeout 10000" "$1" ".backup '$2'"
  local check
  check=$(sqlite3 -readonly "$2" "PRAGMA integrity_check;")
  [ "$check" = "ok" ] || die "integrity check failed on the copy of $1: $check"
}

# tar exits 1 when a file changed while it was being read, which is expected on
# a live directory (e.g. chat_metrics_state.json). Treat 1 as a warning, >1 as failure.
# --atime-preserve=system reads without touching the source files' access times.
archive() {  # archive <output.tar.gz> <source dir> [tar options...]
  local out="$1" src="$2"; shift 2
  local rc=0
  tar czf "$out" --atime-preserve=system --warning=no-file-changed "$@" -C "$src" . || rc=$?
  [ "$rc" -le 1 ] || die "tar failed (exit $rc) archiving $src"
  [ "$rc" -eq 0 ] || log "  warning: a file in $src changed while it was archived (normal on a live system)"
}

kb() { du -sk "$1" | cut -f1; }

# --- Preflight ---------------------------------------------------------------

[ "$(id -u)" -eq 0 ] || die "run as root: sudo bash $0"

for cmd in sqlite3 tar sha256sum docker du df stat realpath; do
  command -v "$cmd" >/dev/null || die "missing command: $cmd"
done

for d in "$OPENWEBUI_DATA" "$UPLOADS_DIR" "$GRAFANA_DATA" "$QDRANT_STORAGE"; do
  [ -d "$d" ] || die "not a directory: $d"
done

# The backup must never be written inside a folder it reads from (or the other
# way round), and BACKUP_ROOT is quoted into sqlite3 commands, so allow only
# plain path characters.
[[ "$BACKUP_ROOT" =~ ^/[A-Za-z0-9/._-]*$ ]] \
  || die "BACKUP_ROOT must be an absolute path of letters, digits and / . _ -"
root_real=$(realpath -m "$BACKUP_ROOT")
for d in "$OPENWEBUI_DATA" "$UPLOADS_DIR" "$GRAFANA_DATA" "$QDRANT_STORAGE" "$(dirname "$ENV_FILE")"; do
  src_real=$(realpath -m "$d")
  case "$root_real/" in "$src_real"/*) die "BACKUP_ROOT ($BACKUP_ROOT) is inside $d" ;; esac
  case "$src_real/" in "$root_real"/*) die "$d is inside BACKUP_ROOT ($BACKUP_ROOT)" ;; esac
done
for f in "$OPENWEBUI_DATA/webui.db" "$GRAFANA_DATA/grafana.db" "$ENV_FILE"; do
  [ -f "$f" ] || die "not found: $f"
done

for c in "$OPENWEBUI_CONTAINER" "$QDRANT_CONTAINER"; do
  [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = "true" ] \
    || die "container not running: $c"
done

if [ ! -d "$BACKUP_ROOT" ]; then
  mkdir -p "$BACKUP_ROOT"
  chmod 700 "$BACKUP_ROOT"
fi
[ "$(stat -c '%u %a' "$BACKUP_ROOT")" = "0 700" ] \
  || die "$BACKUP_ROOT must be owned by root with mode 700 — it holds student data"

# The Qdrant snapshot is written once inside the container and once more when
# copied out, so it counts twice. Add 50% margin for everything else.
need_kb=$(( $(kb "$OPENWEBUI_DATA/webui.db") + $(kb "$UPLOADS_DIR") \
          + $(kb "$GRAFANA_DATA") + 2 * $(kb "$QDRANT_STORAGE") ))
need_kb=$(( need_kb * 3 / 2 ))
# /opt/backup and Qdrant's snapshot directory are on the VM's root disk, the same
# disk production writes to, so leave RESERVE_GB free after the run as well.
[[ "$RESERVE_GB" =~ ^[0-9]+$ ]] || die "RESERVE_GB must be a whole number of GB"
reserve_kb=$(( RESERVE_GB * 1024 * 1024 ))
avail_kb=$(df -Pk "$BACKUP_ROOT" | awk 'NR==2 {print $4}')
[ "$avail_kb" -gt $(( need_kb + reserve_kb )) ] \
  || die "not enough free space in $BACKUP_ROOT: need ~$((need_kb / 1024)) MB plus ${RESERVE_GB} GB reserve, have $((avail_kb / 1024)) MB"

TS=$(date +%Y-%m-%d_%H%M%S)
DEST="$BACKUP_ROOT/$TS"
WORK="$DEST.partial"
[ ! -e "$DEST" ] && [ ! -e "$WORK" ] || die "$DEST already exists"
mkdir "$WORK"

ok=0
snap_in_container=""   # set while a snapshot this run created exists inside Qdrant
cleanup() {
  if [ -n "$snap_in_container" ]; then
    qcurl -X DELETE "$QDRANT_URL/snapshots/$snap_in_container" >/dev/null 2>&1 \
      || echo "Could not delete Qdrant snapshot $snap_in_container inside $QDRANT_CONTAINER — remove it by hand." >&2
  fi
  [ "$ok" -eq 1 ] || echo "Backup FAILED. Incomplete files are in $WORK — delete them." >&2
}
trap cleanup EXIT
# Turn Ctrl-C, kill and a dropped SSH session into an ordinary exit, so the
# EXIT trap above runs and removes the Qdrant snapshot in every case.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

log "Backing up to $WORK (estimated ~$((need_kb / 1024)) MB needed, $((avail_kb / 1024)) MB free)"

# --- 1. Open WebUI database --------------------------------------------------

log "1/6 Open WebUI database: $OPENWEBUI_DATA/webui.db"
sqlite_backup "$OPENWEBUI_DATA/webui.db" "$WORK/webui.db"

# --- 2. Rest of the Open WebUI data directory --------------------------------

# Excluded: webui.db (copied above), webui.db.* (old manual copies) and
# webui.db-journal, and cache/ (~6 GB of models Open WebUI re-downloads itself).
log "2/6 Rest of $OPENWEBUI_DATA (without webui.db*, cache/)"
archive "$WORK/openwebui-data-rest.tar.gz" "$OPENWEBUI_DATA" \
  --exclude='./webui.db*' --exclude='./cache'

# --- 3. Uploaded course files ------------------------------------------------

log "3/6 Uploaded files: $UPLOADS_DIR"
archive "$WORK/uploads.tar.gz" "$UPLOADS_DIR"

# --- 4. Qdrant ---------------------------------------------------------------

log "4/6 Qdrant full snapshot"
collections=$(qcurl "$QDRANT_URL/collections") || die "cannot reach Qdrant at $QDRANT_URL from $OPENWEBUI_CONTAINER"
counts=""
for c in $(printf '%s' "$collections" | grep -o '"name":"[^"]*"' | cut -d'"' -f4); do
  n=$(qcurl -X POST "$QDRANT_URL/collections/$c/points/count" \
        -H 'Content-Type: application/json' -d '{"exact":true}' \
      | sed -n 's/.*"count":\([0-9]*\).*/\1/p') \
    || die "could not count points in Qdrant collection $c"
  [ -n "$n" ] || die "unexpected point-count response from Qdrant for collection $c"
  counts+="  $c: $n points"$'\n'
done

resp=$(qcurl -X POST "$QDRANT_URL/snapshots?wait=true") \
  || die "Qdrant snapshot request failed (a snapshot may still have been written inside $QDRANT_CONTAINER — check $QDRANT_SNAPSHOT_DIR by hand)"
snap=$(printf '%s' "$resp" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p')
# The name is used in a file path and in a DELETE, so accept only a plain file
# name. Anything else is not deleted: it may not be ours.
[[ "$snap" =~ ^[A-Za-z0-9._-]+\.snapshot$ ]] && [[ "$snap" != *..* ]] \
  || die "unexpected snapshot name from Qdrant (not deleted, check by hand): $resp"
snap_in_container="$snap"
docker cp "$QDRANT_CONTAINER:$QDRANT_SNAPSHOT_DIR/$snap" "$WORK/$snap"
# The copy inside the container is no longer needed; free the space.
qcurl -X DELETE "$QDRANT_URL/snapshots/$snap" >/dev/null \
  || die "could not delete $snap inside $QDRANT_CONTAINER"
snap_in_container=""

# --- 5. Grafana --------------------------------------------------------------

log "5/6 Grafana: $GRAFANA_DATA"
sqlite_backup "$GRAFANA_DATA/grafana.db" "$WORK/grafana.db"
archive "$WORK/grafana-rest.tar.gz" "$GRAFANA_DATA" --exclude='./grafana.db*'

# --- 6. Production .env ------------------------------------------------------

log "6/6 Production .env"
cp -p "$ENV_FILE" "$WORK/production.env"
chmod 600 "$WORK/production.env"

# --- Manifest and checksums --------------------------------------------------

{
  echo "created:      $(date -Is)"
  echo "host:         $(hostname)"
  echo "open-webui:   $(qcurl http://localhost:8080/api/version || echo unknown)"
  echo "qdrant:       $(qcurl "$QDRANT_URL/" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')"
  for c in "$OPENWEBUI_CONTAINER" "$QDRANT_CONTAINER" grafana; do
    printf '%-13s %s\n' "image[$c]:" "$(docker inspect -f '{{.Config.Image}}' "$c" 2>/dev/null || echo unknown)"
  done
  echo "qdrant collections, counted just before the snapshot:"
  printf '%s' "$counts"
  echo "qdrant snapshot: $snap"
  echo "excluded:     $OPENWEBUI_DATA/cache (re-downloadable models)"
  echo "              $OPENWEBUI_DATA/webui.db.* (old manual copies)"
  echo "              Prometheus, Redis, Pushgateway (see docs/backup-restore.md)"
} > "$WORK/MANIFEST.txt"

(cd "$WORK" && sha256sum -- * > SHA256SUMS)

mv "$WORK" "$DEST"
ok=1

log "Done: $DEST ($(du -sh "$DEST" | cut -f1))"
ls -lh "$DEST"
echo
cat "$DEST/MANIFEST.txt"

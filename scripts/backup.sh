#!/usr/bin/env bash
# Back up the production FamilyFinanceChat stack (task A2).
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
#   7. Writes MANIFEST.txt (mode, versions, point counts, exclusions) and SHA256SUMS
#
# Two modes:
#   online (default)  Nothing is stopped. Steps 1, 3 and 4 run minutes apart
#                     while students keep working, so a file uploaded or deleted
#                     during the run can be in one of them and not the others.
#                     Fine for a routine safety copy.
#   CONSISTENT=1      Stops the Open WebUI container for steps 1-4 and starts it
#                     again right after (also when the run fails). Open WebUI is
#                     the only service that writes the database, the uploads and
#                     Qdrant, so all three show the same moment.
#   CONSISTENT=1 KEEP_STOPPED=1
#                     The same, but on success Open WebUI stays stopped, so nothing
#                     is written between the backup and the upgrade that follows.
#                     On failure it is started again: no backup, no upgrade.
#                     Use this for the backup taken just before an upgrade.
#
# Otherwise read-only against production: every existing file is only read and
# databases are opened read-only. The other writes outside the backup folder
# are Qdrant's own temporary snapshot file inside its container and a helper
# container (see qcurl below). The script removes both again, even when it fails.
# Output goes to $BACKUP_ROOT/<timestamp>.partial and is renamed to
# $BACKUP_ROOT/<timestamp> only when every step succeeded. A directory still
# ending in .partial is a failed run — delete it.
#
# Procedure, scope decisions and restore notes: docs/backup-restore.md
#
# Usage (on the VM):  sudo bash backup.sh                                online
#                     sudo CONSISTENT=1 bash backup.sh                   consistent
#                     sudo CONSISTENT=1 KEEP_STOPPED=1 bash backup.sh    before an upgrade
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
#   QDRANT_URL           Qdrant URL as seen from inside the Qdrant container
#                                                         (default: http://localhost:6333)
#   QDRANT_SNAPSHOT_DIR  snapshot directory inside the Qdrant container
#                                                         (default: /qdrant/snapshots)
#   RESERVE_GB           free space that must remain on the backup disk after
#                        the run, so production can keep writing   (default: 5)
#   CONSISTENT           1 = stop Open WebUI during steps 1-4, see above (default: 0)
#   KEEP_STOPPED         1 = with CONSISTENT=1, leave Open WebUI stopped after a
#                        successful run, see above                  (default: 0)

set -euo pipefail
umask 077   # everything written here is student data or secrets

CONSISTENT="${CONSISTENT:-0}"
KEEP_STOPPED="${KEEP_STOPPED:-0}"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/backup}"
OPENWEBUI_DATA="${OPENWEBUI_DATA:-/opt/openwebui/data}"
UPLOADS_DIR="${UPLOADS_DIR:-/mnt/gcs/fin602}"
GRAFANA_DATA="${GRAFANA_DATA:-/opt/grafana/data}"
QDRANT_STORAGE="${QDRANT_STORAGE:-/opt/qdrant/storage}"
ENV_FILE="${ENV_FILE:-/opt/openwebui/FamilyFinanceChat/.env}"
OPENWEBUI_CONTAINER="${OPENWEBUI_CONTAINER:-open-webui}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_SNAPSHOT_DIR="${QDRANT_SNAPSHOT_DIR:-/qdrant/snapshots}"
RESERVE_GB="${RESERVE_GB:-5}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Qdrant's port is not published on the VM and the Qdrant image has no curl.
# So a helper container shares the Qdrant container's network and runs curl
# from the Open WebUI image, which is already on the VM. It does not need the
# Open WebUI container itself, so it keeps working while CONSISTENT=1 has
# stopped it. Started in the preflight, removed by cleanup.
HELPER="backup-qdrant-curl-$$"
qcurl() { docker exec "$HELPER" curl -sf "$@"; }

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
[[ "$CONSISTENT" =~ ^[01]$ ]] || die "CONSISTENT must be 0 or 1"
[[ "$KEEP_STOPPED" =~ ^[01]$ ]] || die "KEEP_STOPPED must be 0 or 1"
[ "$KEEP_STOPPED" -eq 0 ] || [ "$CONSISTENT" -eq 1 ] || die "KEEP_STOPPED=1 needs CONSISTENT=1"

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
# Read now: in CONSISTENT mode Open WebUI is stopped when the manifest is written.
ow_version=$(docker exec "$OPENWEBUI_CONTAINER" curl -sf http://localhost:8080/api/version || echo unknown)

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
ow_stopped=0           # 1 while this run has Open WebUI stopped
cleanup() {
  if [ -n "$snap_in_container" ]; then
    qcurl -X DELETE "$QDRANT_URL/snapshots/$snap_in_container" >/dev/null 2>&1 \
      || echo "Could not delete Qdrant snapshot $snap_in_container inside $QDRANT_CONTAINER — remove it by hand." >&2
  fi
  docker rm -f "$HELPER" >/dev/null 2>&1 || true
  if [ "$ow_stopped" -eq 1 ]; then
    if docker start "$OPENWEBUI_CONTAINER" >/dev/null; then
      echo "Started $OPENWEBUI_CONTAINER again." >&2
    else
      echo "COULD NOT START $OPENWEBUI_CONTAINER — production is down. Run: sudo docker start $OPENWEBUI_CONTAINER" >&2
    fi
  fi
  [ "$ok" -eq 1 ] || echo "Backup FAILED. Incomplete files are in $WORK — delete them." >&2
}
trap cleanup EXIT
# Turn Ctrl-C, kill and a dropped SSH session into an ordinary exit, so the
# EXIT trap above runs in every case: it removes the Qdrant snapshot and the
# helper, and starts Open WebUI again if this run stopped it.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# The helper behind qcurl. It ends by itself after a day, in case even cleanup
# never ran (kill -9).
docker run -d --rm --name "$HELPER" --network "container:$QDRANT_CONTAINER" \
  --entrypoint sleep "$(docker inspect -f '{{.Image}}' "$OPENWEBUI_CONTAINER")" 86400 >/dev/null
qcurl "$QDRANT_URL/collections" >/dev/null \
  || die "cannot reach Qdrant at $QDRANT_URL from the helper container"

log "Backing up to $WORK (estimated ~$((need_kb / 1024)) MB needed, $((avail_kb / 1024)) MB free)"

if [ "$CONSISTENT" -eq 1 ]; then
  if [ "$KEEP_STOPPED" -eq 1 ]; then
    log "CONSISTENT=1 KEEP_STOPPED=1: stopping $OPENWEBUI_CONTAINER; it stays stopped if the backup succeeds"
  else
    log "CONSISTENT=1: stopping $OPENWEBUI_CONTAINER until Qdrant is done"
  fi
  ow_stopped=1
  stopped_at=$SECONDS
  docker stop "$OPENWEBUI_CONTAINER" >/dev/null
fi

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
collections=$(qcurl "$QDRANT_URL/collections") || die "cannot reach Qdrant at $QDRANT_URL from the helper container"
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

# Everything that must match is copied; production need not wait for the rest.
# With KEEP_STOPPED=1 it stays stopped; cleanup still starts it if a later step fails.
if [ "$ow_stopped" -eq 1 ] && [ "$KEEP_STOPPED" -eq 0 ]; then
  docker start "$OPENWEBUI_CONTAINER" >/dev/null
  ow_stopped=0
  log "  started $OPENWEBUI_CONTAINER again (stopped for $((SECONDS - stopped_at)) s)"
fi

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
  if [ "$CONSISTENT" -eq 1 ]; then
    echo "mode:         consistent (Open WebUI stopped for steps 1-4)"
  else
    echo "mode:         online (database, uploads and Qdrant copied minutes apart)"
  fi
  echo "open-webui:   $ow_version"
  echo "qdrant:       $(qcurl "$QDRANT_URL/" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')"
  for c in "$OPENWEBUI_CONTAINER" "$QDRANT_CONTAINER" grafana; do
    printf '%-13s %s\n' "image[$c]:" "$(docker inspect -f '{{.Config.Image}}' "$c" 2>/dev/null || echo unknown)"
  done
  if [ "$CONSISTENT" -eq 1 ]; then
    echo "qdrant collections, counted just before the snapshot (exact):"
  else
    echo "qdrant collections, counted just before the snapshot (online: may differ slightly):"
  fi
  printf '%s' "$counts"
  echo "qdrant snapshot: $snap"
  echo "excluded:     $OPENWEBUI_DATA/cache (re-downloadable models)"
  echo "              $OPENWEBUI_DATA/webui.db.* (old manual copies)"
  echo "              Prometheus, Redis, Pushgateway (see docs/backup-restore.md)"
} > "$WORK/MANIFEST.txt"

(cd "$WORK" && sha256sum -- * > SHA256SUMS)

mv "$WORK" "$DEST"
ok=1
ow_stopped=0   # KEEP_STOPPED=1: the backup is good, so cleanup leaves Open WebUI stopped

log "Done: $DEST ($(du -sh "$DEST" | cut -f1))"
ls -lh "$DEST"
echo
cat "$DEST/MANIFEST.txt"
if [ "$KEEP_STOPPED" -eq 1 ]; then
  echo
  log "KEEP_STOPPED=1: $OPENWEBUI_CONTAINER is still stopped. Upgrade now, or start it again with: sudo docker start $OPENWEBUI_CONTAINER"
fi

# Backup and restore

Task **A2** in [`capstone_fall2026/PROJECT_PLAN.md`](../../capstone_fall2026/PROJECT_PLAN.md).

Every upgrade (A9–A13) needs a way back, and the v0.10.0 migration is one-way, so the only
rollback is restoring a backup. This document covers taking a backup with
[`scripts/backup.sh`](../../scripts/backup.sh) and restoring production from it. Proving that a
backup restores is the [restore drill](restore-drill.md).

**Backed up:** the Open WebUI database and data folder, the uploaded course files, the Qdrant
vector index, Grafana, and the production `.env`. **Skipped:** Open WebUI's 6 GB model cache
(re-downloaded automatically), old `webui.db.*` copies, and Redis, Pushgateway and Prometheus,
whose data is temporary or out of scope.

The backup holds student transcripts and API keys. It stays in `/opt/backup` on the VM (root
only), never on a laptop or in git.

---

## Taking a backup

### Modes

The database, the uploads and Qdrant are copied one after the other, a few minutes apart.

| Mode | Command | Open WebUI | Use it for |
|---|---|---|---|
| Online | `sudo bash backup.sh` | keeps running | routine safety copies. A file uploaded or deleted during the run may be in one copy and not the others |
| Consistent | `sudo CONSISTENT=1 bash backup.sh` | stopped while the three are copied (about 30 s), then started again, even if the run fails | an exact backup outside an upgrade, e.g. for a drill |
| Before an upgrade | `sudo CONSISTENT=1 KEEP_STOPPED=1 bash backup.sh` | the same, but after a **successful** run it stays stopped, so nothing is written before the upgrade. After a failed run it is started again | the backup a failed upgrade is restored from |

Open WebUI is the only service that writes the database, the uploads and Qdrant, so stopping it
is enough for an exact backup. Everything else keeps running. In the consistent modes, tell the
team first.

### Run it

On the VM, inside `tmux` (a dropped SSH session aborts the run safely; `tmux` keeps it going).
Copy the script; do not `git pull` the deployed checkout, which pins older images than `main`
(see [`memory/vm-state-differs-from-repo.md`](../memory/vm-state-differs-from-repo.md)).

```bash
gcloud compute scp scripts/backup.sh qasap-vm01:backup.sh --zone us-east4-c --project wsb-hc-qasap-ae2e   # lands in your VM home folder; "~/" fails on Windows
gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e
tmux new -s backup                      # after a disconnect: tmux attach -t backup
sudo bash ~/backup.sh 2>&1 | tee ~/backup-run.log                                 # routine, in a quiet period
sudo CONSISTENT=1 bash ~/backup.sh 2>&1 | tee ~/backup-run.log                    # exact, Open WebUI back afterwards
sudo CONSISTENT=1 KEEP_STOPPED=1 bash ~/backup.sh 2>&1 | tee ~/backup-run.log     # before an upgrade
```

- Run it with `bash`, not `sh`: `sh` stops at once with `Illegal option -o pipefail`.
- If `sudo` refuses to set the variables, put `env` in front: `sudo env CONSISTENT=1 bash ...`.
- A good run ends with `Done: /opt/backup/<timestamp>` and the manifest. A folder still ending in
  `.partial` is a failed run: delete it.
- `/opt/backup` is root-only, so your own shell cannot expand a wildcard in it: use
  `sudo sh -c 'du -sh /opt/backup/*'`, not `sudo du -sh /opt/backup/*`.

### Before an upgrade

1. Tell the team and the instructor: Open WebUI is down from the backup until the upgrade is done.
2. `sudo CONSISTENT=1 KEEP_STOPPED=1 bash ~/backup.sh 2>&1 | tee ~/backup-run.log`. It must end
   with `Done` and `KEEP_STOPPED=1: open-webui is still stopped`. If it ends with `Backup FAILED`,
   Open WebUI is running again: do not upgrade.
3. Upgrade (the runbook for that upgrade, A9–A13).
4. If the upgrade fails, [restore this backup](#restoring-production).

To call the upgrade off after a successful backup: `sudo docker start open-webui`.

### What the script does

It writes nine files to a new folder `/opt/backup/<timestamp>` and only reads everything else
(databases read-only, source access times kept). Outside that folder it creates a Qdrant snapshot
inside the Qdrant container and a helper container to call Qdrant, and removes both even when it
fails or is interrupted. In the consistent modes it also stops and starts `open-webui`. It
refuses to run if the backup folder would be inside a folder it reads, if `/opt/backup` is not
root-only, or if less than 5 GB would stay free on production's disk.

| Lines | What it does |
|---|---|
| 1–63 | Header: what it does, the modes, usage, settings |
| 65–83 | `set -euo pipefail` (stop at the first error), `umask 077` (root-only files), settings with the VM's defaults, `log`, `die` |
| 90–91 | `qcurl`: `curl` from a helper container on Qdrant's network. Qdrant has no open port and no `curl`; the helper uses the Open WebUI image, so it works while `open-webui` is stopped |
| 96–101 | `sqlite_backup`: SQLite's online `.backup` from a read-only source, then `PRAGMA integrity_check` on the copy |
| 106–112 | `archive`: `tar czf`; a file that changed during the copy (exit 1) is a warning, worse is a failure |
| 118–177 | Preflight, nothing copied yet: root, valid settings, tools, folders, files, containers running, Open WebUI version, `/opt/backup` root-only, disk space incl. `RESERVE_GB`, then the new `.partial` folder |
| 179–203 | `cleanup` on every exit (also Ctrl-C, `kill`, dropped SSH): deletes this run's snapshot, removes the helper, starts `open-webui` again unless the backup succeeded with `KEEP_STOPPED=1`, reports failure |
| 207–210 | Starts the helper (it ends itself after a day) and checks Qdrant answers |
| 214–223 | Consistent modes: stops `open-webui` |
| 227–241 | Steps 1–3: `webui.db`, the rest of the data folder (without `webui.db*` and `cache/`), the uploads |
| 245–269 | Step 4: counts the points per collection, takes a full Qdrant snapshot, copies it out, deletes it inside the container |
| 271–277 | `CONSISTENT=1` without `KEEP_STOPPED`: starts `open-webui` again and logs how long it was stopped |
| 281–289 | Steps 5–6: Grafana's database and folder, the production `.env` |
| 293–331 | `MANIFEST.txt` (mode, versions, images, point counts, snapshot name, exclusions), `SHA256SUMS`, rename `.partial` to the final name, print the result |

**Restore into the versions `MANIFEST.txt` lists.** Running a newer release over a backup is an
upgrade, not a restore.

---

## Restoring production

The way back after a failed upgrade: production's data is replaced with a backup. These are the
same steps as the [restore drill](restore-drill.md), so a drill that passed on a backup means
they work on it.

**Before you start:**

- Tell the team and the instructor: Open WebUI, Qdrant and Grafana are down for about 15 minutes.
- Use the backup taken just before the upgrade, with `CONSISTENT=1 KEEP_STOPPED=1`
  (`mode: consistent` in its manifest). Everything written after it is lost. An online backup
  also restores, but may be slightly inconsistent; step 1 warns.
- Nothing is deleted: the broken data is renamed to `*.broken-<T>`, and
  [If a step fails](#if-a-step-fails) puts it back.
- **Start on the manifest's versions.** If the upgrade changed the images in the deployed
  `docker-compose.yml`, put them back first, or Open WebUI migrates the restored database again.
  Step 1 prints both to compare.

### Steps

On the VM, in `tmux`, in one session. Paste the blocks **one at a time**. Each stops at its first
error and ends with `OK <n>: ...`. **No `OK` line: do not paste the next block**; go to
[If a step fails](#if-a-step-fails).

```bash
# 0. Settings for the blocks below. After a disconnect, set them again: T must keep its value
#    (ls /opt/openwebui shows it in data.broken-<T>).
B=/opt/backup/<timestamp>                      # the backup taken just before the upgrade
T=$(date +%Y%m%d_%H%M%S)                       # suffix for the broken data
cd /opt/openwebui/FamilyFinanceChat            # the deployed checkout
```

```bash
# 1. Check the backup. Changes nothing.
sudo bash -euo pipefail -c '
  B=${1%/} checkout=$PWD
  case $B in *.partial) echo "FAILED: $B is an unfinished backup"; exit 1 ;; esac
  cd "$B"
  sha256sum --quiet -c SHA256SUMS
  for f in webui.db openwebui-data-rest.tar.gz grafana.db grafana-rest.tar.gz MANIFEST.txt; do
    [ -f "$f" ] || { echo "FAILED: $f is missing"; exit 1; }
  done
  snap=$(sed -n "s/^qdrant snapshot: *//p" MANIFEST.txt)
  [ -n "$snap" ] && [ -f "$snap" ] || { echo "FAILED: the snapshot in MANIFEST.txt is missing: $snap"; exit 1; }
  tar tzf openwebui-data-rest.tar.gz >/dev/null
  tar tzf grafana-rest.tar.gz >/dev/null
  cat MANIFEST.txt
  echo "--- images the deployed docker-compose.yml will start (compare with image[...] above):"
  (cd "$checkout" && docker compose config --images) || echo "(could not read docker-compose.yml: check its images by hand)"
  grep -q "^mode: *consistent" MANIFEST.txt \
    || echo "WARNING: online backup, see Before you start. Continue only if no consistent one exists."
  echo "OK 1: the backup is complete and undamaged"
' _ "$B"
```

```bash
# 2. Stop only what is restored (Redis, Prometheus and the rest keep running), then move the
#    broken data aside. Rename, never delete.
sudo bash -euo pipefail -c '
  T=$1
  for d in /opt/openwebui/data /opt/qdrant/storage /opt/grafana/data; do
    [ ! -e "$d.broken-$T" ] || { echo "FAILED: $d.broken-$T already exists; pick a new T"; exit 1; }
  done
  docker compose stop open-webui qdrant grafana
  for d in /opt/openwebui/data /opt/qdrant/storage /opt/grafana/data; do
    mv "$d" "$d.broken-$T"
    mkdir "$d"
  done
  echo "OK 2: stopped; the old data is in *.broken-$T"
' _ "$T"
```

```bash
# 3. Put the Open WebUI and Grafana data back.
sudo bash -euo pipefail -c '
  B=$1
  tar xzf "$B/openwebui-data-rest.tar.gz" -C /opt/openwebui/data --exclude=./uploads
  cp "$B/webui.db" /opt/openwebui/data/
  tar xzf "$B/grafana-rest.tar.gz" -C /opt/grafana/data
  cp "$B/grafana.db" /opt/grafana/data/
  chown -R 472:0 /opt/grafana/data
  echo "OK 3: Open WebUI and Grafana data restored"
' _ "$B"
```

```bash
# 4. Qdrant: start it once with the snapshot, on the backup's Qdrant version. It loads the
#    snapshot, then opens its port. Gives up after 10 minutes, or as soon as Qdrant stops.
sudo bash -euo pipefail -c '
  B=$1
  snap=$(sed -n "s/^qdrant snapshot: *//p" "$B/MANIFEST.txt")
  img=$(sed -n "s/^image\[qdrant\]: *//p" "$B/MANIFEST.txt")
  fail() { docker logs --tail 30 qdrant-load || true; docker rm -f qdrant-load >/dev/null; echo "FAILED: $1"; exit 1; }
  [ -z "$(ls -A /opt/qdrant/storage)" ] \
    || { echo "FAILED: /opt/qdrant/storage is not empty (an earlier attempt?); see If a step fails"; exit 1; }
  docker rm -f qdrant-load >/dev/null 2>&1 || true
  docker run -d --name qdrant-load -v /opt/qdrant/storage:/qdrant/storage -v "$B:/backup:ro" \
    --entrypoint ./qdrant "$img" --storage-snapshot "/backup/$snap" >/dev/null
  for i in $(seq 120); do
    if docker exec qdrant-load bash -c "exec 3<>/dev/tcp/127.0.0.1/6333" 2>/dev/null; then
      docker rm -f qdrant-load >/dev/null
      echo "OK 4: Qdrant snapshot loaded"
      exit 0
    fi
    [ "$(docker inspect -f "{{.State.Running}}" qdrant-load)" = true ] \
      || fail "Qdrant stopped while loading the snapshot (log above)"
    sleep 5
  done
  fail "Qdrant did not finish loading within 10 minutes (log above)"
' _ "$B"
```

```bash
# 5. Start production again, but only on restored data.
sudo bash -euo pipefail -c '
  for f in /opt/openwebui/data/webui.db /opt/grafana/data/grafana.db; do
    [ -s "$f" ] || { echo "FAILED: $f is missing; step 3 did not finish"; exit 1; }
  done
  # a finished load leaves collections/; a failed one can leave snapshots_recovery_tmp/
  [ -d /opt/qdrant/storage/collections ] && [ ! -e /opt/qdrant/storage/snapshots_recovery_tmp ] \
    || { echo "FAILED: /opt/qdrant/storage holds no finished load; step 4 did not finish"; exit 1; }
  docker compose up -d open-webui qdrant grafana
  echo "OK 5: production started on the restored data"
'
```

| Step | What it does |
|---|---|
| 1 | Changes nothing. Checks the backup is finished, the checksums match, the files and snapshot exist and the archives unpack. Prints the manifest next to the images the compose file will start |
| 2 | Refuses a `T` that is already used. Stops only `open-webui`, `qdrant`, `grafana`; renames their data folders and creates empty ones |
| 3 | Unpacks the Open WebUI data folder (not `uploads/`, which lives in `/mnt/gcs/fin602`) and Grafana, and copies both databases back. Grafana's files belong to its user, 472 |
| 4 | A full Qdrant snapshot loads only as Qdrant starts. A one-off container of the manifest's Qdrant image loads it into the empty storage folder; checks every 5 s, stops at once if Qdrant exits, gives up after 10 minutes |
| 5 | Refuses to start production unless both databases are there and Qdrant finished loading |

**Uploads and `.env`** are not touched by an upgrade. Restore them only if they were lost or
changed: unpack `uploads.tar.gz` into `/mnt/gcs/fin602`, or copy `production.env` to
`/opt/openwebui/FamilyFinanceChat/.env`.

### If a step fails

- **Step 1:** nothing has changed and production is still running. Use another backup.
- **Steps 2–5:** production is stopped. Put the data from before step 2 back; the half-restored
  copy is kept as `*.failed-restore-<T>`:

  ```bash
  sudo bash -euo pipefail -c '
    T=$1
    docker rm -f qdrant-load >/dev/null 2>&1 || true
    for d in /opt/openwebui/data /opt/qdrant/storage /opt/grafana/data; do
      [ -e "$d.broken-$T" ] || continue
      [ ! -e "$d" ] || mv "$d" "$d.failed-restore-$T"
      mv "$d.broken-$T" "$d"
    done
    echo "OK: the data from before the restore is back"
  ' _ "$T"
  ```

  That is the state of the failed upgrade. Start it as it was
  (`sudo docker compose up -d open-webui qdrant grafana`), or fix the cause and start again from
  step 0 with a new `T`.

### Check it

- `sudo docker exec open-webui curl -s localhost:8080/api/version` shows the manifest's version.
- Straight after step 5, every Qdrant collection has the manifest's point count (after an online
  backup, a few `DIFF` lines are expected):

  ```bash
  sudo bash -c '
    sed -n "s/^  \([A-Za-z0-9._-]*\): \([0-9]*\) points$/\1 \2/p" "$1/MANIFEST.txt" | while read -r c n; do
      got=$(docker exec open-webui curl -sf -X POST "http://qdrant:6333/collections/$c/points/count" \
              -H "Content-Type: application/json" -d "{\"exact\":true}" | sed -n "s/.*\"count\":\([0-9]*\).*/\1/p")
      if [ "$got" = "$n" ]; then echo "ok   $c: $n"; else echo "DIFF $c: ${got:-none} now, $n in the backup"; fi
    done
  ' _ "$B"
  ```

- At production's address: sign in, open an older chat, ask a knowledge-base question.
- Grafana on port 3001 lists the dashboards.

Then tell the team. Remove the `*.broken-<T>` and `*.failed-restore-<T>` folders only when the
team agrees they are not needed to investigate the failure.

# Backup and restore

Task **A2** in [`capstone_fall2026/PROJECT_PLAN.md`](../capstone_fall2026/PROJECT_PLAN.md).

## What we are doing

Every upgrade in Part A (A9–A13) needs a way back. The v0.10.0 migration is one-way, so the
only rollback is restoring a backup, and a backup only counts once it has been restored.
A2 has two halves:

1. **Back up production** with [`scripts/backup.sh`](../scripts/backup.sh). It copies every
   piece of data that cannot be recreated, without stopping anything.
2. **Restore that backup into the test stack** with [`scripts/restore.sh`](../scripts/restore.sh)
   and confirm the users, the chats and a knowledge-base query are all there. See
   [Restoring into the test stack](#restoring-into-the-test-stack).

When a production upgrade fails, the way back is
[Restoring production](#restoring-production): the same steps, done by hand into production's
own folders.

What gets backed up: the Open WebUI database and data folder, the uploaded course files, the
Qdrant vector index, Grafana, and the production `.env`. What is skipped: Open WebUI's 6 GB
model cache (re-downloaded automatically), old `webui.db.*` copies, and Redis, Pushgateway and
Prometheus, whose data is temporary or out of scope.

**What the script does not touch.** It creates nine new files in a new folder under
`/opt/backup` and changes nothing else. Every existing file is only read. Databases are opened
read-only, and `tar` keeps the source files' access times. The one write outside `/opt/backup` is
Qdrant's own temporary snapshot file inside its container. The script deletes it afterwards,
even when a run fails or is interrupted (Ctrl-C, `kill`, a dropped SSH session). It refuses to
run if the backup folder would be inside a folder it reads from, if an existing `/opt/backup` is
not root-only, or if the run would leave less than 5 GB free on the disk production writes to.
It never deletes anything except that one snapshot.

**Run it** on the VM, in a quiet period, inside `tmux`. A dropped SSH session aborts the run
(safely), and `tmux` keeps it running:

```bash
gcloud compute scp scripts/backup.sh qasap-vm01:backup.sh --zone us-east4-c --project wsb-hc-qasap-ae2e   # lands in your VM home folder; "~/" fails on Windows
gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e
tmux new -s backup                      # after a disconnect: tmux attach -t backup
sudo bash ~/backup.sh 2>&1 | tee ~/backup-run.log
```

The script can live anywhere; every path it uses is absolute. Copy it; do not `git pull` the
deployed checkout. `main` pins newer images than production runs (see
[`memory/vm-state-differs-from-repo.md`](memory/vm-state-differs-from-repo.md)). The backup
contains student transcripts and API keys. It stays in `/opt/backup` on the VM, and never goes
on a laptop or into git.

---

## `backup.sh`, line by line

### Setup (lines 1–88)

| Lines | Code | What it does |
|---|---|---|
| 1–43 | header comment | What the script does, how to run it, and the settings you can override |
| 45 | `set -euo pipefail` | Stop on the first failed command (`-e`), on any unset variable (`-u`), and on a failure anywhere in a pipe (`pipefail`) |
| 46 | `umask 077` | Every file the script creates is readable by root only |
| 48–58 | `X="${X:-default}"` | Paths, container names and `RESERVE_GB`. Each defaults to the VM's real value; override with e.g. `sudo RESERVE_GB=10 bash backup.sh` |
| 60 | `log()` | Prints a message with the time |
| 61 | `die()` | Prints an error and exits |
| 65 | `qcurl()` | Runs `curl` inside the `open-webui` container. Qdrant has no port open on the VM and its image has no `curl`, so this is the only way to call its API |
| 70–75 | `sqlite_backup()` | Copies a live SQLite database with SQLite's `.backup`, which is safe while the app writes (waits up to 10 s for a lock). The source is opened `-readonly`, because a read-write open can roll back a leftover journal and so change the file. Then runs `PRAGMA integrity_check` on the copy and fails unless it says `ok` |
| 80–86 | `archive()` | `tar czf` of a folder, keeping source access times (`--atime-preserve=system`). tar exits 1 when a file changed during the copy, which is normal on a live system, so 1 is a warning and only >1 is a failure |
| 88 | `kb()` | Size of a file or folder in KB, used for the disk-space check |

### Preflight checks (lines 92–162): nothing is copied until all pass

| Lines | What it checks |
|---|---|
| 92 | Running as root (needed to read the data folders) |
| 94–96 | `sqlite3`, `tar`, `sha256sum`, `docker`, `du`, `df`, `stat`, `realpath` are installed |
| 98–100 | The data folders exist |
| 105–106 | `BACKUP_ROOT` is an absolute path of plain characters (it is quoted into a sqlite3 command) |
| 107–112 | `BACKUP_ROOT` is not inside any folder being backed up, and none of them is inside it, so the backup can never write into production data |
| 113–115 | `webui.db`, `grafana.db` and `.env` exist |
| 117–120 | The `open-webui` and `qdrant` containers are running |
| 122–127 | Creates `/opt/backup` with mode 700 if missing. An existing `/opt/backup` is never changed; the script stops unless it is root-owned and mode 700 |
| 131–133 | Estimates the space needed: database + uploads + Grafana + 2× Qdrant (the snapshot exists twice for a moment), plus 50% margin |
| 136–140 | Checks `RESERVE_GB` is a whole number, then stops unless `df` shows room for the backup **plus** `RESERVE_GB` (default 5 GB) left over. The backup shares the VM's root disk with production, and production must still be able to write when the run is done |
| 142–146 | Creates `/opt/backup/<timestamp>.partial`. Refuses if the folder already exists |
| 148–157 | `cleanup`, run on every exit: if a Qdrant snapshot from this run is still inside the container, delete it; if the run did not finish, print "Backup FAILED" and name the `.partial` folder to delete |
| 160–162 | Turns a dropped SSH session (`HUP`), Ctrl-C (`INT`) and `kill` (`TERM`) into an ordinary exit (codes 129/130/143), so `cleanup` also runs when the script is interrupted |

### The backup (lines 164–222)

| Lines | Step | What it does |
|---|---|---|
| 164 | | Prints where the backup goes, the space needed and the space free |
| 168–169 | 1/6 Database | `sqlite_backup` of `webui.db`: users, chats, settings, knowledge-base records |
| 175–177 | 2/6 Data folder | `archive` of the rest of `/opt/openwebui/data`, excluding `webui.db*` (already copied, plus old copies) and `cache/` |
| 181–182 | 3/6 Uploads | `archive` of `/mnt/gcs/fin602`, the original uploaded course files |
| 187 | 4/6 Qdrant | Lists Qdrant's collections, failing clearly if Qdrant is unreachable |
| 188–196 | | Counts the points (vectors) in each collection. Stops if a count request fails or returns no number, since a restored Qdrant must be checked against these counts |
| 198–199 | | `POST /snapshots?wait=true` makes Qdrant write a consistent full snapshot. If the request fails, stops and warns that a snapshot may still exist inside the container, to be checked by hand |
| 200–204 | | `sed` extracts the snapshot's file name, which must be a plain `<name>.snapshot` (no `/` or `..`). Otherwise the script stops and deletes nothing |
| 205–206 | | Marks the snapshot for `cleanup`, then `docker cp` copies it out of the container into the backup folder |
| 208–210 | | Deletes the snapshot inside the container (only the one this run created) and clears the mark |
| 214–216 | 5/6 Grafana | `sqlite_backup` of `grafana.db` (dashboards, datasources), then `archive` of the rest of the folder |
| 220–222 | 6/6 `.env` | Copies the production `.env` (API keys, `WEBUI_SECRET_KEY`) as `production.env`, mode 600 |

### Finish (lines 226–250)

| Lines | What it does |
|---|---|
| 226–240 | Writes `MANIFEST.txt`: date, host, Open WebUI and Qdrant versions, image names, Qdrant point counts, snapshot name, what was excluded. **Restore into the same versions it lists** |
| 242 | Writes `SHA256SUMS`, a checksum of every file. Check later with `sha256sum -c SHA256SUMS` |
| 244–245 | Renames `<timestamp>.partial` to `<timestamp>`. Only a folder without `.partial` is a complete backup |
| 247–250 | Prints the backup's size, its files and the manifest |

---

## Restoring into the test stack

### What we are doing

`restore.sh` takes one backup folder and builds a second, private copy of production from it:
the test stack (`docker-compose.test.yml`) plus [`docker-compose.restore.yml`](../docker-compose.restore.yml),
an overlay that adds Grafana and runs **the versions listed in the backup's `MANIFEST.txt`**.
A backup restores safely only into the versions that wrote it. Running a newer release over it
is an upgrade, which is A9–A13's job, not a restore. The script then starts the copy and checks
it automatically. You finish by checking users, chats and a knowledge-base query in the browser.

**What it does not touch.** Production is never touched. The backup folder is only read, and is
mounted read-only into the containers. Everything is written to new Docker volumes of a separate
compose project, `ffc-restore`. The script refuses to start if any test-stack container
(`open-webui-test`, `qdrant-test`, ...) or `ffc-restore` volume already exists, so it never
overwrites anything. If one exists that isn't from your own earlier drill, it is a teammate's
test stack: ask before removing it.

**It is a copy of production.** It holds every student transcript and runs with production's
API keys, so a chat in it costs real tokens. Open WebUI (3003) and Grafana (3004) listen on the
VM only; you reach them through an SSH tunnel. Remove it the same day.

### Run it

On the VM, in `tmux`, from your own clone of the repository (never the deployed checkout):

```bash
git clone -b feat/a2-backup-restore https://github.com/mfedenia/FamilyFinanceChat ~/ffc-drill   # once; "main" after merge
cd ~/ffc-drill
sudo ls /opt/backup                                    # pick a folder without .partial
sudo bash scripts/restore.sh /opt/backup/<timestamp> 2>&1 | tee ~/restore-run.log
```

It takes a few minutes and ends with `Restore finished`, or stops at the first problem with
`ERROR: ...`. After a failure, run `sudo bash scripts/restore.sh --down` and start again.

### Check it by hand

1. On your laptop, open the tunnel and leave it open:
   `gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e -- -L 3003:localhost:3003 -L 3004:localhost:3004`
2. **Users.** Open <http://localhost:3003> and sign in with your *production* account. Under
   Admin Panel → Users, the user list matches production's.
3. **Chats.** In Admin Panel → Users, open a student's chats and read one older than the backup.
4. **Knowledge base.** Start a new chat, type `#`, pick a course knowledge base, and ask a question
   its documents answer. The reply must cite those documents.
5. **Grafana.** Open <http://localhost:3004> and sign in with production's Grafana login. The
   dashboards are listed. They show no data, because Prometheus is not part of the copy.
6. **Remove it:** `sudo bash scripts/restore.sh --down`. This deletes the test containers and the
   `ffc-restore` volumes, which hold only the copy. The backup in `/opt/backup` stays.

Record the date, the backup folder, and who ran the drill in the A2 pull request or `HANDOFF.md`.

### `restore.sh`, line by line

| Lines | What it does |
|---|---|
| 1–32 | Header comment: what the script does, what it never touches, usage, settings |
| 34 | `set -euo pipefail`: stop on the first error, unset variable, or failed pipe |
| 36–37 | `RESERVE_GB` (default 5 GB): free space that must remain for production. `MODELS_GB` (default 7 GB): room for images and the models Open WebUI downloads on first start, because the backup leaves out its model cache |
| 38–40 | The compose project name `ffc-restore`, the repository folder (found from the script's own path), and the name of the temporary Qdrant loader container |
| 42–43 | `log()` prints with the time; `die()` prints an error and exits |
| 46–49 | `dc()`: `docker compose` with the test stack, the restore overlay and the `ffc-restore` project name. Every compose command goes through it |
| 51–52 | Must run as root (the backup is root-only), with exactly one argument |
| 56–64 | `--down`: removes the restored stack's containers and volumes, and only those of project `ffc-restore`. The `RESTORE_*` placeholders are needed only so compose can read the overlay |
| 68–75 | Preflight: the tools exist; `docker compose` is 2.24.4 or newer (the overlay uses `!override`); the compose files are next to the script |
| 77–80 | The backup folder exists, is finished (no `.partial`), and its path has plain characters, since it goes into `docker -v` |
| 81–86 | All eight files are present and `sha256sum -c SHA256SUMS` passes, so a damaged backup is caught before anything is created |
| 88–100 | Reads from `MANIFEST.txt` the Open WebUI and Qdrant versions, the Grafana image, the snapshot name and the point count per collection, and checks each looks right. A `:latest` Grafana tag is refused, because it would not pin a version |
| 102–107 | Refuses if any test-stack container or `ffc-restore` volume already exists |
| 109–115 | Disk check: 3× the backup's size + `MODELS_GB` + `RESERVE_GB` must be free on Docker's disk, which is the disk production writes to |
| 117–118 | Hands the versions and the path of the backup's `production.env` to the overlay |
| 120–128 | `cleanup` runs on every exit: it removes the Qdrant loader container and, if the run did not finish, says how to start again. Ctrl-C, `kill` and a dropped SSH session also run it |
| 134–135 | 1/5: pulls the images |
| 141–147 | 2/5 Open WebUI: a throwaway `open-webui-test` container, with the backup mounted read-only at `/backup`, unpacks the data folder (except `uploads/`), copies `webui.db`, and unpacks the uploads into the uploads volume. The paths are the same as in production |
| 152–158 | 3/5 Grafana: the same with `grafana-test`, run as root to read the backup, then hands the files to Grafana's user (472) |
| 165–180 | 4/5 Qdrant: a full snapshot can only be restored as Qdrant starts (`--storage-snapshot`). It starts `qdrant-test` once that way, waits until it answers `/healthz` (up to 10 min, failing with its log if it stops), then stops and removes it |
| 184–192 | 5/5 `dc up -d` starts the stack and waits up to 10 min for `open-webui-test` to be healthy |
| 195–198 | Check: `open-webui-test` runs the backup's version |
| 199–202 | Check: Open WebUI does not offer first-time setup, which it does only when it has no users, so the restored database is in use |
| 203–209 | Check: every Qdrant collection has exactly the point count recorded in the manifest |
| 210–215 | Check: Grafana reports a working database |
| 217–230 | Marks success and prints the tunnel command, what to check by hand, and how to remove the stack |

**Tested** locally end to end with the real images (Open WebUI 0.8.12, Qdrant 1.17.1,
Grafana 9.0.0) and a mock model provider. `backup.sh` was run on a fake production with
two users, a chat, a knowledge base and a dashboard, then `restore.sh` on the result. All of
them came back: sign-in with the production password, the file contents, the knowledge-base
query, the Qdrant counts and the dashboard. A second run was refused while the first stack
existed, production's data and counts were unchanged, and `--down` left nothing behind.

---

## Restoring production

### What we are doing

This is the way back after a failed upgrade: production's data is replaced with a backup. It
uses the same three steps as `restore.sh` (unpack the archives, copy the databases, load the
Qdrant snapshot as Qdrant starts), so a passing drill means these steps work on that backup.
It is done by hand, not by a script, because it replaces production data and is needed only a
few times a term.

**Before you start:**

- Tell the team and the instructor: Open WebUI, Qdrant and Grafana are down for the restore,
  about 15 minutes.
- Use the backup taken **just before** the upgrade. Everything written after it, such as chats
  from that day, is lost.
- Nothing is deleted. The broken data is renamed to `*.broken-<time>`, so renaming it back
  undoes the whole restore.
- **The restored data must start on the versions in the backup's `MANIFEST.txt`.** If the
  failed upgrade changed the image versions in the deployed `docker-compose.yml`, put them back
  first. Otherwise Open WebUI migrates the restored database again.

### Steps

On the VM, in `tmux`, as one session:

```bash
B=/opt/backup/<timestamp>                      # the backup taken just before the upgrade
sudo cat $B/MANIFEST.txt                       # note the versions and the snapshot name
cd /opt/openwebui/FamilyFinanceChat            # the deployed checkout

# 1. Stop only what is restored (Redis, Prometheus and the rest keep running)
sudo docker compose stop open-webui qdrant grafana

# 2. Move the broken data aside. Rename, never delete
T=$(date +%Y%m%d_%H%M%S)
sudo mv /opt/openwebui/data /opt/openwebui/data.broken-$T
sudo mv /opt/qdrant/storage /opt/qdrant/storage.broken-$T
sudo mv /opt/grafana/data   /opt/grafana/data.broken-$T

# 3. Put the Open WebUI and Grafana data back
sudo mkdir /opt/openwebui/data /opt/qdrant/storage /opt/grafana/data
sudo tar xzf $B/openwebui-data-rest.tar.gz -C /opt/openwebui/data --exclude=./uploads
sudo cp $B/webui.db /opt/openwebui/data/
sudo tar xzf $B/grafana-rest.tar.gz -C /opt/grafana/data
sudo cp $B/grafana.db /opt/grafana/data/
sudo chown -R 472:0 /opt/grafana/data

# 4. Qdrant: start it once with the snapshot. It loads the snapshot, then opens its port
sudo docker compose run -d --name qdrant-load -v $B:/backup:ro --entrypoint ./qdrant \
  qdrant --storage-snapshot /backup/<snapshot name from MANIFEST.txt>
until sudo docker exec qdrant-load bash -c 'exec 3<>/dev/tcp/127.0.0.1/6333' 2>/dev/null; do sleep 5; done
sudo docker rm -f qdrant-load

# 5. Start production again
sudo docker compose up -d open-webui qdrant grafana
```

| Step | What it does |
|---|---|
| 1 | `stop` (not `down`) stops the three containers and leaves everything else running |
| 2 | Renames the current data folders. A new, empty folder is created for each in step 3 |
| 3 | Unpacks the Open WebUI data folder (except `uploads/`, which lives in `/mnt/gcs/fin602`) and copies the database back; the same for Grafana, whose files must belong to Grafana's user, 472 |
| 4 | A full Qdrant snapshot can only be restored while Qdrant starts. A one-off `qdrant` container loads it into the empty `/opt/qdrant/storage`; once its port opens, the load is done and it is removed. If it stops instead, `sudo docker logs qdrant-load` says why |
| 5 | Starts the normal containers on the restored data |

**Uploads and `.env`** are not touched by an upgrade, so they are normally left as they are.
Restore them only if they were actually lost or changed: unpack `uploads.tar.gz` into
`/mnt/gcs/fin602`, or copy `production.env` back to `/opt/openwebui/FamilyFinanceChat/.env`.

### Check it

- `sudo docker exec open-webui curl -s localhost:8080/api/version` shows the manifest's version.
- Sign in at production's address: the users, an older chat and a knowledge-base query work,
  as in [Check it by hand](#check-it-by-hand).
- Grafana on port 3001 lists the dashboards.

Once it is confirmed, tell the team. Remove the `*.broken-<time>` folders only after the team
agrees they are not needed to investigate what went wrong.

# Upgrading Open WebUI

How to move production (`qasap-vm01`) from one Open WebUI release to a newer one. An upgrade
here means **changing the pinned `image:` tag**, nothing else. Open WebUI is a dependency: no
patches, no bind mounts over its code (see
[`COMPATIBILITY_POLICY.md`](../../capstone_fall2026/COMPATIBILITY_POLICY.md)).

These steps follow the procedure that took production from 0.8.12 to v0.11.4 on 2026-10-30
([window record](../../capstone_fall2026/A13_FOLLOWUPS.md)). They use the tested tools in
[`docs/backup/`](../backup/). The longer
[A13 runbook in SETUP.md](../../SETUP.md#a13-production-upgrade-runbook-preparation) is a
stricter variant that was prepared but never run. Use it only if the team asks for its extra
gates.

**Not covered here:** upgrading Qdrant, Valkey, Grafana or Prometheus (separate changes, one at a
time), and moving from SQLite to PostgreSQL.

---

## Why the order matters

- **Migrations are one-way.** On its first start, a new release migrates the database
  (v0.10.0, for example, replaces the `config` table). After that the old image can't read
  the database. **Rolling back means restoring the backup**, not just switching the tag back.
- **Upgrade one minor version at a time** (0.11 → 0.12 → 0.13), and do each step completely
  before starting the next. Releases run their migrations in order anyway, but after a
  multi-version jump you can't tell which release broke something. The 2026-10-30 jump of three
  minor versions at once was an exception, made for time and disk reasons.
- **The VM is not the repo.** The deployed checkout is old and its `docker-compose.yml` was
  edited by hand. Running `docker compose up` from a different checkout can upgrade other
  services too. Always look at what is running (step 2). See
  [`vm-state-differs-from-repo`](../memory/vm-state-differs-from-repo.md).

| Placeholder | Meaning | Example |
|---|---|---|
| `OLD` | the version production runs now (step 2 prints it) | `v0.11.4` |
| `NEW` | the next minor release you upgrade to | `v0.12.x` |

---

## A. Preparation (days before; production keeps running)

### 1. Choose `NEW` and read the release notes

1. Read the release notes for **every** release between `OLD` and `NEW` at
   <https://github.com/open-webui/open-webui/releases>, plus the upstream
   [updating guide](https://docs.openwebui.com/getting-started/updating/).
2. Write down anything that touches us:
   - database migrations, and whether they are one-way;
   - plugin / Functions API changes (we run Filter Functions, e.g.
     [`monitoring/chat_metrics_filter.py`](../../monitoring/chat_metrics_filter.py));
   - renamed or removed environment variables (compare with `docker-compose.yml` and `.env`);
   - REST API changes to `/api/v1/users` and `/api/v1/chats`, which the grading extractor
     ([`grading_feature/backend/extract_chats.py`](../../grading_feature/backend/extract_chats.py)) uses;
   - interface changes that make our screenshots and click-paths wrong (P4 updates the docs).
3. Pick the latest **patch** release of the next minor version (e.g. `v0.12.3`, not `v0.12.0`).

### 2. Record what production runs now

From your laptop (on Windows, `~/` in `scp` targets fails; the commands below avoid it):

```bash
gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e
```

On the VM:

```bash
cd /opt/openwebui/FamilyFinanceChat                               # the deployed checkout
sudo docker exec open-webui curl -s localhost:8080/api/version    # running version = OLD
sudo docker inspect -f '{{.Config.Image}}  {{.Image}}' open-webui # running image and its ID
grep -n 'image: ghcr.io/open-webui/open-webui' docker-compose.yml # must print exactly ONE line, = running image
sudo docker compose config --images                               # every image this file would start
git -c safe.directory="$PWD" log -1 --oneline                     # deployed commit
sudo grep -c '^WEBUI_SECRET_KEY=..' .env                          # must print 1 (prints a count, not the key)
df -h /                                                           # free disk
sudo sh -c 'du -sh /opt/backup/*'                                 # size of earlier backups
```

Stop and fix before going on if:

- **the `grep` for the image prints no line or two lines, or a tag other than the running one.**
  The compose file doesn't describe production, and the edit in step 9 would be wrong.
- **`docker compose config --images` lists an image that differs from what is running**
  (`sudo docker ps --format '{{.Names}}  {{.Image}}'`). `--no-deps` in step 10 keeps those services
  untouched, but record the difference.
- **`WEBUI_SECRET_KEY` prints `0`.** Without a fixed key, every new container signs all users out.
  Add one to `.env` first (`openssl rand -hex 32`); this alone also signs everyone out once.
- **Free disk is too small.** You need room for the new image (several GB), one more backup
  (about the size of the last one) and the 5 GB that `backup.sh` refuses to go below. Before
  deleting the previous upgrade's rollback image or backup, check that its keep-until date has
  passed (see the window record).

### 3. Rehearse on blank data (CI)

On a branch (never on `main`), change the Open WebUI tag to `NEW` in three places:

| File | Line |
|---|---|
| `docker-compose.yml` | `open-webui:` → `image: ghcr.io/open-webui/open-webui:NEW` |
| `docker-compose.test.yml` | `open-webui-test:` → `image: ghcr.io/open-webui/open-webui:NEW` |
| `.github/workflows/smoke-test.yml` | `REHEARSAL_OPENWEBUI_IMAGE: ghcr.io/open-webui/open-webui:NEW` |

```bash
git switch -c chore/openwebui-NEW
# edit the three lines
bash scripts/ci/compat_guards.sh             # pin and fork guards
git commit -am "Bump Open WebUI to NEW"
git push -u origin chore/openwebui-NEW       # opens the smoke-test workflow
```

Open the PR. Say which surface it uses ("pinned image tag") and copy in the release-note
findings from step 1. **The smoke test must be green.** It checks chat, upload, the knowledge
base and retrieval on **empty** data. It proves nothing about migrating production's data,
which is what step 4 is for. Don't merge yet. Merge after the production window (step 14), so
that `main` matches what runs.

### 4. Rehearse on a copy of production's data

This is the real rehearsal: the migration runs on a copy of production's database, on the VM,
without touching production.

1. Take a fresh online backup (production keeps running), see
   [Taking a backup](../backup/backup-restore.md#taking-a-backup):

   ```bash
   # on your laptop, from the repo
   gcloud compute scp scripts/backup.sh qasap-vm01:backup.sh --zone us-east4-c --project wsb-hc-qasap-ae2e
   # on the VM
   tmux new -s upgrade
   sudo bash ~/backup.sh 2>&1 | tee ~/backup-rehearsal.log      # ends with "Done: /opt/backup/<timestamp>"
   ```

2. Run [restore drill](../backup/restore-drill.md) blocks **0–5** on that backup. You now have a
   copy of production on the **old** version (`ffc-restore-webui`, port 3005), and block 5 has
   printed `ok` lines.

3. Upgrade the copy: recreate its Open WebUI container on `NEW`, same arguments as drill
   block 4:

   ```bash
   NEW=v0.12.x                                    # the exact tag
   sudo docker pull ghcr.io/open-webui/open-webui:$NEW
   sudo bash -euo pipefail -c '
     B=$1 NEW=$2
     docker rm -f ffc-restore-webui >/dev/null
     docker run -d --name ffc-restore-webui --network ffc-restore --env-file "$B/production.env" \
       -e VECTOR_DB=qdrant -e QDRANT_URI=http://ffc-restore-qdrant:6333 \
       -e WEBSOCKET_MANAGER= -e REDIS_URL= -e WEBSOCKET_REDIS_URL= -e ENABLE_OTEL=false \
       -v ffc-restore-webui:/app/backend/data -v ffc-restore-uploads:/app/backend/data/uploads \
       -p 127.0.0.1:3005:8080 "ghcr.io/open-webui/open-webui:$NEW" >/dev/null
     for i in $(seq 120); do
       docker exec ffc-restore-webui curl -sf localhost:8080/health >/dev/null 2>&1 && break
       [ "$(docker inspect -f "{{.State.Running}}" ffc-restore-webui)" = true ] \
         || { docker logs --tail 50 ffc-restore-webui; echo "FAILED: Open WebUI stopped (log above)"; exit 1; }
       [ "$i" -lt 120 ] || { docker logs --tail 50 ffc-restore-webui; echo "FAILED: no answer in 10 minutes"; exit 1; }
       sleep 5
     done
     docker exec ffc-restore-webui curl -s localhost:8080/api/version; echo
     echo "OK: the copy runs $NEW on production data"
   ' _ "$B" "$NEW"
   sudo docker logs ffc-restore-webui 2>&1 | grep -iE 'error|traceback|migrat' | head -50   # read every line
   ```

4. Run drill block 5 again. The version line now says `DIFF` (expected: it runs `NEW`). All
   other lines must be `ok`: users present, Qdrant counts as in the backup.
5. Do the drill's [Check by hand](../backup/restore-drill.md#check-by-hand) on
   <http://localhost:3005>: sign in, open an older student chat, ask a knowledge-base question. Also check:
   - Admin Panel → Functions: our Functions are still listed and enabled; open each one and
     check it saved without errors;
   - Workspace → Models: each course model still has its base model, prompt and knowledge base;
   - a chat with each course model gets an answer (a retired base model fails here; that is
     what broke "FA26-cs620 User Interview" last time).
6. Note how long the first start took (the migration). That is the minimum outage in step 10.
7. Remove the copy with drill block **6**. Write the result in the PR: version, migration
   time, anything that broke or changed.

**Stop here if anything failed.** Fix it on the branch or wait for a newer patch release, then
repeat steps 3–4.

### 5. Schedule the window

- Agree a date and time with the instructor, outside any week with a student deadline. Allow
  about 1 hour: backup about 1 min, migration from step 4, checks about 20 min, and time for a
  restore if needed.
- Name two people: one runs the commands, one checks grading extraction (P2).
- Tell students when Open WebUI will be unavailable.

---

## B. The window (production is down from step 8 to step 12)

Everything below runs **on the VM, in one `tmux` session** (`tmux new -s upgrade`; after a
dropped connection, `tmux attach -t upgrade`). Run the blocks one at a time and read the
output of each before the next.

### 6. Set the variables and pull the new image (production still running)

```bash
cd /opt/openwebui/FamilyFinanceChat
OLD=v0.11.4                                   # from step 2
NEW=v0.12.x                                   # the rehearsed tag from step 4
sudo docker pull ghcr.io/open-webui/open-webui:$NEW
sudo docker image inspect -f '{{index .RepoDigests 0}}' ghcr.io/open-webui/open-webui:$NEW   # write down the digest
```

### 7. Keep the old image and the old compose file for rollback

```bash
sudo docker tag "$(sudo docker inspect -f '{{.Image}}' open-webui)" ffc-rollback/open-webui:$OLD
sudo cp -p docker-compose.yml docker-compose.yml.before-$NEW
sudo docker image ls ffc-rollback/open-webui   # the tag is listed
```

The tag stops `docker image prune` from deleting the old image while you might still need it.

### 8. Backup, leaving Open WebUI stopped

Copy the current `scripts/backup.sh` to the VM first if you didn't in step 4 (see there). Then:

```bash
sudo CONSISTENT=1 KEEP_STOPPED=1 bash ~/backup.sh 2>&1 | tee ~/backup-upgrade-$NEW.log
```

It must end with `Done: /opt/backup/<timestamp>` **and**
`KEEP_STOPPED=1: open-webui is still stopped`. Write down the backup folder:

```bash
B=/opt/backup/<timestamp>          # from the line above
```

- If it ends with `Backup FAILED`, Open WebUI has been started again. **Don't upgrade.** Fix
  the cause first, or call off the window.
- To call off the upgrade at this point: `sudo docker start open-webui`.

**Production is down from here on.**

### 9. Change the image tag

```bash
sudo sed -i "s#image: ghcr.io/open-webui/open-webui:$OLD\$#image: ghcr.io/open-webui/open-webui:$NEW#" docker-compose.yml
diff docker-compose.yml.before-$NEW docker-compose.yml   # exactly one changed line: $OLD -> $NEW
```

If `diff` prints nothing, the `sed` didn't match (the line in the file differs from `OLD`):
edit the line by hand with `sudo nano docker-compose.yml`. Change nothing else in the file.

### 10. Start the new version (the migration runs now)

```bash
sudo docker compose up -d --no-deps open-webui
sudo docker logs -f open-webui        # watch the migrations; Ctrl-C once it has started serving
```

`--no-deps` recreates only `open-webui`. Qdrant, Redis, Grafana and the other services keep
running as they are.

Wait until it answers (up to 15 minutes; the first start may download models):

```bash
for i in $(seq 180); do
  curl -sf http://127.0.0.1:3000/health >/dev/null && { echo "healthy after $((i*5)) s"; break; }
  [ "$(sudo docker inspect -f '{{.State.Running}}' open-webui)" = true ] || { echo "open-webui stopped"; break; }
  sleep 5
done
curl -s http://127.0.0.1:3000/api/version; echo        # must show NEW
curl -s http://127.0.0.1:3000/ready; echo
sudo docker logs open-webui 2>&1 | grep -iE 'error|traceback' | head -50
```

If it doesn't become healthy, keeps restarting, or a migration error appears: go to
[Rollback](#rollback). Don't try an older image on the migrated database.

### 11. Check it

Every item must pass. Write the results down as you go.

1. **Qdrant unchanged:** every collection has the point count from the backup's manifest:

   ```bash
   sudo bash -c '
     sed -n "s/^  \([A-Za-z0-9._-]*\): \([0-9]*\) points$/\1 \2/p" "$1/MANIFEST.txt" | while read -r c n; do
       got=$(docker exec open-webui curl -sf -X POST "http://qdrant:6333/collections/$c/points/count" \
               -H "Content-Type: application/json" -d "{\"exact\":true}" | sed -n "s/.*\"count\":\([0-9]*\).*/\1/p")
       if [ "$got" = "$n" ]; then echo "ok   $c: $n"; else echo "DIFF $c: ${got:-none} now, $n in the backup"; fi
     done
   ' _ "$B"
   ```

2. **By hand, at production's address** (`https://<PUBLIC_HOSTNAME>`):
   - sign in with your own account; you are not asked to set up a first admin;
   - Admin Panel → Users: the student list is there;
   - open an older student chat: messages are intact;
   - new chat with each course model: an answer arrives;
   - type `#`, pick a course knowledge base, ask a question its documents answer: the reply
     cites them;
   - upload a small PDF to a chat: it is processed;
   - Admin Panel → Functions: our Functions are listed and enabled.
3. **Metrics:** Grafana shows new data after your test chats, and the Open WebUI health panel
   is up. Grafana is at `https://<PUBLIC_HOSTNAME>/grafana/` behind the proxy, or on port 3001
   if the VM doesn't run the proxy yet.
4. **Grading (P2):** run the grading extractor against production and compare it with a
   known recent chat: the user, the number of question/answer pairs and the content match.
   If you saved an extraction from before the window, compare the two files with
   `python3 scripts/compare_extractions.py <before>.json <after>.json`. The JSON contains student
   data: keep it on the VM or in the grading service, and delete it afterwards.
5. **Optional, automated:** the smoke test with an existing account's token. It creates a test
   knowledge base and file, which you remove afterwards with `scripts/cleanup_smoke.py`. Exact
   commands: [SETUP.md §4 of the A13 runbook](../../SETUP.md#4-baseline-and-per-version-acceptance).

Any failure you can't fix within the window: [Rollback](#rollback).

### 12. Reopen

Tell the team, the instructor and students that Open WebUI is back. Production is up.

### 13. More than one minor version to go?

Repeat steps **1–12** for the next minor version, each with its own rehearsal, backup and checks.
Don't chain versions inside one window without a backup in between.

---

## C. After the window

### 14. Make the repo match production

1. On the step 3 branch, add the window record to the PR description (or to `HANDOFF.md`):
   date, `OLD` → `NEW`, image digest, backup folder, outage minutes, migration observations,
   check results, anything that surprised you.
2. Update the version wherever the repo states what production runs, e.g. the current
   version in [`docs/memory/vm-state-differs-from-repo.md`](../memory/vm-state-differs-from-repo.md)
   and [`capstone_fall2026/reference/system-facts.md`](../../capstone_fall2026/reference/system-facts.md) §1.
3. Merge after review.
4. If the interface changed, give P4 the list of screenshots and click-paths to redo.

### 15. Clean up, about two weeks later

Keep these until the team agrees the new version is fine (last time: 14 days):

- the backup `$B`;
- `/opt/openwebui/FamilyFinanceChat/docker-compose.yml.before-NEW`;
- the image `ffc-rollback/open-webui:OLD`.

Then:

```bash
sudo docker rmi ffc-rollback/open-webui:OLD ghcr.io/open-webui/open-webui:OLD
sudo rm /opt/openwebui/FamilyFinanceChat/docker-compose.yml.before-NEW
sudo rm -r /opt/backup/<timestamp>         # only if a newer backup exists and backup retention allows it
```

---

## Rollback

Use this if the new version fails in step 10 or 11. **Everything written after the backup in
step 8 is lost**. That is nothing if production stayed closed until step 12.

1. Put the old compose file back. The restore must start the **old** image; otherwise Open WebUI
   migrates the restored database again:

   ```bash
   cd /opt/openwebui/FamilyFinanceChat
   sudo cp -p docker-compose.yml.before-$NEW docker-compose.yml
   grep -n 'image: ghcr.io/open-webui/open-webui' docker-compose.yml   # shows OLD
   sudo docker image inspect ghcr.io/open-webui/open-webui:$OLD >/dev/null \
     || sudo docker tag ffc-rollback/open-webui:$OLD ghcr.io/open-webui/open-webui:$OLD
   ```

2. Follow [Restoring production](../backup/backup-restore.md#restoring-production) with
   `B` = the backup from step 8, blocks 0–5, then its [Check it](../backup/backup-restore.md#check-it).
   Block 1 prints the manifest's images next to the compose file's: they must match.
3. Speed-up (optional): after restore block 3, move the model cache back from the failed data
   so it isn't downloaded again (about 6 GB):

   ```bash
   sudo mv /opt/openwebui/data.broken-$T/cache /opt/openwebui/data/
   ```

4. Tell the team. Keep `*.broken-<T>` until someone has looked at why the upgrade failed. Record
   what happened in the PR.

Switching the image back **without** restoring the data does not work: the old release can't
read the migrated database.

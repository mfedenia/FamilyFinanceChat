# Restore drill

Task **A2** in [`capstone_fall2026/PROJECT_PLAN.md`](../../capstone_fall2026/PROJECT_PLAN.md).

A backup only counts once it has been restored. The drill restores one backup (made with
[`scripts/backup.sh`](../../scripts/backup.sh), see [Backup and restore](backup-restore.md)) into a
private copy of production on the VM, and checks that users, chats, files, the knowledge base and
Grafana came back. It uses the same steps as [restoring production](backup-restore.md#restoring-production),
so it is also the rehearsal for that.

- **Production is never touched.** The backup is mounted read-only. Everything the drill creates
  is named `ffc-restore-*` (four containers, four volumes, one network). On its own network the
  copy cannot reach production's Qdrant or Redis. Step 1 refuses to start if anything named
  `ffc-restore-*` already exists.
- **Same versions.** The copy runs the images listed in the backup's `MANIFEST.txt`. (That is why
  it does not use `docker-compose.test.yml`, which pins other versions.)
- **It is a copy of production:** every student transcript, and production's API keys, so a chat
  in it costs real tokens. Open WebUI (port 3005) and Grafana (3006) listen on the VM only.
  Remove it the same day.
- **Disk:** it uses about 2 GB for the data plus up to 7 GB for the models Open WebUI downloads,
  on production's disk. Step 1 refuses unless 5 GB would still be free.

Use a consistent backup (`sudo CONSISTENT=1 bash backup.sh`) so the point counts match exactly.

## Steps

On the VM, in `tmux`, in one session (the blocks use `$B` from block 0). Paste the blocks **one at
a time**; each ends with `OK <n>: ...`. **No `OK` line: do not paste the next block.** Production
is untouched whatever happened: remove the copy (block 6), fix the cause, start again at block 1.

```bash
# 0. The backup to restore
sudo ls /opt/backup                            # pick a folder without .partial
B=/opt/backup/<timestamp>
```

```bash
# 1. Check the backup, the images, the disk and that no earlier drill is left. Changes nothing.
sudo bash -euo pipefail -c '
  B=${1%/}
  case $B in *.partial) echo "FAILED: $B is an unfinished backup"; exit 1 ;; esac
  cd "$B"
  sha256sum --quiet -c SHA256SUMS
  for f in webui.db openwebui-data-rest.tar.gz uploads.tar.gz grafana.db grafana-rest.tar.gz production.env MANIFEST.txt; do
    [ -f "$f" ] || { echo "FAILED: $f is missing"; exit 1; }
  done
  snap=$(sed -n "s/^qdrant snapshot: *//p" MANIFEST.txt)
  [ -n "$snap" ] && [ -f "$snap" ] || { echo "FAILED: the snapshot in MANIFEST.txt is missing: $snap"; exit 1; }
  for f in openwebui-data-rest.tar.gz uploads.tar.gz grafana-rest.tar.gz; do tar tzf "$f" >/dev/null; done
  for s in open-webui qdrant grafana; do
    img=$(sed -n "s/^image\[$s\]: *//p" MANIFEST.txt)
    [ -n "$img" ] && [ "$img" != unknown ] || { echo "FAILED: MANIFEST.txt names no image for $s"; exit 1; }
    docker image inspect "$img" >/dev/null 2>&1 || echo "note: $img is not on the VM; docker will pull it"
    echo "$s: $img"
  done
  [ -z "$(docker ps -aq --filter name=^ffc-restore-)$(docker volume ls -q --filter name=^ffc-restore-)" ] \
    || { echo "FAILED: an earlier drill is still there; remove it with step 6 first"; exit 1; }
  # unpacked data is at most ~3x the backup; Open WebUI downloads ~7 GB of models (not in the
  # backup); and 5 GB must stay free for production, which writes to the same disk
  need=$(( $(du -sk . | cut -f1) * 3 / 1024 / 1024 + 7 + 5 ))
  free=$(df -Pk "$(docker info -f "{{.DockerRootDir}}")" | awk "NR==2 {print int(\$4 / 1024 / 1024)}")
  [ "$free" -ge "$need" ] || { echo "FAILED: Docker disk has $free GB free, the drill needs $need GB"; exit 1; }
  echo "OK 1: the backup is complete and undamaged; $free GB free, $need GB needed"
' _ "$B"
```

```bash
# 2. Put the Open WebUI data, the uploads and Grafana into new volumes.
sudo bash -euo pipefail -c '
  B=$1
  ow=$(sed -n "s/^image\[open-webui\]: *//p" "$B/MANIFEST.txt")
  docker network create ffc-restore >/dev/null
  docker run --rm --user 0:0 -v "$B:/backup:ro" -v ffc-restore-webui:/data \
    -v ffc-restore-uploads:/uploads -v ffc-restore-grafana:/grafana \
    --entrypoint bash "$ow" -euo pipefail -c "
      tar xzf /backup/openwebui-data-rest.tar.gz -C /data --exclude=./uploads
      cp /backup/webui.db /data/
      tar xzf /backup/uploads.tar.gz -C /uploads
      tar xzf /backup/grafana-rest.tar.gz -C /grafana
      cp /backup/grafana.db /grafana/
      chown -R 472:0 /grafana"
  echo "OK 2: Open WebUI data, uploads and Grafana restored into the ffc-restore-* volumes"
' _ "$B"
```

```bash
# 3. Qdrant: start it once with the snapshot, as in production step 4, then start it normally.
sudo bash -euo pipefail -c '
  B=$1
  snap=$(sed -n "s/^qdrant snapshot: *//p" "$B/MANIFEST.txt")
  img=$(sed -n "s/^image\[qdrant\]: *//p" "$B/MANIFEST.txt")
  L=ffc-restore-qdrant-load
  fail() { docker logs --tail 30 $L || true; docker rm -f $L >/dev/null; echo "FAILED: $1"; exit 1; }
  docker run -d --name $L -v ffc-restore-qdrant:/qdrant/storage -v "$B:/backup:ro" \
    --entrypoint ./qdrant "$img" --storage-snapshot "/backup/$snap" >/dev/null
  for i in $(seq 120); do
    docker exec $L bash -c "exec 3<>/dev/tcp/127.0.0.1/6333" 2>/dev/null && break
    [ "$(docker inspect -f "{{.State.Running}}" $L)" = true ] \
      || fail "Qdrant stopped while loading the snapshot (log above)"
    [ "$i" -lt 120 ] || fail "Qdrant did not finish loading within 10 minutes (log above)"
    sleep 5
  done
  docker rm -f $L >/dev/null
  docker run -d --name ffc-restore-qdrant --network ffc-restore \
    -v ffc-restore-qdrant:/qdrant/storage "$img" >/dev/null
  echo "OK 3: Qdrant snapshot loaded, ffc-restore-qdrant running"
' _ "$B"
```

```bash
# 4. Start Open WebUI and Grafana on the restored data, and wait until Open WebUI answers.
sudo bash -euo pipefail -c '
  B=$1
  ow=$(sed -n "s/^image\[open-webui\]: *//p" "$B/MANIFEST.txt")
  gf=$(sed -n "s/^image\[grafana\]: *//p" "$B/MANIFEST.txt")
  docker run -d --name ffc-restore-webui --network ffc-restore --env-file "$B/production.env" \
    -e VECTOR_DB=qdrant -e QDRANT_URI=http://ffc-restore-qdrant:6333 \
    -e WEBSOCKET_MANAGER= -e REDIS_URL= -e WEBSOCKET_REDIS_URL= -e ENABLE_OTEL=false \
    -v ffc-restore-webui:/app/backend/data -v ffc-restore-uploads:/app/backend/data/uploads \
    -p 127.0.0.1:3005:8080 "$ow" >/dev/null
  docker run -d --name ffc-restore-grafana --network ffc-restore --user 472:0 \
    -v ffc-restore-grafana:/var/lib/grafana -p 127.0.0.1:3006:3000 "$gf" >/dev/null
  for i in $(seq 120); do
    docker exec ffc-restore-webui curl -sf localhost:8080/health >/dev/null 2>&1 && break
    [ "$(docker inspect -f "{{.State.Running}}" ffc-restore-webui)" = true ] \
      || { docker logs --tail 30 ffc-restore-webui; echo "FAILED: Open WebUI stopped (log above)"; exit 1; }
    [ "$i" -lt 120 ] \
      || { docker logs --tail 30 ffc-restore-webui; echo "FAILED: Open WebUI did not answer within 10 minutes"; exit 1; }
    sleep 5
  done
  echo "OK 4: the copy is running: Open WebUI on 127.0.0.1:3005, Grafana on 127.0.0.1:3006"
' _ "$B"
```

```bash
# 5. Automatic checks. Every line must start with "ok".
sudo bash -c '
  B=$1 M=$1/MANIFEST.txt
  got=$(docker exec ffc-restore-webui curl -sf localhost:8080/api/version | sed -n "s/.*\"version\": *\"\([^\"]*\)\".*/\1/p")
  want=$(sed -n "s/^open-webui:.*\"version\": *\"\([^\"]*\)\".*/\1/p" "$M")
  if [ -n "$got" ] && [ "$got" = "$want" ]; then echo "ok   Open WebUI $got, as in the backup"; else echo "DIFF Open WebUI runs ${got:-nothing}, the backup is from $want"; fi
  # with no users, Open WebUI offers first-time setup ("onboarding")
  if docker exec ffc-restore-webui curl -sf localhost:8080/api/config | grep -q "\"onboarding\": *true"; then
    echo "DIFF users: none; the restored database is not in use"
  else echo "ok   users: present (no first-time setup offered)"; fi
  sed -n "s/^  \([A-Za-z0-9._-]*\): \([0-9]*\) points$/\1 \2/p" "$M" | while read -r c n; do
    got=$(docker exec ffc-restore-webui curl -sf -X POST "http://ffc-restore-qdrant:6333/collections/$c/points/count" \
            -H "Content-Type: application/json" -d "{\"exact\":true}" | sed -n "s/.*\"count\":\([0-9]*\).*/\1/p")
    if [ "$got" = "$n" ]; then echo "ok   Qdrant $c: $n points"; else echo "DIFF Qdrant $c: ${got:-none} now, $n in the backup"; fi
  done
  for i in $(seq 24); do
    docker exec ffc-restore-grafana wget -qO- localhost:3000/api/health 2>/dev/null | grep -q "\"database\": *\"ok\"" && break
    sleep 5
  done
  if [ "$i" -lt 24 ]; then echo "ok   Grafana database"; else echo "DIFF Grafana did not report a working database: sudo docker logs ffc-restore-grafana"; fi
' _ "$B"
```

| Block | What it does |
|---|---|
| 1 | Changes nothing. Checks the backup (finished, checksums, files, snapshot, archives unpack), that the manifest names all three images, that no earlier drill is left, and the disk space |
| 2 | Creates the network. A throwaway container of the backup's Open WebUI image unpacks the data folder, the uploads and Grafana into new volumes, and copies both databases |
| 3 | A one-off Qdrant container loads the snapshot into a new volume (checks every 5 s, stops at once if Qdrant exits, gives up after 10 minutes), then Qdrant starts normally on it |
| 4 | Starts Open WebUI with the backup's `production.env`, pointed at the copy's Qdrant, without Redis or metrics, and Grafana; waits up to 10 minutes. The first start is slow: Open WebUI downloads its models |
| 5 | Checks the Open WebUI version, that users exist, each Qdrant collection's point count against the manifest (exact after a consistent backup), and Grafana's database |

## Check by hand

1. On your laptop, open a tunnel and leave it open (works in PowerShell, cmd and bash):
   ```
   gcloud compute ssh qasap-vm01 --zone us-east4-c --project wsb-hc-qasap-ae2e --ssh-flag="-L 3005:localhost:3005" --ssh-flag="-L 3006:localhost:3006"
   ```
2. <http://localhost:3005>: sign in with your **production** account. Admin Panel → Users matches
   production; a student's older chat opens.
3. **Knowledge base:** in a new chat, type `#`, pick a course knowledge base, ask one question its
   documents answer. The reply must cite them.
4. **Grafana:** compare the copy's database with production's (read-only, no login needed). Both
   halves must show the same counts and titles:
   ```bash
   sudo bash -c '
     copy=$(docker volume inspect -f "{{.Mountpoint}}" ffc-restore-grafana)/grafana.db
     for db in /opt/grafana/data/grafana.db "$copy"; do
       echo "== $db"
       for t in dashboard user data_source; do echo "  $t: $(sqlite3 -readonly "$db" "select count(*) from $t;")"; done
       sqlite3 -readonly "$db" "select title from dashboard order by title;" | sed "s/^/    /"
     done
   '
   ```
   With production's Grafana login you can also check <http://localhost:3006>: the dashboards are
   listed, without data (Prometheus is not part of the copy).

## Remove it

```bash
# 6. Remove the copy: only the ffc-restore-* containers, volumes and network. The backup stays.
sudo bash -c '
  for c in ffc-restore-webui ffc-restore-grafana ffc-restore-qdrant ffc-restore-qdrant-load; do
    docker rm -f "$c" >/dev/null 2>&1 || true
  done
  for v in ffc-restore-webui ffc-restore-uploads ffc-restore-grafana ffc-restore-qdrant; do
    docker volume rm "$v" >/dev/null 2>&1 || true
  done
  docker network rm ffc-restore >/dev/null 2>&1 || true
  if [ -z "$(docker ps -aq --filter name=^ffc-restore-)$(docker volume ls -q --filter name=^ffc-restore-)" ]; then
    echo "OK 6: the copy is removed"
  else
    echo "FAILED: something is left: sudo docker ps -a; sudo docker volume ls"
  fi
'
```

It removes each container, volume and network by its exact name, so it cannot remove anything
else. Then record the date, the backup folder and who ran the drill in the A2 pull request or
`HANDOFF.md`.

## Runs

- **2026-10-30, qasap-vm01**, backup `2026-10-30_143305` (consistent, Open WebUI stopped 33 s,
  703 MB). Blocks 1–5 all `OK`; block 5: Open WebUI 0.8.12, users present, Qdrant
  `open-webui_files` 1625 and `open-webui_knowledge` 1010 points as in the backup, Grafana
  database ok. Grafana content identical to production: 7 dashboards, 1 user, 1 data source.
- **2026-10-30, local**, the same blocks on a fake production with the production images
  (Open WebUI 0.8.12, Qdrant 1.17.1, Grafana 9.0.0): sign-in, chat, knowledge base, file contents,
  a vector query and the dashboard all came back; a second block 1 was refused while the copy
  existed; block 6 left nothing behind.

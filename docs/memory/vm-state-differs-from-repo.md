---
name: vm-state-differs-from-repo
description: The production VM (qasap-vm01) does not match the repo — older versions running, a staging stack with forbidden core mounts, extra copies of student data. Check the VM, not the compose file.
metadata:
  type: project
---

Observed on qasap-vm01 on 2026-10-17 (`docker ps`, `docker inspect`, `ls`, read-only), while
scoping task A2:

- **Production versions.** Production runs Open WebUI **0.8.12** (custom-built image
  `familyfinancechat-open-webui`) and Qdrant **1.17.1**. The containers were created about six
  months earlier. `main`'s `docker-compose.yml` pins `open-webui:v0.11.4` and `qdrant:v1.19.1`
  (commit `f247c21`, A8), so `docker compose up -d` from an up-to-date checkout would upgrade
  production three minor versions at once, through the one-way v0.10.0 migration.
- **Deployed checkout.** It is at `/opt/openwebui/FamilyFinanceChat`, on `main` at `fca2531`
  (2026-04-25), 42 commits behind `main` as of 2026-10-17. git refuses to run there
  ("dubious ownership"). Use `git -c safe.directory=/opt/openwebui/FamilyFinanceChat ...`; do
  not chown the folder.
- **Staging.** A staging stack is still running (`open-webui-staging` on :3002, plus
  `qdrant-staging` and `redis-staging`), although `HANDOFF.md` says it was removed. It
  bind-mounts `custom-code/main.py`, `config.py`, `observability.py` and `custom_pdf_router.py`
  over `/app/backend/open_webui/`. That is the forbidden fork pattern (policy F1). Its data is
  in a former student's home directory, `/home/mdhanif_wisc_edu/ffc-staging/`.
- **Student data permissions.** `/opt/openwebui/data/webui.db` is mode 777.
  `webui.db.bak_api_key_fix` is a full 633 MB copy of the database that every user on the VM
  can read. Two more older copies sit beside it.
- **Unrelated container.** `ibgw-live-ib-gateway-live-1` (an Interactive Brokers gateway) is
  not part of this project, and it publishes ports 4003 and 5901 (VNC) on all interfaces.
- **Uploads.** `/mnt/gcs/fin602` holds the uploaded files, but `mount` does not list it, so it
  is probably a plain directory on the VM disk rather than a Cloud Storage bucket.
- **Disk.** `/opt/openwebui/data` is 7.4 GB, of which `cache/` is 6.1 GB. The root disk was
  79% full (21 GB free).

None of this was changed. Raising it with the team and instructor was planned for M2.

**Why:** the repo's compose files and docs describe an intended state, not the running one.
Anyone who plans a backup, an upgrade or a cleanup from the repo alone will plan for a system
that isn't there. The version mismatch in particular turns a routine deploy into an
unrehearsed, irreversible upgrade.

**How to apply:** before any VM operation, inspect the running containers
(`sudo docker ps`, `sudo docker inspect <name>`) instead of trusting the compose file. Never
run `docker compose up` from the deployed checkout until its pins match what production
actually runs. When an item here gets resolved, update or delete this file. Backup procedure:
[`docs/backup/backup-restore.md`](../backup/backup-restore.md) and [`docs/backup/restore-drill.md`](../backup/restore-drill.md). Related: [[secrets-never-in-repo]].

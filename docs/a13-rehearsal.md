# A10/A11 image rehearsal setup

The committed CI stack runs Open WebUI v0.9.6 with an empty database. Use this
override to start a **separate** stack at an approved v0.10.x or v0.11.x image,
without reusing CI's fixed container names or published ports. Docker Compose
v2.24.4 or newer is required for `!reset` and `!override`.

```bash
cp .env.test.example .env.test
sed -i '/^WEBUI_SECRET_KEY=/d' .env.test
printf 'WEBUI_SECRET_KEY=%s\n' "$(openssl rand -hex 32)" >> .env.test
chmod 600 .env.test
export REHEARSAL_OPENWEBUI_IMAGE='ghcr.io/open-webui/open-webui:v0.11.4'
docker compose -p ffc-a13-rehearsal -f docker-compose.test.yml \
  -f docker-compose.ci.yml -f docker-compose.rehearsal.yml config --quiet
docker compose -p ffc-a13-rehearsal -f docker-compose.test.yml \
  -f docker-compose.ci.yml -f docker-compose.rehearsal.yml \
  up -d open-webui-test qdrant-test redis-test mock-openai-test
python scripts/smoke_test.py --base-url http://127.0.0.1:43183
```

For v0.10.x, use the **exact patch version and digest approved by A10**, not a
guessed tag. The same applies to A11's v0.11.x image. This command starts with
blank volumes and confirms API compatibility only. A9–A11 acceptance still
requires the A2-restored production-data copy, an existing-user token, and the
grading extraction/metrics checks in `SETUP.md`. The A2 PR must deliver and
verify its restore tooling before that run; `docker-compose.rehearsal.yml` does
not restore data. Run one approved version at a time against a fresh restorable
copy, recording the image digest and results. Never point this project at the
production bind mounts or volumes.

After the blank-data check, remove only this named project:

```bash
export REHEARSAL_OPENWEBUI_IMAGE='ghcr.io/open-webui/open-webui:v0.11.4'  # same image used above
docker compose -p ffc-a13-rehearsal -f docker-compose.test.yml \
  -f docker-compose.ci.yml -f docker-compose.rehearsal.yml down -v
```

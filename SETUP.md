<!--
A24 REMINDER: documentation work still open. Do this LAST, after the team's feature freeze.
  1. Professor-facing guides: rewrite for the v0.11 interface and re-shoot their screenshots.
  2. This file: add every screenshot marked "SCREENSHOT:" and confirm every click-path marked "TODO-UI".
  3. Resolve every "TODO-A19" and "TODO-A24" marker (pending decisions from P1/P2).
  4. Newcomer acceptance test on a fresh VM, then remove this comment and the status note below.
Full checklist: docs/A24_REMAINING_WORK.md
Find everything left:  grep -rn -E "TODO-UI|TODO-A19|TODO-A24|SCREENSHOT:" --include=*.md .
-->

# Setup Guide

How to get FamilyFinanceChat running from scratch on a fresh Linux VM. Follow the steps in
order. Each step tells you what you should see when it worked; if you see something else, check
[Common errors](#common-errors) before moving on.

> **Status (task A24, in progress):** the commands in this guide follow the current `main`
> branch, including the A16 reverse proxy and the A15 hosted grading dashboard. Some menu
> click-paths are still being re-checked against the Open WebUI v0.11 interface, and screenshots
> will be added at the end of the semester. If a menu doesn't match what you see, add a line to
> [`docs/A24_REMAINING_WORK.md`](docs/A24_REMAINING_WORK.md).

<!-- TODO-A24: this guide was merged with main's A15/A16 changes. Walk through every step on a
fresh VM before release to confirm the merged commands. -->

---

## What you will end up with

| Service | Address | What it is |
|---|---|---|
| `proxy` | `https://<PUBLIC_HOSTNAME>` (ports 80 and 443) | Caddy reverse proxy: TLS certificates, routing, and the instructor login for grading. The only public entry point |
| `open-webui` | `https://<PUBLIC_HOSTNAME>/` (`127.0.0.1:3000` on the VM) | The chat app students use (Open WebUI v0.11.4) |
| `grading` | `https://<PUBLIC_HOSTNAME>/grading/` | Grading dashboard for instructors (no port of its own) |
| `grafana` | `https://<PUBLIC_HOSTNAME>/grafana/` (`127.0.0.1:3001` on the VM) | Dashboards |
| `prometheus` | `http://localhost:9090` on the VM | Metrics database (30-day retention) |
| `pushgateway` | `http://localhost:9091` on the VM | Receives chat metrics from the Chat Metrics filter |
| `otel-collector` | `127.0.0.1:8889` on the VM | Receives Open WebUI's built-in OpenTelemetry metrics |
| `cadvisor` | `http://localhost:8080` on the VM | Container CPU/memory metrics |
| `metrics-exporter` | `http://localhost:8001` on the VM | Health-probe sidecar for Open WebUI |
| `qdrant` | internal only | Vector database for document search (RAG) |
| `redis` | internal only | Valkey, for WebSocket session state |

Only the proxy's ports 80 and 443 should ever be reachable from outside the VM. Everything else is
bound to localhost or is internal-only, and most of those services have no login.

<!-- TODO-A19: pushgateway is commented out in docker-compose.yml on main and has no scrape job in
prometheus.yml. P1 must restore both (the team decided to keep the filter). Remove this
comment once merged. -->

---

## Prerequisites

| Requirement | Details |
|---|---|
| Linux VM | Ubuntu 22.04 or 24.04 LTS. GCP `e2-standard-4` (4 vCPU, 16 GB RAM) recommended; Open WebUI alone is allowed up to 10 GB RAM |
| Disk | Boot disk of at least 50 GB, with **at least 30 GB free** before you start. The Open WebUI image is several GB, and upgrades need room for the old image, the new image and a backup at the same time |
| Hostname | A DNS name pointing at the VM's external IP. See [below](#choose-a-hostname-and-open-the-firewall-gcp) |
| Docker | Docker Engine 24+ with Compose v2.24.4+ (`docker compose`, with a space, not `docker-compose`) |
| Git | Any recent version |
| OpenAI API key | Used for chat, document embeddings and grading. Set a spending limit on it |
| Optional | Python 3.10+ (knowledge-base sync tool), Node.js **v20.19.5** with npm **10.8.2** (grading dashboard development on your laptop) |

Check what you have:

```bash
docker --version          # 24.x or newer
docker compose version    # v2.24.4 or newer
git --version
df -h /                   # "Avail" should be 30G or more
```

If `docker` commands fail with `permission denied ... docker.sock`, add yourself to the
`docker` group, then **log out and back in**:

```bash
sudo usermod -aG docker "$USER"
```

### Choose a hostname and open the firewall (GCP)

Everyone reaches the stack through the reverse proxy on ports 80 and 443, at one hostname you
choose. The proxy gets a TLS certificate for that name from Let's Encrypt automatically.

Testing on your own machine instead of a VM? Use `PUBLIC_HOSTNAME=localhost` in step 2a and skip
this section.

1. **Hostname.** Choose the name and create a DNS `A` record pointing it at the VM's external IP
   (shown in the GCP console under **Compute Engine → VM instances**). Use a reserved (static)
   external IP so the record does not go stale when the VM restarts. If no university DNS name
   is available yet, `<external-ip>.sslip.io` resolves to that IP with no setup, and Let's Encrypt
   issues certificates for it. Treat it as temporary: the URL changes if the IP does.
2. **Firewall.** Open TCP 80 and 443 (and UDP 443 for HTTP/3) to the VM. Port 80 is required:
   Let's Encrypt validates over it, and the proxy redirects it to HTTPS.

   ```bash
   gcloud compute instances add-tags <VM_NAME> --tags=http-server,https-server \
     --zone <ZONE> --project <PROJECT_ID>
   ```

   The `http-server` and `https-server` tags only open ports if the project has the default
   `default-allow-http` and `default-allow-https` firewall rules. Check with
   `gcloud compute firewall-rules list`.
3. **Close the old ports.** Remove any firewall rule that opened 3000, 3001, 8000, 8001, 8080,
   8889, 9090 or 9091 to the internet. If you followed an older version of this guide, that
   includes `ffc-allow-web`: `gcloud compute firewall-rules delete ffc-allow-web`.

**Reaching the internal tools.** Prometheus, cAdvisor, the metrics exporter and the OTel collector
are not routed through the proxy because they have no login. To open one from your laptop, use an
SSH tunnel, for example:

```bash
gcloud compute ssh <VM_NAME> --zone <ZONE> -- -L 9090:localhost:9090
```

and then open `http://localhost:9090` while the session is open.

---

## 1. Clone the repository

```bash
git clone https://github.com/mfedenia/FamilyFinanceChat
cd FamilyFinanceChat
```

All later commands are run from this `FamilyFinanceChat` directory unless a step says otherwise.

---

## 2. Create the configuration files

The stack needs some files that are **not** in Git, because they hold secrets or passwords.

### 2a. `.env` (Open WebUI, proxy, Grafana and alerts)

Create it with placeholders:

```bash
cat > .env <<'EOF'
# --- Public address (reverse proxy) ---
PUBLIC_HOSTNAME=REPLACE_ME

# --- Chat model provider ---
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=REPLACE_ME
ENABLE_OLLAMA_API=false

# --- Document search (RAG) ---
RAG_EMBEDDING_ENGINE=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_OPENAI_API_BASE_URL=https://api.openai.com/v1
RAG_OPENAI_API_KEY=REPLACE_ME
RAG_TOP_K=5
RAG_RELEVANCE_THRESHOLD=0.30
RAG_FILE_MAX_SIZE=50
RAG_ALLOWED_FILE_EXTENSIONS=pdf,md,txt,docx

# --- Security ---
WEBUI_SECRET_KEY=REPLACE_ME
ENABLE_PERSISTENT_CONFIG=true

# --- Grafana (read by docker compose) ---
GRAFANA_ADMIN_PASSWORD=REPLACE_ME

# --- Alert email (A20; read by docker compose) ---
ALERT_EMAIL=REPLACE_ME
SMTP_HOST=REPLACE_ME
SMTP_USER=REPLACE_ME
SMTP_PASSWORD=REPLACE_ME
# SMTP_FROM_ADDRESS=   optional; defaults to SMTP_USER
EOF
chmod 600 .env
```

Generate the secret key automatically:

```bash
sed -i "s/^WEBUI_SECRET_KEY=.*/WEBUI_SECRET_KEY=$(openssl rand -hex 32)/" .env
```

Open the file (`nano .env`) and replace the remaining `REPLACE_ME` values:

| Variable | What to put |
|---|---|
| `PUBLIC_HOSTNAME` | The hostname from [Prerequisites](#choose-a-hostname-and-open-the-firewall-gcp), e.g. `fin602.example.edu` (no `https://`, no trailing slash). Use `localhost` on your own machine |
| `OPENAI_API_KEY` | Your OpenAI key (`sk-...`) |
| `RAG_OPENAI_API_KEY` | The same key is fine |
| `GRAFANA_ADMIN_PASSWORD` | A password you choose for Grafana's `admin` user. Save it in a password manager |
| `ALERT_EMAIL` | The person Grafana alerts go to. A real, monitored address |
| `SMTP_HOST` | The mail server Grafana sends alerts through, as `host:port`, e.g. `smtp.gmail.com:587` |
| `SMTP_USER` | The SMTP login. For Gmail, the account address |
| `SMTP_PASSWORD` | The SMTP password. For Gmail, an [app password](https://myaccount.google.com/apppasswords), not the account password |

Check that nothing is left:

```bash
grep -n REPLACE_ME .env    # should print nothing
```

Do **not** add `QDRANT_URI`, `REDIS_URL`, `WEBSOCKET_REDIS_URL`, `VECTOR_DB`, `WEBUI_URL`,
`CORS_ALLOW_ORIGIN` or the `OTEL_*` variables. `docker-compose.yml` already sets them.

> **Important:** with `ENABLE_PERSISTENT_CONFIG=true`, most Open WebUI settings (connections,
> document settings, the WebUI URL) are copied from `.env` into the database **on first start
> only**. After that, editing `.env` does nothing for those settings; change them in the Admin
> Panel instead.

`.env` is already in `.gitignore`. Never commit it.

<!-- TODO-A24 (P1): compare this template with the production .env (values redacted) and commit
it as .env.example, so this step becomes "cp .env.example .env". -->

### 2b. `grading_feature/.env` (grading dashboard)

```bash
cat > grading_feature/.env <<'EOF'
OPENAI_API_KEY=REPLACE_ME
OPENWEBUI_API_KEY=FILL_IN_AT_STEP_7
EOF
chmod 600 grading_feature/.env
nano grading_feature/.env    # put your OpenAI key in OPENAI_API_KEY
```

Leave `OPENWEBUI_API_KEY` as it is for now. That key doesn't exist until you create an admin
account, and you will come back to it in [step 7](#7-create-an-api-key-for-the-grading-backend).
Details of the hosted grading service are in
[`grading_feature/README.md`](grading_feature/README.md#hosted-docker).

Note: the grading backend quietly switches to fake scores if `OPENAI_API_KEY` is empty. Make sure
the key is filled in.

Older versions of this guide used `grading_feature/backend/grading.env`. That file is no longer
read.

<!-- TODO-A24 (P2): confirm against grading_feature/README.md whether the hosted service also
reads OPENAI_MODEL and MOCK_SCORER (the old grading.env had both) and add them here if so. -->

### 2c. Instructor logins for the grading dashboard

The proxy protects `/grading/` with one username and password per instructor.

```bash
cp proxy/grading_users.caddy.example proxy/grading_users.caddy
docker compose run --rm --no-deps proxy caddy hash-password   # prompts for a password, prints a hash
```

The command prompts for the password, so it never reaches your shell history. Run it once per
instructor, and add one line per instructor to `proxy/grading_users.caddy`:

```
<username> <bcrypt hash>
```

Then protect the file:

```bash
sudo chmod 600 proxy/grading_users.caddy
```

The file is in `.gitignore`. Without it the dashboard answers 401 to everyone, and the rest of the
site still works.

### 2d. Check the configuration before downloading anything

```bash
docker compose config --quiet && echo "Compose config OK"
```

You should see `Compose config OK`. This catches a missing `.env`, a missing `PUBLIC_HOSTNAME`,
`GRAFANA_ADMIN_PASSWORD`, `ALERT_EMAIL` or `SMTP_*` value and YAML mistakes in seconds, before step 4 downloads several GB. It
does **not** check whether your keys are valid; the `grep REPLACE_ME` check in 2a covers the
placeholders.

Always keep `--quiet`. Without it, the command prints the full configuration **including your
secrets**, so never paste its output into chat, issues or pull requests.

---

## 3. Prepare host directories

```bash
sudo mkdir -p /opt/openwebui/data /opt/qdrant/storage /mnt/gcs/fin602
```

- `/opt/openwebui/data` holds Open WebUI's database and app data.
- `/opt/qdrant/storage` holds the document vectors.
- `/mnt/gcs/fin602` holds uploaded files. For a new or test stack, the plain local directory
  created above is all you need.

Prometheus, Grafana, Redis, the grading dashboard and the proxy's certificates use Docker named
volumes, so there is nothing to create for them.

<!-- TODO-A24 (P1): document the production uploads storage. The 2026-10-17 inventory found
/mnt/gcs/fin602 was a plain directory on the VM, not a GCS mount. State the decision here and,
if it is a bucket, the exact mount steps. -->

---

## 4. Start the stack

First check that nothing else on the VM is using ports 80 or 443 (other projects share some VMs):

```bash
sudo ss -ltnp '( sport = :80 or sport = :443 )'    # must list no listeners
```

Then start everything:

```bash
docker compose pull
docker compose up -d --build
```

The first run downloads several GB of images and builds the local images (the grading dashboard
and `metrics-exporter`). Expect 5–15 minutes.

Check that everything is up:

```bash
docker compose ps
```

You should see these services with status `running` (or `Up`): `proxy`, `qdrant`, `redis`,
`otel-collector`, `open-webui`, `grading`, `cadvisor`, `prometheus`, `pushgateway`,
`metrics-exporter`, `grafana`. `open-webui` may show `health: starting` for the first minute or
two while it creates its database.

Then check Open WebUI directly on the VM:

```bash
curl -s http://localhost:3000/health
```

You should see `{"status":true}`. If you get nothing or an error, wait a minute and retry. If it
still fails, look at the logs:

```bash
docker compose logs open-webui --tail 100
```

Finally, check that the proxy got its certificate (skip this on a workstation with
`PUBLIC_HOSTNAME=localhost`):

```bash
docker compose logs -f proxy        # wait for "certificate obtained successfully", then Ctrl+C
curl -s https://<PUBLIC_HOSTNAME>/ready
```

The `curl` should print `{"status":true}`. If the certificate never arrives, see
[Common errors](#common-errors).

---

## 5. Create the admin account

1. Open `https://<PUBLIC_HOSTNAME>/` in your browser.
2. Click **Get started** and sign up. **The first account created becomes the administrator.** Use
   an email you control and save the password.

<!-- SCREENSHOT: setup/05-first-signup.png | first-run sign-up page -->
<!-- TODO-UI: confirm the first-run button and page wording on v0.11.4 -->

You should land on an empty chat page with a model selector at the top.

---

## 6. Send a test chat

1. Open the model selector and choose `gpt-4o-mini` (any OpenAI model in the list works).
2. Type `Hello` and send it.

You should get a reply within a few seconds.

<!-- SCREENSHOT: setup/06-model-picker.png | v0.11 model picker with OpenAI models listed -->
<!-- TODO-UI: confirm where the model picker is in v0.11.4 (A11 says it moved) -->

**If the model list is empty:** go to **Admin Panel → Settings → Connections** and check that
the OpenAI connection is enabled and shows your key. You reach the Admin Panel by clicking your
name or avatar in the bottom-left corner. <!-- TODO-UI -->

**If the page loads but the reply never arrives:** see the WebSocket row in
[Common errors](#common-errors).

---

## 7. Create an API key for the grading backend

The grading dashboard and the knowledge-base sync tool talk to Open WebUI through its API.

1. Allow API keys: **Admin Panel → Settings → General → Enable API Keys** → on → **Save**.
   <!-- TODO-UI: confirm the toggle name and location on v0.11.4 -->
2. Create a key: click your name (bottom-left) → **Settings → Account → API keys → Create new
   key**. Copy the key; it starts with `sk-`. <!-- TODO-UI -->

<!-- SCREENSHOT: setup/07-api-key.png | Settings > Account > API keys, key blurred -->

3. Put it in the grading dashboard's config. Open `grading_feature/.env`, replace
   `FILL_IN_AT_STEP_7` with the key, then recreate the container so it reads the new value:

```bash
docker compose up -d grading      # "restart" does NOT re-read env files; use "up -d"
curl -s -o /dev/null -w '%{http_code}\n' -u <instructor-user> https://<PUBLIC_HOSTNAME>/grading/api/health
```

`curl` asks for the instructor password from step 2c, and should print `200`.

This key has your admin account's permissions. Keep it private.

---

## 8. Install the Chat Metrics filter

**Why this is a manual step:** Open WebUI's built-in OpenTelemetry export (already running) does
not provide all the chat metrics we need, so the team kept our own filter. See
<!-- TODO-A19: link to the A19 justification paragraph --> the A19 decision for exactly which
metrics. Open WebUI has no documented way to install a filter from the command line, so it is
pasted in through the admin screens.

**When to do it:** once on every new deployment. The filter is stored in Open WebUI's
database, so it survives restarts and upgrades of the same data. Redo it whenever you start from
an empty database. <!-- TODO-A24: confirm on v0.11.4 that the filter survives an upgrade -->

1. Go to **Admin Panel → Functions**. <!-- TODO-UI: older guides say "Workspace > Functions";
   confirm the v0.11.4 location -->
2. Click **+** (new function).
3. Delete the example code, then paste the **entire** contents of
   `monitoring/chat_metrics_filter.py`. To print the file on the VM:
   `cat monitoring/chat_metrics_filter.py`.
4. Give it the name `Chat Metrics`, then click **Save**.
5. Back in the Functions list, switch **Chat Metrics** on, then open its **⋯** menu and turn
   **Global** on, so it runs for every model and every user. <!-- TODO-UI -->
6. Click the gear icon next to it (**Valves**) and check that `pushgateway_url` is
   `http://pushgateway:9091` (this is the default). Click **Save**. <!-- TODO-UI -->

<!-- SCREENSHOT: setup/08-functions-list.png | Chat Metrics enabled and Global on -->
<!-- SCREENSHOT: setup/08-valves.png | Valves with pushgateway_url -->

**Check that it works:** send one chat message (as in step 6), then on the VM run:

```bash
curl -s http://localhost:9091/metrics | grep openwebui_chat
```

You should see lines starting with `openwebui_chat`. Nothing appears until at least one chat
message has been sent after the filter was enabled.

---

## 9. Load the course documents (knowledge base)

### 9a. Create the knowledge base in the browser

1. Go to **Workspace → Knowledge → +** (new knowledge). Give it a name, for example
   `FIN602 Client Data`. <!-- TODO-UI -->
2. Upload files by dragging them onto the page. Wait until each file finishes processing.
3. Note the knowledge base's ID: it is the last part of the page address, after
   `/workspace/knowledge/`. You only need it for the sync tool in 9b. <!-- TODO-UI -->

<!-- SCREENSHOT: setup/09-knowledge-upload.png -->

### 9b. Bulk upload with the sync tool (optional)

For many files, use the command-line tool instead of dragging:

```bash
cd tools/kb_sync
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
```

Set the three values in `tools/kb_sync/.env`:

| Variable | Value |
|---|---|
| `OPENWEBUI_BASE_URL` | `http://localhost:3000` if you run the tool on the VM, or `https://<PUBLIC_HOSTNAME>` from your laptop. **The example file points at the production server; change it, or you will upload into production** |
| `OPENWEBUI_API_KEY` | The key from step 7 |
| `OPENWEBUI_KB_ID` | The ID from step 9a |

```bash
python sync_kb.py list                                  # should list the KB's files
python sync_kb.py sync ./path/to/documents/ --dry-run   # shows what would change
python sync_kb.py sync ./path/to/documents/             # does it
deactivate && cd ../..
```

### 9c. Check that document search works

Attach the knowledge base to a model: **Workspace → Models → (edit or create a model) →
Knowledge → select your knowledge base → Save**. <!-- TODO-UI --> Then start a new chat with that
model and ask a question that only the uploaded documents can answer. The answer should use the
document, and a source citation should appear under the reply.

The course's role-play "client" models (system prompts and attached documents) are set up by the
professor; see the professor guide. <!-- TODO-A24: link the rewritten professor guide -->

---

## 10. Grafana and alerting

1. Open `https://<PUBLIC_HOSTNAME>/grafana/` and log in as `admin` with the
   `GRAFANA_ADMIN_PASSWORD` from your `.env`.
2. Go to **Dashboards → FamilyFinanceChat → Open WebUI OpenTelemetry**.

The Prometheus data source, the dashboards, the alert rules and the alert email are all created
automatically from `monitoring/grafana/provisioning/` at startup. There is nothing to click. The
panels (requests per second, 95th percentile response time) fill in within about a minute of
using the chat.

<!-- SCREENSHOT: setup/10-grafana-dashboard.png -->

To check that Prometheus is collecting from every source, run this on the VM:

```bash
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"[a-z]*"' | sort | uniq -c
```

Every target should be `"health":"up"`.

### Alerts (A20)

Grafana emails `ALERT_EMAIL` when one of these rules (in
`monitoring/grafana/provisioning/alerting/rules.yml`) fires:

| Rule | Fires when |
|---|---|
| Open WebUI memory above 80% of its limit | the `open-webui` container stays above 80% of its 10 GB limit for 10 minutes, or the memory metric disappears |
| Chat completion error rate above 5% | more than 5% of `/api/chat/completions` requests return 400 or 5xx over 10 minutes (at least 5 requests), for 5 minutes |

Change alerts by editing those files, not the Grafana UI; provisioned rules are read-only there.

The error-rate rule cannot see model-provider failures in web-UI chats: Open WebUI processes
those in a background task and reports the error over the WebSocket after the HTTP request has
already returned 200. See `HANDOFF.md`.

**Check that email works:** in Grafana, go to **Alerting → Contact points → `on-call-email` →
Test**. <!-- TODO-UI --> A test email should reach `ALERT_EMAIL` within a minute or two.

### External uptime check (once per GCP project)

Grafana runs on the same VM it watches, so it cannot report the VM going down. A GCP uptime check
against Open WebUI's `/ready` (which returns 200 only when the database and Redis are both up)
covers that. Create it from a machine with `gcloud` signed in to the project:

```bash
GCP_PROJECT=<PROJECT_ID> UPTIME_HOST=<PUBLIC_HOSTNAME> ALERT_EMAIL=<person> \
  bash scripts/gcp_uptime_alert.sh
```

It is safe to re-run. It checks `https://<PUBLIC_HOSTNAME>:443/ready` through the reverse proxy,
so ports 80 and 443 must be open to the addresses listed by `gcloud monitoring uptime list-ips`
(the firewall step in [Prerequisites](#choose-a-hostname-and-open-the-firewall-gcp) already
allows them). Opening port 3000 would not help: it is bound to `127.0.0.1`.

**Acceptance test.** Run it after any change to alerting, and once per semester:

```bash
docker compose stop redis      # /ready starts returning 503
# within 5 minutes: GCP "FamilyFinanceChat /ready down" email to ALERT_EMAIL
docker compose start redis     # the incident closes; GCP sends a recovery email
```

Students can't chat while Redis is stopped, so do this outside class hours.

### Grafana password

**Forgot the Grafana password?** Changing `GRAFANA_ADMIN_PASSWORD` later has no effect, because
it is only applied the first time Grafana starts. Reset it instead. This reads the new password
from the keyboard, so it stays out of your shell history:

```bash
read -rs GF_PW && printf '%s\n' "$GF_PW" | \
  docker compose exec -T grafana grafana-cli admin reset-admin-password --password-from-stdin
unset GF_PW
```

---

## 11. Grading dashboard

The `grading` service started with the rest of the stack in step 4. It has no port of its own;
the proxy is the only way in.

1. Open `https://<PUBLIC_HOSTNAME>/grading/` and sign in with an instructor login from step 2c.
2. Click **Refresh** to pull the latest chats from Open WebUI.

**Refresh costs money:** it scores student questions through the OpenAI API and can take several
minutes on a full class. Click it once and wait.

<!-- SCREENSHOT: grading/home.png and grading/refresh.png (re-shoot; current ones are pre-v0.11) -->

**VM that ran the old `grading-api` container (A14):** move its keys from
`grading_feature/backend/grading.env` into `grading_feature/.env` (the old file is no longer
read), then remove the leftover container and volume:

```bash
docker compose up -d --remove-orphans
docker volume ls                              # find the exact name of the old grading-data volume
docker volume rm <project>_grading-data
```

The `grading` service uses a new volume, `grading-dashboard-data`, so the old root-owned one is
not needed.

### Developers: running the dashboard on your laptop

<!-- TODO-A24: consider moving this subsection to grading_feature/README.md. -->

This runs the dashboard on your own machine and connects it to a deployed Open WebUI. It uses its
own `grading_feature/.env` **on your laptop**, separate from the one on the VM.

1. Install Python 3.10+, and Node.js **exactly v20.19.5** with npm **exactly 10.8.2**.
   `run_app.sh` refuses other versions. The easiest way is [nvm](https://github.com/nvm-sh/nvm):
   `nvm install 20.19.5 && nvm use 20.19.5`, then check with `node --version` and `npm --version`.
2. Configure it:

```bash
cd grading_feature
cp .env.example .env
nano .env
```

| Variable | Value |
|---|---|
| `OPENWEBUI_BASE_URL` | `https://<PUBLIC_HOSTNAME>` |
| `OPENWEBUI_API_KEY` | Uncomment the line and paste the key from step 7 |
| `OUTPUT_PATH` and `DATA_PATH` | The same file path on your laptop, for example `/home/you/ffc-data/chats.json` (the folder must exist) |
| `OPENAI_API_KEY` | Add this line with your OpenAI key; it is used for scoring |

3. Start it:

```bash
./run_app.sh
```

The backend starts on port 9500 and the frontend dev server starts after it. Open the address
the frontend prints (normally `http://localhost:5173`); it forwards `/api` calls to the backend.
Then click **Refresh** (same cost warning as above).

---

## You're done when

- [ ] `docker compose ps` shows every service from step 4 running
- [ ] `curl -sI http://<PUBLIC_HOSTNAME>/ | head -1` shows a 308 redirect, and
      `https://<PUBLIC_HOSTNAME>/` opens in a browser with no certificate warning
- [ ] A chat message gets a reply (step 6)
- [ ] A question about an uploaded document gets an answer with a citation (step 9c)
- [ ] `curl -s http://localhost:9091/metrics | grep openwebui_chat` prints lines (step 8)
- [ ] Grafana's **Open WebUI OpenTelemetry** dashboard shows data (step 10)
- [ ] The Grafana contact-point test email reaches `ALERT_EMAIL`, and the GCP uptime check
      exists (step 10)
- [ ] `https://<PUBLIC_HOSTNAME>/grading/` asks for a password, and an instructor login works
      (steps 7 and 11)

---

## Day-to-day commands

```bash
docker compose ps                         # what is running
docker compose logs <service> --tail 100  # recent logs, e.g. open-webui or proxy
docker compose up -d <service>            # apply a changed config/env file to one service
docker compose down                       # stop everything (data is kept)
docker compose up -d                      # start everything again
docker compose exec proxy caddy reload --config /etc/caddy/Caddyfile   # apply edited instructor logins, no downtime
```

**Never run `docker compose down -v` on a real deployment.** `-v` deletes the named volumes
(Redis, Prometheus, Grafana and grading data, and the proxy's certificates).

---

## Reverse proxy reference (A16)

The `proxy` service (Caddy, configuration in [`proxy/Caddyfile`](proxy/Caddyfile)) is the only
container published beyond `127.0.0.1`. It terminates TLS and routes by path:

| URL | Goes to | Who signs in, and how |
|---|---|---|
| `https://<PUBLIC_HOSTNAME>/` | Open WebUI | Students and staff, with Open WebUI's own login |
| `https://<PUBLIC_HOSTNAME>/grading/` | Grading dashboard | Instructors, with HTTP basic auth held by the proxy |
| `https://<PUBLIC_HOSTNAME>/grafana/` | Grafana | Maintainers, with Grafana's own login |

Prometheus, cAdvisor, the metrics exporter and the OTel collector are not routed: they have no
login. Reach them over SSH (see [Prerequisites](#choose-a-hostname-and-open-the-firewall-gcp)).

Open WebUI extension surface: reverse-proxy composition (policy A9). The only Open WebUI
settings involved are `WEBUI_URL` and `CORS_ALLOW_ORIGIN`, both documented environment
variables. Nothing here needs to change on the next upstream release.

### Checks after any change

```bash
H=<PUBLIC_HOSTNAME>
curl -sI http://$H/ | head -1                                       # 308 redirect to https
curl -s https://$H/ready                                             # {"status":true}
curl -s -o /dev/null -w '%{http_code}\n' https://$H/grading/          # 401 without a login
curl -s -o /dev/null -w '%{http_code}\n' -u <user> https://$H/grading/api/health   # 200
curl -s -o /dev/null -w '%{http_code}\n' https://$H/grafana/login     # 200
```

Then open the site in a browser and send one chat message. Chat replies stream over a WebSocket
through the proxy, and a failure there only shows up in a browser.

### Operations

- **Add, remove or change an instructor:** edit `proxy/grading_users.caddy` (step 2c), then apply
  it with no downtime: `docker compose exec proxy caddy reload --config /etc/caddy/Caddyfile`.
- **Certificates** renew automatically. They live in the `caddy-data` volume; do not delete it,
  or every recreate requests a new certificate and can hit Let's Encrypt's rate limits.
- **Recovery route if the proxy is down:** `ssh -L 3000:localhost:3000 <user>@<vm>` and open
  `http://localhost:3000` (Open WebUI), or forward 3001 and open `http://localhost:3001/grafana/`.
  The grading dashboard has no such route, because it is built for `/grading/` behind the proxy.

### Moving an existing install behind the proxy

Only for a VM that ran the stack before A16:

1. **Open WebUI's stored URL.** `WEBUI_URL` is saved in Open WebUI's database on first start, so
   the compose value does not replace an existing one. Set **Admin Panel → Settings → General →
   WebUI URL** to `https://<PUBLIC_HOSTNAME>`. <!-- TODO-UI -->
2. **Grafana password.** `GRAFANA_ADMIN_PASSWORD` only applies to a new Grafana database;
   production's Grafana still accepted `admin`/`admin` in October 2026. Reset it with the command
   in [step 10](#grafana-password), then check that `admin`/`admin` is refused at
   `https://<PUBLIC_HOSTNAME>/grafana/login`.
3. **Cut-over.** Students who bookmarked `http://<VM-IP>:3000` lose access when this lands,
   because that port becomes localhost-only. Announce the new URL first, and deploy outside a
   week with a student deadline. Open WebUI accounts, chats and sessions are unaffected: same
   container, same data.

### Single sign-on

The task asked for university SSO (OIDC) if it could be arranged. It needs the university to
register the application as an OIDC client, which was not arranged this term, so HTTP basic auth
is the documented fallback. When a client registration exists:

- **Open WebUI** supports OIDC natively through documented environment variables
  (`ENABLE_OAUTH_SIGNUP`, `OPENID_PROVIDER_URL`, `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`,
  `OAUTH_PROVIDER_NAME`). The redirect URI is `https://<PUBLIC_HOSTNAME>/oauth/oidc/callback`.
- **The grading dashboard** would sit behind an `oauth2-proxy` sidecar, wired in with Caddy's
  `forward_auth` in place of `basic_auth` and limited to instructor accounts. Nothing in the
  dashboard itself changes.

---

## Common errors

| What you see | Likely cause | Fix |
|---|---|---|
| `required variable PUBLIC_HOSTNAME is missing` | Root `.env` has no `PUBLIC_HOSTNAME` | Add it (step 2a); use `localhost` on a workstation |
| `docker compose up` says `Set GRAFANA_ADMIN_PASSWORD` | Missing from `.env` | Add `GRAFANA_ADMIN_PASSWORD=...` to `.env` (step 2a) |
| `docker compose up` says `Set ALERT_EMAIL...` or `Set SMTP_...` | Alert email variables missing from `.env` | Add them (step 2a) |
| `cp: cannot stat '.env.staging.example'` | Old instructions; that file doesn't exist | Use the template in step 2a |
| `permission denied ... docker.sock` | Your user isn't in the `docker` group | `sudo usermod -aG docker "$USER"`, then log out and in |
| `no space left on device` while pulling | Images need several GB | Free space (`docker image prune` removes unused images; check first with `docker images`), or enlarge the disk |
| Browser can't reach `https://<PUBLIC_HOSTNAME>` at all | DNS not pointing at the VM, ports 80/443 blocked, or `proxy` not running | Check the `A` record and firewall ([Prerequisites](#choose-a-hostname-and-open-the-firewall-gcp)) and `docker compose ps` |
| Browser warns the certificate is invalid; proxy log shows ACME errors | DNS does not point at the VM yet, or port 80 is blocked | Fix the `A` record or firewall, then `docker compose restart proxy` |
| `open-webui` keeps restarting | Bad or missing `OPENAI_API_KEY` / `WEBUI_SECRET_KEY` | Check `.env`, then `docker compose logs open-webui --tail 100` |
| Qdrant connection refused | `/opt/qdrant/storage` doesn't exist or has wrong permissions | Create the directory (step 3); check `docker compose logs qdrant` |
| Model list is empty | OpenAI connection off or key wrong | **Admin Panel → Settings → Connections** (step 6) |
| Chat page loads but replies never arrive | The WebSocket is blocked, or the page's origin is not in `CORS_ALLOW_ORIGIN` | Use the `https://<PUBLIC_HOSTNAME>` URL; check `docker compose logs open-webui` for `is not an accepted origin` |
| Changed `.env` but Open WebUI behaves the same | Persistent config: `.env` only applies on first start | Change the setting in the Admin Panel instead |
| Changed `grading_feature/.env` but nothing changed | `docker compose restart` doesn't re-read env files | `docker compose up -d grading` |
| Uploading a document fails or search returns nothing | `RAG_EMBEDDING_ENGINE` not `openai`, or bad `RAG_OPENAI_API_KEY` | Fix in **Admin Panel → Settings → Documents** (persistent config) |
| No `openwebui_chat` lines from the Pushgateway | Filter not installed, not enabled, not Global, or no chat sent yet | Redo step 8 and send a message |
| `curl localhost:9091` refused | `pushgateway` service not running | `docker compose ps`; if it's not listed, your checkout predates the A19 change; pull the latest `main` |
| Grafana shows "No data" | No chat traffic yet, or a Prometheus target is down | Send a chat; run the targets check in step 10 |
| Can't log into Grafana | Password set on first start only | Reset it (step 10) |
| Grafana alert email never arrives | Wrong SMTP settings (Gmail needs an app password) | `docker compose logs grafana \| grep -i smtp`; then use the contact point **Test** button (step 10) |
| `/grading/` keeps asking for a password | No `proxy/grading_users.caddy`, a wrong hash, or the proxy was not reloaded | Re-create the line with `caddy hash-password` (step 2c), then reload the proxy |
| Grading scores look generic or identical for every question | Mock scoring: `OPENAI_API_KEY` empty in the grading config | Set a real key in `grading_feature/.env`, then `docker compose up -d grading` (or restart `run_app.sh`) |
| Grading **Refresh** fails with 401 after you have signed in | Open WebUI API key wrong, expired, or API keys disabled | Redo step 7 |
| `run_app.sh` says the npm or Node version is wrong | Exact versions are required | `nvm install 20.19.5 && nvm use 20.19.5` |
| `docker compose up` fails with "port is already allocated" | Something else uses 80 or 443 (proxy), or 3000, 3001, 8001, 8080, 8889, 9090 or 9091 on localhost | Stop the other program, or change the host port in `docker-compose.yml` |
| Chat hangs after 7+ messages | Known Open WebUI bug in earlier versions | Start a new chat. <!-- TODO-A24: re-test on v0.11.4; delete this row if fixed --> |

<!-- TODO-A24: add every stuck point from task 0.6 that isn't covered above. -->

---

## Upgrading Open WebUI

The short version of a safe upgrade:

1. **Back up** the database and data (`scripts/backup.sh`; see `docs/backup/backup-restore.md`).
2. **Free disk space** for the new image (`df -h /`). You need room for the old and the new image
   at the same time.
3. **Keep the old image** for rollback. Don't prune it until the new version is accepted.
4. Pull the new image, **stop** `open-webui`, change the `image:` tag in `docker-compose.yml`,
   and **start** it again (`docker compose up -d open-webui`).
5. **Test:** old chats and users are still there, a new chat works, document search works,
   metrics still arrive, and the grading dashboard can still extract chats.

Upgrade one minor version at a time. Some migrations are **one-way** (v0.10 changes the config
table), so going back means restoring the backup, not just switching the image. For production,
follow the full [A13 production upgrade runbook](#a13-production-upgrade-runbook-preparation)
below.

<!-- TODO-A24 (P1): this runbook is long for a setup guide. Consider moving it to
docs/runbooks/upgrade.md. If moved, update the anchor link in HANDOFF.md. -->

---

## CI Smoke Test

`.github/workflows/smoke-test.yml` runs on every push. It starts the isolated
`docker-compose.test.yml` stack with the `docker-compose.ci.yml` override. The
override routes chat and embeddings to a deterministic OpenAI-compatible test
provider, so CI needs no external API key or paid model calls. This checks
OpenWebUI model routing and Qdrant retrieval, but does not check live OpenAI
availability or credentials. No production volume or port is used.

The test waits for `/health` and `/ready`, registers the first admin user, checks a
non-empty chat response, uploads a text document, adds it to a new knowledge base,
and queries that collection for a unique marker in the uploaded text. It fails at
the first broken step and prints the test stack logs on failure. The stack and
volumes are removed at the end of each run.

To run the CI stack locally, copy `.env.test.example` to `.env.test`. Then run:

```bash
docker compose -f docker-compose.test.yml -f docker-compose.ci.yml up -d --build open-webui-test qdrant-test redis-test mock-openai-test
python -m pip install requests==2.32.5
python scripts/smoke_test.py --model gpt-4o-mini
docker compose -f docker-compose.test.yml -f docker-compose.ci.yml down -v
```

To test a real provider locally, omit the CI override and set real
`OPENAI_API_KEY` and `RAG_OPENAI_API_KEY` values plus
`RAG_EMBEDDING_ENGINE=openai` in `.env.test`.

For the A3 red/green demonstration, push a commit on the feature branch with
`SMOKE_MODEL` set to a nonexistent model and capture the failed Actions run.
Restore the configured model in a follow-up commit, push, and capture the passing
run. Keep both commits on the PR so the failure and recovery are reviewable.

## A13 production upgrade runbook (preparation)

**Status: preparation for review; no production upgrade or restore has been executed
by this PR.** A13 closes only after the maintenance window, smoke checks, P2's
grading acceptance, and the execution record below. The target for this preparation
is the production Compose tag `v0.11.4`. A tag in Git is not evidence of the image
running on the server. The current isolated CI stack uses `v0.9.6`; its green run
does not validate a production-data migration to `v0.11.4`.

### 1. Review gates and deployment inventory

P1 and P2 must fill this table with evidence before scheduling execution. All entries
are **pending verification** in this preparation PR.

| Gate | Required evidence / decision | Record |
|---|---|---|
| A2 | Successful restore rehearsal; backup locations, retention and measured restore time | Pending |
| A9 | Restored-data v0.9.x smoke and concurrent metrics results; exact image tag/digest | Pending |
| A10 | Restored-data v0.10.x smoke; isolated downgrade failure and recovery record | Pending |
| A11 | Restored-data v0.11.4 smoke; UI changes handed to P4 | Pending |
| A12 | Team's SQLite/PostgreSQL decision and reasoning in HANDOFF.md | Pending |
| Window | Instructor-approved date/time/timezone, outside student deadline weeks; outage and rollback deadline | Pending |
| Operators | Named P1 executor and P2 grading reviewer, both available throughout | Pending |
| Upload protection | Verify whether the source is a bucket mount or local directory; matching snapshot or cold archive, restore procedure and retention | Pending |
| Deployment | Host/project, active Compose files/overrides, Git SHA, actual image tags/IDs/repository digests, DB backend, all mounts | Pending |
| Baseline | Existing-user smoke, known saved chats/files, P2 extraction and metrics results before change | Pending |

The commands below are for **Bash on the Linux deployment host**, with Docker
Compose v2, Python 3, GNU tar, sudo and enough space for backups, saved images and
a restore alongside the failed data. Rehearse this exact procedure on isolated
copies first. Do not paste the whole section as a script: there are human evidence
gates between blocks. Keep a durable, access-restricted backup outside the repository
and off the VM as required by A2; a VM-local archive alone is not disaster recovery.

Start with the actual deployed checkout, not a fresh clone of `main`. The VM inventory
from 2026-10-17 found an older, differently owned checkout, Open WebUI v0.8.12, no
hosted grading service, and an uploads directory that was not a mount. Recheck each
fact on the day of execution. Stop and revise/rehearse this procedure if the active
Compose files, services, mounts, or database differ from the inventory below.

```bash
set -euo pipefail
umask 077
read -rp 'Absolute path of the deployed checkout: ' REPO
REPO=$(realpath "$REPO")
test -f "$REPO/docker-compose.yml" && test -f "$REPO/.env"
# The deployed checkout has different ownership. Do not chown it or set a global exception.
deployed_git() { git -c "safe.directory=$REPO" -C "$REPO" "$@"; }
test "$(deployed_git rev-parse --show-toplevel)" = "$REPO"
test -z "$(deployed_git status --porcelain)"
read -rp 'Running Open WebUI container name: ' WEBUI_CONTAINER
test -n "$WEBUI_CONTAINER"
PROJECT=$(docker inspect "$WEBUI_CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project"}}')
WEBUI_SERVICE=$(docker inspect "$WEBUI_CONTAINER" --format '{{index .Config.Labels "com.docker.compose.service"}}')
test -n "$PROJECT" && test "$PROJECT" != '<no value>'
test -n "$WEBUI_SERVICE" && test "$WEBUI_SERVICE" != '<no value>'
test "$WEBUI_SERVICE" = open-webui || {
  echo 'Open WebUI service name differs from this runbook; stop and adapt the overrides' >&2
  exit 1
}
COMPOSE_FILES=$(docker inspect "$WEBUI_CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}')
test "$COMPOSE_FILES" = "$REPO/docker-compose.yml" || {
  echo "Active Compose files differ from the deployed checkout; stop and adapt the runbook" >&2
  exit 1
}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="/var/backups/familyfinancechat/$STAMP"
sudo install -d -m 700 -o "$(id -u)" -g "$(id -g)" "$BACKUP"
export BACKUP STAMP
dc() { docker compose -p "$PROJECT" --env-file "$REPO/.env" -f "$REPO/docker-compose.yml" "$@"; }
deployed_git rev-parse HEAD > "$BACKUP/git-sha.txt"
printf '%s\n' "$PROJECT" > "$BACKUP/project.txt"
dc config --quiet
dc ps -a > "$BACKUP/containers.txt"
dc config --services | sort > "$BACKUP/config-services.txt"
docker ps --filter "label=com.docker.compose.project=$PROJECT" \
  --format '{{.Label "com.docker.compose.service"}}' | sort -u > "$BACKUP/running-services.txt"
diff -u "$BACKUP/config-services.txt" "$BACKUP/running-services.txt" || {
  echo 'Running services differ from the deployed Compose file; stop and reconcile' >&2; exit 1;
}
for service in $(dc config --services); do
  cid=$(dc ps -a -q "$service")
  test -n "$cid" || { echo "Configured service $service is absent; stop and reconcile Compose" >&2; exit 1; }
  docker inspect "$cid" --format '{{.Name}} image={{.Config.Image}} id={{.Image}} mounts={{json .Mounts}}'
  docker image inspect "$(docker inspect "$cid" --format '{{.Image}}')" --format '{{json .RepoDigests}}'
done > "$BACKUP/images-and-mounts.txt"
dc exec -T "$WEBUI_SERVICE" python -c 'import os; u=os.getenv("DATABASE_URL", ""); print("database_backend=" + (u.split(":", 1)[0] if u else "default: verify SQLite path"))'
```

Inspect the recorded mounts and the private deployment configuration. Never print
the full container environment or resolved Compose configuration into PR logs.
Confirm `DATA_DIR`, any database URL, Qdrant and Redis endpoints actually match the
local services below. These backup/restore commands support **SQLite stored under
`/app/backend/data` only**; an external DB, custom SQLite location, external vector
store or A12 PostgreSQL migration requires a separate reviewed backup/restore plan.
Do not read or alter Open WebUI's database schema to perform these checks.

| Service | Expected persistent mount |
|---|---|
| `open-webui` | `/opt/openwebui/data` → `/app/backend/data` (SQLite and local app data) |
| `open-webui` uploads | Inspect the source of `/app/backend/data/uploads`; `/mnt/gcs/fin602` may be a plain VM directory |
| `qdrant` | `/opt/qdrant/storage` → `/qdrant/storage` |
| `redis` | Compose volume `redis-data` → `/data` |
| grading, if deployed | `grading-api` uses `grading-data` → `/app/data`; A15 `grading` uses `grading-dashboard-data` → `/data` |
| `prometheus`, `grafana` | `/opt/prometheus/data`, `/opt/grafana/data` |

Use the inspected volume names, which include the project prefix; do not guess them.
Confirm any named volumes are local Docker volumes without custom driver options and
no other process writes them. If uploads are a mount, identify the backing bucket and
test its point-in-time restore. If uploads are a plain directory, include them in the
cold archive. Every directory that rollback moves aside must be an ordinary directory,
not a symlink or mountpoint.

```bash
check_bind() {
  local service=$1 destination=$2 expected=$3 cid actual
  cid=$(dc ps -q "$service")
  test -n "$cid"
  actual=$(docker inspect "$cid" --format "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.Source}}{{end}}{{end}}")
  test "$actual" = "$expected" || {
    echo "Unexpected mount for $service:$destination ($actual); stop and adapt" >&2; exit 1;
  }
}
check_bind open-webui /app/backend/data /opt/openwebui/data
check_bind qdrant /qdrant/storage /opt/qdrant/storage
check_bind prometheus /prometheus /opt/prometheus/data
check_bind grafana /var/lib/grafana /opt/grafana/data
REDIS_VOLUME=$(docker inspect "$(dc ps -q redis)" --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')
test -n "$REDIS_VOLUME"
VOLUME_ITEMS=("redis:$REDIS_VOLUME")
GRADING_SERVICE=''
GRADING_VOLUME_KEY=''
if dc config --services | grep -qx 'grading' && dc config --services | grep -qx 'grading-api'; then
  echo 'Both grading services are configured; stop and choose the active data store' >&2
  exit 1
fi
if dc config --services | grep -qx 'grading'; then
  GRADING_SERVICE=grading; GRADING_VOLUME_KEY=grading-dashboard-data; GRADING_DATA_DEST=/data
elif dc config --services | grep -qx 'grading-api'; then
  GRADING_SERVICE=grading-api; GRADING_VOLUME_KEY=grading-data; GRADING_DATA_DEST=/app/data
fi
if [ -n "$GRADING_SERVICE" ]; then
  GRADING_CID=$(dc ps -q "$GRADING_SERVICE")
  test -n "$GRADING_CID" || { echo "Grading service is configured but not running" >&2; exit 1; }
  GRADING_VOLUME=$(docker inspect "$GRADING_CID" --format "{{range .Mounts}}{{if eq .Destination \"$GRADING_DATA_DEST\"}}{{.Name}}{{end}}{{end}}")
  test -n "$GRADING_VOLUME"
  VOLUME_ITEMS+=("grading:$GRADING_VOLUME")
fi
printf '%s\n' "${VOLUME_ITEMS[@]}" > "$BACKUP/volume-items.txt"
printf '%s\n' "$GRADING_VOLUME_KEY" > "$BACKUP/grading-volume-key.txt"
docker volume inspect "${VOLUME_ITEMS[@]#*:}" > "$BACKUP/volumes.json"
python3 - <<'PY'
import json, os, pathlib
volumes = json.loads((pathlib.Path(os.environ['BACKUP']) / 'volumes.json').read_text())
if not volumes or any(v.get('Driver') != 'local' or v.get('Options') for v in volumes):
    raise SystemExit('Non-local or custom-driver volume; stop and adapt the backup plan')
PY
UPLOADS_SOURCE=$(docker inspect "$WEBUI_CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/app/backend/data/uploads"}}{{.Source}}{{end}}{{end}}')
test -n "$UPLOADS_SOURCE" && test -d "$UPLOADS_SOURCE"
BIND_PATHS=(/opt/openwebui/data /opt/qdrant/storage /opt/prometheus/data /opt/grafana/data)
if mountpoint -q "$UPLOADS_SOURCE"; then
  UPLOADS_KIND=mount
else
  UPLOADS_KIND=local
  for path in "${BIND_PATHS[@]}"; do
    case "$UPLOADS_SOURCE/" in "$path/"* ) echo 'Uploads overlap another backup path; stop and adapt' >&2; exit 1;; esac
    case "$path/" in "$UPLOADS_SOURCE/"* ) echo 'Uploads contain another backup path; stop and adapt' >&2; exit 1;; esac
  done
  BIND_PATHS+=("$UPLOADS_SOURCE")
fi
printf '%s\n' "${BIND_PATHS[@]}" > "$BACKUP/bind-paths.txt"
printf 'source=%s\nkind=%s\n' "$UPLOADS_SOURCE" "$UPLOADS_KIND" > "$BACKUP/uploads.txt"
for path in "${BIND_PATHS[@]}"; do
  sudo test -d "$path"
  test ! -L "$path"
  if mountpoint -q "$path"; then echo "Unexpected data mountpoint: $path" >&2; exit 1; fi
done
```

### 2. Freeze the rollback release and approve the version sequence

Keep the deployed release, secrets (private backup only), and every running image,
including locally built grading and metrics images. Local rollback tags
below refer to the exact running image IDs; repository digests can be empty for local
builds. Do not prune them during the window. The helper image is only for archiving
volumes; record it and include it in the image archive.

```bash
mkdir "$BACKUP/release"
deployed_git archive HEAD | tar -xf - -C "$BACKUP/release"
install -m 600 "$REPO/.env" "$BACKUP/release/.env"
# Save only env files that the active Compose configuration actually uses.
for env_file in grading_feature/backend/grading.env grading_feature/.env; do
  if [ -f "$REPO/$env_file" ]; then install -m 600 "$REPO/$env_file" "$BACKUP/release/$env_file"; fi
done
: > "$BACKUP/frozen-images.tsv"
for service in $(dc config --services); do
  cid=$(dc ps -a -q "$service")
  image_id=$(docker inspect "$cid" --format '{{.Image}}')
  tag="localhost/ffc-a13/$service:$STAMP"
  docker tag "$image_id" "$tag"
  printf '%s\t%s\n' "$service" "$tag" >> "$BACKUP/frozen-images.tsv"
done
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['BACKUP'])
services = {s: {'image': image} for s, image in
            (line.split() for line in (p / 'frozen-images.tsv').read_text().splitlines())}
(p / 'frozen-images.json').write_text(json.dumps({'services': services}, indent=2))
PY
docker pull alpine:3.21.3
docker image inspect alpine:3.21.3 --format '{{.Id}} {{json .RepoDigests}}' > "$BACKUP/archive-helper.txt"
# All local data and the archive must fit on the same filesystem for this procedure.
# Reserve room for the cold archive, restored copy, saved images and 5 GiB headroom.
DATA_DEVICE=$(stat -c %d "${BIND_PATHS[0]}")
test "$(stat -c %d "$BACKUP")" = "$DATA_DEVICE" || {
  echo 'Backup and data are on different disks; recalculate space and rehearse a revised plan' >&2; exit 1;
}
data_kb=0
for path in "${BIND_PATHS[@]}"; do
  test "$(stat -c %d "$path")" = "$DATA_DEVICE" || {
    echo "Data is on another disk: $path; revise the space check" >&2; exit 1;
  }
  data_kb=$((data_kb + $(sudo du -sk "$path" | cut -f1)))
done
volume_kb=0
for item in "${VOLUME_ITEMS[@]}"; do
  volume=${item#*:}
  mount=$(docker volume inspect "$volume" --format '{{.Mountpoint}}')
  test "$(stat -c %d "$mount")" = "$DATA_DEVICE" || {
    echo "Volume $volume is on another disk; revise the space check" >&2; exit 1;
  }
  volume_kb=$((volume_kb + $(sudo du -sk "$mount" | cut -f1)))
done
image_kb=0
for service in $(dc config --services); do
  cid=$(dc ps -a -q "$service")
  bytes=$(docker image inspect "$(docker inspect "$cid" --format '{{.Image}}')" --format '{{.Size}}')
  image_kb=$((image_kb + (bytes + 1023) / 1024))
done
reserve_kb=$((5 * 1024 * 1024))
need_kb=$((2 * (data_kb + volume_kb) + image_kb + reserve_kb))
free_kb=$(df -Pk "$BACKUP" | awk 'NR==2 {print $4}')
test "$free_kb" -gt "$need_kb" || {
  echo "Insufficient space: need ${need_kb} KiB, have ${free_kb} KiB; stop before saving images" >&2
  exit 1
}
printf 'data_kb=%s\nvolume_kb=%s\nimage_kb=%s\nreserve_kb=%s\nfree_kb=%s\n' \
  "$data_kb" "$volume_kb" "$image_kb" "$reserve_kb" "$free_kb" > "$BACKUP/space-check.txt"
docker image save -o "$BACKUP/images.tar" alpine:3.21.3 $(cut -f2 "$BACKUP/frozen-images.tsv")
```

P1/P2 now write the ordered **remaining** upgrade tags and digests in the execution
record. Use the fixed patch versions already rehearsed for A9–A11. The v0.10.x
patch is deliberately unspecified here because its rehearsal evidence is pending.
End at v0.11.4; include each intervening minor version from the real starting version.
Do not downgrade a server already at or above a planned step to recreate the sequence.
If already on v0.11.4, assess and record validation-only work; if newer, stop and
re-scope. A green blank-data v0.9.6 CI run does not fill A9–A11's restored-data gates.
Pull the approved images before the window and record their digests. Do not broadly
pull or rebuild the entire deployment; Qdrant/Valkey upgrades are separate changes.

### 3. Fresh, consistent backup inside the maintenance window

After the teacher-approved window starts, block public ingress and background jobs
using the deployment's reviewed maintenance procedure. Stop external grading refresh,
upload sync and all other writers. Run the baseline checks in section 4 and record
P2's baseline before stopping the stack. Record the outage start and rollback deadline.

The A2 `scripts/backup.sh` is a **live** application backup. Once A2 is complete,
run that reviewed script and verify its manifest and restore-drill record before the
window. This runbook retains the **cold** backup below because it also freezes the
other running services, local volumes and exact images at one rollback point. Do not
substitute one archive for the other. If A2's script or its restore drill is absent,
the A2 gate above remains pending and the upgrade cannot start.

```bash
read -rp 'Reviewed A2 backup.sh absolute path: ' A2_SCRIPT
test -f "$A2_SCRIPT"
sudo bash "$A2_SCRIPT"
read -rp 'Completed A2 backup directory (not .partial): ' A2_BACKUP
sudo test -d "$A2_BACKUP"
sudo test -f "$A2_BACKUP/SHA256SUMS"
case "$A2_BACKUP" in *.partial) echo 'Incomplete A2 backup' >&2; exit 1;; esac
sudo bash -euo pipefail -c 'cd -- "$1"; sha256sum -c SHA256SUMS' bash "$A2_BACKUP"
printf '%s\n' "$A2_BACKUP" > "$BACKUP/a2-backup-path.txt"
```

```bash
dc stop
test -z "$(dc ps --status running -q)"
sudo tar --numeric-owner --acls --xattrs -cpf "$BACKUP/bind-data.tar" -C / \
  "${BIND_PATHS[@]#/}"
sudo chown "$(id -u):$(id -g)" "$BACKUP/bind-data.tar"
for item in "${VOLUME_ITEMS[@]}"; do
  name=${item%%:*}; volume=${item#*:}
  docker run --rm --network none \
    --mount "type=volume,src=$volume,dst=/source,readonly" \
    --mount "type=bind,src=$BACKUP,dst=/backup" alpine:3.21.3 \
    tar -cpf "/backup/$name.tar" -C /source .
  sudo chown "$(id -u):$(id -g)" "$BACKUP/$name.tar"
done
tar -tf "$BACKUP/bind-data.tar" > "$BACKUP/bind-data.contents.txt"
checksum_files=(bind-data.tar images.tar)
for item in "${VOLUME_ITEMS[@]}"; do
  name=${item%%:*}
  tar -tf "$BACKUP/$name.tar" > "$BACKUP/$name.contents.txt"
  checksum_files+=("$name.tar")
done
(cd "$BACKUP" && sha256sum "${checksum_files[@]}" > SHA256SUMS)
(cd "$BACKUP" && sha256sum -c SHA256SUMS)
```

If `UPLOADS_KIND=local`, `bind-data.tar` includes the uploads directory. If it is
`mount`, the archive excludes it: while writers remain stopped, take the fresh
bucket snapshot/version manifest using the A2-tested procedure and record its
matching restore point. Confirm retention protects deleted/overwritten files and
restoration to that point. Copy the backup bundle to the A2 off-host destination
and verify its checksums.
Also take the fresh VM/disk snapshot required by A2 and record its ID. **Stop before
upgrade if these steps, archive inspection, or the rehearsed restore are unverified.**
Checksums confirm archive integrity, not application recoverability.

### 4. Baseline and per-version acceptance

Install the reviewed A13 `scripts/smoke_test.py` and `requests==2.32.5` in a dedicated
operator Python environment beforehand. Set `SMOKE_SCRIPT` to that reviewed absolute
path and `SMOKE_PYTHON` to its interpreter (the saved old release may lack this flag).
Use a dedicated existing test account with chat, upload, knowledge and retrieval
permissions; P2's extraction credential separately requires the users/chats API.
Enter only the token value, without a `Bearer ` prefix. Disable shell tracing.

```bash
set +x
read -rp 'Reviewed A13 smoke script absolute path: ' SMOKE_SCRIPT
read -rp 'Smoke Python interpreter absolute path: ' SMOKE_PYTHON
test -f "$SMOKE_SCRIPT" && test -x "$SMOKE_PYTHON"
read -rp 'Enabled smoke model ID: ' SMOKE_MODEL
read -rp 'Reviewed check_pushgateway_metric.py absolute path: ' METRICS_SCRIPT
read -rp 'Host-reachable Pushgateway URL: ' METRICS_URL
test -f "$METRICS_SCRIPT" && test -n "$METRICS_URL"
read -rsp 'Existing smoke account token: ' OPENWEBUI_SMOKE_TOKEN; printf '\n'
export OPENWEBUI_SMOKE_TOKEN
read -rp 'Evidence label (baseline or version, no slash): ' CHECK_LABEL
[[ "$CHECK_LABEL" =~ ^[a-zA-Z0-9._-]+$ ]]
SMOKE_STARTED=$(date +%s)
"$SMOKE_PYTHON" "$SMOKE_SCRIPT" --base-url http://127.0.0.1:3000 \
  --model "$SMOKE_MODEL" --existing-token-env OPENWEBUI_SMOKE_TOKEN \
  2>&1 | tee "$BACKUP/smoke-$CHECK_LABEL.log"
unset OPENWEBUI_SMOKE_TOKEN
python3 "$METRICS_SCRIPT" --url "$METRICS_URL" --since "$SMOKE_STARTED" --model "$SMOKE_MODEL" \
  2>&1 | tee "$BACKUP/metrics-$CHECK_LABEL.log"
```

This waits for `/health` and `/ready`, requires a non-empty chat answer, processes an
uploaded document, attaches it to a knowledge base, and retrieves its unique marker.
It uses the deployment's real model/embedding provider (normal API usage applies).
A missing/blank environment variable fails before any HTTP request; HTTP 401/403
fails immediately without falling back to signup. Leave signup mode for the isolated
blank CI stack shown above. Never run the CI `down -v` command against production.

`Smoke run` identifies the run; `Smoke resource` lines record created user (fresh
mode only), knowledge and file IDs as soon as available, including on later failure.
The chat request uses `smoke-test-<run ID>` but does not create a saved chat fixture.
After retaining results, use `scripts/cleanup_smoke.py` with the printed run,
knowledge and file IDs. It first reads both resources and checks their exact names
and owner; it is a dry run unless `--execute` is supplied. Run the dry run, review
the IDs, then repeat with `--execute`. It never removes the smoke user. If a
timed-out create returned no ID, inspect the account's resources by the run marker
manually; the tool will not guess an ID. Tokens and raw HTTP error bodies are not
printed by the clients. Keep evidence private.

```bash
read -rp 'Reviewed cleanup_smoke.py absolute path: ' CLEANUP_SCRIPT
read -rp 'Smoke run ID: ' RUN_ID
read -rp 'Smoke knowledge ID: ' KNOWLEDGE_ID
read -rp 'Smoke file ID: ' FILE_ID
read -rsp 'Existing smoke account token: ' OPENWEBUI_SMOKE_TOKEN; printf '\n'
export OPENWEBUI_SMOKE_TOKEN
"$SMOKE_PYTHON" "$CLEANUP_SCRIPT" --base-url http://127.0.0.1:3000 \
  --run-id "$RUN_ID" --knowledge-id "$KNOWLEDGE_ID" --file-id "$FILE_ID" \
  --token-env OPENWEBUI_SMOKE_TOKEN
# Only after inspecting the dry-run result and retaining the evidence:
"$SMOKE_PYTHON" "$CLEANUP_SCRIPT" --base-url http://127.0.0.1:3000 \
  --run-id "$RUN_ID" --knowledge-id "$KNOWLEDGE_ID" --file-id "$FILE_ID" \
  --token-env OPENWEBUI_SMOKE_TOKEN --execute
unset OPENWEBUI_SMOKE_TOKEN
```

P2 must compare extraction against known **saved** baseline chats with user ownership,
message-pair counts and representative question/answer content. A successful exit or
an empty JSON alone is not acceptance. The smoke's synthetic completion alone is not
a saved conversation. Run the reviewed extractor from the operator's Python
environment into the restricted backup directory. This works when grading runs on
a laptop, as `grading-api`, or as A15's `grading` service; it does not overwrite any
dashboard data. Install its dependencies beforehand. The extraction token must have
access to the users/chats API.

```bash
read -rsp 'P2 extraction bearer token: ' OPENWEBUI_BEARER_TOKEN; printf '\n'
export OPENWEBUI_BEARER_TOKEN
read -rp 'Reviewed extract_chats.py absolute path: ' EXTRACT_SCRIPT
read -rp 'Extractor Python interpreter absolute path: ' EXTRACT_PYTHON
test -f "$EXTRACT_SCRIPT" && test -x "$EXTRACT_PYTHON"
OUTPUT_PATH="$BACKUP/extraction-$CHECK_LABEL.json" \
  OPENWEBUI_BASE_URL=http://127.0.0.1:3000 \
  "$EXTRACT_PYTHON" "$EXTRACT_SCRIPT" \
  > "$BACKUP/extraction-$CHECK_LABEL.log" 2>&1
unset OPENWEBUI_BEARER_TOKEN
test -s "$BACKUP/extraction-$CHECK_LABEL.json"
if [ "$CHECK_LABEL" != baseline ]; then
  read -rp 'Reviewed compare_extractions.py absolute path: ' COMPARE_SCRIPT
  test -f "$COMPARE_SCRIPT"
  python3 "$COMPARE_SCRIPT" \
    "$BACKUP/extraction-baseline.json" "$BACKUP/extraction-$CHECK_LABEL.json"
fi
```

P2 privately inspects `extraction-<label>.json`, confirms the comparison and the
agreed fixtures/counts, and signs each row below. Protect the JSON and logs: they
contain student data. Remove them after acceptance per the team's retention policy.
Verify the existing metrics filter still emits a **new** response-time sample after
the chat and check the grading UI's health; a previously pushed sample is insufficient.
The isolated CI stack has no installed metrics Filter, so this check runs only against
an environment where that Filter is installed and Pushgateway is reachable.
The checker matches the model and the full Pushgateway grouping labels (including
job and instance) to a fresh push. It does not identify an individual request:
keep other chat traffic and producers for this metrics job stopped during the check.
For each supported version, rehearse the installed Filter's inlet/outlet lifecycle;
client and Filter unit tests alone do not prove the server invokes both hooks.

### 5. Execute one approved version at a time

Only proceed after sections 1–4's baseline and backup gates are signed. Keep ingress
and other writers blocked through validation. Each iteration requires a new evidence
label and P2 sign-off before the next version. Pin the selected rehearsed digest in
the step override; the frozen override keeps every other service on its original image.

```bash
read -rp 'Next approved full image reference (tag@sha256:digest): ' NEXT_IMAGE
[[ "$NEXT_IMAGE" =~ ^ghcr.io/open-webui/open-webui:v[0-9]+\.[0-9]+\.[0-9]+@sha256:[a-f0-9]{64}$ ]]
export NEXT_IMAGE
docker pull "$NEXT_IMAGE"
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['BACKUP'])
(p / 'step.json').write_text(json.dumps({'services': {'open-webui': {'image': os.environ['NEXT_IMAGE']}}}))
PY
upgrade() { docker compose -p "$PROJECT" --env-file "$BACKUP/release/.env" \
  -f "$BACKUP/release/docker-compose.yml" -f "$BACKUP/frozen-images.json" \
  -f "$BACKUP/step.json" "$@"; }
upgrade config --quiet
test -d "$UPLOADS_SOURCE"
if [ "$UPLOADS_KIND" = mount ]; then mountpoint -q "$UPLOADS_SOURCE"; fi
upgrade up -d --no-build --pull never
upgrade ps
docker inspect "$WEBUI_CONTAINER" --format '{{.Config.Image}} {{.Image}}'
```

Re-run section 4 (health, chat, upload, retrieval, metrics and P2 extraction) after
**every** version, including v0.11.4. Record the selected reference, actual image ID,
timestamps and evidence before replacing `step.json` for the next step. Preserve each
approved step override with its evidence. Any migration error, unhealthy service,
failed assertion, extraction mismatch or rollback-deadline breach stops the sequence.
Keep maintenance in place, capture private logs, and use the rollback below if the
issue cannot be resolved within the agreed window. Never start an older image on the
migrated data as a production experiment; the A10 downgrade experiment belongs in test.

### 6. Roll back images AND the matching data

Restore the **pre-window** snapshot as one set: SQLite/local app data, Qdrant,
Redis, grading data when present, and matching uploads. Returning only the old
image can fail on the migrated schema. Keep ingress and all external writers blocked.
These commands retain failed bind data and use new Docker volumes instead of erasing
the originals. Verify space and the backup manifest before moving any directories.

Use the same shell variables from the window; after reconnecting, explicitly recover
`BACKUP`, `STAMP`, `PROJECT`, `UPLOADS_SOURCE`, `UPLOADS_KIND`, `BIND_PATHS` and
`VOLUME_ITEMS` from the recorded bundle and redefine the functions. Review every
value before running a mutating command.

```bash
upgrade stop
test -z "$(upgrade ps --status running -q)"
(cd "$BACKUP" && sha256sum -c SHA256SUMS)
docker image load -i "$BACKUP/images.tar"
mapfile -t BIND_PATHS < "$BACKUP/bind-paths.txt"
mapfile -t VOLUME_ITEMS < "$BACKUP/volume-items.txt"
GRADING_VOLUME_KEY=$(cat "$BACKUP/grading-volume-key.txt")
UPLOADS_SOURCE=$(sed -n 's/^source=//p' "$BACKUP/uploads.txt")
UPLOADS_KIND=$(sed -n 's/^kind=//p' "$BACKUP/uploads.txt")
test -n "$UPLOADS_SOURCE" && [[ "$UPLOADS_KIND" == local || "$UPLOADS_KIND" == mount ]]
export GRADING_VOLUME_KEY
for path in "${BIND_PATHS[@]}"; do
  sudo test -d "$path"
  test ! -L "$path"
  if mountpoint -q "$path"; then echo "Unexpected data mountpoint: $path" >&2; exit 1; fi
  sudo test ! -e "$path.failed-$STAMP"
done
for path in "${BIND_PATHS[@]}"; do
  sudo mv -- "$path" "$path.failed-$STAMP"
done
sudo tar --numeric-owner --acls --xattrs -xpf "$BACKUP/bind-data.tar" -C /
for item in "${VOLUME_ITEMS[@]}"; do
  name=${item%%:*}
  volume="ffc-a13-$name-$STAMP"
  if docker volume inspect "$volume" >/dev/null 2>&1; then
    echo "Restore volume already exists; stop and inspect before retrying" >&2
    exit 1
  fi
  docker volume create "$volume"
  docker run --rm --network none \
    --mount "type=volume,src=$volume,dst=/restore" \
    --mount "type=bind,src=$BACKUP,dst=/backup,readonly" alpine:3.21.3 \
    tar -xpf "/backup/$name.tar" -C /restore
done
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['BACKUP'])
keys = {'redis': 'redis-data', 'grading': os.environ.get('GRADING_VOLUME_KEY', '')}
names = [line.split(':', 1)[0] for line in (p / 'volume-items.txt').read_text().splitlines()]
volumes = {keys[name]: {'name': f"ffc-a13-{name}-{os.environ['STAMP']}", 'external': True}
           for name in names}
(p / 'restored-volumes.json').write_text(json.dumps({'volumes': volumes}))
PY
```

**Before starting any service**, if uploads were mounted, restore/verify the bucket
to the matching recorded point using the A2-approved procedure and verify the mount
and representative old uploads. If uploads were local, confirm they came back from
`bind-data.tar`. This is a hard gate even though the archive passed checksums.

```bash
rollback() { docker compose -p "$PROJECT" --env-file "$BACKUP/release/.env" \
  -f "$BACKUP/release/docker-compose.yml" -f "$BACKUP/frozen-images.json" \
  -f "$BACKUP/restored-volumes.json" "$@"; }
rollback config --quiet
test -d "$UPLOADS_SOURCE"
if [ "$UPLOADS_KIND" = mount ]; then mountpoint -q "$UPLOADS_SOURCE"; fi
rollback up -d --no-build --pull never
rollback ps
```

Run section 4 against the restored instance and obtain P2 sign-off before reopening
access. If it fails, keep maintenance active and escalate to P1/P2; do not continue
upgrading. Record the failure and retained `.failed-<timestamp>` directories/old volumes.
Rollback is not idempotent: inspect partial progress before retrying any restore step.
Record the **exact Compose invocation and bundle path** as the active deployment.
After rollback, bare `docker compose up` from the old checkout would select the
original volumes/images again; maintenance must use the recorded rollback overrides
until a reviewed follow-up reconciles the deployment configuration.

### 7. Execution record and closeout

Copy this table into the private window record. Put redacted outcomes and evidence
links in the PR/HANDOFF; never attach tokens, environment files, backups or student data.

| Stage / UTC start–end | Image tag, digest and Git SHA | Backup / snapshot / log location | Health / chat / upload / retrieval / metrics | P2 extraction result and signer | Exception / action / executor |
|---|---|---|---|---|---|
| Pre-window inventory and gates | Pending | Pending | Pending | Pending | Pending |
| Baseline and fresh backup | Pending | Pending | Pending | Pending | Pending |
| Each approved remaining minor step (one row each) | Pending | Pending | Pending | Pending | Pending |
| v0.11.4 final acceptance | Pending | Pending | Pending | Pending | Pending |
| Rollback, if needed | Pending | Pending | Pending | Pending | Pending |
| Reopen access and clean test resources | Pending | Pending | Pending | Pending | Pending |

Record the teacher's window confirmation, P1/P2 names, time budget, DB decision,
uploads restore point, archive checksums, off-host copy, actual migration observations
and any data loss/reconciliation needed after rollback. On success, retain the final
`step.json` and frozen images override as part of the recorded deployment command;
reconcile the checked-in deployment with that state in review before discarding the
overrides. Re-enable ingress/jobs only after both operators accept the result.

Local preparation checks (no deployment required):

```bash
python -m pip install requests==2.32.5
python -m unittest discover -s tests -p test_smoke_test.py -v
bash scripts/ci/compat_guards.sh
git diff --check
```

The mock tests validate both authentication paths and failure handling. CI additionally
runs the existing blank-data integration stack. Neither constitutes a restored-data
rehearsal or production acceptance. No Open WebUI core files are modified: the
extension surfaces are the public REST API and deployment configuration.
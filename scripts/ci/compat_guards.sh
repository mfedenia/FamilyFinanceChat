#!/usr/bin/env bash
# CI compatibility guards: enforces the upstream-compatibility policy
# (see capstone_fall2026/). Run locally: bash scripts/ci/compat_guards.sh
# The gitleaks secret scan runs as its own job in .github/workflows/compat-guards.yml.
set -uo pipefail
fail=0
err() { echo "::error::$1"; fail=1; }

# Tracked files only, skipping reference-only folders
files() { git ls-files | grep -vE '^(legacy|archive)/'; }
COMPOSE=$(files | grep -E '(^|/)(docker-)?compose[^/]*\.ya?ml$')
DOCKERFILES=$(files | grep -E '(^|/)Dockerfile[^/]*$')

# 1. No mounts into Open WebUI's code or built frontend
for f in $COMPOSE; do
  grep -nE '(:|target:\s*)/app/(backend/open_webui|build)\b' "$f" \
    && err "$f mounts into /app/backend/open_webui or /app/build"
done

# 2. Open WebUI must run the official image (no custom builds)
for f in $COMPOSE; do
  grep -nE '^\s*image:.*open-webui' "$f" | grep -vE 'image:\s*["'\'']?ghcr\.io/open-webui/open-webui:' \
    && err "$f: Open WebUI image is not ghcr.io/open-webui/open-webui"
done
if [ -f Dockerfile ]; then
  grep -iE '^\s*FROM\s' Dockerfile | grep -viE '^\s*FROM\s+ghcr\.io/open-webui/open-webui:' \
    && err "Dockerfile: base image is not ghcr.io/open-webui/open-webui"
fi

# 3. No direct SQLite/SQLAlchemy access to Open WebUI's data directory
for f in $(files | grep '\.py$'); do
  if grep -qE 'sqlite3\.connect|create_engine|sessionmaker' "$f" \
     && grep -qE 'webui\.db|/app/backend/data|openwebui/data' "$f"; then
    err "$f opens a database in Open WebUI's data directory"
  fi
done

# 5. No floating :latest image tags (check 4, gitleaks, runs in the workflow)
for f in $COMPOSE $DOCKERFILES; do
  grep -inE '^\s*(image:|FROM)\s+\S+:latest["'\'']?\s*(#.*)?$' "$f" && err "$f uses a :latest image tag"
done

# 6. Production stack: only the reverse proxy is published beyond localhost (A16). Anything
#    else must bind 127.0.0.1, or it bypasses TLS and the proxy's authentication.
if [ -f docker-compose.yml ]; then
  exposed=$(awk '
    /^services:/                 { in_svc = 1; next }
    /^[^ #]/                     { in_svc = 0 }
    in_svc && /^  [A-Za-z0-9_.-]+:[ \t]*$/ { svc = $1; sub(":", "", svc) }
    in_svc && /^[ \t]*-[ \t]*["'\'']?([0-9.]+:)?[0-9]+(-[0-9]+)?:[0-9]+(-[0-9]+)?(\/(tcp|udp))?["'\'']?[ \t]*(#.*)?$/ {
      if ($0 !~ /127\.0\.0\.1:/ && svc != "proxy") printf "  %s (line %d): %s\n", svc, NR, $0
    }' docker-compose.yml)
  [ -n "$exposed" ] && { echo "$exposed"; err "docker-compose.yml publishes a non-proxy port on all interfaces; bind it to 127.0.0.1"; }
fi

[ "$fail" -eq 0 ] && echo "All compatibility guards passed."
exit "$fail"

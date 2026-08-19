#!/usr/bin/env bash
set -e

# 找到脚本所在目录（CS620 根）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

if [ $# -lt 1 ]; then
  echo "Usage: $0 {install|start}"
  exit 1
fi

cmd="$1"

case "$cmd" in
  install)
    echo "==> Installing backend dependencies..."
    cd "$BACKEND_DIR"
    # 这里只安装依赖，不自动 npm init，避免各种兼容问题
    npm install express cors dotenv openai
    echo "==> Done."
    ;;

    start)
    echo "==> Starting backend server (node server.js)..."
    cd "$BACKEND_DIR"

    # Configuration comes from backend/.env (loaded by dotenv in server.js).
    # Do NOT export OPENAI_API_KEY here -- a key in a committed script is how
    # this repo leaked one before, and exporting it also overrides .env.
    if [ ! -f .env ]; then
      echo "ERROR: backend/.env not found. Run: cp .env.example .env  (then add your key)" >&2
      exit 1
    fi

    node server.js
    ;;


  *)
    echo "Usage: $0 {install|start}"
    exit 1
    ;;
esac

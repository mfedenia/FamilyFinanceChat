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

    # 在这里给 Node 进程设置环境变量（bash 写法）
    export OPENAI_MODEL="gpt-4o-mini"      # 可选，和 server.js 默认一致即可
    # 如果你有自定义 base url，这里也可以加：
    # export OPENAI_BASE_URL="https://api.openai.com/v1"

    node server.js
    ;;


  *)
    echo "Usage: $0 {install|start}"
    exit 1
    ;;
esac

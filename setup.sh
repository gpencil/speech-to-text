#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================="
echo "  FunASR 语音识别 — 环境准备"
echo "========================================="

if ! command -v ffmpeg &>/dev/null; then
  echo "安装 ffmpeg（需要 brew）…"
  arch -arm64 brew install ffmpeg
else
  echo "ffmpeg: ok"
fi

PY="${PYTHON:-}"
if [[ -z "$PY" ]] && [[ -x "/opt/homebrew/bin/python3" ]]; then
  PY="/opt/homebrew/bin/python3"
elif [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

VENV="${ROOT}/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "创建 venv: $VENV"
  "$PY" -m venv "$VENV"
fi
echo "使用: $VENV/bin/python"
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"

echo ""
echo "启动服务:"
echo "  $VENV/bin/python $ROOT/server.py"
echo "  浏览器打开 http://localhost:8888"

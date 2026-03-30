#!/bin/bash
# 构建 speech-to-text 服务

set -e

CGO_CFLAGS="-I/opt/homebrew/include" \
CGO_LDFLAGS="-L/opt/homebrew/lib" \
go build -mod=vendor -o speech-to-text .

echo "构建成功: ./speech-to-text"
echo "启动命令: ./speech-to-text"
echo "模型路径: WHISPER_MODEL=./models/ggml-base.bin (默认)"

#!/bin/bash
set -e

echo "开始安装 Vosk API..."

# 安装到用户目录
INSTALL_DIR="$HOME/.local"
mkdir -p "$INSTALL_DIR"

# 检查是否已安装
if [ -f "$INSTALL_DIR/include/vosk_api.h" ]; then
    echo "Vosk API 似乎已经安装在 $INSTALL_DIR"
    exit 0
fi

# 创建临时目录
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "下载 Vosk API..."
git clone --depth 1 https://github.com/alphacep/vosk-api.git
cd vosk-api

echo "编译 Vosk API..."
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
make install

# 清理
cd /
rm -rf "$TMP_DIR"

echo "Vosk API 安装完成！"
echo ""
echo "请设置以下环境变量："
echo "export CGO_LDFLAGS=-L$INSTALL_DIR/lib"
echo "export CGO_CFLAGS=-I$INSTALL_DIR/include"
echo "export DYLD_LIBRARY_PATH=$INSTALL_DIR/lib:\$DYLD_LIBRARY_PATH"
echo ""
echo "可以将以上内容添加到 ~/.zshrc 或 ~/.bashrc 中"

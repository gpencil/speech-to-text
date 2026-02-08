#!/bin/bash
set -e

echo "========================================="
echo "  离线语音识别工具 - 安装脚本"
echo "========================================="
echo ""

# 检查 ffmpeg
echo "1. 检查 ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "   ffmpeg 未安装，正在安装..."
    brew install ffmpeg
else
    echo "   ffmpeg 已安装"
fi

# 检查 Python3
echo ""
echo "2. 检查 Python3..."
if ! command -v python3 &> /dev/null; then
    echo "   错误: 未找到 Python3，请先安装"
    exit 1
else
    python3 --version
fi

# 安装 faster-whisper
echo ""
echo "3. 安装 faster-whisper..."
pip3 install faster-whisper

# 编译 Go 程序
echo ""
echo "4. 编译 Go 程序..."
go build -o speech-to-text

echo ""
echo "========================================="
echo "  安装完成！"
echo "========================================="
echo ""
echo "运行以下命令启动服务器："
echo "  ./speech-to-text"
echo ""
echo "然后访问: http://localhost:8888"

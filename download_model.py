#!/usr/bin/env python3
"""
下载 Whisper 模型
使用国内镜像加速下载
"""
import sys
import os
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print(json.dumps({"error": "faster-whisper not installed", "success": False}))
    sys.exit(1)

def download_model(model_size="base"):
    """下载指定大小的模型"""
    print(f"正在下载 Whisper 模型: {model_size}")
    print("模型会自动缓存到本地，后续使用时无需重新下载")
    print("-" * 50)

    # 设置 Hugging Face 镜像（使用 hf-mirror）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    try:
        # 创建模型会自动触发下载
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models"
        )
        print("✓ 模型下载完成!")
        print(f"✓ 模型已保存到: {Path('./models').absolute()}")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n解决方案:")
        print("1. 使用镜像下载: export HF_ENDPOINT=https://hf-mirror.com")
        print("2. 配置代理: export https_proxy=your_proxy")
        print("3. 手动下载模型: 访问 https://hf-mirror.com/guillaumekln/faster-whisper-base")
        return False

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="下载 Whisper 模型")
    parser.add_argument("--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="模型大小 (默认: base)")

    args = parser.parse_args()

    download_model(args.model)

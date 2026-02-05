#!/usr/bin/env python3
"""
Whisper 语音识别脚本
使用 faster-whisper 库进行离线语音识别
"""

import sys
import json
import argparse
import os
from pathlib import Path
import io

# 设置 Hugging Face 镜像（用于国内网络）
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# 重定向所有日志输出
class SuppressOutput:
    """抑制 stdout 和 stderr 的上下文管理器"""
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

try:
    from faster_whisper import WhisperModel
except ImportError:
    print(json.dumps({"error": "faster-whisper not installed", "success": False}))
    sys.exit(1)


def recognize_audio(audio_path, model_size="base", language=None):
    """
    识别音频文件

    Args:
        audio_path: 音频文件路径
        model_size: 模型大小 (tiny, base, small, medium, large)
        language: 语言代码 (zh, en, auto)

    Returns:
        识别的文本
    """
    # 抑制 faster-whisper 的所有输出
    with SuppressOutput():
        # 使用 CPU 推理
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models"
        )

        # 语言映射
        lang_map = {
            "cn": "zh",
            "zh": "zh",
            "en": "en",
            "english": "en",
            "chinese": "zh",
        }

        if language and language.lower() in lang_map:
            language = lang_map[language.lower()]
        elif language == "auto":
            language = None

        # 执行识别
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        # 合并所有段落
        text = " ".join(segment.text for segment in segments)

        return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Whisper 离线语音识别")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--model", default="base", help="模型大小 (tiny, base, small, medium, large)")
    parser.add_argument("--lang", default=None, help="语言代码 (zh, en, auto)")

    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(json.dumps({"error": f"音频文件不存在: {args.audio}", "success": False}))
        sys.exit(1)

    try:
        text = recognize_audio(args.audio, args.model, args.lang)
        # 只输出 JSON，没有其他内容
        print(json.dumps({"text": text, "success": True}, ensure_ascii=False))
    except Exception as e:
        error_msg = str(e)
        # 提供更有用的错误信息
        if "cannot find the appropriate snapshot folder" in error_msg or "internet connection" in error_msg or "LocalEntryNotFoundError" in error_msg:
            error_msg = "模型未找到，请先运行: arch -arm64 python3 download_model.py"
        elif "No such file or directory" in error_msg:
            error_msg = f"音频文件无法访问: {args.audio}"
        print(json.dumps({"error": error_msg, "success": False}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

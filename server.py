#!/usr/bin/env python3
"""
FunASR HTTP 服务：启动时加载模型，提供调试页与 /api/transcribe 上传识别。
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

# Hugging Face 镜像（国内）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speech-to-text")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 不在此处加载 FunASR：首次拉模型可能数十分钟，会导致 uvicorn 一直不监听端口。
    # 模型在首次 /api/transcribe 时懒加载；调试页可立即打开。
    yield


def _ffprobe_duration(path: str) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return 0.0
    s = (r.stdout or "").strip()
    if not s or s == "N/A":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


app = FastAPI(title="speech-to-text", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ready", response_model=None)
def api_ready() -> JSONResponse:
    try:
        from funasr_backend import is_model_loaded, load_model

        if not is_model_loaded():
            return JSONResponse({"ready": False, "model": None})
        _, label = load_model()
        return JSONResponse({"ready": True, "model": label})
    except Exception as e:
        log.exception("/api/ready failed")
        return JSONResponse(
            status_code=500,
            content={
                "ready": False,
                "model": None,
                "error": f"{type(e).__name__}: {e}",
            },
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    hotword: str = Form(""),
) -> JSONResponse:
    from funasr_backend import transcribe_file

    suffix = Path(audio.filename or "audio.wav").suffix
    if not suffix or len(suffix) > 8:
        suffix = ".wav"

    data = await audio.read()
    tmp_path: str | None = None
    audio_sec = 0.0
    result: dict | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        audio_sec = _ffprobe_duration(tmp_path)
        try:
            result = await run_in_threadpool(
                transcribe_file,
                tmp_path,
                hotword.strip() or None,
                300.0,
            )
        except Exception as e:
            log.exception("transcribe threadpool failed")
            tb = traceback.format_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "detail": tb[-8000:] if len(tb) > 8000 else tb,
                    "text": "",
                    "audio_seconds": audio_sec,
                    "wall_seconds": 0.0,
                    "rtf": 0.0,
                    "model": "",
                },
            )
    except OSError as e:
        log.exception("temp file failed")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "text": "",
                "audio_seconds": 0.0,
                "wall_seconds": 0.0,
                "rtf": 0.0,
                "model": "",
            },
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if result is None:
        log.error("transcribe: no result (unexpected)")
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "internal_error: no transcription result",
                "text": "",
                "audio_seconds": audio_sec,
                "wall_seconds": 0.0,
                "rtf": 0.0,
                "model": "",
            },
        )

    if result.get("ok") is False:
        log.warning("transcribe failed: %s", result.get("error", ""))
        return JSONResponse(
            {
                "ok": False,
                "error": result.get("error", "unknown"),
                "text": result.get("text", ""),
                "audio_seconds": audio_sec,
                "wall_seconds": result.get("wall_seconds", 0),
                "rtf": 0.0,
                "model": result.get("model", ""),
            },
            status_code=500,
        )

    wall = float(result.get("wall_seconds", 0))
    rtf = wall / audio_sec if audio_sec > 0 else 0.0
    return JSONResponse(
        {
            "ok": True,
            "text": result.get("text", ""),
            "audio_seconds": audio_sec,
            "wall_seconds": wall,
            "rtf": rtf,
            "model": result.get("model", ""),
            "filename": audio.filename or "",
        }
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8888"))
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=os.environ.get("UVICORN_RELOAD", "").lower() in ("1", "true"),
    )


if __name__ == "__main__":
    main()

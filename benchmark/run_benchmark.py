#!/usr/bin/env python3
"""
FunASR-only benchmark: RTF + CER vs reference .txt files (same stem as audio).

  python benchmark/run_benchmark.py --dir /path/to/audio_and_txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


def normalize_for_cer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def levenshtein_chars(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_n = normalize_for_cer(ref)
    hyp_n = normalize_for_cer(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return levenshtein_chars(ref_n, hyp_n) / len(ref_n)


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
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


def discover_pairs(audio_dir: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for p in sorted(audio_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
            continue
        ref = p.with_suffix(".txt")
        pairs.append((p, ref if ref.is_file() else None))
    return pairs


def main() -> int:
    from funasr_backend import load_model, transcribe_file

    ap = argparse.ArgumentParser(description="FunASR benchmark (CER + RTF)")
    ap.add_argument("--dir", type=Path, required=True, help="Audio + same-name .txt refs")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    audio_dir = args.dir.expanduser().resolve()
    if not audio_dir.is_dir():
        print(f"Not a directory: {audio_dir}", file=sys.stderr)
        return 1

    pairs = discover_pairs(audio_dir)
    if not pairs:
        print(f"No audio in {audio_dir}", file=sys.stderr)
        return 1

    ref_by_stem: dict[str, str] = {}
    for audio, ref in pairs:
        if ref is not None:
            ref_by_stem[audio.stem] = ref.read_text(encoding="utf-8")

    t_load = time.perf_counter()
    _, model_label = load_model()
    load_s = time.perf_counter() - t_load

    results: list[dict[str, Any]] = []
    print()
    print(f"Directory: {audio_dir}")
    print(f"Model load: {load_s:.3f}s  ({model_label})")
    print()

    cers: list[float] = []
    rtfs: list[float] = []

    for audio, _ref in pairs:
        audio_sec = ffprobe_duration(audio)
        tr = transcribe_file(audio)
        wall = tr.get("wall_seconds", 0.0)
        text = tr.get("text", "")
        err = tr.get("error")
        rtf = wall / audio_sec if audio_sec > 0 else 0.0
        stem = audio.stem
        ref_text = ref_by_stem.get(stem)
        c_val: float | None = None
        if ref_text is not None and normalize_for_cer(ref_text) and tr.get("ok") and not err:
            c_val = cer(ref_text, text)
            cers.append(c_val)

        if not err and tr.get("ok"):
            rtfs.append(rtf)

        row = {
            "file": str(audio),
            "text": text,
            "wall_seconds": wall,
            "audio_seconds": audio_sec,
            "rtf": rtf,
            "cer": c_val,
            "error": err,
        }
        results.append(row)

        if err:
            print(f"  {audio.name}: ERROR {err}")
        else:
            cs = f"  CER={c_val:.4f}" if c_val is not None else "  CER=n/a"
            print(f"  {audio.name}: RTF={rtf:.3f}  wall={wall:.2f}s  audio={audio_sec:.2f}s{cs}")

    if cers:
        print(f"  Average CER: {sum(cers)/len(cers):.4f}  (n={len(cers)})")
    if rtfs:
        print(f"  Average RTF: {sum(rtfs)/len(rtfs):.4f}  (n={len(rtfs)})")

    out = {"backend": "funasr", "model": model_label, "load_seconds": load_s, "results": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

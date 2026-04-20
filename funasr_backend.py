"""
FunASR inference: Paraformer + VAD + punctuation (offline file ASR).
Used by server.py and benchmark/run_benchmark.py.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_load_lock = threading.Lock()

# Optional mirrors (China-friendly ModelScope / HF mirror)
if not os.environ.get("MODELSCOPE_CACHE"):
    os.environ.setdefault(
        "MODELSCOPE_CACHE",
        str(Path(__file__).resolve().parent / "models" / "modelscope"),
    )


def _apply_perf_debug_defaults() -> None:
    """
    TODO(revert-after-perf-debug):
      - 删除本函数及文件末尾对它的调用；或把默认改为 FUNASR_PERF_DEBUG=0。

    临时本地调试：FUNASR_PERF_DEBUG=1（默认）时，在 macOS 上未指定设备则尝试 ``mps``。
    不再默认 FUNASR_SIMPLE：仅 ASR 会跳过标点模型，输出无标点；需要标点请保持 FUNASR_SIMPLE 未开启。
    关闭整段调试：export FUNASR_PERF_DEBUG=0
    """
    if os.environ.get("FUNASR_PERF_DEBUG", "1") != "1":
        return
    if sys.platform == "darwin":
        os.environ.setdefault("FUNASR_DEVICE", "mps")


_apply_perf_debug_defaults()

_mps_torch_patched: bool = False

_model: Any = None
_model_label: str = ""


def _device_str_is_mps(device: str) -> bool:
    d = (device or "").strip().lower()
    return d == "mps" or d.startswith("mps:")


def _configure_torch_for_mps_if_needed(device: str) -> None:
    """
    MPS does not support float64. FunASR / numpy audio often yields float64; torch.from_numpy
    keeps float64 and later ops fail. Force float32 tensors from numpy when using MPS.
    """
    global _mps_torch_patched
    if not _device_str_is_mps(device):
        return

    import torch

    torch.set_default_dtype(torch.float32)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if _mps_torch_patched:
        return
    _orig_from_numpy = torch.from_numpy
    _orig_as_tensor = torch.as_tensor

    def _from_numpy_float32(ndarray: Any) -> Any:
        t = _orig_from_numpy(ndarray)
        if t.dtype == torch.float64:
            return t.float()
        return t

    def _as_tensor_float32(obj: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Any:
        t = _orig_as_tensor(obj, dtype=dtype, device=device, **kwargs)
        if t.dtype == torch.float64:
            return t.float()
        return t

    torch.from_numpy = _from_numpy_float32  # type: ignore[method-assign]
    torch.as_tensor = _as_tensor_float32  # type: ignore[method-assign]
    _mps_torch_patched = True


_configure_torch_for_mps_if_needed(os.environ.get("FUNASR_DEVICE", "cpu"))


def _build_kwargs() -> tuple[dict[str, Any], str, bool]:
    """
    - Main ASR on HF: use full repo id ``funasr/paraformer-zh`` (short ``paraformer-zh`` may not register).
    - VAD / punctuation: use **short ids** ``fsmn-vad``, ``ct-punc`` and pass ``hub=hf`` — FunASR resolves
      them on Hugging Face. Do **not** use ``funasr/fsmn-vad`` here; some code paths wrongly hit
      ModelScope and 404 (MS has no ``funasr/`` namespace).
    - For ``hub=ms``, optional full ``iic/...`` ids via env overrides.
    """
    device = os.environ.get("FUNASR_DEVICE", "cpu")
    hub = (os.environ.get("FUNASR_HUB", "hf") or "").strip()
    simple = os.environ.get("FUNASR_SIMPLE", "").lower() in ("1", "true", "yes")

    hub_l = hub.lower() if hub else "hf"
    use_hf_repo_ids = hub_l in ("hf", "huggingface", "")

    if os.environ.get("FUNASR_MODEL"):
        model_id = os.environ["FUNASR_MODEL"]
    elif use_hf_repo_ids:
        model_id = "funasr/paraformer-zh"
    else:
        model_id = "paraformer-zh"

    kw: dict[str, Any] = {
        "model": model_id,
        "device": device,
    }
    if hub:
        kw["hub"] = hub
    kw["disable_update"] = True
    if not simple:
        if os.environ.get("FUNASR_VAD_MODEL"):
            vad = os.environ["FUNASR_VAD_MODEL"]
        elif use_hf_repo_ids:
            vad = "fsmn-vad"
        else:
            vad = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        kw["vad_model"] = vad
        kw["vad_kwargs"] = {"max_single_segment_time": 30000}
        if os.environ.get("FUNASR_PUNC_MODEL"):
            punc = os.environ["FUNASR_PUNC_MODEL"]
        elif use_hf_repo_ids:
            punc = "ct-punc"
        else:
            punc = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
        kw["punc_model"] = punc

    return kw, model_id, simple


def is_model_loaded() -> bool:
    return _model is not None


def _mps_cpu_fallback_enabled() -> bool:
    return os.environ.get("FUNASR_MPS_FALLBACK_CPU", "1").lower() not in ("0", "false", "no")


def _reset_model_cache() -> None:
    global _model, _model_label
    with _load_lock:
        _model = None
        _model_label = ""


def load_model() -> tuple[Any, str]:
    """Load (or return cached) AutoModel. Returns (model, display label). Thread-safe."""
    global _model, _model_label
    if _model is not None:
        return _model, _model_label

    with _load_lock:
        if _model is not None:
            return _model, _model_label

        kw, model_id, simple = _build_kwargs()
        _configure_torch_for_mps_if_needed(str(kw.get("device", "cpu")))

        from funasr import AutoModel

        label = model_id if simple else f"{model_id}+vad+punc"

        def _instantiate(kw_in: dict[str, Any]) -> Any:
            k = dict(kw_in)
            try:
                return AutoModel(**k)
            except TypeError:
                k.pop("disable_update", None)
                try:
                    return AutoModel(**k)
                except TypeError:
                    k.pop("hub", None)
                    return AutoModel(**k)

        try:
            _model = _instantiate(kw)
        except Exception as e:
            if _mps_cpu_fallback_enabled() and _device_str_is_mps(str(kw.get("device", "cpu"))):
                log.warning("AutoModel failed on MPS (%s), falling back to CPU", e)
                os.environ["FUNASR_DEVICE"] = "cpu"
                kw, model_id, simple = _build_kwargs()
                _configure_torch_for_mps_if_needed(str(kw.get("device", "cpu")))
                label = model_id if simple else f"{model_id}+vad+punc"
                _model = _instantiate(kw)
            else:
                raise

        _model_label = label
        return _model, _model_label


def result_to_text(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, list):
        if not out:
            return ""
        item = out[0]
        if isinstance(item, dict):
            t = item.get("text")
            if t is not None:
                return str(t).strip()
        return str(item).strip()
    if isinstance(out, dict):
        t = out.get("text")
        if t is not None:
            return str(t).strip()
    return str(out).strip()


def transcribe_file(
    audio_path: str | Path,
    hotword: str | None = None,
    batch_size_s: float = 300.0,
) -> dict[str, Any]:
    """
    Run ASR on a file path. Returns dict with text, wall_seconds, error (if any).
    """
    path = Path(audio_path)
    gen_kw: dict[str, Any] = {
        "input": str(path),
        "batch_size_s": batch_size_s,
    }
    if hotword and hotword.strip():
        gen_kw["hotword"] = hotword.strip()

    device_before = os.environ.get("FUNASR_DEVICE", "cpu")
    label = ""
    t0 = time.perf_counter()

    for attempt in range(2):
        try:
            model, label = load_model()
            out = model.generate(**gen_kw)
            text = result_to_text(out)
        except Exception as e:
            wall = time.perf_counter() - t0
            if (
                attempt == 0
                and _mps_cpu_fallback_enabled()
                and _device_str_is_mps(device_before)
            ):
                log.warning("FunASR failed on MPS (%s), retrying on CPU", e)
                _reset_model_cache()
                os.environ["FUNASR_DEVICE"] = "cpu"
                device_before = "cpu"
                continue
            return {
                "ok": False,
                "error": str(e),
                "text": "",
                "wall_seconds": wall,
                "model": label,
            }
        wall = time.perf_counter() - t0
        return {
            "ok": True,
            "text": text,
            "wall_seconds": wall,
            "model": label,
        }

    wall = time.perf_counter() - t0
    return {
        "ok": False,
        "error": "transcribe exhausted retries",
        "text": "",
        "wall_seconds": wall,
        "model": label,
    }

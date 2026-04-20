# 语音识别（FunASR）

基于 [FunASR](https://github.com/modelscope/FunASR) 的离线语音识别：**Paraformer + VAD + 标点**，默认从 Hugging Face 拉取模型（可通过环境变量改用 ModelScope）。

## 环境

- macOS（Apple Silicon 或 Intel 均可）
- Python 3.10+（推荐 Homebrew `/opt/homebrew/bin/python3`）
- ffmpeg（用于音频时长与解码）

## 安装

```bash
./setup.sh
```

或手动：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 启动

```bash
.venv/bin/python server.py
```

浏览器打开 **http://localhost:8888** — 调试页支持拖拽上传、可选热词、展示识别文本与 RTF。

服务启动后应很快出现 `Application startup complete` 并监听端口。**模型在首次调用「开始识别」时加载**（避免启动阶段长时间下载导致浏览器无法连接）。`/api/ready` 可查看是否已加载完成。

环境变量（可选）：

| 变量 | 含义 |
|------|------|
| `FUNASR_MODEL` | 未设置时：`FUNASR_HUB=hf` 下为 **`funasr/paraformer-zh`**（HF 全名）；`ms` 下为 `paraformer-zh` |
| `FUNASR_VAD_MODEL` / `FUNASR_PUNC_MODEL` | 可选覆盖；默认 hf 下为短名 `fsmn-vad`、`ct-punc`（由 hub 解析） |
| `FUNASR_DEVICE` | 默认 `cpu`；有 NVIDIA 可试 `cuda:0`，Apple Silicon 可试 `mps` |
| `FUNASR_HUB` | `hf`（默认）或 `ms`（ModelScope，国内常用） |
| `FUNASR_SIMPLE` | 设为 `1` 则仅 ASR，不加载 VAD/标点（更快、略简） |
| `FUNASR_PERF_DEBUG` | **临时性能调试**：默认 `1` 时仅在 macOS 上未指定设备则尝试 `mps`（**不再**默认 `FUNASR_SIMPLE`，以免跳过标点）。调试结束后请设为 **`0`** 或见下方 TODO。 |
| `FUNASR_MPS_FALLBACK_CPU` | 默认 `1`：`FUNASR_DEVICE=mps` 时若加载或推理失败，自动清空缓存并改用 **CPU** 重试一次。设为 `0` 可关闭。 |
| `PORT` | 默认 `8888` |

### 性能调试（临时，结束后请改回）

当前在 `funasr_backend.py` 里默认开启了 **`FUNASR_PERF_DEBUG=1`**（macOS 上尽量用 `mps`；**完整 ASR + VAD + 标点**）。**调试结束后请：**

1. 运行前执行：`export FUNASR_PERF_DEBUG=0`（不再自动选 `mps`，设备仍可由 `FUNASR_DEVICE` 指定）；或  
2. 按源码中 **`TODO(revert-after-perf-debug)`** 注释删掉 `_apply_perf_debug_defaults` 及其调用，并把正式环境的默认策略写回 README。

若 `mps` 仍报错：本仓库在 `FUNASR_DEVICE=mps` 时会将 `torch.from_numpy` / `torch.as_tensor` 产生的 float64 转为 float32（MPS 不支持 float64），并默认开启 `PYTORCH_ENABLE_MPS_FALLBACK=1`。若仍失败，可 `export FUNASR_DEVICE=cpu`。

## 基准测试（可选）

```bash
.venv/bin/python benchmark/run_benchmark.py --dir /path/to/wavs_and_txts --json-out artifacts/out.json
```

同目录下每段音频需有同名 `.txt` 参考文本方可计算 CER。

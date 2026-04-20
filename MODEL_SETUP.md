# 模型与缓存

- **FunASR** 首次运行会从 `FUNASR_HUB` 指定的源下载子模型（ASR / VAD / 标点），体积较大，请保持网络畅通。
- **`hub=hf`（默认）**：主 ASR 用 **`funasr/paraformer-zh`**；VAD / 标点请用短名 **`fsmn-vad`**、**`ct-punc`**（由 `hub=hf` 解析到 Hugging Face）。不要写 **`funasr/fsmn-vad`**，否则部分版本会错误地走 ModelScope 导致 404。
- **`hub=ms`**：默认 VAD / 标点为 ModelScope 上的 **`iic/...`** 全名；主模型仍可用 `paraformer-zh` 等短名（由 MS 解析）。
- **ModelScope 缓存** 默认目录：`models/modelscope/`（可通过环境变量 `MODELSCOPE_CACHE` 覆盖）。
- 仅 ASR、不加载 VAD/标点：设置 `FUNASR_SIMPLE=1`（启动更快、占用更小）。

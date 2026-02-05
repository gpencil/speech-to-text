# Whisper 模型下载指南

首次使用需要下载 Whisper 模型。

## 方法 1: 自动下载（推荐）

运行模型下载脚本（已配置国内镜像）：

```bash
arch -arm64 python3 download_model.py
```

默认下载 `base` 模型（约 150MB），支持中英文识别。

如需下载其他大小的模型：

```bash
# tiny 模型（约 40MB，最快但准确率较低）
arch -arm64 python3 download_model.py --model tiny

# small 模型（约 500MB，准确率较高）
arch -arm64 python3 download_model.py --model small

# medium 模型（约 1.5GB）
arch -arm64 python3 download_model.py --model medium

# large 模型（约 3GB，最准确）
arch -arm64 python3 download_model.py --model large
```

## 方法 2: 手动下载（使用镜像）

如果自动下载失败，可以使用国内镜像手动下载：

1. 访问 [HF-Mirror - Faster Whisper Base](https://hf-mirror.com/guillaumekln/faster-whisper-base)
2. 下载所有文件
3. 放到 `./models/guillaumekln/faster-whisper-base/` 目录

或使用其他镜像站：
- https://hf-mirror.com
- https://huggingface.co （官方，可能需要代理）

## 方法 3: 使用代理

如果有代理，可以设置环境变量：

```bash
export https_proxy=http://127.0.0.1:7890
arch -arm64 python3 download_model.py
```

## 模型说明

| 模型 | 大小 | 速度 | 准确率 | 推荐场景 |
|------|------|------|--------|----------|
| tiny | ~40MB | 最快 | 较低 | 测试、快速转录 |
| base | ~150MB | 快 | 中等 | 日常使用（推荐） |
| small | ~500MB | 中等 | 较高 | 重要内容 |
| medium | ~1.5GB | 慢 | 高 | 专业使用 |
| large | ~3GB | 最慢 | 最高 | 最高要求 |

## 验证模型

下载完成后，可以验证模型是否可用：

```bash
arch -arm64 python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('base'); print('✓ 模型可用')"
```

## 网络问题

如果下载失败：
1. 确保使用 ARM64 Python：`arch -arm64 python3`
2. 检查网络连接
3. 尝试使用镜像（脚本已默认使用 hf-mirror.com）
4. 如果在中国大陆，可能需要配置代理或使用其他镜像

下载完成后，模型会缓存到本地，后续使用无需重新下载。

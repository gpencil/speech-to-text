# 离线语音识别工具

基于 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 的离线语音识别 Web 服务，支持中英文识别。模型在服务启动时一次性加载到内存，后续请求直接推理，无重复加载开销。

## 功能

- 离线推理，无需联网
- 支持中文（简体）、英文及多语言自动检测
- 单文件识别 → 直接下载同名 `.txt`
- 批量识别（文件夹模式）→ 打包为 `.zip` 下载
- 支持拖拽上传
- 使用 Metal GPU 加速（Apple Silicon）

## 系统要求

- macOS（Apple Silicon）
- Go 1.24+
- [whisper-cpp](https://github.com/ggml-org/whisper.cpp)（提供 C++ 库和头文件）
- ffmpeg（音频格式转换）

## 安装

### 1. 安装依赖

```bash
arch -arm64 brew install whisper-cpp ffmpeg
```

### 2. 下载模型

模型文件为 GGML 格式（`.bin`），放入 `models/` 目录：

```bash
mkdir -p models
# 国内镜像
curl -L -o models/ggml-base.bin \
  "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
```

可选模型（精度从低到高，速度从快到慢）：

| 文件名 | 大小 | 推荐场景 |
|---|---|---|
| `ggml-tiny.bin` | 75 MB | 快速预览 |
| `ggml-base.bin` | 141 MB | 日常使用（默认）|
| `ggml-small.bin` | 465 MB | 更高精度 |
| `ggml-medium.bin` | 1.5 GB | 高精度中文 |
| `ggml-large-v3.bin` | 3.1 GB | 最高精度 |

### 3. 编译

```bash
./build.sh
```

或手动执行：

```bash
CGO_CFLAGS="-I/opt/homebrew/include" \
CGO_LDFLAGS="-L/opt/homebrew/lib" \
go build -mod=vendor -o speech-to-text .
```

## 启动

```bash
./speech-to-text
```

服务监听 `http://localhost:8888`。

默认加载 `./models/ggml-base.bin`，可通过环境变量指定其他模型：

```bash
WHISPER_MODEL=./models/ggml-medium.bin ./speech-to-text
```

## 使用

打开 `http://localhost:8888`：

1. 选择模式：**单个文件** 或 **文件夹批量**
2. 上传音频（支持 WAV、MP3、M4A、FLAC 等 ffmpeg 可解码的格式）
3. 选择语言（中文 / English / 自动检测）
4. 点击**开始识别**
5. 自动下载识别结果（单文件 → `.txt`，批量 → `.zip`）

## 项目结构

```
.
├── asr/
│   └── asr.go          # ASR 封装包（模型加载、推理、格式转换）
├── main.go             # HTTP 服务
├── vendor/             # Go 依赖（含修改过的 whisper.cpp binding）
├── models/             # GGML 模型文件（不提交到 git）
├── tmp/                # 临时文件（自动清理）
├── build.sh            # 构建脚本
└── go.mod
```

## 工作原理

```
启动时：加载模型到内存（Metal GPU，一次性，约 10s）
         ↓
每次请求：ffmpeg 转为 16kHz mono WAV
         ↓
         whisper.cpp 推理（约 3s / 文件）
         ↓
         单文件返回 .txt，多文件返回 .zip
```

## 健康检查

```bash
curl http://localhost:8888/health
```

## 常见问题

**编译报 `whisper.h not found`**

确认 whisper-cpp 已通过 homebrew 安装，并使用正确的 CGO 路径：
```bash
arch -arm64 brew install whisper-cpp
# 然后用 build.sh 编译
```

**启动报 `模型加载失败`**

确认 `models/` 目录下有对应的 `.bin` 文件，参考上方下载步骤。

**中文识别出现繁体字**

选择语言时选"中文"而非"自动检测"，服务会自动加 Simplified Chinese 引导提示。

## License

MIT

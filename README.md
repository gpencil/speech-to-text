# 离线语音识别工具

基于 Whisper 的离线语音识别 Web 服务，支持中英文语音识别。

## 功能特点

- 🎤 离线语音识别，无需联网
- 🌏 支持中文、英文及多语言自动检测
- 📁 简单的 Web 界面，支持拖拽上传
- 🗑️ 自动清理临时文件
- 🚀 基于 faster-whisper，快速高效

## 系统要求

- Go 1.24+
- Python 3.8+
- ffmpeg

## 安装步骤

### 1. 安装 ffmpeg

```bash
brew install ffmpeg
```

### 2. 安装 Python 依赖

```bash
pip3 install faster-whisper
```

### 3. 编译 Go 程序

```bash
go build -o speech-to-text
```

## 使用方法

### 启动服务器

```bash
./speech-to-text
```

服务器将在 http://localhost:8080 启动。

### 使用 Web 界面

1. 打开浏览器访问 http://localhost:8080
2. 点击或拖拽音频文件到上传区域
3. 选择识别语言（中文/英文/自动检测）
4. 点击"开始识别"按钮
5. 等待识别完成，查看结果

## 支持的音频格式

- WAV
- MP3
- M4A
- FLAC
- 其他 ffmpeg 支持的格式

## 项目结构

```
.
├── main.go              # Web 服务器
├── recognize.py         # Python 语音识别脚本
├── speech/              # Go 语音识别模块
├── tmp/                 # 临时文件目录
└── models/              # 模型下载目录（自动）
```

## 工作原理

1. 用户上传音频文件
2. 文件保存到 `./tmp` 临时目录
3. 调用 Python faster-whisper 进行识别
4. 返回识别结果
5. 自动删除临时音频文件

## 模型说明

首次运行时，faster-whisper 会自动下载 Whisper 模型到 `./models` 目录。

可选模型：
- `tiny` - 最小最快，精度较低
- `base` - 推荐，平衡速度和精度
- `small` - 更高精度
- `medium` - 高精度
- `large` - 最高精度，速度较慢

## 配置

可以通过修改 `recognize.py` 中的模型大小参数：

```python
model = WhisperModel(
    "base",  # 可改为 tiny, small, medium, large
    device="cpu",
    compute_type="int8",
)
```

## 常见问题

### 1. Python 依赖检查失败

确保已安装 faster-whisper：
```bash
pip3 install faster-whisper
```

### 2. ffmpeg 未找到

```bash
brew install ffmpeg
```

### 3. 识别速度慢

可以尝试更小的模型（如 `tiny`），或者使用 GPU（需要修改代码）。

## 开发

### 运行开发服务器

```bash
go run main.go
```

### 健康检查

```bash
curl http://localhost:8080/health
```

## License

MIT

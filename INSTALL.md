# 离线语音识别工具 - 安装说明

## 依赖安装

### 1. 安装 ffmpeg
```bash
brew install ffmpeg
```

### 2. 安装 Vosk API

Vosk API 需要从源码编译安装：

```bash
# 克隆 Vosk API 仓库
git clone https://github.com/alphacep/vosk-api.git
cd vosk-api

# 编译并安装
mkdir build && cd build
cmake ..
make
sudo make install

# 安装到系统库路径
sudo cp -r /usr/local/lib/vosk /usr/local/lib/
sudo ldconfig  # Linux
# macOS 上通常不需要 ldconfig
```

### 3. 安装 CGO 依赖
确保系统已安装 Xcode Command Line Tools：
```bash
xcode-select --install
```

## 运行项目

```bash
# 下载 Go 依赖
go mod download

# 运行服务器
go run main.go
```

服务器将在 http://localhost:8080 启动。

## 使用方法

1. 打开浏览器访问 http://localhost:8080
2. 上传音频文件（WAV, MP3, M4A 等）
3. 点击"开始识别"
4. 等待识别完成，查看结果

## 音频格式要求

- **最佳格式**: WAV PCM, 16kHz, 单声道, 16-bit
- **其他格式**: 会自动使用 ffmpeg 转换

## 临时文件

上传的音频文件会保存到 `./tmp` 目录，识别完成后自动删除。

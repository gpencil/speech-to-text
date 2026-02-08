package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var (
	pythonAvailable bool
	pythonChecked   bool
	checkMu         sync.Once
)

func main() {
	// 检查 Python 依赖
	checkPythonDeps()

	// 设置路由
	http.HandleFunc("/", handleHome)
	http.HandleFunc("/upload", handleUpload)
	http.HandleFunc("/health", handleHealth)

	// 静态文件
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))

	port := ":8888"
	fmt.Printf("服务器启动在 http://localhost%s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}

func checkPythonDeps() {
	checkMu.Do(func() {
		// 检查 Python3
		pythonPath, err := exec.LookPath("python3")
		if err != nil {
			log.Println("警告: 未找到 python3")
			return
		}

		// 检查 faster-whisper (使用 ARM64 Python)
		cmd := exec.Command("arch", "-arm64", pythonPath, "-c", "import faster_whisper")
		cmd.Stdout = nil
		cmd.Stderr = nil
		if err = cmd.Run(); err != nil {
			log.Println("警告: faster-whisper 未安装或架构不匹配")
			log.Println("请运行: arch -arm64 pip3 install --force-reinstall faster-whisper ctranslate2 onnxruntime")
			return
		}

		pythonAvailable = true
		pythonChecked = true
	})
}

func handleHome(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(htmlTemplate))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           "ok",
		"python_available": pythonAvailable,
	})
}

func handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if !pythonAvailable {
		http.Error(w, "Python 依赖不可用，请先安装 faster-whisper", http.StatusInternalServerError)
		return
	}

	// 解析表单 (最大 500MB，支持多个文件)
	if err := r.ParseMultipartForm(500 << 20); err != nil {
		http.Error(w, "解析表单失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 获取语言参数
	lang := r.FormValue("lang")
	if lang == "" {
		lang = "auto"
	}

	// 检查是否有文件上传
	files := r.MultipartForm.File["audio"]
	if len(files) == 0 {
		http.Error(w, "未找到文件", http.StatusBadRequest)
		return
	}

	// 创建临时目录
	tmpDir := "./tmp"
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		http.Error(w, "创建临时目录失败", http.StatusInternalServerError)
		return
	}

	// 创建一个缓冲区来存储zip文件
	var zipBuffer bytes.Buffer
	zipWriter := zip.NewWriter(&zipBuffer)

	// 处理每个文件
	successCount := 0
	failCount := 0
	var errors []string

	for _, fileHeader := range files {
		// 打开上传的文件
		file, err := fileHeader.Open()
		if err != nil {
			failCount++
			errors = append(errors, fmt.Sprintf("%s: 打开文件失败", fileHeader.Filename))
			continue
		}

		// 保存到临时位置
		tmpFileName := fmt.Sprintf("%s_%d%s", filepath.Base(fileHeader.Filename), time.Now().UnixNano(), filepath.Ext(fileHeader.Filename))
		tmpFilePath := filepath.Join(tmpDir, tmpFileName)

		dst, err := os.Create(tmpFilePath)
		if err != nil {
			file.Close()
			failCount++
			errors = append(errors, fmt.Sprintf("%s: 创建临时文件失败", fileHeader.Filename))
			continue
		}

		if _, err := io.Copy(dst, file); err != nil {
			dst.Close()
			file.Close()
			os.Remove(tmpFilePath)
			failCount++
			errors = append(errors, fmt.Sprintf("%s: 保存文件失败", fileHeader.Filename))
			continue
		}
		dst.Close()
		file.Close()

		// 确保最后删除临时文件
		defer os.Remove(tmpFilePath)

		// 执行识别
		text, err := recognizeWithPython(tmpFilePath, lang)
		if err != nil {
			failCount++
			errors = append(errors, fmt.Sprintf("%s: %s", fileHeader.Filename, err.Error()))
			continue
		}

		// 生成txt文件名（与音频文件同名）
		txtFileName := filepath.Base(fileHeader.Filename)
		txtFileName = strings.TrimSuffix(txtFileName, filepath.Ext(txtFileName)) + ".txt"

		// 将文本添加到zip中
		writer, err := zipWriter.Create(txtFileName)
		if err != nil {
			failCount++
			errors = append(errors, fmt.Sprintf("%s: 创建zip条目失败", fileHeader.Filename))
			continue
		}

		_, err = writer.Write([]byte(text))
		if err != nil {
			failCount++
			errors = append(errors, fmt.Sprintf("%s: 写入zip失败", fileHeader.Filename))
			continue
		}

		successCount++
	}

	// 关闭zip writer
	err := zipWriter.Close()
	if err != nil {
		http.Error(w, "创建zip文件失败", http.StatusInternalServerError)
		return
	}

	// 如果全部失败
	if successCount == 0 {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "所有文件处理失败",
			"errors":  errors,
		})
		return
	}

	// 设置响应头，下载zip文件
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="transcripts.zip"`)
	w.Write(zipBuffer.Bytes())
}

func recognizeWithPython(audioPath, lang string) (string, error) {
	pythonPath, err := exec.LookPath("python3")
	if err != nil {
		return "", fmt.Errorf("未找到 python3")
	}

	scriptPath := "./recognize.py"
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return "", fmt.Errorf("recognize.py 不存在")
	}

	// 执行 Python 脚本 (使用 ARM64 Python)
	cmd := exec.Command("arch", "-arm64", pythonPath, scriptPath, audioPath, "--lang", lang)

	output, err := cmd.CombinedOutput()
	if err != nil {
		// 尝试解析输出中的错误信息
		var result struct {
			Error   string `json:"error"`
			Success bool   `json:"success"`
		}
		if json.Unmarshal(output, &result) == nil && result.Error != "" {
			return "", fmt.Errorf("%s", result.Error)
		}
		return "", fmt.Errorf("执行识别失败: %w", err)
	}

	// 解析 JSON 结果
	var result struct {
		Text    string `json:"text"`
		Success bool   `json:"success"`
		Error   string `json:"error"`
	}

	if err := json.Unmarshal(output, &result); err != nil {
		return "", fmt.Errorf("解析结果失败: %w\n原始输出: %s", err, string(output))
	}

	if !result.Success {
		errMsg := result.Error
		if errMsg == "" {
			errMsg = "未知错误"
		}
		return "", fmt.Errorf("%s", errMsg)
	}

	return result.Text, nil
}

// HTML 模板
const htmlTemplate = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8ebff;
            transform: scale(1.02);
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .upload-text {
            color: #667eea;
            font-size: 16px;
            font-weight: 500;
        }
        .upload-hint {
            color: #999;
            font-size: 12px;
            margin-top: 10px;
        }
        input[type="file"] {
            display: none;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            margin-top: 20px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .result-area {
            margin-top: 30px;
            display: none;
        }
        .result-area.show {
            display: block;
        }
        .result-title {
            color: #333;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .result-box {
            background: #f8f9ff;
            border: 2px solid #e8ebff;
            border-radius: 10px;
            padding: 20px;
            min-height: 120px;
            color: #333;
            line-height: 1.8;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .file-info {
            margin-top: 20px;
            padding: 15px;
            background: #f0f2ff;
            border-radius: 10px;
            display: none;
        }
        .file-info.show {
            display: block;
        }
        .file-name {
            color: #667eea;
            font-weight: 500;
            word-break: break-all;
        }
        .language-select {
            margin-top: 20px;
        }
        .language-select label {
            color: #666;
            font-size: 14px;
            display: block;
            margin-bottom: 8px;
        }
        .language-select select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e8ebff;
            border-radius: 10px;
            font-size: 14px;
            cursor: pointer;
        }
        .error-message {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        .error-message.show {
            display: block;
        }
        .mode-select {
            margin-bottom: 20px;
        }
        .mode-select label {
            color: #666;
            font-size: 14px;
            display: block;
            margin-bottom: 8px;
        }
        .mode-buttons {
            display: flex;
            gap: 10px;
        }
        .mode-btn {
            flex: 1;
            background: #f8f9ff;
            border: 2px solid #e8ebff;
            color: #667eea;
            padding: 12px;
            border-radius: 10px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .mode-btn:hover {
            background: #f0f2ff;
        }
        .mode-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        .progress-container {
            margin-top: 20px;
            display: none;
        }
        .progress-container.show {
            display: block;
        }
        .progress-item {
            background: #f8f9ff;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 13px;
        }
        .progress-item.success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        .progress-item.error {
            background: #ffebee;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <p class="subtitle">基于 Whisper，支持中英文语音识别</p>

        <div class="mode-select">
            <label>选择模式:</label>
            <div class="mode-buttons">
                <button class="mode-btn active" id="singleFileBtn">单个文件</button>
                <button class="mode-btn" id="folderBtn">文件夹批量</button>
            </div>
        </div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📁</div>
            <div class="upload-text" id="uploadText">点击或拖拽音频文件到此处</div>
            <div class="upload-hint" id="uploadHint">支持 WAV, MP3, M4A 等格式</div>
        </div>

        <input type="file" id="fileInput" accept="audio/*">
        <input type="file" id="folderInput" accept="audio/*" webkitdirectory directory multiple style="display: none;">

        <div class="file-info" id="fileInfo">
            <div class="file-name" id="fileName"></div>
        </div>

        <div class="progress-container" id="progressContainer">
            <div style="color: #666; margin-bottom: 10px;">处理进度:</div>
            <div id="progressList"></div>
        </div>

        <div class="language-select">
            <label for="language">选择语言:</label>
            <select id="language">
                <option value="zh">中文</option>
                <option value="en">English</option>
                <option value="auto">自动检测</option>
            </select>
        </div>

        <button class="btn" id="recognizeBtn" disabled>开始识别</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>正在识别中，请稍候...</div>
        </div>

        <div class="error-message" id="errorMessage"></div>

        <div class="result-area" id="resultArea">
            <div class="result-title">识别结果:</div>
            <div class="result-box" id="resultText"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const folderInput = document.getElementById('folderInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const recognizeBtn = document.getElementById('recognizeBtn');
        const loading = document.getElementById('loading');
        const resultArea = document.getElementById('resultArea');
        const resultText = document.getElementById('resultText');
        const errorMessage = document.getElementById('errorMessage');
        const languageSelect = document.getElementById('language');
        const singleFileBtn = document.getElementById('singleFileBtn');
        const folderBtn = document.getElementById('folderBtn');
        const uploadText = document.getElementById('uploadText');
        const uploadHint = document.getElementById('uploadHint');
        const progressContainer = document.getElementById('progressContainer');
        const progressList = document.getElementById('progressList');

        let selectedFiles = [];
        let isFolderMode = false;

        singleFileBtn.addEventListener('click', () => {
            isFolderMode = false;
            singleFileBtn.classList.add('active');
            folderBtn.classList.remove('active');
            uploadText.textContent = '点击或拖拽音频文件到此处';
            uploadHint.textContent = '支持 WAV, MP3, M4A 等格式';
            selectedFiles = [];
            updateFileInfo();
        });

        folderBtn.addEventListener('click', () => {
            isFolderMode = true;
            folderBtn.classList.add('active');
            singleFileBtn.classList.remove('active');
            uploadText.textContent = '点击选择文件夹';
            uploadHint.textContent = '将处理文件夹中的所有音频文件';
            selectedFiles = [];
            updateFileInfo();
        });

        uploadArea.addEventListener('click', () => {
            if (isFolderMode) {
                folderInput.click();
            } else {
                fileInput.click();
            }
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        folderInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                // 过滤音频文件
                const audioFiles = Array.from(e.target.files).filter(file => file.type.startsWith('audio/'));
                if (audioFiles.length === 0) {
                    showError('所选文件夹中没有音频文件');
                    return;
                }
                handleMultipleFiles(audioFiles);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('audio/')) {
                showError('请选择音频文件');
                return;
            }
            selectedFiles = [file];
            updateFileInfo();
            recognizeBtn.disabled = false;
            hideError();
        }

        function handleMultipleFiles(files) {
            selectedFiles = files;
            updateFileInfo();
            recognizeBtn.disabled = false;
            hideError();
        }

        function updateFileInfo() {
            if (selectedFiles.length === 0) {
                fileInfo.classList.remove('show');
                return;
            }
            fileInfo.classList.add('show');
            if (selectedFiles.length === 1) {
                fileName.textContent = '已选择: ' + selectedFiles[0].name;
            } else {
                fileName.textContent = '已选择 ' + selectedFiles.length + ' 个音频文件';
            }
        }

        recognizeBtn.addEventListener('click', async () => {
            if (selectedFiles.length === 0) return;

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('audio', file);
            });
            formData.append('lang', languageSelect.value);

            loading.classList.add('show');
            resultArea.classList.remove('show');
            progressContainer.classList.remove('show');
            progressList.innerHTML = '';
            recognizeBtn.disabled = true;
            hideError();

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.error || '处理失败');
                }

                // 检查响应类型
                const contentType = response.headers.get('Content-Type');
                if (contentType && contentType.includes('application/zip')) {
                    // 下载zip文件
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'transcripts.zip';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);

                    resultText.textContent = '成功处理 ' + selectedFiles.length + ' 个文件！\n\n已下载 transcripts.zip，解压后将生成与音频文件同名的txt文件。';
                    resultArea.classList.add('show');
                } else {
                    const data = await response.json();
                    throw new Error(data.error || '处理失败');
                }
            } catch (error) {
                showError('处理失败: ' + error.message);
            } finally {
                loading.classList.remove('show');
                recognizeBtn.disabled = false;
            }
        });

        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.classList.add('show');
        }

        function hideError() {
            errorMessage.classList.remove('show');
        }
    </script>
</body>
</html>
`

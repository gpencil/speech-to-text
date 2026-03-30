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
	"path/filepath"
	"strings"
	"sync"

	"github.com/gpencil/speech-to-text/asr"
)

var transcriber *asr.Transcriber

func main() {
	modelPath := os.Getenv("WHISPER_MODEL")
	if modelPath == "" {
		modelPath = "./models/ggml-base.bin"
	}

	log.Printf("加载模型: %s", modelPath)
	var err error
	transcriber, err = asr.New(asr.Config{ModelPath: modelPath})
	if err != nil {
		log.Fatalf("模型加载失败: %v\n\n请先下载模型:\n  mkdir -p models\n  curl -L -o models/ggml-base.bin https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", err)
	}
	defer transcriber.Close()
	log.Printf("模型加载完成")

	http.HandleFunc("/", handleHome)
	http.HandleFunc("/upload", handleUpload)
	http.HandleFunc("/health", handleHealth)

	port := ":8888"
	fmt.Printf("服务启动: http://localhost%s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
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

func handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := r.ParseMultipartForm(500 << 20); err != nil {
		http.Error(w, "解析表单失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	lang := r.FormValue("lang")
	if lang == "" {
		lang = "auto"
	}

	files := r.MultipartForm.File["audio"]
	if len(files) == 0 {
		http.Error(w, "未找到文件", http.StatusBadRequest)
		return
	}

	type result struct {
		txtName string
		text    string
		err     string
	}

	// Process files concurrently; pool in Transcriber limits actual parallelism.
	results := make([]result, len(files))
	var wg sync.WaitGroup
	for i, fh := range files {
		i, fh := i, fh
		wg.Add(1)
		go func() {
			defer wg.Done()
			f, err := fh.Open()
			if err != nil {
				results[i].err = fmt.Sprintf("%s: 打开文件失败", fh.Filename)
				return
			}
			data, err := io.ReadAll(f)
			f.Close()
			if err != nil {
				results[i].err = fmt.Sprintf("%s: 读取文件失败", fh.Filename)
				return
			}

			text, err := transcriber.Transcribe(data, lang)
			if err != nil {
				results[i].err = fmt.Sprintf("%s: %s", fh.Filename, err.Error())
				return
			}

			results[i].txtName = strings.TrimSuffix(filepath.Base(fh.Filename), filepath.Ext(fh.Filename)) + ".txt"
			results[i].text = text
		}()
	}
	wg.Wait()

	// Collect successes and errors.
	var ok []result
	var errMsgs []string
	for _, r := range results {
		if r.err != "" {
			errMsgs = append(errMsgs, r.err)
		} else {
			ok = append(ok, r)
		}
	}

	if len(ok) == 0 {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "所有文件处理失败",
			"errors":  errMsgs,
		})
		return
	}

	// Single file: return .txt directly.
	if len(ok) == 1 {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, ok[0].txtName))
		w.Write([]byte(ok[0].text))
		return
	}

	// Multiple files: return zip.
	var zipBuffer bytes.Buffer
	zipWriter := zip.NewWriter(&zipBuffer)
	for _, r := range ok {
		zw, err := zipWriter.Create(r.txtName)
		if err != nil {
			continue
		}
		zw.Write([]byte(r.text))
	}
	if err := zipWriter.Close(); err != nil {
		http.Error(w, "创建 zip 失败", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="transcripts.zip"`)
	w.Write(zipBuffer.Bytes())
}

const htmlTemplate = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
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
        h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        .upload-area:hover { border-color: #764ba2; background: #f0f2ff; }
        .upload-area.dragover { border-color: #764ba2; background: #e8ebff; transform: scale(1.02); }
        .upload-icon { font-size: 48px; margin-bottom: 15px; }
        .upload-text { color: #667eea; font-size: 16px; font-weight: 500; }
        .upload-hint { color: #999; font-size: 12px; margin-top: 10px; }
        input[type="file"] { display: none; }
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
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102,126,234,0.4); }
        .btn:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        .result-area { margin-top: 30px; display: none; }
        .result-area.show { display: block; }
        .result-title { color: #333; font-size: 18px; font-weight: 600; margin-bottom: 15px; }
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
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .file-info { margin-top: 20px; padding: 15px; background: #f0f2ff; border-radius: 10px; display: none; }
        .file-info.show { display: block; }
        .file-name { color: #667eea; font-weight: 500; word-break: break-all; }
        .language-select { margin-top: 20px; }
        .language-select label { color: #666; font-size: 14px; display: block; margin-bottom: 8px; }
        .language-select select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e8ebff;
            border-radius: 10px;
            font-size: 14px;
            cursor: pointer;
        }
        .error-message { background: #fee; color: #c33; padding: 15px; border-radius: 10px; margin-top: 20px; display: none; }
        .error-message.show { display: block; }
        .mode-select { margin-bottom: 20px; }
        .mode-select label { color: #666; font-size: 14px; display: block; margin-bottom: 8px; }
        .mode-buttons { display: flex; gap: 10px; }
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
        .mode-btn:hover { background: #f0f2ff; }
        .mode-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
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
            <div class="upload-icon">&#128193;</div>
            <div class="upload-text" id="uploadText">点击或拖拽音频文件到此处</div>
            <div class="upload-hint" id="uploadHint">支持 WAV, MP3, M4A 等格式</div>
        </div>

        <input type="file" id="fileInput" accept="audio/*">
        <input type="file" id="folderInput" accept="audio/*" webkitdirectory directory multiple style="display:none">

        <div class="file-info" id="fileInfo">
            <div class="file-name" id="fileName"></div>
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
            isFolderMode ? folderInput.click() : fileInput.click();
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });
        folderInput.addEventListener('change', (e) => {
            const audioFiles = Array.from(e.target.files).filter(f => f.type.startsWith('audio/'));
            if (audioFiles.length === 0) { showError('所选文件夹中没有音频文件'); return; }
            handleMultipleFiles(audioFiles);
        });

        function handleFile(file) {
            if (!file.type.startsWith('audio/')) { showError('请选择音频文件'); return; }
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
            if (selectedFiles.length === 0) { fileInfo.classList.remove('show'); return; }
            fileInfo.classList.add('show');
            fileName.textContent = selectedFiles.length === 1
                ? '已选择: ' + selectedFiles[0].name
                : '已选择 ' + selectedFiles.length + ' 个音频文件';
        }

        recognizeBtn.addEventListener('click', async () => {
            if (selectedFiles.length === 0) return;

            const formData = new FormData();
            selectedFiles.forEach(file => formData.append('audio', file));
            formData.append('lang', languageSelect.value);

            loading.classList.add('show');
            resultArea.classList.remove('show');
            recognizeBtn.disabled = true;
            hideError();

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.error || '处理失败');
                }
                const contentType = response.headers.get('Content-Type') || '';
                if (contentType.includes('text/plain')) {
                    // Single file: download .txt directly
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const disposition = response.headers.get('Content-Disposition') || '';
                    const match = disposition.match(/filename="?([^"]+)"?/);
                    const filename = match ? match[1] : 'transcript.txt';
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    resultText.textContent = '识别完成，已下载 ' + filename;
                    resultArea.classList.add('show');
                } else if (contentType.includes('application/zip')) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'transcripts.zip';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    resultText.textContent = '成功处理 ' + selectedFiles.length + ' 个文件！\n\n已下载 transcripts.zip，解压后将生成与音频文件同名的 txt 文件。';
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

        function showError(msg) {
            errorMessage.textContent = msg;
            errorMessage.classList.add('show');
        }
        function hideError() {
            errorMessage.classList.remove('show');
        }
    </script>
</body>
</html>
`

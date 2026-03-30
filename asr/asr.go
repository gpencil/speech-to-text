// Package asr provides offline speech-to-text transcription using whisper.cpp.
//
// The model is loaded once and held in memory. Inference contexts are pooled
// for safe concurrent use.
//
// Minimal example:
//
//	t, err := asr.New(asr.Config{ModelPath: "models/ggml-base.bin"})
//	if err != nil { log.Fatal(err) }
//	defer t.Close()
//
//	data, _ := os.ReadFile("audio.mp3")
//	text, err := t.Transcribe(data, "zh")
package asr

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os/exec"
	"strings"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// Config holds Transcriber configuration.
type Config struct {
	// ModelPath is the path to the GGML model file (e.g. "models/ggml-base.bin").
	ModelPath string

	// Concurrency is the number of inference contexts pre-allocated in the pool,
	// which controls how many Transcribe calls can run simultaneously.
	// Each context consumes roughly the same memory as the model itself.
	// Defaults to 1 if unset.
	Concurrency int
}

// Transcriber holds a loaded Whisper model and a pool of inference contexts.
// It is safe for concurrent use by multiple goroutines.
type Transcriber struct {
	model   whisper.Model
	ctxPool chan whisper.Context
}

// New loads the GGML model file and pre-allocates inference contexts.
// This call is slow (several seconds); invoke once at application startup.
func New(cfg Config) (*Transcriber, error) {
	model, err := whisper.New(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("load model %q: %w", cfg.ModelPath, err)
	}

	n := cfg.Concurrency
	if n <= 0 {
		n = 1
	}

	pool := make(chan whisper.Context, n)
	for i := 0; i < n; i++ {
		ctx, err := model.NewContext()
		if err != nil {
			for len(pool) > 0 {
				<-pool
			}
			model.Close()
			return nil, fmt.Errorf("create context %d/%d: %w", i+1, n, err)
		}
		pool <- ctx
	}

	return &Transcriber{model: model, ctxPool: pool}, nil
}

// Transcribe converts audio bytes to text.
//
// audio may be in any format that ffmpeg can decode (WAV, MP3, M4A, FLAC, …).
// lang is a BCP-47 language code ("zh", "en", …), or "" / "auto" for
// automatic language detection.
//
// If all inference contexts are busy, Transcribe blocks until one is available.
func (t *Transcriber) Transcribe(audio []byte, lang string) (string, error) {
	// Decode audio to 16 kHz mono float32 PCM entirely in memory.
	samples, err := toFloat32PCM(audio)
	if err != nil {
		return "", err
	}

	// Acquire a context from the pool.
	ctx := <-t.ctxPool
	defer func() { t.ctxPool <- ctx }()

	// Per-call configuration.
	if lang == "" {
		lang = "auto"
	}
	if err := ctx.SetLanguage(lang); err != nil {
		return "", fmt.Errorf("set language %q: %w", lang, err)
	}
	if lang == "zh" {
		ctx.SetInitialPrompt("以下是普通话的句子，使用简体中文。")
	} else {
		ctx.SetInitialPrompt("")
	}

	var sb strings.Builder
	if err := ctx.Process(samples, nil, func(seg whisper.Segment) {
		sb.WriteString(seg.Text)
	}, nil); err != nil {
		return "", fmt.Errorf("process audio: %w", err)
	}

	return strings.TrimSpace(sb.String()), nil
}

// Close releases the model and all pooled inference contexts.
func (t *Transcriber) Close() error {
	for i := 0; i < cap(t.ctxPool); i++ {
		<-t.ctxPool
	}
	return t.model.Close()
}

// toFloat32PCM decodes audio to 16 kHz mono float32 PCM via ffmpeg.
// Uses stdin/stdout pipes — no temporary files are written to disk.
// ffmpeg outputs raw f32le, skipping WAV encoding/decoding entirely.
func toFloat32PCM(audio []byte) ([]float32, error) {
	cmd := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "error",
		"-i", "pipe:0", // read from stdin
		"-ar", "16000", // 16 kHz
		"-ac", "1",     // mono
		"-f", "f32le",  // raw float32 little-endian, no container overhead
		"pipe:1",       // write to stdout
	)
	cmd.Stdin = bytes.NewReader(audio)

	var out bytes.Buffer
	cmd.Stdout = &out

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg decode: %w", err)
	}

	data := out.Bytes()
	if len(data) == 0 {
		return nil, fmt.Errorf("ffmpeg produced no audio output")
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("unexpected output size: %d bytes", len(data))
	}

	samples := make([]float32, len(data)/4)
	for i := range samples {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		samples[i] = math.Float32frombits(bits)
	}
	return samples, nil
}

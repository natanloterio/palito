// Package palito lets you run GGUF LLMs locally from Go with custom CUDA
// kernels, token streaming, and structured JSON output — no llama.cpp required.
//
//	engine, err := palito.Open("model.gguf", nil)
//	if err != nil { ... }
//	defer engine.Close()
//	reply, err := engine.Chat(ctx, []palito.Message{{Role: "user", Content: "Hello!"}},
//		&palito.Generation{OnText: func(text string) bool { fmt.Print(text); return true }})
//
// libpalito is loaded at run time from $PALITO_LIBRARY, then the installed
// `palito` command (<bundle>/lib/libpalito.so), then the system loader, so
// building needs no link flags (cgo and a C compiler are still required).
package palito

// #cgo linux LDFLAGS: -ldl
// #include <stdlib.h>
// #include "shim.h"
import "C"

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/cgo"
	"sync"
	"time"
	"unsafe"
)

// Device selects where the model runs.
type Device int

const (
	// DeviceDefault is CUDA when the library has it, else CPU.
	DeviceDefault Device = iota
	DeviceCPU
	DeviceCUDA
)

// Options configures Open; the zero value uses Palito's defaults.
type Options struct {
	Device Device
	// 0: the model's trained context, fitted to GPU memory.
	ContextTokens uint32
	Threads       uint32
	GPUDevice     uint32
	// 0: free GPU memory minus 2 GiB.
	GPUMemoryLimitBytes uint64
	KernelCacheDir      string
}

// Message is one chat turn.
type Message struct {
	Role    string // "system", "user" or "assistant"
	Content string
}

// Generation configures one request; the zero value (or nil) is greedy,
// reasoning on, no token cap and no stop sequences.
type Generation struct {
	MaxTokens  uint32 // 0: until the model stops or the context is full
	NoThinking bool   // turn off Qwen3-style reasoning
	// 0: greedy.
	Temperature float32
	TopP        float32 // 0 or 1: off
	TopK        uint32  // 0: off
	MinP        float32 // in [0, 1]; 0: off
	// llama.cpp repetition penalties over the last RepeatLastN tokens
	// (prompt included). RepeatPenalty 0 or 1: off; RepeatLastN 0: 64.
	RepeatPenalty    float32
	RepeatLastN      uint32
	PresencePenalty  float32 // 0: off
	FrequencyPenalty float32 // 0: off
	// Token id -> bias added to its logit; math.Inf(-1) bans the token.
	LogitBias map[int32]float32
	// nil: a different seed each call.
	Seed *uint64
	// Generation ends at the first of these; it is not part of the text.
	Stop []string
	// OnText receives text as it is generated; returning false stops it.
	OnText func(text string) bool
}

// Completion is the result of Chat or Complete.
type Completion struct {
	Text string
	// "stop", "length", "context_full" or "cancelled".
	FinishReason    string
	PromptTokens    int
	GeneratedTokens int
	TimeToFirst     time.Duration
	TokensPerSecond float64
}

// Error is an error reported by the Palito library.
type Error struct{ Message string }

func (e *Error) Error() string { return "palito: " + e.Message }

// ErrClosed is returned when using an engine after Close.
var ErrClosed = errors.New("palito: engine is closed")

var finishReasons = [...]string{"stop", "length", "context_full", "cancelled"}

// LibraryPath reports where libpalito is loaded from.
func LibraryPath() string {
	if path := os.Getenv("PALITO_LIBRARY"); path != "" {
		return path
	}
	name := libraryName()
	if command, err := exec.LookPath("palito"); err == nil {
		if real, err := filepath.EvalSymlinks(command); err == nil {
			bundled := filepath.Join(filepath.Dir(real), "lib", name)
			if info, err := os.Stat(bundled); err == nil && !info.IsDir() {
				return bundled
			}
		}
	}
	return name
}

func libraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libpalito.dylib"
	case "windows":
		return "palito.dll"
	}
	return "libpalito.so"
}

func load() error {
	path := C.CString(LibraryPath())
	defer C.free(unsafe.Pointer(path))
	if failure := C.palito_go_load(path); failure != nil {
		return &Error{fmt.Sprintf("cannot load %s: %s (install Palito or set PALITO_LIBRARY)",
			LibraryPath(), C.GoString(failure))}
	}
	return nil
}

// Version returns the library version.
func Version() (string, error) {
	if err := load(); err != nil {
		return "", err
	}
	return C.GoString(C.palito_go_version()), nil
}

// Engine is an opened model. Requests run one at a time; concurrent calls wait.
type Engine struct {
	mu     sync.Mutex
	handle *C.palito_engine
}

// Open loads a GGUF model; options may be nil.
func Open(modelPath string, options *Options) (*Engine, error) {
	if err := load(); err != nil {
		return nil, err
	}
	if options == nil {
		options = &Options{}
	}
	var raw C.palito_options
	C.palito_go_options_init(&raw)
	raw.device = C.int(options.Device)
	raw.context_tokens = C.uint32_t(options.ContextTokens)
	raw.threads = C.uint32_t(options.Threads)
	raw.gpu_device = C.uint32_t(options.GPUDevice)
	raw.gpu_memory_limit_bytes = C.uint64_t(options.GPUMemoryLimitBytes)
	if options.KernelCacheDir != "" {
		dir := C.CString(options.KernelCacheDir)
		defer C.free(unsafe.Pointer(dir))
		raw.kernel_cache_dir = dir
	}
	path := C.CString(modelPath)
	defer C.free(unsafe.Pointer(path))
	var failure *C.char
	handle := C.palito_go_open(path, &raw, &failure)
	if handle == nil {
		return nil, takeError(failure)
	}
	engine := &Engine{handle: handle}
	runtime.SetFinalizer(engine, (*Engine).Close)
	return engine, nil
}

// ContextTokens is the context actually opened.
func (e *Engine) ContextTokens() uint32 {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return 0
	}
	return uint32(C.palito_go_context_tokens(e.handle))
}

// Profile is "cpu", "int8-fast", "f32-tiled" or "f32".
func (e *Engine) Profile() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return ""
	}
	return C.GoString(C.palito_go_profile(e.handle))
}

// Chat replies to messages using the model's chat template. Cancelling ctx
// stops generation (the result then has FinishReason "cancelled").
func (e *Engine) Chat(ctx context.Context, messages []Message, generation *Generation) (Completion, error) {
	count := len(messages)
	size := C.size_t(unsafe.Sizeof(C.palito_message{}))
	array := (*C.palito_message)(C.calloc(C.size_t(max(count, 1)), size))
	defer C.free(unsafe.Pointer(array))
	rows := unsafe.Slice(array, max(count, 1))
	var strings []*C.char
	defer func() {
		for _, s := range strings {
			C.free(unsafe.Pointer(s))
		}
	}()
	for i, message := range messages {
		role, content := C.CString(message.Role), C.CString(message.Content)
		strings = append(strings, role, content)
		rows[i].role, rows[i].content = role, content
	}
	return e.run(ctx, generation, func(gen *C.palito_generation, handle C.uintptr_t,
		result *C.palito_result, failure **C.char) C.int {
		return C.palito_go_chat(e.handle, array, C.size_t(count), gen, handle, result, failure)
	})
}

// Complete continues prompt without a chat template.
func (e *Engine) Complete(ctx context.Context, prompt string, generation *Generation) (Completion, error) {
	text := C.CString(prompt)
	defer C.free(unsafe.Pointer(text))
	return e.run(ctx, generation, func(gen *C.palito_generation, handle C.uintptr_t,
		result *C.palito_result, failure **C.char) C.int {
		return C.palito_go_complete(e.handle, text, gen, handle, result, failure)
	})
}

type call func(*C.palito_generation, C.uintptr_t, *C.palito_result, **C.char) C.int

func (e *Engine) run(ctx context.Context, generation *Generation, invoke call) (Completion, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if generation == nil {
		generation = &Generation{}
	}
	var raw C.palito_generation
	C.palito_go_generation_init(&raw)
	raw.max_tokens = C.uint32_t(generation.MaxTokens)
	if generation.NoThinking {
		raw.enable_thinking = 0
	}
	raw.temperature = C.float(generation.Temperature)
	if generation.TopP != 0 {
		raw.top_p = C.float(generation.TopP)
	}
	raw.top_k = C.uint32_t(generation.TopK)
	raw.min_p = C.float(generation.MinP)
	if generation.RepeatPenalty != 0 {
		raw.repeat_penalty = C.float(generation.RepeatPenalty)
	}
	if generation.RepeatLastN != 0 {
		raw.repeat_last_n = C.uint32_t(generation.RepeatLastN)
	}
	raw.presence_penalty = C.float(generation.PresencePenalty)
	raw.frequency_penalty = C.float(generation.FrequencyPenalty)
	if n := len(generation.LogitBias); n > 0 {
		biases := (*C.palito_logit_bias)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof(C.palito_logit_bias{}))))
		defer C.free(unsafe.Pointer(biases))
		items := unsafe.Slice(biases, n)
		i := 0
		for token, bias := range generation.LogitBias {
			items[i] = C.palito_logit_bias{token: C.int32_t(token), bias: C.float(bias)}
			i++
		}
		raw.logit_bias, raw.logit_bias_count = biases, C.size_t(n)
	}
	if generation.Seed != nil {
		raw.seed = C.int64_t(*generation.Seed & (1<<63 - 1))
	}
	if n := len(generation.Stop); n > 0 {
		stops := (**C.char)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof((*C.char)(nil)))))
		defer C.free(unsafe.Pointer(stops))
		items := unsafe.Slice(stops, n)
		for i, stop := range generation.Stop {
			items[i] = C.CString(stop)
			defer C.free(unsafe.Pointer(items[i]))
		}
		raw.stop, raw.stop_count = stops, C.size_t(n)
	}
	var panicked any
	sink := textSink(func(text string) (keepGoing bool) {
		if ctx.Err() != nil {
			return false
		}
		if generation.OnText == nil {
			return true
		}
		defer func() {
			if p := recover(); p != nil {
				panicked, keepGoing = p, false
			}
		}()
		return generation.OnText(text)
	})
	handle := cgo.NewHandle(sink)
	defer handle.Delete()

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return Completion{}, ErrClosed
	}
	var result C.palito_result
	var failure *C.char
	if invoke(&raw, C.uintptr_t(handle), &result, &failure) != 0 {
		return Completion{}, takeError(failure)
	}
	defer C.palito_go_result_free(&result)
	if panicked != nil {
		panic(panicked)
	}
	completion := Completion{
		Text:            C.GoString(result.text),
		FinishReason:    finishReasons[result.finish_reason],
		PromptTokens:    int(result.prompt_tokens),
		GeneratedTokens: int(result.generated_tokens),
		TimeToFirst:     time.Duration(float64(result.ttft_ms) * float64(time.Millisecond)),
		TokensPerSecond: float64(result.tokens_per_second),
	}
	if completion.FinishReason == "cancelled" && ctx.Err() != nil {
		return completion, ctx.Err()
	}
	return completion, nil
}

// Close releases the model; it waits for a running request.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != nil {
		C.palito_go_close(e.handle)
		e.handle = nil
	}
	return nil
}

func takeError(failure *C.char) error {
	if failure == nil {
		return &Error{"unknown error"}
	}
	defer C.palito_go_string_free(failure)
	return &Error{C.GoString(failure)}
}

// Seed returns a pointer for Generation.Seed.
func Seed(seed uint64) *uint64 { return &seed }

package palito

// Run: PALITO_TEST_MODEL=$PWD/../../models/SmolLM2-135M-Instruct-Q4_K_M.gguf go test ./...

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

var hello = []Message{{Role: "user", Content: "Say hello."}}

func openTest(t *testing.T) *Engine {
	model := os.Getenv("PALITO_TEST_MODEL")
	if model == "" {
		t.Skip("set PALITO_TEST_MODEL to a small GGUF model")
	}
	engine, err := Open(model, &Options{Device: DeviceCPU, ContextTokens: 512})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { engine.Close() })
	return engine
}

func TestHeaderMatchesTheLibrary(t *testing.T) {
	upstream, err := os.ReadFile("../../include/palito.h")
	if err != nil {
		t.Skip("not inside the Palito repository")
	}
	local, _ := os.ReadFile("palito.h")
	if !bytes.Equal(upstream, local) {
		t.Fatal("bindings/go/palito.h differs from include/palito.h; copy it")
	}
}

func TestStreamingStopSamplingCancel(t *testing.T) {
	engine := openTest(t)
	ctx := context.Background()
	var streamed strings.Builder
	reply, err := engine.Chat(ctx, hello, &Generation{MaxTokens: 24,
		OnText: func(text string) bool { streamed.WriteString(text); return true }})
	if err != nil || streamed.String() != reply.Text || reply.GeneratedTokens == 0 {
		t.Fatalf("reply %+v, streamed %q, err %v", reply, streamed.String(), err)
	}
	cut, err := engine.Complete(ctx, "1, 2, 3, 4,", &Generation{MaxTokens: 24, Stop: []string{" 7"}})
	if err != nil || cut.FinishReason != "stop" || strings.Contains(cut.Text, " 7") {
		t.Fatalf("stop: %+v %v", cut, err)
	}
	sampled := &Generation{MaxTokens: 16, Temperature: 0.9, Seed: Seed(4)}
	a, _ := engine.Chat(ctx, hello, sampled)
	b, _ := engine.Chat(ctx, hello, sampled)
	if a.Text != b.Text {
		t.Fatalf("seeded sampling differs: %q vs %q", a.Text, b.Text)
	}
	ones := "1, 1, 1, 1, 1, 1,"
	plain, _ := engine.Complete(ctx, ones, &Generation{MaxTokens: 4})
	penalized, err := engine.Complete(ctx, ones, &Generation{MaxTokens: 4, FrequencyPenalty: 2,
		PresencePenalty: 0.5, RepeatPenalty: 1.1})
	if err != nil || !strings.HasPrefix(plain.Text, " 1") || strings.HasPrefix(penalized.Text, " 1") {
		t.Fatalf("penalties: %q then %q, %v", plain.Text, penalized.Text, err)
	}
	minP := &Generation{MaxTokens: 8, Temperature: 1, MinP: 0.05, Seed: Seed(3)}
	c, _ := engine.Complete(ctx, ones, minP)
	d, _ := engine.Complete(ctx, ones, minP)
	if c.Text != d.Text {
		t.Fatalf("seeded min-p differs: %q vs %q", c.Text, d.Text)
	}
	if _, err := engine.Complete(ctx, ones, &Generation{MaxTokens: 2,
		LogitBias: map[int32]float32{1 << 30: 1}}); err == nil {
		t.Fatal("out-of-vocabulary logit bias accepted")
	}
	cancelled, cancel := context.WithCancel(ctx)
	got, err := engine.Complete(cancelled, "Once upon a time", &Generation{
		OnText: func(string) bool { cancel(); return true }})
	if !errors.Is(err, context.Canceled) || got.FinishReason != "cancelled" {
		t.Fatalf("cancel: %+v %v", got, err)
	}
	if _, err := engine.Chat(ctx, hello, &Generation{Temperature: -1}); err == nil {
		t.Fatal("negative temperature accepted")
	}
	if _, err := Open("/nonexistent.gguf", nil); err == nil {
		t.Fatal("missing model opened")
	}
	engine.Close()
	if _, err := engine.Complete(ctx, "Hi", nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed engine: %v", err)
	}
}

// usage: go run ./example model.gguf "message"
package main

import (
	"context"
	"fmt"
	"os"

	palito "github.com/natanloterio/palito/bindings/go"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, `usage: go run ./example model.gguf "message"`)
		os.Exit(2)
	}
	engine, err := palito.Open(os.Args[1], nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer engine.Close()
	version, _ := palito.Version()
	fmt.Fprintf(os.Stderr, "palito %s, %s, context %d\n", version, engine.Profile(), engine.ContextTokens())
	reply, err := engine.Chat(context.Background(),
		[]palito.Message{{Role: "user", Content: os.Args[2]}},
		&palito.Generation{NoThinking: true, OnText: func(text string) bool {
			fmt.Print(text)
			return true
		}})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "\n[%d tokens, %.1f tok/s, %s]\n", reply.GeneratedTokens, reply.TokensPerSecond, reply.FinishReason)
}

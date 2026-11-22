# palito (Go)

Go binding for [Palito](../../README.md). It loads `libpalito.so` at run time, so building needs cgo and a C compiler but no link flags.

```sh
go get github.com/natanloterio/palito/bindings/go
```

```go
engine, err := palito.Open("model.gguf", nil)
if err != nil {
	log.Fatal(err)
}
defer engine.Close()
reply, err := engine.Chat(ctx, []palito.Message{{Role: "user", Content: "Hello!"}},
	&palito.Generation{Temperature: 0.7, Stop: []string{"\n\n"},
		OnText: func(text string) bool { fmt.Print(text); return true }})
```

- **Cancellation:** cancelling `ctx` stops generation, and the call then returns `ctx.Err()`.
- **Concurrency:** calls on one engine wait their turn.
- **Header:** `palito.h` is a copy of `include/palito.h`, and a test keeps the two identical.

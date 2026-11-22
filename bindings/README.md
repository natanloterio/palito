# Palito language bindings

Each binding wraps the C API (`include/palito.h`, `libpalito.so`) in the same
process. There is no server and no copying of the model. They all have the same shape:

| | Python | Node.js | Java 22+ | C# (.NET 8+) | Go |
|---|---|---|---|---|---|
| Open | `Palito(path)` | `await Palito.open(path)` | `Palito.open(path, options)` | `PalitoEngine.Open(path)` | `palito.Open(path, nil)` |
| Chat | `chat(messages)` | `await chat(messages)` | `chat(messages, gen)` | `Chat` / `ChatAsync` | `Chat(ctx, messages, gen)` |
| Stream | `on_text=` or `stream_chat()` | `onText` or `for await` `streamChat()` | `onText(...)` | `OnText` or `await foreach` `StreamChatAsync` | `OnText` |
| Stop early | return `False` / close the generator | return `false` / `break` | return `false` | return `false` / cancel the token | return `false` / cancel `ctx` |
| Native calls via | ctypes (stdlib) | [koffi](https://koffi.dev) | FFM (`java.lang.foreign`) | P/Invoke | cgo + `dlopen` |

All of them accept the same generation settings:
- `max_tokens`
- `thinking`
- `temperature` (0, the default, is greedy)
- `top_p`, `top_k` and `seed`
- `stop` sequences

They also report the same result: text, finish reason, token counts, time to the first token, and tok/s.

**Finding the library.** The bindings look in three places, in order:
1. `$PALITO_LIBRARY`.
2. The installed `palito` command, which points to `<bundle>/lib/libpalito.so`.
3. The system loader.

`libpalito.so` finds the NVRTC bundled beside it on its own, so `LD_LIBRARY_PATH` is not needed.

**Threads.** One request runs at a time per engine, and calls from other threads wait their turn. Open several engines to run requests in parallel (each holds its own copy of the model).

**Tests.** Each binding has tests that run against a small model:

```sh
export PALITO_TEST_MODEL=$PWD/models/SmolLM2-135M-Instruct-Q4_K_M.gguf
(cd bindings/python && python3 tests/test_palito.py)
(cd bindings/node && npm install && npm test)
(cd bindings/java && mvn test)
(cd bindings/csharp && dotnet test)
(cd bindings/go && go test ./...)
```

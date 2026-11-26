# palito (Python)

Python binding for [Palito](../../README.md). It uses only the standard library (ctypes).

```sh
pip install palito                # from PyPI
pip install ./bindings/python     # or from a clone of the repository
```

It needs the native `libpalito` library: install the [Linux bundle](https://github.com/natanloterio/palito#download) or point `PALITO_LIBRARY` at your own build (see [bindings](https://github.com/natanloterio/palito/blob/master/bindings/README.md)).

```python
import palito

with palito.Palito("model.gguf") as llm:
    reply = llm.chat([{"role": "user", "content": "Hello!"}],
                     on_text=lambda text: print(text, end="", flush=True))
    print(reply.finish_reason, reply.tokens_per_second)

    for text in llm.stream_chat([{"role": "user", "content": "A haiku about tea"}],
                                temperature=0.7, top_p=0.9, stop=["\n\n"]):
        print(text, end="", flush=True)
```

- **Generation keywords:** `max_tokens`, `thinking`, `temperature` (0: greedy), `top_p`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `logit_bias` (`{token id: bias}`; `float("-inf")` bans), `seed`, `stop` and `json_schema` (a dict or JSON string; the reply is JSON matching it, with reasoning off; `{"type": "object"}` for any object).
- **Results:** a `Completion` dataclass.
- **Errors:** raise `palito.PalitoError`.

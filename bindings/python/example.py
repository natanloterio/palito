"""usage: python3 example.py model.gguf "message" """
import sys

import palito

if len(sys.argv) != 3:
    sys.exit(__doc__)
with palito.Palito(sys.argv[1]) as llm:
    print(f"palito {palito.version()}, {llm.profile}, context {llm.context_tokens}",
          file=sys.stderr)
    reply = llm.chat([{"role": "user", "content": sys.argv[2]}], thinking=False,
                     on_text=lambda text: print(text, end="", flush=True))
    print(f"\n[{reply.generated_tokens} tokens, {reply.tokens_per_second:.1f} tok/s,"
          f" {reply.finish_reason}]", file=sys.stderr)

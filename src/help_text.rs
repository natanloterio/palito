//! `Examples:` footers for `--help`.

pub(crate) const ROOT_EXAMPLES: &str = "\
Examples:
  palito chat     --model qwen3-8b.palito
  palito serve    --model qwen3-8b.palito --port 8080
  palito generate --model model.gguf --prompt 'Once upon a time' --stream
  palito convert  -i qwen3-8b-Q8_0.gguf -o qwen3-8b.palito";

pub(crate) const CHAT_EXAMPLES: &str = "\
Examples:
  palito chat --model qwen3-8b.palito
  palito chat --model qwen3-8b.palito --system 'Be brief' --temperature 0.7
  echo 'What is 2+2?' | palito chat --model model.gguf --no-stream";

pub(crate) const SERVE_EXAMPLES: &str = "\
Examples:
  palito serve --model qwen3-8b.palito
  palito serve --model qwen3-8b.palito --host 0.0.0.0 --api-key \"$KEY\" -np 4
  curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}'";

pub(crate) const CONVERT_EXAMPLES: &str = "\
Examples:
  palito convert -i qwen3-8b-Q8_0.gguf -o qwen3-8b.palito
  palito convert -i qwen3-8b-Q8_0.gguf -o qwen3-8b.palito --bits 2 --force";

pub(crate) const GENERATE_EXAMPLES: &str = "\
Examples:
  palito generate --model model.gguf --prompt 'Once upon a time' --stream
  palito generate --model model.gguf --prompt-file prompt.txt --tokens 128 --json
  cat prompt.txt | palito generate --model model.gguf --prompt-file - --stream";

pub(crate) const TOKENIZE_EXAMPLES: &str = "\
Examples:
  palito tokenize --model model.gguf --prompt 'Hello world'
  echo 'Hello world' | palito tokenize --model model.gguf --prompt-file -";

pub(crate) const DETOKENIZE_EXAMPLES: &str = "\
Examples:
  palito detokenize --model model.gguf --tokens 9906,1917
  echo '[9906,1917]' | palito detokenize --model model.gguf --input-tokens-file -";

pub(crate) const BENCH_EXAMPLES: &str = "\
Examples:
  palito bench --model qwen3-8b.palito --prompt 'Hello' --tokens 128
  palito bench --model model.gguf --backend cpu --prompt-file p.txt --repetitions 5";

pub(crate) const LOGITS_EXAMPLES: &str = "\
Examples:
  palito logits --model model.gguf --prompt 'The capital of France is'
  palito logits --model model.gguf --input-tokens-file ids.json";

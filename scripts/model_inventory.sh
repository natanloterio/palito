#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    printf 'Usage: bash %s <llama.cpp-checkout>\n' "$0" >&2
    exit 2
fi

revision=6b01a4767f8d3d0c416c410a49861333965c0982
git --no-pager -C "$1" show "$revision:src/llama-arch.cpp" |
    awk '
        BEGIN { print "architecture\tdefault_cpu\tcpu_reference\tcuda\tlimitation" }
        /LLM_ARCH_NAMES = \{/ { registry = 1; next }
        registry && /^};/ { registry = 0; closed = 1 }
        registry && /LLM_ARCH_[A-Z0-9_]+[[:space:]]*,[[:space:]]*"/ {
            split($0, parts, "\"")
            arch = parts[2]
            if (arch == "(unknown)") next
            ++rows
            if (arch == "llama")
                print arch "\tpartial\tpartial\tpartial\tSmolLM2 135M Q4_K_M, Q4_1, Q5_1 and IQ4_NL/Q3_K; Llama 3.2 1B Q8_0 on reference and CUDA f32 only"
            else if (arch == "qwen3")
                print arch "\tnot supported\tpartial\tpartial\tQwen3 0.6B F16, Q4_K_M, BF16 and pure Q2_K causal text; BF16/Q2_K CUDA f32 without graphs"
            else if (arch == "qwen3moe")
                print arch "\tnot supported\tpartial\tpartial\tSelected causal text fixtures validated"
            else if (arch == "gemma3")
                print arch "\tnot supported\tpartial\tpartial\tGemma 3 270M IT Q8_0 causal logits and greedy text validated"
            else if (arch == "gemma2")
                print arch "\tnot supported\tpartial\tpartial\tGemma 2 2B IT Q8_0 short-context causal text; CUDA f32 without graphs"
            else if (arch == "phi3")
                print arch "\tnot supported\tpartial\tpartial\tPhi-3 Mini 4K Instruct Q8_0 and Q4_K_M; Mini 128K Q4_K_M short/long LongRoPE selections with short inputs; CUDA f32 without graphs"
            else if (arch == "starcoder2")
                print arch "\tnot supported\tpartial\tpartial\tStarCoder2-3B Q4_K_M short causal code; CUDA f32 without graphs"
            else if (arch == "gpt2")
                print arch "\tnot supported\tpartial\tpartial\tGPT-2 small Q8_0 causal text incl. 1024-token boundary; CUDA f32 without graphs"
            else if (arch == "bloom")
                print arch "\tnot supported\tpartial\tpartial\tBloom-560M Q4_K_M causal ALiBi, short prompts; CUDA f32 without graphs"
            else if (arch == "gptneox")
                print arch "\tnot supported\tpartial\tpartial\tPythia-70M Q8_0 partial RoPE and parallel residual, short prompts; CUDA f32 without graphs"
            else if (arch == "olmo")
                print arch "\tnot supported\tpartial\tpartial\tOLMo-1B-hf Q4_K_M non-affine LayerNorm, no QKV clamp, short prompts; CUDA f32 without graphs"
            else if (arch == "olmo2")
                print arch "\tnot supported\tpartial\tpartial\tOLMo-2-0425-1B Q4_K_M full attention and post-RMSNorm, short prompts; CUDA f32 without graphs"
            else if (arch == "olmoe")
                print arch "\tnot supported\tpartial\tpartial\tOLMoE-1B-7B-0924-Instruct Q4_K_M full-expert softmax/top-8, short prompts; CUDA f32 without graphs"
            else if (arch == "qwen35")
                print arch "\tnot supported\tpartial\tpartial\tQwen3.5-0.8B Q4_K_M text-only hybrid, short prompts, 32-token configured boundary and recurrent reset/replay; CUDA f32 without graphs"
            else if (arch == "mamba")
                print arch "\tnot supported\tpartial\tnot supported\tMamba-130M-hf Q4_K_M selective scan without dt/B/C RMSNorm; short prompts and reset/replay on CPU reference"
            else if (arch == "qwen2")
                print arch "\tnot supported\tpartial\tpartial\tQwen2.5 0.5B Instruct Q8_0, Q4_K_M and Q4_0 causal text; CUDA f32 without graphs"
            else if (arch == "clip")
                print arch "\tnot supported\tnot supported\tnot supported\tAuxiliary GGUF entry; not a standalone language model"
            else
                print arch "\tnot supported\tnot supported\tnot supported\tNo native execution path"
        }
        END { if (!rows || !closed) exit 1 }
    '

# Relatório: desempenho de decode CUDA vs llama.cpp

Data: 2026-11-21 · Modelo: Qwen3-30B-A3B Q4_K_M (`qwen3moe`, 48 camadas, 128 experts, top-8) · GPU: RTX 5090 32 GB (sm_120, ~1,79 TB/s) · llama.cpp: revisão `6b01a476`.

Fontes: profiling do nosso motor com nsys, leitura do código CUDA do llama.cpp e pesquisa web (2025–2026). Onde um número é inferência ou não foi verificado, isso está marcado.

## 1. Resumo

- **A diferença está na geração de tokens: o llama.cpp é ~4× mais rápido** (~355 tok/s contra 80–98 tok/s, ou ~140 tok/s com `--cuda-matmul f32-tiled --cuda-decode-graph`).
- **O gargalo é a qualidade dos kernels, não o custo de lançamento nem o tráfego com o host.** A GPU fica ociosa só 3–5% do tempo, mas nossos kernels leem os pesos a ~300 GB/s (17% do pico). O llama.cpp chega a ~680 GB/s (38%).
- **O roteamento MoE já está certo:** os IDs dos experts ficam na GPU, sem ida e volta ao host, igual ao llama.cpp.
- **O maior ganho do llama.cpp vem de ativações em int8 (q8_1) com dp4a**, e isso esbarra na nossa política de qualidade "F32 estrito" (`runtime/results/llama-gap-investigation/analysis.json`). **Essa decisão precisa ser sua** (seção 6).
- Mesmo sem mudar a precisão, há bastante ganho disponível: kernels com um warp por linha, fusões e attention split-K. Pela estimativa do profiling (inferência), o decode chegaria a 2–3× o desempenho atual.

## 2. Medições

### 2.1 Comparação direta, mesmo prompt de chat, greedy, contexto 4096

| Cenário | Nosso motor (padrão) | llama.cpp | Diferença |
|---|---|---|---|
| Com raciocínio, turno 1 (~170 tokens) | 86,8 tok/s | ~357 tok/s | 4,1× |
| Com raciocínio, turno 2 (~200 tokens) | 79,3 tok/s | ~356 tok/s | 4,5× |
| `--no-thinking` (11 tokens) | ~98 tok/s | ~337 tok/s | 3,4× |
| Tempo até o 1º token (~25 tokens de prompt) | 92–109 ms | 12–15 ms | ~7× |

### 2.2 Modos do nosso motor (bench, 128 tokens; os tokens gerados foram idênticos nos 4 modos)

| Modo | Decode | TTFT |
|---|---|---|
| padrão (`f32`) | ~92 tok/s | 50 ms |
| `--cuda-matmul f32-tiled` | ~135 tok/s | 61 ms |
| `--cuda-decode-graph` | ~94 tok/s | 50 ms |
| tiled + graph | **~140 tok/s** | 61 ms |

O README registra 165–170 tok/s para tiled + graph, num harness diferente (prompt e tamanho próprios). A diferença não foi investigada. As hipóteses são o custo de attention crescer com o contexto e a GPU estar sendo usada pelo desktop.

### 2.3 Profiling por token (nsys, tiled + graph)

- 965 kernels por token, 6,54 ms de parede, 6,32 ms com a GPU ocupada.
- Os pesos lidos por token somam ~1,92 GB: experts 1097 MB, Q/K/V/O 516 MB, logits 255 MB e router 50 MB. Com o pico de banda, o teto ficaria em ~1,07 ms por token (~930 tok/s).

| Grupo | Fração do tempo |
|---|---|
| matvecs dos experts (gate/up/down) | 42% |
| projeções Q/K/V/O | 18% |
| attention | 10% |
| logits + router | 8% |
| RMSNorm (97 lançamentos de 1 bloco) | 7,5% |
| router top-k | 5,7% |
| outros kernels pequenos | ~7% |
| GPU ociosa | 3% |

| Kernel | µs cada | ms/passo | Banda |
|---|---|---|---|
| expert down (largura 768) | 28,4 | 1,36 | 306 GB/s |
| expert gate + up | 14,1 | 1,36 | 502 GB/s |
| `attention_fused_decode` (32 blocos) | 13,9 | 0,67 | — |
| `normalize` (1 bloco) | 5,1 | 0,49 | — |
| projeção Q | 10,2 | 0,49 | 463 GB/s |
| router top-k (1 bloco) | 7,7 | 0,37 | — |
| logits (151 936 linhas) | 296 | 0,30 | 862 GB/s |

## 3. Diagnóstico do nosso motor (com referências)

1. **O down dos experts deixa 5 dos 8 warps ociosos.** O loop em `runtime/src/cuda/kernels.cu:267` avança de 2048 em 2048, mas a largura 768 tem só 3 superblocos. Depois vem a redução do bloco inteiro de 256 threads (`kernels.cu:145`).
2. **Todo matvec usa um bloco de 256 threads por linha de saída**, com redução no nível do bloco (`kernels.cu:206-245`, despacho em `cuda.rs:1306-1330` e `1640-1700`). No modo padrão, cada elemento ainda passa pela função escalar `weight()` (`kernels.cu:28-104`), que decodifica a escala e o mínimo do bloco Q4_K de novo a cada elemento.
3. **O NVRTC compila com `fmad: Some(false)`** (`cuda.rs:878`), o que quase dobra as instruções de ponto flutuante. Está assim para manter a paridade bit a bit.
4. **A attention usa só 32 blocos, um por head** (`cuda/attention.cu:3-60`), e cresce linearmente com o contexto. O caminho com CUDA graph tem limite de 1024 tokens, porque o kernel guarda os scores num `scores[1024]` em memória compartilhada.
5. **Cada RMSNorm é um único bloco**, lançado 97 vezes por token (`kernels.cu:501`).
6. **O router top-k roda num único bloco**, em 8 passadas com barreiras (`kernels.cu:760-819`). A matriz do router está em F32.
7. **Sem CUDA graph, cada token copia ~608 KB de logits para o host**, sincroniza e faz argmax na CPU (`cuda.rs:1796-1810`). O modo com graph já faz o argmax na GPU.
8. **Já existem versões de pesquisa** atrás de features de compilação: `router_warp128`, `qkv_head_norm_rope_cache` e um gate+up fundido (`cuda.rs:1270`, `1360`, `1601`, `1631`). Elas não estão ativadas por padrão.

## 4. O que o llama.cpp faz (e nós não)

| # | Técnica | Onde (llama.cpp) | Precisão |
|---|---|---|---|
| 1 | GEMV quantizado `mul_mat_vec_q`: 128 threads por linha, leitura direta dos blocos empacotados, escalas decodificadas sem branches, 4 linhas por bloco quando K < 2048 | `mmvq.cu:618`, `:1137`; `vecdotq.cuh:973` | **muda** (q8_1 + dp4a) |
| 2 | Ativação quantizada para q8_1 antes de cada matmul | `quantize.cu:54`, `mmvq.cu:1548` | **muda** |
| 3 | gate + up + SwiGLU num único kernel, compartilhando a leitura da ativação | `mmvq.cu:680-860`, `ggml-cuda.cu:4871` | igual |
| 4 | Router fundido: softmax, top-8 e renormalização num warp por token | `topk-moe.cu:156` | igual |
| 5 | Soma ponderada dos 8 experts num único kernel (16 nós do grafo) | `moe-weighted-reduction.cu:3` | igual |
| 6 | Fusões rms_norm+mul(+add), Q/K-norm+rope+escrita no KV e matmul+residual | `ggml-cuda.cu:5106-5131`, `3391`, `5060` | igual |
| 7 | CUDA graph do passo inteiro, estável porque o KV é preenchido em múltiplos de 256 e posições, slots e máscara são tensores de entrada | `ggml-cuda.cu:5418`, `3244` | igual |
| 8 | PDL (programmatic dependent launch): o prólogo de um kernel se sobrepõe ao fim do anterior (sm_90+) | `common.cuh:1525-1575`, `mmvq.cu:657` | igual |
| 9 | Flash-attention vetorial no decode, com split-K dimensionado para ocupar as ondas de SMs, mais uma etapa de combinação | `fattn-vec.cuh:21`, `fattn-common.cuh:1201` | KV em f16 (**muda**) |
| 10 | Prefill com MMA int8 (`mma.sync`) e ordenação dos experts na GPU | `mmq.cu:448`, `mmid.cu:35` | **muda** |

O llama.cpp também deixa espaço para ganhar dele. Q, K e V quantizam a mesma entrada três vezes. E o servidor copia os logits inteiros e faz greedy na CPU por padrão, mesmo tendo um ARGMAX na GPU.

## 5. Técnicas novas (pesquisa web)

A revisão `6b01a476` já tem a maior parte do estado da arte: EAGLE-3, MTP, DFlash, n-gram, NVFP4/MXFP4, PDL, CUDA graphs, KV quantizado e FP8. **Não tem megakernels persistentes nem suffix decoding.** Segundo a discussão #17621, os ~352 tok/s do llama.cpp vieram de 247 tok/s depois das fusões e de `GGML_CUDA_GRAPH_OPT`. Isso equivale a ~40% do teto de banda, e a pesquisa atribui a sobra aos GEMVs pequenos e ao custo entre kernels.

| Técnica | Ganho reportado | No llama.cpp? | Esforço para nós |
|---|---|---|---|
| Kernel persistente por bloco MoE (estilo MonoMoE, arXiv 2609.04244) | 1,17–1,54× no bloco MoE, batch 1–8, H200 | não | médio |
| Megakernel do modelo inteiro (Hazy "No Bubbles", MPK/OSDI'26, Cohere) | 1,5–1,7× em outras GPUs e modelos (parte não verificada); há armadilhas de corretude documentadas (arXiv 2611.00495) | não | alto |
| Decodificação especulativa: n-gram/suffix, depois EAGLE-3 | ~1,37× com EAGLE-3 no 30B-A3B (batch não informado); no MoE o ganho é menor, porque verificar k tokens ativa mais experts | sim, exceto suffix | baixo (n-gram) a médio |
| Contexto longo: flash-decoding split-K + KV em FP8/q8 | de 234 para 111 tok/s entre 4K e 32K no llama.cpp; acima de ~8K o KV pesa mais que os pesos | parcial | médio |
| NVFP4 | ~8% a menos de bytes que Q4_K_M; o ganho de +87% citado na web é implausível | sim | médio, baixa prioridade |
| Pular canais ou neurônios do MoE sem treino | 1,6–1,9× no bloco MoE, com perda de qualidade | não | médio, com perdas |

## 6. Plano proposto

Cada fase só termina com o mesmo gate de validação que já usamos: logits e IDs gerados contra a referência, mais tok/s medido no `bench` e no `chat`.

**Fase 0. Ganhos imediatos, sem mudar a precisão**
- Ativar tiled + graph por padrão no `chat`. Ganho conhecido: ~92 → ~140 tok/s.
- Fazer o argmax na GPU também fora do modo graph e eliminar o limite de 1024 tokens do modo graph.

**Fase 1. Kernels melhores, mantendo F32 estrito**
1. Matvec com um warp por linha e 4–8 linhas por bloco, loads de 16 bytes, escalas decodificadas uma vez por sub-bloco e redução só com shuffles. Isso corrige os itens 3.1 e 3.2 e é o maior ganho desta fase.
2. Fundir gate + up + SwiGLU e Q + K + V, reaproveitando as versões de pesquisa.
3. Router top-k num warp, fundido com softmax e pesos, e a soma ponderada dos experts fundida com o residual.
4. RMSNorm com redução por warp, fundida com o residual e com a cadeia head_norm → rope → escrita no KV.
5. Attention split-K com etapa de combinação, sem o limite de 1024.

Estimativa (inferência): ~2–4 ms por token, ou seja, 250–450 tok/s, dependendo de quanto banda os kernels F32 conseguirem usar.

**Fase 2. Exige mudar a política de precisão (decisão sua)**
- `fmad` ligado.
- Ativações q8_1 com dp4a, como o llama.cpp. Quantizar uma vez só para Q, K e V.
- KV em f16.
- Prefill com MMA int8. É o que explica os 12 ms de TTFT do llama.cpp.

**Fase 3. Além do llama.cpp**
- Kernel persistente por bloco MoE.
- Decodificação especulativa n-gram/suffix com verificação que leve o MoE em conta.
- KV em FP8 para contexto longo.

## 7. Decisão necessária

A política atual mantém tudo em F32 estrito: `fmad` desligado, ativações e KV em F32 e validação bit a bit. Ela bloqueia a Fase 2, que é o que mais aproxima a gente do llama.cpp. As opções são:

1. Manter F32 estrito e fazer só as Fases 0–1. O teto fica abaixo do llama.cpp.
2. Criar um perfil "rápido" opcional (`--cuda-matmul q8`, por exemplo), validado por tolerância de logits e concordância de tokens, como o próprio llama.cpp é validado. O perfil F32 estrito continua como padrão.
3. Tornar o perfil rápido o padrão.

## Artefatos

- Profiling: `nsys-rep`, `sqlite`, CSVs e o script `an.py` no scratchpad da sessão (`/tmp/claude-1000/-media-loterio-workspace-workspace-gaia-local-inference-rust/721ecc6f-f284-4ab0-ac83-caf255203e06/scratchpad/perfil/`). Esse diretório é temporário.
- Benchmark do llama.cpp: `llama_chat.py`, no mesmo scratchpad.

## 8. Progresso da implementação (2026-11-21)

Todas as mudanças estão no perfil opt-in `--cuda-matmul f32-tiled`. O perfil padrão `f32` não mudou: ~90 tok/s, com os mesmos IDs gerados.

| # | Mudança | Arquivos | Efeito medido |
|---|---|---|---|
| 1 | `matvec_packed_rows`: 1 a 8 warps por linha, com a divisão escolhida pela ocupação da GPU | `kernels.cu`, `cuda.rs` (`matvec_grid`) | down 28→10 µs, gate/up 14→10 µs, logits 296→157 µs |
| 2 | Attention split-K (64 chaves por split) com etapa de combinação. O modo graph passa a aceitar qualquer contexto no `f32-tiled` | `attention.cu` | graph funciona com contexto 4096 |
| 3 | `router_warp128` promovido. É o mesmo algoritmo do `router` | `kernels.cu` | 7,6→3,2 µs |
| 4 | `normalize_warp` com 1024 threads | `kernels.cu` | 5,0→2,2 µs |
| 5 | gate + up + SwiGLU fundidos | `kernels.cu` | −1 kernel por camada |
| 6 | head_norm + RoPE + escrita no KV fundidos (só Qwen3: dim 128, RoPE completo) | `kernels.cu` | −4 kernels por camada |
| 7 | Q, K e V num único lançamento | `kernels.cu` | −2 kernels por camada |

**Resultado (Qwen3-30B-A3B, RTX 5090):**

| Medição | Antes | Depois | llama.cpp |
|---|---|---|---|
| bench, 128 tokens | 92 tok/s (padrão) / 140 tok/s (tiled + graph) | ~241 tok/s | — |
| chat, com raciocínio | 86,8 tok/s | 236 tok/s | ~357 tok/s |
| chat, `--no-thinking` | ~98 tok/s | 259 tok/s | ~337 tok/s |

**Validação:**
- Gate de logits contra o oráculo independente, em 20 casos (SmolLM2 e Qwen3-30B, de 1 a 150 tokens, chunks 1 e 3): erro máximo 0,0003 (limite 0,01), cosseno 1,0 e argmax idêntico.
- IDs gerados idênticos em todas as configurações.
- 15 testes de integração CUDA, testes de kernel e 42 testes de CPU passando.
- 37 testes de pesquisa falham porque dependem de variáveis de ambiente com fixtures (`M8_*`, `QWEN35_GGUF`). Já falhavam antes das mudanças.

**Próximos passos:**
- Fundir residual + RMSNorm e combine + RMSNorm.
- PDL (programmatic dependent launch).
- Usar `f32-tiled` por padrão no `chat`.
- Fase 2: depende da decisão sobre a política de precisão.

## 9. Paridade com o llama.cpp (2026-11-21, segunda rodada)

Tudo continua no perfil opt-in `--cuda-matmul f32-tiled`, com aritmética F32 e `fmad` desligado. O perfil padrão `f32` não mudou: ~94 tok/s, com os mesmos IDs gerados. Os números de llama.cpp foram medidos na mesma sessão e nas mesmas condições térmicas.

| Cenário (Qwen3-30B-A3B, RTX 5090, greedy) | Nosso (`f32-tiled` + `--cuda-decode-graph`) | llama.cpp | Diferença |
|---|---|---|---|
| chat com raciocínio, 2 turnos (167 e 232 tokens) | 373–377 tok/s | 359–363 tok/s | **+3–4%** |
| chat `--no-thinking` (11 tokens) | 375–377 tok/s | 344–346 tok/s | **+9%** |
| prompt cru curto (10 tokens), 128 tokens gerados | 375–376 tok/s | 358–363 tok/s | **+3–4%** |
| prompt cru longo (~2.860 tokens), 128 tokens gerados | 315–317 tok/s | 307–312 tok/s | **+1–3%** |

### O que mudou nesta rodada

| # | Mudança | Arquivos | Efeito medido |
|---|---|---|---|
| 1 | **Q4_K "wide"**: 8 lanes por bloco, uma carga de 16 bytes por lane, 4 blocos por passo de warp | `kernels.cu` (`q4k_wide_*`, `matvec_warps_per_row` por tipo) | gate 10,8→8,0 µs, Q 6,2→4,3 µs, O 6,2→4,5 µs (microbench); 288→311 tok/s |
| 2 | Conversão exata de inteiro pequeno para float por bit-trick em vez de I2F | `kernels.cu` (`small_uint_float`) | neutro; bit-idêntico |
| 3 | **residual + RMSNorm** e **combine MoE + RMSNorm** num único bloco de 1024 threads, com entradas pré-carregadas em registradores | `kernels.cu`, `cuda.rs` | −2 lançamentos por camada; 331→342 tok/s |
| 4 | **Attention de decode em chunks** dimensionados no device pela contagem viva (grafo continua válido), mais combine paralelo por chunk com cargas vetoriais | `attention.cu` | contexto longo 135→294 tok/s |
| 5 | **Attention GQA com um warp por head de query**: um bloco por (head KV, chunk), os 8 warps leem as mesmas linhas K/V (L1 absorve o reúso), softmax online local ao warp e nenhuma barreira | `attention.cu` (`attention_split_group`) | contexto longo 294→316 tok/s; curto 372→376 tok/s |
| 6 | **Replay do grafo em pipeline**: até 2 passos em voo; cada token é copiado para um anel em memória pinned e o host para no primeiro EOG. Acaba a sincronização a cada 8 tokens e o desperdício de até 7 passos especulativos | `decode.rs` | chat sem raciocínio 233→375 tok/s |

### Tentado e descartado (medido)

- **`fmad` ligado:** neutro (281 contra 284 tok/s), portanto o gargalo não é FP.
- **Cargas de gate e up intercaladas** num único laço: pior.
- **Q4_K com cargas vetorizadas** mas escalas em vetor local: muito pior (206 tok/s), por spill.
- **Última-bloco-termina** (atomic + fence) para o RMSNorm fundido: 4,7 µs, mais lento que os 2 kernels originais.
- **PDL** (programmatic dependent launch): microbench mostra só −0,15 µs por kernel (~3% no total); não compensou o encanamento no cudarc.
- **Q6_K wide** (16 lanes por bloco): só ajuda a matriz de saída (180→159 µs); `down` e `V` pioram.
- **Prefetch de V junto com K** na attention: +1% no curto e −2,5% no longo.

### Validação

- Gate de logits completos contra o oráculo independente, nos 20 casos: max_abs ≤ 0,00033, cosseno 1,0 e argmax idêntico.
- IDs gerados idênticos em todos os perfis e modos, incluindo o prompt longo.
- 15 testes de integração CUDA passam, entre eles `captured_decode_*` (grafo, EOG e `kv_live_bytes`).
- Testes de kernel e os 41 testes de CPU passam.

### O que ainda falta

- **Prefill de prompts longos** (TTFT): o caminho de prefill de attention (`attention_scores`/`attention_values`) é muito lento em ~3k tokens. É o próximo alvo.
- O `chat` ainda usa o perfil `f32` por padrão. O `f32-tiled` com grafo precisa de flag.

## 10. Prefill (TTFT) de prompts longos

Prompt de 2.863 tokens no Qwen3-30B-A3B, RTX 5090, perfil `f32-tiled`, tempo até o primeiro token:

| Etapa | TTFT |
|---|---|
| início (chunk 64, attention `attention_scores`/`attention_values`) | 14,0 s |
| attention de prefill estilo flash (`attention_prefill_flash4`) | 7,2 s |
| chunks de 512 tokens (padrão do `chat` em CUDA `f32-tiled`) | 2,8 s |
| GEMM empacotada com blocagem em registradores (`gemm.cu`) | 1,24 s |
| GEMM com K=16 e double buffering, agrupamento de experts paralelo | **~1,06 s** |
| llama.cpp (MMQ int8 em tensor cores) | 0,18 s |

O que entrou:

- **Attention de prefill causal estilo flash.** Um bloco cobre um head KV e 4 tokens, com tiles K/V de 32 chaves em shared memory, que servem os 4 tokens × 8 heads de query. São 4 warps por token, com softmax online.
- **GEMM F32 para Q4_K/Q6_K.** Tiles de 64×64 (denso) e 64×32 (experts), com 4×4 acumuladores por thread e K=16. Os bytes do tile seguinte são lidos para registradores enquanto o tile atual é calculado. A decodificação usa a mesma aritmética de `projection_weights`, e cada saída soma K em ordem crescente, então os experts dão resultado bit a bit igual ao kernel anterior.
- **Agrupamento de experts paralelo** (`group_experts_fast32`): `__match_any_sync` mais prefixos por warp. São 52 ms a menos e o layout dos descritores é idêntico.
- **`chat` em CUDA:** usa por padrão `f32-tiled` com o grafo de decode e chunks de 512 tokens. Para voltar ao comportamento antigo, use `--cuda-matmul f32`, `--no-cuda-decode-graph`, `--prefill-chunk` e `--batch`.

Tentado e descartado: tiles maiores (128×64 e 128×128 no denso, 128×32 e 64×32 com 8×4 nos experts) e K de 8, 32 ou 64. Todos foram mais lentos.

Validação:

- gate de logits nos 20 casos;
- gate novo de 600 tokens com chunks de 64 e 512, max_abs ≤ 0,00003;
- IDs gerados idênticos no prompt longo;
- 15 testes de integração CUDA passam.

**Limite físico em F32.** O prefill desse prompt precisa de ~19 TFLOP nas projeções e ~3,2 TFLOP na attention. Com `fmad` desligado, a 5090 faz ~52 TFLOP/s em F32 nos CUDA cores, o que dá um piso teórico de ~0,45 s. Chegar aos 0,18 s do llama.cpp exige tensor cores, seja com ativações int8 (q8_1), seja com TF32 em três produtos. Isso é uma decisão de precisão.

## 11. Prefill com tensor cores: perfil `int8-prefill`

Com o critério de qualidade relaxado, o prefill passou a usar tensor cores. O decode continua sendo o do `f32-tiled`. O perfil é `--cuda-matmul int8-prefill`, e o `chat` em CUDA passa a usá-lo por padrão. GPUs abaixo de sm_80 caem para `f32-tiled`.

Prompt de 2.863 tokens, Qwen3-30B-A3B, RTX 5090, tempo até o primeiro token:

| Perfil | TTFT |
|---|---|
| `f32` | 15,4 s |
| `f32-tiled` | 1,06 s |
| `int8-prefill` | **0,54 s** |
| llama.cpp | 0,18 s |

O que entrou:

- **Projeções em INT8 MMA** (`mmq.cu`). Os pesos Q4_K/Q6_K são desempacotados para s8. As ativações são quantizadas em 3 componentes int8 por bloco de 32 valores (`residual_quantize32_precise<3>`). Tiles de 128 linhas × 64 tokens (denso), × 16 tokens (K/V, poucas linhas) ou × 32 atribuições (experts). Os bytes do passo K seguinte são lidos para registradores durante o cálculo do atual.
- **Attention de prefill em FP16 MMA** (`attention_mma.cu`). Q, K, V e as probabilidades são divididos em parte alta e baixa em FP16, com acumulação e softmax em F32. Com FP16 simples, o erro passava de 1,0 nos logits. Só com P em FP16, os gates MoE longos ficavam com cos 0,999988.

Critério relaxado: mesmo argmax, cos ≥ 0,99999 e max_abs ≤ 0,15 contra o oráculo.

| Caso | max_abs | cos |
|---|---|---|
| texto real de 256 tokens | 0,092 | 0,9999981 |
| MoE long600, chunks 64/512 | 0,00003 | 1,0 |
| 20 casos curtos e long150 | ≤ 0,0008 | 1,0 |

Os 32 primeiros tokens gerados são idênticos aos do F32 no prompt curto e no longo. O decode fica em ~377 tok/s, igual ao `f32-tiled`. Os 15 testes de integração CUDA passam. Os perfis `f32` e `f32-tiled` não mudaram.

Tentado e descartado: blocos de 64 atribuições por expert (0,75 s contra 0,65 s, porque com chunks de 512 cada expert recebe em média 32 atribuições) e 1 ou 2 componentes nas ativações (1 componente dá max_abs 0,98).

Tempo de GPU no prompt longo (nsys), 534 ms no total:

| Kernel | Tempo |
|---|---|
| projeções dos experts (`residual_matmul_mmq_grouped`) | 218 ms |
| attention (`attention_prefill_mma`) | 107 ms |
| projeções densas com poucas linhas, K/V (`residual_matmul_mmq16`) | 107 ms |
| projeções densas (`residual_matmul_mmq`) | 43 ms |
| restante (router, quantização, normas) | ~60 ms |

As projeções MMQ e a attention são os próximos alvos.

## 12. Prefill `int8-prefill` na 0.2.3: 0,54 s → 0,35 s

Prompt de 2.863 tokens, Qwen3-30B-A3B Q4_K_M, RTX 5090. O llama.cpp faz 0,18 s.

| Mudança | TTFT |
|---|---|
| 0.2.2 | 0,54 s |
| par residual com fator inteiro nas projeções de atenção | 0,45 s |
| K/V convertidos uma vez por camada para FP16 alto/baixo, attention com 8 warps | 0,426 s |
| tile de 32 tokens nas projeções com poucas linhas (K/V, saída da attention) | 0,40 s |
| capacidade de prefill de 512 para 4096 (prompt inteiro num chunk) | 0,35 s |

- **Par residual** (`quantize32_residual_pair`): x ≈ d·(K·q1 + q2), com K inteiro entre 1 e 32768 por bloco de 32 valores. O kernel combina os dois MMAs em inteiro (K·dot1 + dot2 cabe em 32 bits) e usa 2 MMAs em vez de 3. Um fator potência de dois dava max_abs 0,19 e um split int16 uniforme dava 1,05.
- **FFN e experts continuam com três componentes.** Com o par também nelas o TTFT cai para 0,30 s, mas o max_abs no prompt longo vai a 0,73 e o primeiro token muda: o erro troca experts do roteamento MoE, e nesse prompt os dois primeiros tokens estão a 0,022 de logit.
- **Capacidade 4096:** o buffer de scores por token só é reservado para attention não-flash (BLOOM, Gemma). O custo é ~0,7 GB a mais de workspace no chat com contexto 4096.

Validação: mesmos 32 tokens do F32 nos prompts curto e longo; texto real de 256 tokens com max_abs 0,092 e cos 0,9999981; MoE long600 com chunks 64/512/600 com max_abs ≤ 0,055; 15 testes CUDA; `f32` e `f32-tiled` inalterados; decode ~373 tok/s.

Descartado: KS 128 nos tiles de 64 tokens, KS 64/256 nos experts, attention pré-convertida com 4 warps.

**0.2.4 — NVRTC 13.** O pacote Linux leva o NVRTC 13.3, e com ele o TTFT era 0,42 s: o kernel dos experts (`mmq_c3_grouped`) fazia 3,7 vezes mais spill de registradores com o limite de 128 registros de 2 blocos por SM. As medições acima usaram o NVRTC 12.8 do sistema, carregado pelo nome `libnvrtc.so`. Agora esse kernel usa 1 bloco por SM quando o NVRTC é 13 ou maior: 0,35 s com as duas versões, mesmos tokens e mesmos gates.

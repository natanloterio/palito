# Concorrência no `palito serve` — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `palito serve --parallel N` atende até N requisições ao mesmo tempo, em rodízio na GPU, sem mudar o caso `--parallel 1`.

**Architecture:** O `CudaBackend` ganha N conjuntos de estado de sequência (K/V, `State`, logits, prefixo, decode graph) e troca o conjunto ativo por `mem::swap`, sem tocar nos kernels nem nos 28 pontos que usam `self.keys`/`self.values`. Uma máquina de estados por requisição (`SlotRun`) avança um passo por vez, e um `Scheduler` numa única thread dá a vez a cada slot. Com `--parallel 1`, o worker segue o laço atual.

**Tech Stack:** Rust 2024, axum, tokio, cudarc, clap, llguidance.

**Spec:** `docs/superpowers/specs/2026-11-23-server-concurrency-design.md`

## Global Constraints

- `--parallel N`, `-np N`, `PALITO_PARALLEL`: de 1 a 64, padrão 1. Fora disso, falha na inicialização.
- `contexto_do_slot = ceil_256(C / N)` com divisão inteira, para `N > 1`. Com `N = 1`, o contexto não é arredondado.
- Limite de 65.535 tokens por slot. Sem `--context`, `C` é o contexto treinado e o slot desce em passos de 256 até caber.
- Prompt em blocos fixos de até 512 tokens, alinhados a partir da posição 0. Geração: 1 token por vez com mais de um slot ativo, rajadas de no máximo 8 com um só.
- `--parallel > 1` só com `CudaBackend`. CPU, backend de referência e Qwen3.5 híbrido falham na inicialização com mensagem clara.
- `--queue` conta só pedidos em espera. 503 quando os N slots estão ocupados e `queue` pedidos já esperam. `--queue 0` vale 1.
- Não alterar kernels (`*.cu`), o trait `Backend`, nem o caminho de `--parallel 1`, exceto o refactor sem mudança de comportamento da Tarefa 3.
- Mensagens de commit no formato `<type>: <description>`, sem atribuição. Sem push.
- Todo trabalho na GPU usa o prefixo abaixo, chamado de `$GPU` nos passos:

```bash
GPU='flock /tmp/palito-gpu.lock env LD_LIBRARY_PATH=/media/loterio/workspace/workspace/gaia/local_inference_rust/palito/target/dist/palito-0.7.0-linux-x86_64/lib:$LD_LIBRARY_PATH PALITO_SLOT_MODEL=/media/loterio/workspace/workspace/gaia/local_inference_rust/palito/models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf'
```

## Review Focus

1. Cliente desconecta durante a avaliação de um prompt enorme, antes de qualquer token: o slot é liberado na próxima vez dele.
2. Prompt reaproveitado do cache, inclusive de tamanho múltiplo exato de 512: gera os mesmos tokens que a avaliação a frio.
3. Fim de geração ou cancelamento com replays especulativos em voo, seguido de troca imediata de slot: o outro slot não é afetado.
4. Falha numérica de um slot marca o `status` compartilhado: o slot seguinte não herda o erro.
5. Pedido na fila cujo cliente já saiu nunca toca a GPU, e o limite de 503 é exato com os N slots ocupados.

---

### Task 1: Regras de slot e flag `--parallel`

**Files:**
- Create: `src/slots.rs`
- Modify: `src/lib.rs` (`pub mod slots;`), `src/main.rs:520`, `src/server/mod.rs:46-85`

**Interfaces:**
- Produces, em `palito::slots`:

```rust
pub const MAX_SLOTS: u32 = 64;
pub const PROMPT_TURN_TOKENS: u32 = 512;
pub const SOLO_BURST_TOKENS: u32 = 8;
pub fn slot_context(total: u32, slots: u32) -> Result<u32>;
#[derive(Clone, Copy)]
pub struct SlotCandidate { pub slot: usize, pub restorable: usize, pub last_used: u64 }
pub fn pick_slot(free: &[SlotCandidate]) -> Option<usize>;
```

- Produces, em `src/main.rs`: `fn normalize_args(args: impl Iterator<Item = OsString>) -> Vec<OsString>`.
- Produces: `ServeArgs.parallel: u32`.

- [ ] **Step 1: Escrever os testes que falham**

```rust
// src/slots.rs
#[test] fn one_slot_keeps_the_context() { assert_eq!(slot_context(1000, 1).unwrap(), 1000); }
#[test] fn context_rounds_up_to_256() {
    assert_eq!(slot_context(4096, 4).unwrap(), 1024);
    assert_eq!(slot_context(4097, 4).unwrap(), 1024); // divisão inteira antes
    assert_eq!(slot_context(4100, 4).unwrap(), 1280);
}
#[test] fn context_smaller_than_the_slots_fails() { assert!(slot_context(3, 4).is_err()); }
#[test] fn picks_most_restorable_then_oldest() {
    let c = |slot, restorable, last_used| SlotCandidate { slot, restorable, last_used };
    assert_eq!(pick_slot(&[c(0, 0, 1), c(1, 512, 9)]), Some(1));
    assert_eq!(pick_slot(&[c(0, 512, 9), c(1, 512, 2)]), Some(1));
    assert_eq!(pick_slot(&[]), None);
}

// src/main.rs, mod tests
#[test] fn np_is_an_alias_of_parallel() {
    let parsed = |a: &[&str]| normalize_args(a.iter().map(OsString::from));
    assert_eq!(parsed(&["palito", "serve", "-np", "4"]), ["palito", "serve", "--parallel", "4"]);
    assert_eq!(parsed(&["palito", "serve", "-np=4"]), ["palito", "serve", "--parallel=4"]);
    assert_eq!(parsed(&["palito", "chat", "-np", "4"]), ["palito", "chat", "-np", "4"]);
}
#[test] fn parallel_is_limited_to_64() {
    let base = ["palito", "serve", "--model", "m.gguf", "--parallel"];
    assert!(Cli::try_parse_from([&base[..], &["64"]].concat()).is_ok());
    assert!(Cli::try_parse_from([&base[..], &["0"]].concat()).is_err());
    assert!(Cli::try_parse_from([&base[..], &["65"]].concat()).is_err());
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --lib slots && cargo test --bin palito parallel`
Expected: erro de compilação, `slot_context` e `normalize_args` não existem.

- [ ] **Step 3: Implementar**

`slot_context`: com `slots == 1` retorna `total`. Com `slots > 1`, `total / slots`, erro `Invalid` se der 0, senão arredonda para cima ao múltiplo de 256. `pick_slot`: maior `restorable`, depois menor `last_used`.

`--parallel` usa `#[arg(long, env = "PALITO_PARALLEL", default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=64))]`. O clap não aceita `-np` como flag curta de duas letras, então `normalize_args` reescreve `-np` e `-np=N` somente quando o subcomando é `serve`, e `main` usa `Cli::parse_from(normalize_args(std::env::args_os()))`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --lib slots && cargo test --bin palito`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/slots.rs src/lib.rs src/main.rs src/server/mod.rs
git commit -m "feat: add slot rules and --parallel flag"
```

---

### Task 2: Vários slots no `CudaBackend` (portão de risco)

Esta tarefa prova a premissa do desenho: alternar sequências não altera os tokens. **Se o teste de intercalação falhar por estado escondido no `Workspace`, pare e leve o achado ao usuário.**

**Files:**
- Create: `src/cuda/slots.rs`
- Modify: `src/slots.rs`, `src/cuda.rs:115-145,881-899,939-950,1395-1485,2303-2400`, `src/cuda/decode.rs:46-76`, `src/cuda/prefix.rs:28-46`
- Test: `tests/slots.rs`

**Interfaces:**
- Consumes: nada de tarefas anteriores além do módulo `slots`.
- Produces:

```rust
// src/slots.rs
pub trait SlotBackend: Backend {
    fn slot_count(&self) -> usize;
    /// Espera a GPU terminar e torna `slot` o ativo.
    fn select_slot(&mut self, slot: usize) -> Result<()>;
    /// Tokens de `prompt` que o cache de `slot` restauraria, sem alterá-lo.
    fn restorable(&self, slot: usize, prompt: &[i32]) -> usize;
    /// Descarta o prefixo do slot e limpa o erro deixado na GPU.
    fn abandon_slot(&mut self, slot: usize) -> Result<()>;
    /// Err quando o contexto CUDA não responde mais.
    fn device_alive(&mut self) -> Result<()>;
}
// src/cuda.rs
pub struct CudaOptions { /* ... */ pub slots: u32 } // Default: 1
pub fn open_slot_backend(path: &Path, config: &Config, options: CudaOptions)
    -> Result<Box<dyn SlotBackend>>;
// src/cuda/prefix.rs
pub(super) fn restorable(&self, prompt: &[i32]) -> usize;
```

- [ ] **Step 1: Escrever os testes que falham**

Em `tests/slots.rs`, com `#![cfg(feature = "cuda")]`, `#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]`, abrindo o backend com `slots: 2`, `decode_graph: true`, `prefix_cache: true`. O helper `greedy(backend, prompt, count)` avalia e gera um token por chamada.

```rust
#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn interleaved_slots_match_isolated_runs() {
    let (a, b) = (prompt("The capital of France"), prompt("Write a haiku about rain"));
    let mut backend = open(2);
    let alone_a = run_alone(&mut *backend, 0, &a, 32);
    let alone_b = run_alone(&mut *backend, 1, &b, 32);
    backend.abandon_slot(0).unwrap();
    backend.abandon_slot(1).unwrap();
    let (mixed_a, mixed_b) = run_interleaved(&mut *backend, &a, &b, 32);
    assert_eq!(mixed_a, alone_a);
    assert_eq!(mixed_b, alone_b);
}

#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn switching_after_an_interrupted_burst_keeps_the_other_slot() { /* Review Focus 3:
    slot 1 gera 4 tokens; slot 0 faz rajada de 8 e o sink retorna Err(Cancelled) no 2º;
    select_slot(1); os tokens seguintes do slot 1 são iguais aos da execução isolada */ }

#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn abandoning_a_slot_clears_its_failure_for_the_next() { /* Review Focus 4:
    sobrescreve o status via helper de teste #[doc(hidden)] `poison_status_for_test`,
    abandon_slot(0), select_slot(1), geração do slot 1 igual à isolada */ }

#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn reservation_counts_every_slot() {
    assert!(open(1).memory_usage().unwrap().numerical_reservation_bytes
        < open(2).memory_usage().unwrap().numerical_reservation_bytes);
}
```

Em `src/cuda/prefix.rs`:

```rust
#[test] fn restorable_matches_restore_without_changing_the_cache() {
    let mut cache = PrefixCache::default();
    prefill(&mut cache, &ids(0..20), 8);
    let prompt = ids(0..18);
    let peeked = cache.restorable(&prompt);
    assert_eq!(cache.cached_tokens(), 20);
    assert_eq!(cache.restore(&prompt), peeked);
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --features cuda --test slots --no-run`
Expected: erro de compilação, `open_slot_backend` não existe.

- [ ] **Step 3: Implementar**

`src/cuda/slots.rs` define `ParkedSlot { decode, keys, values, logits, state, prefix }`, com `decode` como **primeiro campo**, porque o grafo guarda endereços crus e deve ser destruído antes dos buffers. `CudaBackend` ganha `parked: Vec<ParkedSlot>` logo depois de `decode` e `active_slot: usize`.

`parked` tem uma entrada por slot, e a entrada do slot ativo fica com os buffers do slot que estava ativo na abertura até a primeira troca. `select_slot(i)`: `stream.synchronize()`; se `i != active_slot`, faz `mem::swap` dos seis campos ativos com `parked[active_slot]` (devolve o atual) e depois com `parked[i]` (traz o novo). Capture o grafo do slot no primeiro uso (`decode.graph.is_none()`), reaproveitando `capture_decode_graph`.

`device_reservation`: multiplique `kv_bytes` e os bytes de `DecodeState` por `options.slots`. K/V e `DecodeState` são alocados na abertura. Se a admissão falhar com `slots > 1`, a mensagem cita N e o contexto do slot.

`open_slot_backend` recusa a arquitetura atendida por `reference::qwen35::device::open`.

- [ ] **Step 4: Rodar e ver passar**

Run: `eval "$GPU cargo test --release --features cuda --test slots -- --ignored --test-threads=1" && cargo test --features cuda --lib prefix`
Expected: 4 testes de GPU e os de prefixo passam.

- [ ] **Step 5: Commit**

```bash
git add src/slots.rs src/cuda.rs src/cuda/slots.rs src/cuda/decode.rs src/cuda/prefix.rs tests/slots.rs
git commit -m "feat: hold several sequences in the CUDA backend"
```

---

### Task 3: Saídas de resposta compartilhadas (refactor)

**Files:**
- Create: `src/engine/output.rs`
- Modify: `src/stop.rs:31-45`, `src/engine.rs:284-387`, `src/engine/tools.rs:30-232`

**Interfaces:**
- Produces:

```rust
// src/stop.rs: StopFilter deixa de ter lifetime e guarda Vec<String>
pub struct StopFilter { stops: Vec<String>, held: String, stopped: bool }
pub fn new(stops: &[String]) -> Self;
// src/engine/output.rs
pub(crate) struct RunStats { pub cached_tokens: usize, pub ttft_ns: u64, pub tokens_per_second: f64 }
pub(crate) trait ReplyOutput {
    /// Err(Cancelled) encerra a geração, por stop ou por cliente.
    fn token(&mut self, token: i32) -> Result<()>;
    /// Libera o texto retido. Chamado só quando a geração terminou sem Cancelled.
    fn drain(&mut self) -> Result<()>;
    fn finish(self: Box<Self>, stats: Option<RunStats>) -> Result<SlotReply>;
}
pub enum SlotReply { Text(Completion), Tools(ToolCompletion) }
pub(crate) struct TextOutput<S: TextSink>; // usada por Engine::generate
pub(crate) struct ToolOutput<F>;           // usada por Engine::chat_tools
```

`Engine.tokenizer` passa a ser `Arc<Tokenizer>`. `TextOutput` e `ToolOutput` guardam um clone dele, a lista de IDs de fim de geração (`Arc<[i32]>`), `budget`, `room` e `prompt_tokens`, de modo que não emprestam nada do `Engine` e podem viver numa `Box<dyn ReplyOutput>`. `RoleGate` passa a guardar `Rc<Cell<TokenRole>>`, compartilhado com `ToolOutput`.

- [ ] **Step 1: Registrar a linha de base**

Run: `cargo test 2>&1 | tail -3`
Expected: 0 failed. Anote o total de testes.

- [ ] **Step 2: Escrever o teste que falha**

```rust
// src/stop.rs
#[test] fn filter_outlives_the_stop_list() {
    let mut filter = { let stops = vec!["END".to_owned()]; StopFilter::new(&stops) };
    assert_eq!(filter.push("abEND"), "ab");
    assert!(filter.stopped());
}
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `cargo test --lib filter_outlives`
Expected: erro de compilação, `stops` não vive o bastante.

- [ ] **Step 4: Implementar**

Mova, sem alterar a lógica, o corpo do callback e o cálculo de `finish_reason` de `Engine::generate` para `TextOutput`, e os de `chat_tools` para `ToolOutput`. Os dois métodos passam a chamar o executor com `|token| output.token(token)`.

- [ ] **Step 5: Rodar tudo**

Run: `cargo test 2>&1 | tail -3 && eval "$GPU PALITO_TOOL_MODEL=\$PALITO_SLOT_MODEL cargo test --release --features cuda --test tools -- --ignored --test-threads=1"`
Expected: mesmo total da linha de base mais 1, 0 failed, e os testes de ferramentas passam.

- [ ] **Step 6: Commit**

```bash
git add src/stop.rs src/engine.rs src/engine/output.rs src/engine/tools.rs
git commit -m "refactor: share reply output between generation paths"
```

---

### Task 4: `SlotRun`, a geração em passos

**Files:**
- Create: `src/slots/run.rs` (com `mod run;` e `pub use run::{Advance, SlotRun};` em `src/slots.rs`)

**Interfaces:**
- Consumes: `SlotBackend` (Task 2), `ReplyOutput`, `RunStats`, `SlotReply` (Task 3), `PROMPT_TURN_TOKENS`, `SOLO_BURST_TOKENS` (Task 1).
- Produces:

```rust
pub enum Advance { Working, Finished }
pub struct SlotRun { /* tokens, fase, sampler, constraint, output, budget, relógio */ }
impl SlotRun {
    pub(crate) fn new(prompt: Vec<i32>, budget: u32, sampler: Sampler,
        constraint: Option<Box<dyn LogitConstraint>>, output: Box<dyn ReplyOutput>,
        eog: Arc<[i32]>) -> Self;
    pub fn prompt(&self) -> &[i32];
    /// Faz um passo no slot já selecionado.
    pub fn advance(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<Advance>;
    pub fn into_reply(self) -> Result<SlotReply>;
}
```

- [ ] **Step 1: Escrever os testes que falham**

Use um `FakeSlots: SlotBackend` em `#[cfg(test)]`, que registra as chamadas `(tokens, position)` e gera tokens por uma função pura de `(slot, position)`.

```rust
#[test] fn prompt_is_evaluated_in_512_token_turns_from_zero() {
    let (mut fake, mut run) = setup(prompt_of(1300), 2);
    run.advance(&mut fake, false).unwrap();
    run.advance(&mut fake, false).unwrap();
    run.advance(&mut fake, false).unwrap();
    assert_eq!(fake.evaluated(), [(512, 0), (512, 512), (276, 1024)]);
}
#[test] fn exact_multiple_of_512_has_no_empty_turn() { /* 1024 tokens: [(512,0),(512,512)] */ }
#[test] fn restored_prefix_keeps_the_alignment() { /* Review Focus 2:
    restaura 512 de 1300: [(512,512),(276,1024)] e tokens iguais à execução a frio */ }
#[test] fn generates_one_token_per_turn_when_shared() {
    let (mut fake, mut run) = ready_to_generate(16);
    run.advance(&mut fake, false).unwrap();
    assert_eq!(fake.last_sample_maximum(), 1);
}
#[test] fn bursts_at_most_8_tokens_when_alone() { /* maximum == 8 */ }
#[test] fn burst_never_exceeds_the_budget() { /* restam 3: maximum == 3 */ }
#[test] fn sampling_uses_host_logits_one_token_at_a_time() { /* alone=true, nenhum sample_tokens_streaming */ }
#[test] fn stops_at_end_of_generation_and_at_budget() { /* Finished nos dois casos */ }
#[test] fn cancelled_output_finishes_without_error() { /* token() -> Err(Cancelled): Ok(Finished), drain não chamado */ }
#[test] fn stepped_path_matches_the_executor() { /* mesmo Fake via Executor::generate_streaming e via
    SlotRun, com stop, greedy e seed: mesmos tokens, texto e finish_reason */ }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --lib slots::run`
Expected: erro de compilação, `SlotRun` não existe.

- [ ] **Step 3: Implementar**

Fases: `Prompt { next }`, `First`, `Generate`, `Done`.

- `Prompt`: na primeira vez, `reset()` e `restore_prefix(prompt)`. Avalia até `prefill_chunk` tokens, limitado a `PROMPT_TURN_TOKENS`. No último bloco usa `evaluate` se houver amostragem ou restrição, senão `evaluate_for_sampling`, e chama `store_prefix`.
- `First` e `Generate` repetem as regras de `Executor::generate_streaming` e `generate_constrained` (`src/lib.rs:157-264`), com `maximum = min(restante, if alone { SOLO_BURST_TOKENS } else { 1 })`. O próximo passo avalia o último token em `prompt.len() + gerados - 1`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --lib slots`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/slots.rs src/slots/run.rs
git commit -m "feat: add stepped generation for server slots"
```

---

### Task 5: `Engine` sobre slots

**Files:**
- Create: `src/engine/slots.rs`
- Modify: `src/engine.rs:39-66,156-225`, `src/engine/open.rs`, `src/engine/tools.rs:122`, `src/lib.rs` (se necessário para acessar o backend do `Executor`)
- Test: `tests/slots.rs`

**Interfaces:**
- Consumes: `open_slot_backend` (Task 2), `TextOutput`, `ToolOutput` (Task 3), `SlotRun` (Task 4), `slot_context` (Task 1).
- Produces:

```rust
pub struct Engine<B: Backend + ?Sized = dyn Backend> { executor: Executor<Box<B>>, /* ... */ }
pub type SlotEngine = Engine<dyn SlotBackend>;
impl SlotEngine {
    pub fn open_slots(path: &Path, options: &EngineOptions, slots: u32) -> Result<Self>;
    pub fn slots(&self) -> usize;
    pub fn backend_mut(&mut self) -> &mut dyn SlotBackend;
    pub fn prepare_chat(&mut self, messages: &[Message], options: &GenerationOptions,
        on_text: Box<dyn FnMut(&str) -> bool>) -> Result<SlotRun>;
    pub fn prepare_complete(&mut self, prompt: &str, options: &GenerationOptions,
        on_text: Box<dyn FnMut(&str) -> bool>) -> Result<SlotRun>;
    pub fn prepare_chat_tools(&mut self, messages: &[Message], spec: &ToolSpec,
        options: &GenerationOptions, on_event: Box<dyn FnMut(ToolEvent) -> bool>) -> Result<SlotRun>;
}
```

`Engine::open` e `from_backend` ficam em `impl Engine<dyn Backend>`, e os métodos comuns em `impl<B: Backend + ?Sized> Engine<B>`.

- [ ] **Step 1: Escrever os testes que falham**

```rust
// tests/slots.rs
#[test] fn cpu_refuses_several_slots() {
    let error = SlotEngine::open_slots(&smol(), &cpu_options(), 2).unwrap_err();
    assert!(error.to_string().contains("--parallel"));
}
#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn context_is_split_between_slots() {
    let engine = SlotEngine::open_slots(&model(), &with_context(4096), 4).unwrap();
    assert_eq!(engine.context_tokens(), 1024);
}
#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn prompt_over_the_slot_context_is_refused() {
    let mut engine = SlotEngine::open_slots(&model(), &with_context(1024), 4).unwrap();
    let error = engine.prepare_complete(&"word ".repeat(600), &options(), sink()).err().unwrap();
    assert!(error.to_string().contains("256-token context"));
}
#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn stepped_chat_matches_engine_chat_for_short_prompts() { /* greedy, prompt < 512:
    Engine::chat e SlotRun no mesmo modelo produzem o mesmo texto */ }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --features cuda --test slots --no-run`
Expected: erro de compilação, `SlotEngine` não existe.

- [ ] **Step 3: Implementar**

Em `open.rs`, `open_cuda_slots` calcula o contexto com `slot_context(options.context_tokens.unwrap_or(trained), slots)`, aplica `.min(65535)` só quando não há `--context` (com `--context`, um slot acima do limite falha como hoje), ajusta `fit_context` como no caminho atual e usa `batch_tokens` e `prefill_chunk` limitados a `PROMPT_TURN_TOKENS`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --features cuda --test slots cpu_refuses && eval "$GPU cargo test --release --features cuda --test slots -- --ignored --test-threads=1" && cargo test`
Expected: PASS, sem regressão.

- [ ] **Step 5: Commit**

```bash
git add src/engine.rs src/engine/open.rs src/engine/slots.rs src/engine/tools.rs src/lib.rs tests/slots.rs
git commit -m "feat: open the engine with server slots"
```

---

### Task 6: `Scheduler`

**Files:**
- Create: `src/slots/scheduler.rs` (com `pub use scheduler::{Round, Scheduler, SlotTask};`)

**Interfaces:**
- Consumes: `SlotBackend`, `Advance`, `pick_slot`, `SlotCandidate`.
- Produces:

```rust
pub trait SlotTask {
    fn prompt(&self) -> &[i32];
    fn abandoned(&self) -> bool;
    fn advance(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<Advance>;
    /// Entrega a resposta ao cliente.
    fn complete(self);
    fn fail(self, error: String);
}
pub enum Round { Idle, Ran, DeviceLost }
pub struct Scheduler<T: SlotTask> { /* slots, last_used, cursor */ }
impl<T: SlotTask> Scheduler<T> {
    pub fn new(slots: usize) -> Self;
    pub fn has_free_slot(&self) -> bool;
    pub fn is_idle(&self) -> bool;
    pub fn admit(&mut self, backend: &dyn SlotBackend, task: T);
    /// Dá uma vez a cada slot ativo, em ordem circular.
    pub fn round(&mut self, backend: &mut dyn SlotBackend) -> Round;
    pub fn fail_all(&mut self, error: &str);
}
```

- [ ] **Step 1: Escrever os testes que falham**

```rust
#[test] fn gives_one_turn_per_active_slot_in_order() { /* 3 tasks: seleção [0,1,2,0,1,2] */ }
#[test] fn short_task_finishes_before_a_long_one() { /* 4 passos vs 100: complete do curto primeiro */ }
#[test] fn lone_task_runs_with_alone_true() { /* e alone=false assim que entra outra */ }
#[test] fn admit_prefers_the_slot_with_the_longest_prefix() {}
#[test] fn abandoned_task_frees_its_slot_without_advancing() { /* Review Focus 1: abandoned()=true
    durante o prompt: nenhum advance, has_free_slot()=true e nenhuma chamada a abandon_slot */ }
#[test] fn request_error_ends_only_that_task() { /* Err(Invalid): fail no A, abandon_slot(A), B continua */ }
#[test] fn backend_error_with_a_live_device_keeps_the_others() { /* Err(Backend), device_alive Ok */ }
#[test] fn lost_device_is_reported() { /* device_alive Err: Round::DeviceLost */ }
#[test] fn fail_all_reaches_every_active_task() {}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --lib slots::scheduler`
Expected: erro de compilação, `Scheduler` não existe.

- [ ] **Step 3: Implementar**

`round` percorre os slots a partir de `cursor`. Para cada ativo: se `abandoned()`, libera o slot sem tocar a GPU, preservando o cache. Senão, `select_slot` e `advance(backend, ativos == 1)`. `Finished` chama `complete`. Em erro, `fail`, `abandon_slot`, depois `device_alive`, e `DeviceLost` encerra a volta. Uma falha de `select_slot` ou `abandon_slot` também vale como `DeviceLost`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --lib slots`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/slots.rs src/slots/scheduler.rs
git commit -m "feat: schedule server slots round-robin"
```

---

### Task 7: Worker e servidor com slots

**Files:**
- Create: `src/server/slot_worker.rs`
- Modify: `src/server/worker.rs:56-143`, `src/server/mod.rs:1-6,76-78,97-126,191-211`
- Test: `tests/serve.rs`

**Interfaces:**
- Consumes: `SlotEngine` (Task 5), `Scheduler`, `SlotTask`, `Round` (Task 6), `SlotReply` (Task 3).
- Produces:

```rust
// worker.rs
pub fn start(path: PathBuf, options: EngineOptions, queue: usize, parallel: u32)
    -> palito::Result<(Self, ModelReport)>;
pub struct ModelReport { /* ... */ pub slots: u32 }
// slot_worker.rs
pub(super) fn serve_slots(engine: SlotEngine, inbox: Receiver<Job>);
struct ServerTask { run: SlotRun, events: UnboundedSender<Event>, /* splitter, ids */ }
impl SlotTask for ServerTask { /* abandoned() = events.is_closed() */ }
```

O `sync_channel(queue)` existente é a fila em memória do desenho: o laço só retira um pedido quando há slot livre, então o canal contém exatamente os pedidos em espera e o `try_send` cheio continua sendo o 503.

- [ ] **Step 1: Escrever os testes que falham**

```rust
// tests/serve.rs
#[test] #[ignore = "requires models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn parallel_on_cpu_fails_at_startup() {
    let status = serve_command(&["--backend", "cpu", "--parallel", "2"]).status().unwrap();
    assert!(!status.success());
}

#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn health_reports_slots_and_slot_context() {
    let server = gpu_server(&["--parallel", "4", "--context", "4096"]);
    let (_, body) = request(&server, "GET", "/health", None, None);
    let health: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((health["slots"].as_u64(), health["context_tokens"].as_u64()), (Some(4), Some(1024)));
}

#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn queue_limit_is_exact_with_all_slots_busy() { /* Review Focus 5: --parallel 2 --queue 1:
    2 longos + 1 na fila aceitos; o 4º recebe 503 com Retry-After */ }

#[test] #[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn queued_request_of_a_gone_client_never_runs() { /* --parallel 2: pedido P termina e deixa seu
    prefixo num slot; 2 longos ocupam os slots; um pedido Q de prompt diferente entra na fila e seu
    cliente fecha o socket; quando os longos terminam, reenviar P informa `cached_tokens` > 0 no
    slot que Q teria sobrescrito */ }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --test serve --no-run && eval "$GPU cargo test --release --features cuda --test serve health_reports -- --ignored"`
Expected: falha, `--parallel` ainda não muda `/health`.

- [ ] **Step 3: Implementar**

`Worker::start`: com `parallel > 1`, abre `SlotEngine::open_slots` e chama `serve_slots`; com 1, mantém o laço atual sem alteração.

Laço de `serve_slots`:

```text
enquanto houver slot livre: try_recv → descarta cliente ausente → prepare_* → admit
    (erro de prepare vira Event::Done(Err) sem ocupar slot)
se ocioso: recv bloqueante; canal fechado encerra
Round::DeviceLost: fail_all, e daí em diante todo pedido recebe Done(Err)
```

`ServerTask` reaproveita de `worker.rs` o `ReasoningSplitter`, `send_text`, `flush_text` e `build_reply`, por meio de um estado compartilhado (`Rc<RefCell<JobEvents>>`) entre o callback e `complete`. `/health` ganha `"slots"`. Atualize a doc de `queue` e o cabeçalho de `server/mod.rs`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test && cargo test --test serve parallel_on_cpu -- --ignored && eval "$GPU cargo test --release --features cuda --test serve -- --ignored --test-threads=1"`
Expected: PASS. Testes que exigem SmolLM2 no checkout precisam do modelo em `models/`; um link simbólico para o checkout principal resolve.

- [ ] **Step 5: Commit**

```bash
git add src/server tests/serve.rs
git commit -m "feat: serve concurrent requests with --parallel"
```

---

### Task 8: Aceite ponta a ponta, medição e documentação

**Files:**
- Create: `scripts/validate_concurrency.py`, `benchmarks/concurrency/` (resultados brutos em JSON)
- Modify: `README.md`, `docs/technical-reference.md`

**Interfaces:**
- Consumes: `palito serve --parallel N`.
- Produces: `scripts/validate_concurrency.py --base-url URL --model NAME --out FILE`, com saída JSON e código de saída diferente de zero se um critério falhar.

- [ ] **Step 1: Escrever o validador**

Com o SDK oficial `openai` e threads, um caso por critério:

| Caso | Critério do desenho | Passa quando |
|---|---|---|
| `short_beats_long` | 2 | pedido de 16 tokens enviado com um longo já gerando: primeiro token em menos de 1 s e termina antes do longo |
| `isolation` | 3 | 4 pedidos simultâneos, um com mais de 512 tokens, greedy e com `seed`: tokens iguais aos de cada um sozinho |
| `throughput` | 4 | registra tok/s total com 4 pedidos, e `ratio` contra 1 pedido; `ratio < 0.80` marca `target_missed` sem esconder o valor |
| `disconnect` | 5 | fechar um stream no meio libera o slot, e os demais terminam corretos |
| `features` | 6 | JSON com schema, `tools` e `stop` corretos com slots ocupados |

- [ ] **Step 2: Rodar contra o modelo pequeno**

Run: `eval "$GPU target/release/palito serve --model \$PALITO_SLOT_MODEL --parallel 4 --port 8091" &` e depois `python scripts/validate_concurrency.py --base-url http://127.0.0.1:8091/v1 --model Qwen3-0.6B-Q4_K_M --out benchmarks/concurrency/qwen3-0.6b.json`
Expected: todos os casos `pass`, exceto `throughput`, que pode marcar `target_missed`.

- [ ] **Step 3: Rodar no Qwen3-30B-A3B e comparar `--parallel 1` com `master`**

Repita o Step 2 com `models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf`. Meça tok/s de `--parallel 1` nesta branch e no `master` (`git worktree` temporária ou binário em `master-target`), com 3 execuções de cada, e salve em `benchmarks/concurrency/qwen3-30b.json`.
Expected: diferença de no máximo 3%. **Se passar disso, pare e investigue antes de seguir.** Registre também a diferença de tokens entre `--parallel 1` e `--parallel 4` em prompts acima de 512 tokens.

- [ ] **Step 4: Documentar**

README: seção curta com `--parallel`, a divisão do contexto por slot e os números medidos, incluindo `ratio`. `docs/technical-reference.md`: atualizar "one sequence", descrever rodízio, tamanhos de bloco, escolha de slot, falhas e o que não é suportado (KV unificado, `/slots`, `id_slot`).

- [ ] **Step 5: Suíte completa**

Run: `cargo test && cargo clippy --all-targets --features cuda -- -D warnings && eval "$GPU cargo test --release --features cuda -- --ignored --test-threads=1"`
Expected: 0 failed. Qualquer falha é reportada pelo nome.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_concurrency.py benchmarks/concurrency README.md docs/technical-reference.md
git commit -m "docs: document and measure concurrent slots"
```

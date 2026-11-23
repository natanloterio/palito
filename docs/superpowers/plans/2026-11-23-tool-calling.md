# Tool calling no `palito serve`: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Um cliente OpenAI completa o ciclo de ferramentas no `palito serve` (declarar, receber a chamada, devolver o resultado, receber a resposta), com e sem streaming, em modelos com o formato JSON `<tool_call>` do Qwen.

**Architecture:** Um `ToolGate` por requisição decide por ID de token quando os logits ficam livres e quando o llguidance restringe a chamada. Um `ToolStream` transforma os mesmos tokens em eventos de texto e de chamada. O engine ganha um caminho separado, `chat_tools`, e `chat` continua intacto para requisições sem ferramentas.

**Tech Stack:** Rust 2024, axum, minijinja, llguidance 1.9.1 (Lark com `%json`), serde_json com `preserve_order`.

**Spec:** `docs/superpowers/specs/2026-11-23-tool-calling-design.md`

## Global Constraints

- Requisições sem `tools` e sem ferramentas no histórico continuam em `Engine::chat`, com a mesma saída e o mesmo caminho de decode.
- O prompt é tokenizado inteiro por `Executor::tokenize`, reconhecendo texto especial, como o `llama-server`. Não há caminho de texto protegido.
- A decisão de entrar e sair de uma chamada é sempre por ID de token, nunca por texto.
- `response_format` mantém a validação estrita atual. A política tolerante vale só para `tools`.
- Formato aceito: template com `<tool_call>` e `<function-name>`, sem `<function=`, e `<tool_call>`/`</tool_call>` como tokens únicos.
- Ordem das chaves na chamada: `name`, depois `arguments`.
- `arguments` na resposta é sempre string.
- API C, bindings, `palito chat` e `/v1/completions` não ganham ferramentas.
- Commits com `git add` por caminho explícito. Nunca adicionar `models`.

## Review Focus

1. Nome de ferramenta com aspas, barra invertida ou Unicode: a gramática aceita exatamente esse nome e a resposta o devolve igual (Task 3).
2. Argumento cujo valor contém o texto `</tool_call>`: a chamada não termina no meio da string (Task 5).
3. Caractere multibyte dividido entre tokens nos argumentos: a concatenação dos deltas é idêntica à resposta completa (Task 5).
4. `required` com raciocínio que nunca fecha: termina em `length`, sem chamada e sem erro (Task 4).
5. Cliente desconecta no meio de uma chamada: a geração para e a GPU fica livre para a próxima requisição (Task 8).

---

### Task 1: Template com ferramentas e detecção de formato

**Files:**
- Modify: `src/chat.rs`
- Modify: `src/ffi.rs`, `src/chat_repl.rs`, `src/server/openai.rs` somente se a mudança de `Message` exigir
- Test: `src/chat.rs` (`mod tests`)

**Interfaces:**
- Produces:
  - `Message { role, content, tool_calls: Option<serde_json::Value>, tool_call_id: Option<String>, reasoning_content: Option<String> }`, com `skip_serializing_if = "Option::is_none"` nos três novos campos. `Message::new` os inicializa com `None`.
  - `ChatTemplate::render_tools(&self, messages: &[Message], tools: Option<&[Value]>, enable_thinking: bool) -> Result<String>`, sempre com prompt de geração.
  - `ChatTemplate::tool_format(&self) -> Option<ToolFormat>` e `pub struct ToolFormat { pub string_arguments: bool }`.

- [ ] **Step 1: Escrever os testes que falham**

```rust
#[test]
fn renders_tools_calls_and_results_with_the_qwen3_template() {
    let template = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "<|im_end|>").unwrap();
    let tools = [json!({"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}})];
    let mut call = Message::assistant("");
    call.tool_calls = Some(json!([{"id": "call_1", "type": "function",
        "function": {"name": "weather", "arguments": "{\"city\":\"Rio\"}"}}]));
    let messages = [Message::user("Tempo?"), call, Message::new("tool", "28C")];
    let prompt = template.render_tools(&messages, Some(&tools), true).unwrap();
    assert!(prompt.contains("<tools>"));
    assert!(prompt.contains("{\"name\": \"weather\", \"arguments\": {\"city\":\"Rio\"}}"));
    assert!(prompt.contains("<tool_response>\n28C\n</tool_response>"));
}

#[test]
fn render_without_tool_fields_is_unchanged() {
    let template = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "<|im_end|>").unwrap();
    let messages = [Message::system("S"), Message::user("Oi")];
    assert_eq!(
        template.render_tools(&messages, None, true).unwrap(),
        template.render_thinking(&messages, true, true).unwrap()
    );
}

#[test]
fn detects_only_the_qwen_json_tool_format() {
    let qwen3 = ChatTemplate::new(QWEN3_TOOLS_TEMPLATE, "", "").unwrap();
    assert_eq!(qwen3.tool_format(), Some(ToolFormat { string_arguments: true }));
    let qwen25 = ChatTemplate::new("<tool_call> <function-name> {{ a | tojson }}", "", "").unwrap();
    assert_eq!(qwen25.tool_format(), Some(ToolFormat { string_arguments: false }));
    let qwen35 = ChatTemplate::new("<tool_call><function=x> <function-name>", "", "").unwrap();
    assert_eq!(qwen35.tool_format(), None);
    assert_eq!(ChatTemplate::new("{{ messages }}", "", "").unwrap().tool_format(), None);
}
```

`QWEN3_TOOLS_TEMPLATE` é uma constante de teste com o trecho do template do Qwen3-30B que trata `tools`, `tool_calls` e `role == "tool"`, incluindo o ramo `tool_call.arguments is string`.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --lib chat::`
Expected: erro de compilação, `render_tools` e `ToolFormat` não existem.

- [ ] **Step 3: Implementar em `src/chat.rs`**

Guardar a fonte do template em `ChatTemplate`. `string_arguments` é verdadeiro quando a fonte contém `arguments is string`. Passar as mensagens com `TemplateValue::from_serialize`, para campos `None` ficarem indefinidos. Sem ferramentas, `tools` fica indefinido.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked --lib chat:: && cargo test --locked --no-run`
Expected: PASS, e todos os alvos compilam.

- [ ] **Step 5: Commit**

```bash
git add src/chat.rs src/ffi.rs src/chat_repl.rs src/server/openai.rs
git commit -m "feat: render tools and tool history through the chat template"
```

---

### Task 2: Definição das ferramentas e política de schema

**Files:**
- Create: `src/tools.rs`, `src/tools/spec.rs`
- Modify: `src/lib.rs` (`pub mod tools;`)
- Test: `src/tools/spec.rs`

**Interfaces:**
- Produces:
  - `pub enum ToolMode { Auto, Required, Named(String), Forbidden }`
  - `pub struct ToolDef { pub name: String, pub parameters: Value }`
  - `pub struct ToolSpec { pub raw: Vec<Value>, pub defs: Vec<ToolDef>, pub mode: ToolMode, pub parallel: bool, pub warnings: Vec<String> }`
  - `ToolSpec::parse(tools: &[Value], choice: Option<&Value>, parallel: Option<bool>) -> Result<Self, String>`
  - `ToolSpec::forbidden() -> Self`, para histórico com ferramentas sem `tools` na requisição.
  - `ToolSpec::template_tools(&self) -> Option<&[Value]>`, que devolve `None` em `Forbidden`.
  - `pub(crate) fn tolerant_schema(tool: &str, schema: &Value) -> (Value, Vec<String>)`

- [ ] **Step 1: Escrever os testes que falham**

```rust
fn tool(name: &str, parameters: Value) -> Value {
    json!({"type": "function", "function": {"name": name, "parameters": parameters}})
}

#[test]
fn parses_every_tool_choice() {
    let tools = [tool("a", json!({"type": "object"})), tool("b", json!({"type": "object"}))];
    assert_eq!(ToolSpec::parse(&tools, None, None).unwrap().mode, ToolMode::Auto);
    assert_eq!(ToolSpec::parse(&tools, Some(&json!("required")), None).unwrap().mode, ToolMode::Required);
    let none = ToolSpec::parse(&tools, Some(&json!("none")), None).unwrap();
    assert_eq!(none.mode, ToolMode::Forbidden);
    assert!(none.template_tools().is_none());
    let named = json!({"type": "function", "function": {"name": "b"}});
    assert_eq!(ToolSpec::parse(&tools, Some(&named), None).unwrap().mode, ToolMode::Named("b".into()));
    assert!(ToolSpec::parse(&tools, None, None).unwrap().parallel);
    assert!(!ToolSpec::parse(&tools, None, Some(false)).unwrap().parallel);
}

#[test]
fn rejects_invalid_definitions() {
    let ok = tool("a", json!({"type": "object"}));
    for (tools, choice) in [
        (vec![json!({"type": "web", "function": {"name": "a"}})], None),
        (vec![tool("", json!({"type": "object"}))], None),
        (vec![ok.clone(), ok.clone()], None),
        (vec![tool("a", json!("x"))], None),
        (vec![ok.clone()], Some(json!({"type": "function", "function": {"name": "z"}}))),
        (vec![], Some(json!("required"))),
    ] {
        assert!(ToolSpec::parse(&tools, choice.as_ref(), None).is_err());
    }
}

#[test]
fn missing_parameters_mean_any_object_and_names_keep_any_character() {
    let tools = [json!({"type": "function", "function": {"name": "a\"b\\ç"}})];
    let spec = ToolSpec::parse(&tools, None, None).unwrap();
    assert_eq!(spec.defs[0].name, "a\"b\\ç");
    assert_eq!(spec.defs[0].parameters, json!({"type": "object"}));
}

#[test]
fn removes_unsupported_keywords_but_not_properties_with_the_same_name() {
    let schema = json!({"type": "object", "not": {"required": ["x"]}, "properties": {
        "not": {"type": "string"},
        "list": {"type": "array", "uniqueItems": true, "items": {"type": "integer"}}
    }});
    let (clean, warnings) = tolerant_schema("t", &schema);
    assert!(clean.get("not").is_none());
    assert_eq!(clean["properties"]["not"], json!({"type": "string"}));
    assert!(clean["properties"]["list"].get("uniqueItems").is_none());
    assert!(warnings.iter().any(|w| w.contains("t") && w.contains("uniqueItems")));
}

#[test]
fn a_schema_that_still_does_not_compile_becomes_any_object() {
    let (clean, warnings) = tolerant_schema("t", &json!({"$ref": "#/missing"}));
    assert_eq!(clean, json!({"type": "object"}));
    assert_eq!(warnings.len(), 1);
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --lib tools::spec`
Expected: erro de compilação, o módulo não existe.

- [ ] **Step 3: Implementar `src/tools/spec.rs`**

A recursão só visita posições de schema: valores de `properties`, `$defs`, `definitions`, `patternProperties`, `items`, `additionalProperties` quando objeto, e elementos de `anyOf`, `oneOf`, `allOf`. Palavras removidas: `uniqueItems`, `not`, `if`, `then`, `else`, `contains`, `minContains`, `maxContains`, `dependentRequired`, `dependentSchemas`, `propertyNames`, `unevaluatedProperties`, `unevaluatedItems`, `multipleOf`. Depois da limpeza, validar com `JsonSchema::new`. Se falhar, usar `{"type":"object"}` e adicionar um aviso.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked --lib tools::spec`
Expected: PASS. Se o llguidance aceitar ou rejeitar outra palavra, ajustar a lista e manter os testes.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/tools.rs src/tools/spec.rs
git commit -m "feat: parse tool definitions with a tolerant schema policy"
```

---

### Task 3: Gramática de uma chamada

**Files:**
- Create: `src/tools/call_grammar.rs`
- Modify: `src/grammar.rs`, `src/grammar/tests.rs`

**Interfaces:**
- Consumes: `ToolDef`, `ToolMode` (Task 2).
- Produces:
  - `pub(crate) fn call_grammar(defs: &[ToolDef], mode: &ToolMode) -> String`, em Lark.
  - `JsonDecoder::lark_constraint(&self, grammar: &str) -> Result<Constraint>`
  - `Constraint::fresh(&self) -> Self`, uma cópia no estado atual via `Matcher::deep_clone`.

- [ ] **Step 1: Escrever os testes que falham**

Com o vocabulário de teste de `src/grammar/tests.rs` e um helper `accepts(grammar, text) -> bool` que consome os bytes e exige `is_complete()`:

```rust
#[test]
fn call_grammar_fixes_key_order_names_and_schema() {
    let defs = [ToolDef { name: "weather".into(), parameters: json!({
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": false}) }];
    let g = call_grammar(&defs, &ToolMode::Auto);
    assert!(accepts(&g, "\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Rio\"}}\n"));
    assert!(!accepts(&g, "{\"arguments\": {\"city\": \"Rio\"}, \"name\": \"weather\"}"));
    assert!(!accepts(&g, "{\"name\": \"other\", \"arguments\": {\"city\": \"Rio\"}}"));
    assert!(!accepts(&g, "{\"name\": \"weather\", \"arguments\": {\"city\": 1}}"));
}

#[test]
fn call_grammar_escapes_any_tool_name() {
    let defs = [ToolDef { name: "a\"b\\ç".into(), parameters: json!({"type": "object"}) }];
    let text = format!("{{\"name\": {}, \"arguments\": {{}}}}", serde_json::to_string("a\"b\\ç").unwrap());
    assert!(accepts(&call_grammar(&defs, &ToolMode::Auto), &text));
}

#[test]
fn named_mode_allows_only_that_tool() {
    let object = json!({"type": "object"});
    let defs = [ToolDef { name: "a".into(), parameters: object.clone() }, ToolDef { name: "b".into(), parameters: object }];
    let g = call_grammar(&defs, &ToolMode::Named("b".into()));
    assert!(accepts(&g, "{\"name\": \"b\", \"arguments\": {}}"));
    assert!(!accepts(&g, "{\"name\": \"a\", \"arguments\": {}}"));
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --lib call_grammar`
Expected: erro de compilação.

- [ ] **Step 3: Implementar**

Uma alternativa por ferramenta: `"{\"name\": <nome JSON>, \"arguments\": " %json <schema> "}"`, com `/\s{0,2}/` antes e depois do objeto. Literais Lark são produzidos com `serde_json::to_string` sobre o texto já em JSON, para cobrir nomes com aspas e barras. Aplicar o limite `MAX_WHITESPACE` dentro de `%json` pelas mesmas opções de `JsonSchema::grammar`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked --lib grammar`
Expected: PASS, inclusive os testes existentes de `response_format`.

- [ ] **Step 5: Commit**

```bash
git add src/tools.rs src/tools/call_grammar.rs src/grammar.rs src/grammar/tests.rs
git commit -m "feat: grammar for one tool call with fixed key order"
```

---

### Task 4: Controlador `ToolGate`

**Files:**
- Create: `src/tools/gate.rs`
- Modify: `src/grammar.rs`, `src/lib.rs` (`generate_constrained`, `draw`)
- Test: `src/tools/gate.rs`

**Interfaces:**
- Consumes: `Constraint::fresh`, `ToolMode`.
- Produces:
  - `pub trait LogitConstraint { fn sample(&mut self, logits: &[f32], pick: &mut dyn FnMut(&[f32]) -> Result<i32>) -> Result<i32>; }`, implementado por `Constraint` e `ToolGate`.
  - `Executor::generate_constrained(..., constraint: Option<&mut dyn LogitConstraint>, ...)`
  - `pub struct ToolTokens { pub call_open: i32, pub call_close: i32, pub think_open: Option<i32>, pub think_close: Option<i32>, pub eog: Vec<i32>, pub whitespace: Vec<bool> }`
  - `pub enum GatePhase { Free, Reasoning, Call, AfterCall }`
  - `ToolGate::new(tokens: ToolTokens, call: Option<Constraint>, mode: ToolMode, parallel: bool, prompt_opens_reasoning: bool) -> Self`, com `call: None` só em `Forbidden`.
  - `pub enum TokenRole { Text, CallOpen, CallBody, CallClose }`
  - `ToolGate::phase(&self) -> GatePhase`, `ToolGate::call_started(&self) -> bool` e `ToolGate::last_role(&self) -> TokenRole`, o papel do último token amostrado. `<tool_call>` dentro do raciocínio tem papel `Text`.

- [ ] **Step 1: Escrever os testes que falham**

Helper `allowed(&mut gate) -> Vec<i32>`, que chama `sample` com logits zero e captura os índices não mascarados. Helper `feed(&mut gate, token)`, que faz `pick` devolver `token`.

```rust
#[test] fn auto_is_free_until_the_call_token_and_reasoning_ignores_it() {
    let mut gate = gate(ToolMode::Auto, true);
    assert!(allowed(&mut gate).contains(&WORD));
    feed(&mut gate, THINK_OPEN);
    assert_eq!(gate.phase(), GatePhase::Reasoning);
    feed(&mut gate, CALL_OPEN);
    assert_eq!(gate.phase(), GatePhase::Reasoning);
    feed(&mut gate, THINK_CLOSE);
    feed(&mut gate, CALL_OPEN);
    assert_eq!(gate.phase(), GatePhase::Call);
}

#[test] fn after_a_call_only_space_another_call_or_the_end() {
    let mut gate = gate_after_one_call(ToolMode::Auto, true);
    let next = allowed(&mut gate);
    assert!(next.contains(&CALL_OPEN) && next.contains(&EOG) && next.contains(&NEWLINE));
    assert!(!next.contains(&WORD));
}

#[test] fn without_parallel_a_second_call_is_forbidden() {
    let mut gate = gate_after_one_call(ToolMode::Auto, false);
    assert!(!allowed(&mut gate).contains(&CALL_OPEN));
}

#[test] fn required_forbids_the_end_and_text_before_the_call() {
    let mut gate = gate(ToolMode::Required, true);
    let first = allowed(&mut gate);
    assert!(first.contains(&THINK_OPEN) && first.contains(&CALL_OPEN));
    assert!(!first.contains(&EOG) && !first.contains(&WORD));
    feed(&mut gate, THINK_OPEN);
    let inside = allowed(&mut gate);
    assert!(inside.contains(&WORD) && !inside.contains(&EOG));
    feed(&mut gate, THINK_CLOSE);
    let after = allowed(&mut gate);
    assert!(after.contains(&CALL_OPEN) && !after.contains(&EOG) && !after.contains(&WORD));
}

#[test] fn forbidden_only_masks_the_call_token() {
    let mut gate = gate(ToolMode::Forbidden, true);
    let free = allowed(&mut gate);
    assert!(!free.contains(&CALL_OPEN) && free.contains(&WORD) && free.contains(&EOG));
}

#[test] fn a_prompt_that_opens_reasoning_starts_in_reasoning() {
    assert_eq!(gate_with_open_reasoning().phase(), GatePhase::Reasoning);
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --lib tools::gate`
Expected: erro de compilação.

- [ ] **Step 3: Implementar `LogitConstraint` e `ToolGate`**

Em `Call`, delegar a `Constraint`. Quando `is_complete()` for verdadeiro, permitir apenas `call_close`. Se o sampler escolher um token proibido por arredondamento, trocar pelo melhor token permitido, como `Constraint::best_allowed`. Em `Required`/`Named`, `think_open` só é permitido no primeiro token, e espaço em branco continua permitido depois de `</think>`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked --lib && cargo test --locked --no-run --features cuda`
Expected: PASS, e o build CUDA compila com a nova assinatura.

- [ ] **Step 5: Commit**

```bash
git add src/tools.rs src/tools/gate.rs src/grammar.rs src/lib.rs
git commit -m "feat: token-level gate for lazy tool call constraints"
```

---

### Task 5: `ToolStream`, de tokens para eventos

**Files:**
- Create: `src/tools/stream.rs`
- Test: `src/tools/stream.rs`

**Interfaces:**
- Consumes: `TokenRole` (Task 4).
- Produces:
  - `pub enum ToolEvent { Text(String), CallStart { index: usize, name: String }, CallArguments { index: usize, text: String } }`
  - `pub struct ToolCall { pub name: String, pub arguments: String, pub complete: bool }`
  - `ToolStream::new() -> Self`
  - `ToolStream::push(&mut self, role: TokenRole, piece: &[u8]) -> Vec<ToolEvent>`
  - `ToolStream::finish(&mut self) -> Vec<ToolEvent>`
  - `ToolStream::calls(&self) -> &[ToolCall]`
  - `ToolStream::in_call(&self) -> bool`

Recebe bytes, porque um caractere multibyte pode vir dividido entre tokens. O papel de cada token vem de `ToolGate::last_role` (Task 4), então o stream nunca decide por texto e `<tool_call>` no raciocínio chega como `TokenRole::Text`.

- [ ] **Step 1: Escrever os testes que falham**

```rust
const WHOLE: &str = "\n{\"name\": \"w\", \"arguments\": {\"t\": \"</tool_call> ç 😊\", \"n\": {\"a\": [1]}}}\n";

#[test] fn argument_deltas_match_for_every_cut() {
    for cut in 1..12 {
        let (events, calls) = run_cut(&[(TEXT, b"Vou ver.")], WHOLE, cut);
        assert_eq!(calls[0].name, "w");
        assert_eq!(calls[0].arguments, "{\"t\": \"</tool_call> ç 😊\", \"n\": {\"a\": [1]}}");
        assert!(calls[0].complete);
        assert_eq!(concat_arguments(&events, 0), calls[0].arguments, "cut {cut}");
        assert_eq!(first_call_event(&events), ToolEvent::CallStart { index: 0, name: "w".into() });
        assert_eq!(concat_text(&events), "Vou ver.");
    }
}

#[test] fn markers_never_reach_text_and_parallel_calls_get_indexes() {
    let (events, calls) = run_two_calls();
    assert_eq!(calls.len(), 2);
    assert!(!concat_text(&events).contains("tool_call"));
    assert!(events.contains(&ToolEvent::CallStart { index: 1, name: "b".into() }));
}

#[test] fn truncated_call_keeps_partial_arguments_and_truncated_name_is_dropped() {
    let (_, calls) = run_truncated("{\"name\": \"w\", \"arguments\": {\"t\": \"par");
    assert_eq!((calls[0].arguments.as_str(), calls[0].complete), ("{\"t\": \"par", false));
    let (events, calls) = run_truncated("{\"name\": \"w");
    assert!(calls.is_empty() && !events.iter().any(|e| matches!(e, ToolEvent::CallStart { .. })));
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --lib tools::stream`
Expected: erro de compilação.

- [ ] **Step 3: Implementar `src/tools/stream.rs`**

Dentro de uma chamada, acumular bytes e usar um varredor JSON mínimo (string, escape, profundidade) para achar o nome e o início e o fim de `arguments`. Emitir só até a última fronteira UTF-8 válida, segurando o `}` final e o espaço depois dele até `call_close`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked --lib tools::stream`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools.rs src/tools/stream.rs
git commit -m "feat: stream tool call events from generated tokens"
```

---

### Task 6: `Engine::chat_tools`

**Files:**
- Create: `src/engine/tools.rs`
- Modify: `src/engine.rs`, `src/stop.rs` somente se precisar de um acessor
- Test: `tests/tools.rs` (`#[ignore]`, modelo real)

**Interfaces:**
- Consumes: `ChatTemplate::render_tools`, `tool_format`, `ToolSpec`, `call_grammar`, `ToolGate`, `ToolStream`.
- Produces:
  - `pub struct ToolCompletion { pub completion: Completion, pub calls: Vec<ToolCall> }`
  - `Engine::supports_tools(&self) -> bool`
  - `Engine::render_chat_tools(&self, messages: &[Message], spec: &ToolSpec, enable_thinking: bool) -> Result<String>`
  - `Engine::chat_tools(&mut self, messages: &[Message], spec: &ToolSpec, options: &GenerationOptions, on_event: impl FnMut(ToolEvent) -> bool) -> Result<ToolCompletion>`

- [ ] **Step 1: Criar acesso aos modelos e escrever os testes que falham**

Run: `ln -s ../../../models models && git status --short models`
Se `models` aparecer no status, adicionar a linha `models` ao `info/exclude` do repositório comum e não versionar.

Em `tests/tools.rs`, com `Qwen3-0.6B-Q4_K_M.gguf` em CUDA:

```rust
#[test] #[ignore] fn required_produces_a_valid_call() {
    let (mut engine, spec) = engine_with_weather(ToolMode::Required);
    let done = engine.chat_tools(&ask("Tempo no Rio?"), &spec, &capped(256), |_| true).unwrap();
    assert!(done.calls[0].complete);
    assert_eq!(done.calls[0].name, "weather");
    let args: Value = serde_json::from_str(&done.calls[0].arguments).unwrap();
    assert!(args["city"].is_string());
}

#[test] #[ignore] fn forbidden_never_emits_the_call_token() {
    let (mut engine, spec) = engine_with_weather(ToolMode::Forbidden);
    let done = engine.chat_tools(&history_with_a_call(), &spec, &capped(128), |_| true).unwrap();
    assert!(done.calls.is_empty() && !done.completion.text.contains("<tool_call>"));
}

#[test] #[ignore] fn stop_does_not_cut_a_call() {
    let (mut engine, spec) = engine_with_weather(ToolMode::Required);
    let options = GenerationOptions { stop: vec!["\n".into()], ..capped(256) };
    assert!(engine.chat_tools(&ask("Tempo no Rio?"), &spec, &options, |_| true).unwrap().calls[0].complete);
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `flock /tmp/palito-gpu.lock cargo test --release --locked --features cuda --test tools -- --ignored --test-threads=1`
Expected: erro de compilação, `chat_tools` não existe.

- [ ] **Step 3: Implementar `src/engine/tools.rs`**

Fluxo separado de `generate`, sempre com `generate_constrained`. `ToolTokens` vem de `tokenize` de cada marcador, exigindo um único ID; `whitespace` vem de `Tokenizer::token_bytes`. O filtro de `stop` só vê texto em fase livre de `Auto` e `Forbidden`. Em `Required` e `Named` ele não é aplicado até a primeira chamada. Registrar `spec.warnings` no stderr. Reusar `TokenClock` e `decode_rate`.

- [ ] **Step 4: Rodar e ver passar**

Run: o mesmo comando do Step 2, depois `cargo test --locked`
Expected: PASS nos dois.

- [ ] **Step 5: Commit**

```bash
git add src/engine.rs src/engine/tools.rs src/stop.rs tests/tools.rs
git commit -m "feat: constrained tool calling in the engine"
```

---

### Task 7: Requisição HTTP

**Files:**
- Modify: `src/server/openai.rs`, `src/server/mod.rs`
- Test: `src/server/openai.rs`

**Interfaces:**
- Consumes: `ToolSpec`, `Message` (Task 1).
- Produces:
  - Campos em `ChatRequest`: `tools: Option<Vec<Value>>`, `tool_choice: Option<Value>`, `parallel_tool_calls: Option<bool>`.
  - Campos em `ChatMessage`: `tool_calls: Option<Value>`, `tool_call_id: Option<String>`, `reasoning_content: Option<String>`.
  - `ChatRequest::tool_spec(&self) -> Result<Option<ToolSpec>, String>`
  - `ChatRequest::messages_for(&self, string_arguments: bool) -> Result<Vec<Message>, String>`
  - `ChatRequest::uses_tools(&self) -> bool`
  - `ModelReport.tool_format: Option<ToolFormat>`

- [ ] **Step 1: Escrever os testes que falham**

```rust
#[test] fn tool_spec_rules() {
    let tool = json!({"type": "function", "function": {"name": "a"}});
    let base = json!({"messages": [{"role": "user", "content": "x"}]});
    let with = |extra: Value| { let mut b = base.clone(); b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone()); chat(b) };
    assert!(with(json!({})).tool_spec().unwrap().is_none());
    assert!(with(json!({"tools": []})).tool_spec().unwrap().is_none());
    assert!(with(json!({"tools": [], "tool_choice": "required"})).tool_spec().is_err());
    assert!(with(json!({"tools": [tool], "response_format": {"type": "json_object"}})).tool_spec().is_err());
    assert!(with(json!({"tools": [tool], "tool_choice": "none", "response_format": {"type": "json_object"}})).tool_spec().is_ok());
}

#[test] fn history_without_tools_is_forbidden_mode() {
    let request = chat(json!({"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"}]}));
    assert!(request.uses_tools());
    assert_eq!(request.tool_spec().unwrap().unwrap().mode, ToolMode::Forbidden);
}

#[test] fn string_arguments_become_objects_only_when_the_template_needs_it() {
    let body = |args: &str| chat(json!({"messages": [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "a", "arguments": args}}]}]}));
    let arguments = |m: &Message| m.tool_calls.as_ref().unwrap()[0]["function"]["arguments"].clone();
    assert_eq!(arguments(&body("{\"x\":1}").messages_for(true).unwrap()[0]), json!("{\"x\":1}"));
    assert_eq!(arguments(&body("{\"x\":1}").messages_for(false).unwrap()[0]), json!({"x": 1}));
    assert_eq!(arguments(&body("{\"x\":").messages_for(false).unwrap()[0]), json!("{\"x\":"));
}

#[test] fn malformed_history_calls_are_rejected() {
    for call in [json!({"function": {"arguments": "{}"}}), json!({"function": {"name": "a", "arguments": 1}})] {
        let request = chat(json!({"messages": [{"role": "assistant", "content": "", "tool_calls": [call]}]}));
        assert!(request.messages_for(true).is_err());
    }
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --bin palito openai`
Expected: erro de compilação.

- [ ] **Step 3: Implementar e ligar em `chat_completions`**

`uses_tools` é verdadeiro com `tools` não vazio ou com ferramentas no histórico. Se `uses_tools` e `report.tool_format` for `None`, responder 400 com `this model's chat template has no supported tool call format`, exceto em `tools` com `tool_choice: "none"` sem ferramentas no histórico, que segue pelo fluxo atual sem renderizar `tools`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked --bin palito`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server/openai.rs src/server/mod.rs src/server/worker.rs
git commit -m "feat: accept tools, tool_choice and tool history in chat requests"
```

---

### Task 8: Resposta, streaming e cancelamento

**Files:**
- Modify: `src/server/worker.rs`, `src/server/mod.rs`, `src/server/stream.rs`, `src/server/openai.rs`
- Test: `src/server/stream.rs`, `src/server/mod.rs`, `tests/serve.rs` (`#[ignore]`)

**Interfaces:**
- Consumes: `Engine::chat_tools`, `ToolEvent`, `ToolCall`, `ChatRequest::tool_spec`.
- Produces:
  - `Event::CallStart { index: usize, id: String, name: String }` e `Event::CallArguments { index: usize, text: String }`
  - `Reply.tool_calls: Vec<ReplyCall>`, com `ReplyCall { id, name, arguments, complete }`
  - `Prompt::ChatTools(Vec<Message>, ToolSpec)`
  - `openai::finish_reason_with_calls(reason: FinishReason, calls: &[ReplyCall]) -> &'static str`

- [ ] **Step 1: Escrever os testes que falham**

```rust
#[test] fn finish_reason_reports_tool_calls_only_for_complete_calls_on_stop() {
    let complete = [ReplyCall { id: "c".into(), name: "a".into(), arguments: "{}".into(), complete: true }];
    let partial = [ReplyCall { complete: false, ..complete[0].clone() }];
    assert_eq!(finish_reason_with_calls(FinishReason::Stop, &complete), "tool_calls");
    assert_eq!(finish_reason_with_calls(FinishReason::Length, &partial), "length");
    assert_eq!(finish_reason_with_calls(FinishReason::Stop, &[]), "stop");
}

#[test] fn final_body_has_null_content_and_string_arguments() {
    let body = final_body(Kind::Chat, "id", "m", &reply_with_call(""));
    let message = &body["choices"][0]["message"];
    assert!(message["content"].is_null());
    assert_eq!(message["tool_calls"][0]["type"], "function");
    assert!(message["tool_calls"][0]["function"]["arguments"].is_string());
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
}

#[test] fn call_chunks_follow_the_openai_delta_shape() {
    let start = call_start_chunk(0, "call_1", "weather");
    assert_eq!(start["choices"][0]["delta"]["tool_calls"][0],
        json!({"index": 0, "id": "call_1", "type": "function", "function": {"name": "weather", "arguments": ""}}));
    let delta = call_arguments_chunk(0, "{\"c");
    assert_eq!(delta["choices"][0]["delta"]["tool_calls"][0],
        json!({"index": 0, "function": {"arguments": "{\"c"}}));
}
```

Em `tests/serve.rs`, `#[ignore]`, com Qwen3-0.6B em CUDA:

```rust
#[test] #[ignore] fn client_disconnect_during_a_call_frees_the_engine() {
    // Abre um stream com tool_choice required e max_tokens 2000, lê até o primeiro
    // delta de tool_calls e fecha a conexão. Uma segunda requisição precisa
    // responder 200 em menos de 30 s.
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cargo test --locked --bin palito`
Expected: erro de compilação.

- [ ] **Step 3: Implementar**

O worker envia `ToolEvent::Text` pelo `ReasoningSplitter` existente, e os eventos de chamada diretamente. Antes de enviar o primeiro evento de uma chamada, descarregar o que o splitter estiver segurando. O `id` é `openai::response_id("call")`. `content` é `null` quando há chamadas e nenhum texto. O retorno `false` de `on_event` em cliente fechado é o que cancela a geração.

- [ ] **Step 4: Rodar e ver passar**

Run: `cargo test --locked && flock /tmp/palito-gpu.lock cargo test --release --locked --features cuda --test serve client_disconnect -- --ignored --test-threads=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server tests/serve.rs
git commit -m "feat: OpenAI tool call responses with streaming"
```

---

### Task 9: Validação ponta a ponta, compatibilidade e documentação

**Files:**
- Create: `scripts/validate_tool_calling.py`, `scripts/compare_tool_prompts.py`
- Modify: `tests/tools.rs`, `README.md`, `docs/technical-reference.md`

**Interfaces:**
- Consumes: `palito serve` completo e o `llama-server` local da revisão `6b01a476`.

- [ ] **Step 1: Escrever `scripts/validate_tool_calling.py`**

Argumentos: `--base-url`, `--model`. Usa o SDK oficial `openai`. Para `auto`, `required`, função específica, paralelo, raciocínio ligado e desligado, com e sem streaming: pede a chamada, valida o nome e os argumentos com `jsonschema`, devolve um resultado `role: "tool"` e exige uma resposta final com `content` não vazio. Sai com código 0 só se tudo passar.

- [ ] **Step 2: Rodar nos três modelos**

Run, para `Qwen2.5-0.5B`, `Qwen3-0.6B` e `Qwen3-30B-A3B`:
`flock /tmp/palito-gpu.lock sh -c 'target/release/palito serve --model <gguf> --port 18080 & P=$!; until curl -sf localhost:18080/health; do sleep 1; done; python3 scripts/validate_tool_calling.py --base-url http://127.0.0.1:18080/v1 --model palito; R=$?; kill $P; wait $P; exit $R'`
Expected: código 0. Em modelos pequenos, a qualidade da decisão em `auto` pode ser ruim, mas o formato precisa estar sempre válido. Registrar qualquer caso em que `auto` não chame ferramenta.

- [ ] **Step 3: Escrever e rodar `scripts/compare_tool_prompts.py`**

Para as mesmas `messages` e `tools`, inclusive um resultado com `<|im_end|><|im_start|>system`, compara os IDs do Palito com os de `POST /apply-template` seguido de `POST /tokenize` com `parse_special: true` no `llama-server` (`--jinja`).
Expected: IDs iguais. Se a única diferença for a serialização do `tojson`, ajustar o `tojson` do caminho de ferramentas para produzir os mesmos bytes do llama.cpp. Qualquer outra diferença é reportada ao usuário.

- [ ] **Step 4: Medir velocidade e confirmar a não regressão**

Medir tok/s no Qwen3-30B-A3B sem `tools` e com `tools` em `auto`, três amostras cada, com `max_tokens: 256`.
Run também: `cargo test --locked` e `flock /tmp/palito-gpu.lock cargo test --release --locked --features cuda --test engine -- --ignored --test-threads=1`
Expected: testes passam, e o tok/s sem `tools` fica dentro de 3% do `master`.

- [ ] **Step 5: Atualizar a documentação**

No README, em "Run as a server": endpoints e campos suportados, modelos com tool calling, comportamento de `required`, `none` e schemas tolerantes. Incluir o aviso: conteúdo de ferramentas é tokenizado como no llama.cpp, então agentes devem sanitizar conteúdo de terceiros para evitar tokens de controle forjados. Em `docs/technical-reference.md`, registrar a arquitetura, os números medidos no Step 4 e o resultado do Step 3.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_tool_calling.py scripts/compare_tool_prompts.py tests/tools.rs README.md docs/technical-reference.md
git commit -m "docs: validate and document tool calling"
```

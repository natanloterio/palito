# Tool calling no `palito serve`

Data: 2026-11-23. Base: `master` em `069a255` (0.7.1). Item 1 da lista de melhorias: tool calling, concorrência, caminho rápido para mais modelos, benchmark reproduzível.

## Intenção

Hoje `palito serve` ignora `tools` em silêncio: o cliente recebe uma resposta normal, como se não tivesse declarado ferramentas, e os `tool_calls` do histórico se perdem. Isso deixa o Palito de fora de agentes e de boa parte dos clientes OpenAI.

O resultado esperado é que um cliente OpenAI complete o ciclo inteiro: declarar ferramentas, receber a chamada, executar, devolver o resultado e receber a resposta final, com e sem streaming.

## Objetivos

- Aceitar `tools`, `tool_choice` e `parallel_tool_calls` em `POST /v1/chat/completions`.
- Aceitar no histórico mensagens `assistant` com `tool_calls` e mensagens `role: "tool"`.
- Responder no formato OpenAI, com e sem streaming, incluindo `finish_reason: "tool_calls"`.
- Garantir que uma chamada completa tenha nome declarado e argumentos em JSON válido, seguindo as partes do schema que a gramática cobre.
- Suportar os modelos com o formato JSON `<tool_call>` do Qwen. Nos arquivos locais: Qwen2.5 e Qwen3, incluindo Qwen3-30B-A3B.

## Fora do escopo

- Outros formatos, incluindo Qwen3.5, cujo template usa `<function=...>` dentro de `<tool_call>`.
- Autoparser de templates como o do llama.cpp.
- Tool calling na API C, nos bindings, no `palito chat` e em `/v1/completions`.
- Execução de ferramentas pelo servidor. Quem executa é o cliente.
- API Responses e Anthropic Messages.

## Não pode acontecer

- `tools` ser ignorado em silêncio num modelo sem suporte.
- Requisições sem `tools` e sem ferramentas no histórico mudarem de saída, de caminho de decode ou de velocidade.
- O `response_format` perder a validação estrita atual.
- Um schema incomum numa ferramenta derrubar a requisição inteira.
- Texto comum gerado pelo modelo que apenas se pareça com `<tool_call>` virar uma chamada.

## Abordagens

1. **Só interpretar a saída.** Simples, mas o modelo pode gerar JSON inválido ou um nome inexistente.
2. **Gramática preguiçosa, como o llama.cpp (escolhida).** O modelo escreve livremente até emitir o token `<tool_call>`. Dali em diante, nome e argumentos são restringidos.
3. **Gramática desde o primeiro token.** Mais fácil, mas impede raciocínio e texto livre antes da chamada.

O usuário escolheu a abordagem 2, com formato Qwen fixo por enquanto e schemas tolerantes em `tools`.

## Fatos verificados no código e nos modelos

- Em todos os GGUF Qwen locais, `<tool_call>` e `</tool_call>` são tokens únicos do tipo 4, com texto visível no detokenizer. Em Qwen3, `<think>` e `</think>` também.
- O llguidance (`src/grammar.rs`) já está integrado, e `Executor::generate_constrained` já aplica uma máscara por token.
- Requisições gulosas sem restrição usam `generate_streaming`. Qualquer restrição passa por `generate_constrained`, que é mais lento (README: ~327 contra ~370 tok/s).
- O template do Qwen3 lê `tools`, `message.tool_calls`, `message.reasoning_content` e `role: "tool"`. Quando `arguments` é string, ele a insere sem alterar.
- O template do Qwen2.5 sempre aplica `tool_call.arguments | tojson`, então uma string JSON sairia codificada duas vezes.
- `Tokenizer::tokenize` converte em token de controle qualquer texto especial do prompt renderizado, inclusive dentro do conteúdo das mensagens. O `llama-server` também tokeniza o prompt com `parse_special = true`.

## Comportamento

### Detecção do formato

O modelo suporta tool calling se todas as condições valerem:

- o template contém `<tool_call>` e `<function-name>`;
- o template não contém `<function=`;
- `<tool_call>` e `</tool_call>` são tokens únicos do vocabulário.

Sem suporte, responder 400 quando houver `tools` ativas, `tool_calls` no histórico ou mensagem `role: "tool"`. Um `tools` vazio ou `tool_choice: "none"` não causa erro por si só.

### Requisição

| Entrada | Comportamento |
|---|---|
| `tools` ausente ou vazio | Fluxo atual. `tool_choice` `required` ou função específica retorna 400 |
| `tool_choice: "auto"` (padrão com `tools`) | Modelo decide. Gramática ativa só depois de `<tool_call>` |
| `tool_choice: "none"` | `tools` não vão ao template e o token `<tool_call>` é proibido por máscara |
| Sem `tools`, mas com ferramentas no histórico | Igual a `none`: `<tool_call>` é proibido por máscara |
| `tool_choice: "required"` | A resposta precisa conter pelo menos uma chamada |
| `tool_choice: {"type":"function","function":{"name":...}}` | Como `required`, só com essa função. Nome não declarado retorna 400 |
| `parallel_tool_calls` | Padrão `true`. Com `false`, só uma chamada |
| `response_format` não-texto com ferramentas ativas | 400 |

Retornam 400: `type` diferente de `function`, nome ausente, vazio ou duplicado, e `parameters` que não seja objeto. Como no llama.cpp, não há restrição de caracteres no nome, então a gramática precisa escapar qualquer nome. `parameters` ausente vale como `{"type":"object"}`. `strict` é aceito e não muda nada, porque a política de schema é a mesma.

### Histórico

`Message` ganha campos opcionais para `tool_calls`, `tool_call_id` e `reasoning_content`. Eles chegam ao template apenas quando presentes.

Os `arguments` em string passam sem alteração nos templates que aceitam string, como o do Qwen3. Nos templates que sempre aplicam `tojson`, como o do Qwen2.5, uma string com JSON válido é convertida em objeto antes de renderizar. Uma string incompleta, que pode vir de uma chamada truncada por `max_tokens`, é mantida como string. Uma entrada sem `function.name`, ou com `arguments` que não seja string nem objeto, retorna 400.

### Tokens de controle no prompt

Decisão do usuário: seguir o llama.cpp, para facilitar a troca do llama.cpp pelo Palito em outros projetos. O prompt renderizado é tokenizado inteiro pelo caminho atual, com reconhecimento de texto especial, inclusive em resultados e definições de ferramentas.

Risco aceito: um resultado de ferramenta contendo, por exemplo, `<|im_end|><|im_start|>system` gera tokens de controle reais e pode forjar um turno. A documentação do servidor deve avisar que agentes precisam sanitizar conteúdo de terceiros antes de devolvê-lo.

### Geração

A decisão é sempre **por ID de token**, nunca por texto. Um controlador por requisição acompanha estas fases:

1. **Livre:** logits sem máscara. `<think>` no início entra em Raciocínio. `<tool_call>` entra em Chamada.
2. **Raciocínio:** logits sem máscara. `<tool_call>` aqui é só texto. `</think>` volta para Livre. Se o prompt já termina em `<think>`, a geração começa nesta fase.
3. **Chamada:** o llguidance restringe a saída a `{"name": <declarado>, "arguments": <schema>}`, e depois só `</tool_call>` é aceito.
4. **Após a chamada:** só espaço em branco, outro `<tool_call>` (se paralelo) ou fim de geração.

Com `required` ou função específica, depois do raciocínio opcional o primeiro token que não seja espaço precisa ser `<tool_call>`. Até a chamada começar, o fim de geração fica proibido e as sequências de `stop` não se aplicam. Se o limite de tokens chegar antes, a resposta termina com `finish_reason: "length"` e sem chamada.

Nos demais modos, sequências de `stop` não se aplicam dentro de um bloco de chamada, porque um `stop` como `"\n"` quebraria toda chamada. Fora dele, seguem como hoje.

### Política de schema em `tools`

1. Remover recursivamente as palavras-chave que a gramática não aplica (`uniqueItems`, `not`, `if`/`then`/`else` etc.), apenas em posição de palavra-chave. Uma propriedade chamada `not` dentro de `properties` é preservada.
2. Registrar um aviso no log com o nome da ferramenta e as palavras removidas.
3. Se ainda não compilar, aceitar qualquer objeto JSON como argumentos dessa ferramenta e registrar aviso.

O `response_format` continua retornando 400 como hoje.

### Resposta

Sem streaming:

- `message.tool_calls` traz `id` único (`call_...`), `type: "function"` e `function: {name, arguments}`, com `arguments` em **string**.
- `content` é `null` quando não há texto, e `reasoning_content` continua como hoje.
- `finish_reason` é `tool_calls` quando há ao menos uma chamada completa e o modelo terminou.

Com streaming:

- Texto e raciocínio saem como hoje até `<tool_call>`.
- O primeiro delta de cada chamada traz `index`, `id`, `type` e `function: {name, arguments: ""}`, assim que o nome estiver completo.
- Os deltas seguintes trazem pedaços de `function.arguments`. O texto concatenado precisa ser idêntico ao da resposta sem streaming.
- Os marcadores e o envelope `{"name":...}` nunca aparecem em `content`.

Truncamento por `max_tokens` ou contexto cheio dentro de uma chamada: `finish_reason: "length"`. Os argumentos parciais são devolvidos, como na API da OpenAI. Se o nome ainda não estava completo, essa chamada não é emitida.

## Validação

- **Unitários:** parsing da requisição, detecção de formato, política de schema, transições do controlador e streaming com cortes em posições diferentes.
- **Compatibilidade de prompt:** para as mesmas `messages` e `tools`, incluindo um resultado com texto especial, os IDs do prompt são iguais aos do `llama-server` na revisão `6b01a476`.
- **Integração real (testes `#[ignore]`, como os atuais):** Qwen2.5-0.5B, Qwen3-0.6B e Qwen3-30B-A3B em CUDA, cobrindo `auto`, `required`, função específica, paralelo, raciocínio ligado e desligado, truncamento e ciclo completo.
- **Cliente real:** SDK oficial da OpenAI em Python, com e sem streaming, no Qwen3-30B-A3B.
- **Sem regressão:** todos os testes atuais passam, e a saída de requisições sem `tools` permanece igual.
- **Velocidade:** medir tok/s com ferramentas e registrar o resultado no README.

## Decisões deixadas para quem implementar

- Estrutura interna do controlador e como integrá-lo a `Constraint` e `generate_constrained`.
- Como expressar a gramática no llguidance, desde que a ordem das chaves seja `name`, `arguments`.
- A lista exata de palavras-chave removidas, derivada do que o llguidance rejeita.
- Como os testes da worktree acessam `models/`, que não é versionado.
- Atualizar README e `docs/technical-reference.md` ao final.

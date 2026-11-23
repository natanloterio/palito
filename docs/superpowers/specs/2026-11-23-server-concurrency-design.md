# Concorrência no `palito serve` (slots alternados)

## Intenção

Hoje o servidor atende uma requisição por vez. Um pedido curto espera um pedido longo terminar, o que dificulta o uso em produção. O objetivo do usuário é que **nenhum usuário fique travado esperando outro**, e que projetos possam trocar o `llama-server` pelo Palito com pouco esforço.

## Objetivos

- `palito serve --parallel N` (também `-np N`) atende até N requisições ao mesmo tempo, com semântica de slots como a do `llama-server`.
- Requisições ativas avançam em rodízio, de modo que uma resposta longa não bloqueia uma curta.
- Todos os tipos de requisição funcionam em qualquer slot: greedy, amostragem, `stop`, JSON, tool calling, streaming e cancelamento.
- A resposta de um slot não depende do que os outros estão fazendo.

## Fora do escopo

- Decodificação em lote de várias sequências no mesmo passo da GPU (caminho B). Pode vir depois, se a medição justificar.
- KV unificado entre slots, `--kv-unified`, `-np -1`, context shift, `id_slot`, salvar e restaurar slots, e os endpoints `/slots` e `/props`.
- Concorrência em `palito chat`, na API C e nas bindings. Elas continuam com uma sequência.
- Concorrência no backend de CPU.

## O que não pode acontecer

- Piorar o caso de uma requisição: com `--parallel 1`, o padrão, o comportamento e a velocidade ficam como hoje.
- Um slot contaminar a saída de outro.
- Um cliente desconectado manter a GPU ocupada.
- Reescrever os kernels de decode.

## Abordagens consideradas

- **A. Slots alternados por token (escolhida pelo usuário).** Pesos compartilhados, um cache K/V por slot, e a GPU atende um slot de cada vez. O total de tok/s continua próximo ao de hoje e é dividido entre os pedidos. Risco baixo para os kernels.
- **B. Decodificação em lote.** Aumenta o total de tok/s, mas exige reescrever decode, atenção e MoE. Fica para depois.
- **Vários motores.** Descartada: cada um carregaria os pesos de novo, 18 GB no Qwen3-30B.

## Comportamento externo

### Flags

| Flag | Comportamento |
|---|---|
| `--parallel N`, `-np N`, `PALITO_PARALLEL` | Número de slots, de 1 a 64. Padrão: 1. Valores fora disso falham na inicialização. |
| `--context C` | Contexto **total**, como `-c` no llama.cpp. Sem a flag, é o contexto que o modelo e a memória da GPU permitem hoje. |
| `--queue` | Conta só os pedidos em espera, como hoje. Retorna 503 quando os N slots estão ocupados e `queue` pedidos já esperam. `--queue 0` continua valendo 1. |

O padrão é 1, e não o automático do `llama-server` (4 slots com KV unificado), porque sem KV unificado 4 slots reduziriam o contexto de cada pedido a um quarto sem o usuário ter pedido. Decisão do autor do desenho.

### Contexto por slot

Como no `llama-server` com `-np N` explícito: `contexto_do_slot = ceil_256(C / N)`, isto é, `C / N` com divisão inteira, arredondado para cima ao múltiplo de 256. O total alocado é `N * contexto_do_slot`.

- Sem `--context`, `C` é o contexto treinado do modelo. Se `N * contexto_do_slot` não couber na GPU, o contexto do slot desce em passos de 256 até caber, como o ajuste atual.
- Com `--context`, a admissão verifica o total real, `N * contexto_do_slot`, depois do arredondamento. Se não couber, a inicialização falha com a mensagem atual de memória, citando N e o contexto do slot. Não há busca automática por outro N.
- O limite atual de 65.535 tokens vale por slot. `Workspace` e lotes são dimensionados uma vez, para o contexto do slot.
- Se o contexto do slot ficar abaixo de 256 tokens, a inicialização falha com mensagem clara.
- Um prompt maior que o contexto do slot retorna 400, com a mesma mensagem de hoje citando o contexto do slot.
- `GET /health` e `GET /v1/models` passam a informar o contexto do slot em `context_tokens` e `context_length`. `/health` ganha `"slots": N`.

### Escolha do slot

1. Entre os slots livres, escolher o que pode **restaurar** mais tokens do prompt pelo cache de prefixo (início de bloco já gravado, não o prefixo bruto).
2. Em empate, inclusive sem prefixo em comum, escolher o usado há mais tempo.
3. Sem slot livre, o pedido espera na fila em ordem de chegada.

### Rodízio

O worker dá uma vez a cada slot ativo, em ordem circular:

- **Prompt:** avalia um bloco fixo de até 512 tokens por vez, alinhado a partir da posição 0, para que um prompt enorme não pare os outros. O cache de prefixo do slot registra esses inícios de bloco.
- **Geração:** gera um token por vez quando há mais de um slot ativo.
- **Um só slot ativo:** usa os mesmos replays do decode graph, em rajadas de no máximo 8 tokens (cerca de 22 ms a 360 tok/s). A rajada não é interrompida. Entre rajadas, o worker verifica se chegou trabalho novo e termina os replays em voo antes de trocar de slot.
- Um novo pedido entra no rodízio ao fim da vez atual, que dura no máximo uma rajada ou um bloco de prompt.

### Isolamento e falhas

- Num servidor com o mesmo `--parallel N > 1`, greedy com o mesmo prompt produz **exatamente os mesmos tokens** com ou sem outros slots ativos. A base de comparação é esse servidor com o pedido sozinho, porque o bloco de 512 tokens é a forma canônica de avaliar prompts nesse modo.
- Com `seed` fixa, a amostragem também, porque amostrador, restrição e filtro de `stop` pertencem ao pedido.
- **Não se promete** igualdade bit a bit entre `--parallel 1` e `--parallel N` em prompts maiores que 512 tokens, porque os kernels não são invariantes ao tamanho do lote. O teste mede e registra essa diferença.
- Desconexão do cliente libera o slot na próxima vez dele. O K/V do prompt já avaliado continua reutilizável como cache de prefixo.
- Um erro de requisição (por exemplo, restrição sem token permitido) encerra só aquele pedido.
- Um erro de dispositivo ou numérico encerra o pedido que estava na GPU e invalida o cache de prefixo do slot. Em seguida o worker sincroniza o stream CUDA. Se a sincronização funcionar, os outros slots continuam, porque seu K/V está em buffers próprios. Se falhar, o contexto é considerado perdido: todos os pedidos ativos e em espera recebem erro, e os seguintes também, como hoje quando o motor falha.

### Backends

`--parallel` maior que 1 só é aceito em modelos servidos pelo `CudaBackend`. Em CPU ou backend de referência, a inicialização falha com uma mensagem clara. Com `--parallel 1`, todos os backends seguem o caminho atual.

## Arquitetura

- **Estado por slot no `CudaBackend`:** `keys`, `values`, `State`, `PrefixCache`, o token amostrado e o decode graph. O grafo captura ponteiros de K/V, então cada slot tem o seu, capturado no primeiro uso do slot. `Workspace`, pesos e kernels são compartilhados e não guardam dados entre tokens fora do slot ativo.
- **Interface:** o trait `Backend` não muda. Um trait separado, implementado só pelo `CudaBackend`, expõe a operação por slot, e os testes usam uma implementação falsa dele. `json_decoder` e `tool_tokens` do `Engine` são compartilhados, e o estado de gramática é por pedido.
- **Troca de slot:** antes de trocar, a GPU precisa ter terminado o trabalho do slot atual, incluindo replays em voo. Os logits do host são por slot quando o pedido usa amostragem. Modelos sem decode graph geram cada token pelo caminho atual de avaliação.
- **Geração em passos:** o laço de `Executor` e `Engine::generate`/`chat_tools` hoje roda até o fim com callback. Para `--parallel > 1`, uma máquina de estados por requisição (`preparar`, `avaliar prompt`, `gerar`, `concluir`) cobre chat, completion, JSON e ferramentas, mantendo uma única thread dona da GPU. Com `--parallel 1`, o servidor continua no caminho atual. São dois caminhos de propósito, para não arriscar o caso de um pedido. A lógica de texto, `stop`, gate e motivo de término é compartilhada, e um teste compara os dois caminhos.
- **Worker:** com `--parallel > 1`, é dono de uma fila em memória limitada por `--queue`, atribui pedidos a slots e executa as voltas. Bloqueia sem consumir CPU quando não há trabalho.

## Critérios de aceite

1. `--parallel 1`: tok/s dentro de 3% do `master` no Qwen3-30B-A3B.
2. `--parallel 4`, um pedido longo já em andamento sozinho: um pedido curto de 16 tokens que chega depois recebe o primeiro token em menos de 1 s e termina antes do longo.
3. Tokens de cada slot idênticos aos do mesmo pedido sozinho num servidor `--parallel 4`, em greedy e com `seed`, com 4 pedidos simultâneos e prompts diferentes, incluindo um maior que 512 tokens.
4. Total de tok/s com 4 pedidos simultâneos: registrar o valor. Meta de pelo menos 80% do tok/s de um pedido só. Se não atingir, reportar ao usuário em vez de esconder.
5. Desconexão no meio de um pedido libera o slot, e os demais terminam corretamente.
6. JSON, tool calling e `stop` passam com slots simultâneos.
7. README e `docs/technical-reference.md` documentam flags, divisão do contexto e números medidos.

## O que fica para quem implementar

Nomes de tipos, divisão em módulos, formato interno da máquina de estados e como testar com um backend falso.

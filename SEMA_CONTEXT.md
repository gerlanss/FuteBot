# SEMA_CONTEXT

Este arquivo e a entrada explicita e obrigatoria para IA neste projeto.
Toda IA deve comecar por aqui antes de abrir codigo cru ou tentar mexer no projeto.
Se a IA ignorar este arquivo, ela ja esta comecando errado.

- Gerado em: `2026-04-03T18:08:12.018Z`
- Escopo: `projeto`
- Base do projeto: `c:\GitHub\FuteBot`

## Ordem minima para IA

1. Ler `SEMA_CONTEXT.md`.
2. Ler `SEMA_BRIEF.md`.
3. Ler `SEMA_INDEX.json`.
4. Usar a CLI `sema` como interface publica principal do projeto.
5. So depois subir para o contrato alvo e para o codigo vivo.

## Papel deste arquivo

- Este arquivo ja e valido como guia inicial do projeto mesmo sem contrato ativo.
- Quando houver um arquivo `.sema` ativo, a preparacao pode incorporar contexto detalhado do modulo sem perder esta base.
- Se algum bloco embutido abaixo entrar em conflito com esta abertura, a abertura deste arquivo prevalece.

## Resumo do projeto

```txt
Resumo IA-first do projeto gerado em c:\GitHub\FuteBot\.tmp\sema-resumo

PROJETO: FuteBot
MODULOS: 5
ENTRADA_IA: llms.txt -> SEMA_BRIEF.micro.txt -> SEMA_INDEX.json -> AGENTS.md
TOP_MODULOS: futebot.previsao, futebot.integracoes, futebot.operacao (+2)
TOP_RISCOS: acompanhar_previsao_live:alto, altera_persistencia, aplicar_quarentena_seletiva:alto (+15)
TOP_LACUNAS: audit_ausente, authz_frouxa, dados_nao_classificados (+3)
GERADO_EM: 2026-04-03T18:08:10.341Z
```

## Prompt curto do projeto

```txt
Voce esta operando Sema em modo IA-first.

Isto nao e material feito para humano; e contexto comprimido para IA.

Capacidade alvo: media
Modo da tarefa: mudanca

Regras:
- comece pelo resumo compacto abaixo
- se a tarefa pedir mais contexto, abra `SEMA_INDEX.json`
- nao tente ler o repo inteiro se o resumo ja disser onde tocar
- preserve contrato, risco, lacuna e checks sugeridos

Contexto do projeto:
PROJETO: FuteBot
BASE: c:\GitHub\FuteBot
MODULOS: 5
ENTRADA_IA: llms.txt -> SEMA_BRIEF.curto.txt -> SEMA_INDEX.json -> AGENTS.md -> README.md
TOP_MODULOS: futebot.previsao, futebot.integracoes, futebot.operacao, futebot.quarentena, futebot.telegram
TOP_RISCOS: acompanhar_previsao_live:alto, altera_persistencia, aplicar_quarentena_seletiva:alto, avaliar_degradacao_modelo:alto, baixar_fixtures_historicos:alto, baixar_stats_historicos:alto (+12)
TOP_LACUNAS: audit_ausente, authz_frouxa, dados_nao_classificados, execucao_critica_sem_bloco, execucao_implicita, proibicoes_ausentes
TOP_ARQUIVOS: c:\GitHub\FuteBot\data\bulk_download.py, c:\GitHub\FuteBot\data\database.py, c:\GitHub\FuteBot\models\learner.py, c:\GitHub\FuteBot\pipeline\scanner.py, c:\GitHub\FuteBot\pipeline\scheduler.py, c:\GitHub\FuteBot\services\apifootball.py (+2)
GERADO_EM: 2026-04-03T18:08:10.403Z
```

## Starter IA

```txt
Starter de IA da Sema

Modo IA-first da instalacao atual
- Use `sema` como interface publica principal.
- A Sema entra em projeto novo, projeto ja semantizado e adocao incremental em legado sem contrato inicial.
- Nao assuma monorepo, `node pacotes/cli/dist/index.js`, `npm run project:check` ou uma pasta `exemplos` externa ao projeto atual.
- Se a IA tiver contexto curto, comece por `sema resumo` e `sema prompt-curto`.
- Se a IA aguentar mais contexto, suba para `sema drift --json` e `sema contexto-ia`.
- So leia `ast.json` e `ir.json` completos quando a capacidade da IA realmente aguentar esse volume.
- Documentos locais empacotados: `AGENT_STARTER.md`, `como-ensinar-a-sema-para-ia.md`, `prompt-base-ia-sema.md`, `fluxo-pratico-ia-sema.md`.

Voce esta trabalhando com Sema, um Protocolo de Governanca de Intencao para IA sobre software vivo em backend e front consumer.

Importante:
- a Sema se apresenta publicamente como protocolo e funciona tecnicamente como linguagem de intencao
- a Sema e protocolo de governanca semantica desenhado para IA, nao para ergonomia humana
- leitura humana e bonus toleravel, nao objetivo de produto
- a Sema nao e gerador magico que deveria fazer tudo
- a Sema modela contratos, estados, fluxos, erros, efeitos, garantias, vinculos e execucao
- a Sema gera codigo e scaffolding real para TypeScript, Python e Dart
- a Sema usa `importar` para bootstrap revisavel, nao para contrato final automatico
- a Sema usa `impl` para ligar task a simbolo real do runtime
- a Sema usa `vinculos` para ligar contrato a arquivo, simbolo, recurso e superficie real
- a Sema usa `execucao` para explicitar timeout, retry, compensacao e criticidade
- a Sema usa `drift` para medir diferenca entre contrato e codigo vivo com score, confianca e lacunas
- a Sema usa `resumo` e `prompt-curto` para IA pequena ou gratuita
- a Sema usa `contexto-ia` para gerar `ast.json`, `ir.json`, `drift.json`, `briefing.json` e artefatos compactos antes da edicao
- a Sema pode servir de base para interfaces graficas elegantes e coerentes
- a Sema nao gera uma interface completa sozinha no estado atual
- trate a Sema como cerebro semantico da aplicacao, nao como gerador magico de front-end pronto
- se a tarefa envolver UI, prefira pedir Sema + React + TypeScript ou Sema + arquitetura de front-end
- evite pedir HTML unico solto quando a intencao for testar a Sema de verdade

Regras:
- nao invente sintaxe fora da gramatica e dos exemplos oficiais
- se a IA for pequena, nao tente abrir tudo de uma vez
- use `sema resumo` e `briefing.min.json` antes de subir para o pacote completo
- trate `ir --json` como fonte de verdade semantica
- trate `briefing.json` como plano de intervencao antes de editar projeto vivo
- trate `diagnosticos --json` como fonte de correcao
- use `sema formatar` como fonte unica de estilo
- preserve a intencao do contrato
- nao cobre da Sema adivinhacao de negocio que nao esta no contrato nem no codigo

Comandos essenciais:
- resumo compacto por capacidade: `sema resumo <arquivo-ou-pasta> [--micro|--curto|--medio] [--para <resumo|onboarding|review|mudanca|bug|arquitetura>]`
- prompt curto para IA pequena: `sema prompt-curto <arquivo-ou-pasta> [--micro|--curto|--medio] [--para <resumo|onboarding|review|mudanca|bug|arquitetura>]`
- descoberta do projeto: `sema inspecionar [arquivo-ou-pasta] --json`
- auditoria do contrato vivo: `sema drift <arquivo-ou-pasta> [--json]`
- contexto completo do modulo: `sema contexto-ia <arquivo.sema>`
- estrutura sintatica: `sema ast <arquivo.sema> --json`
- estrutura semantica: `sema ir <arquivo.sema> --json`
- validacao: `sema validar <arquivo.sema> --json`
- diagnosticos: `sema diagnosticos <arquivo.sema> --json`
- formatacao: `sema formatar <arquivo.sema>`
- importacao assistida de legado: `sema importar <nestjs|fastapi|flask|nextjs|nextjs-consumer|react-vite-consumer|angular-consumer|flutter-consumer|firebase|dotnet|java|go|rust|cpp|typescript|python|dart> <diretorio> --saida <diretorio>`
- geracao de codigo: `sema compilar <arquivo-ou-pasta> --alvo <typescript|python|dart|lua> --saida <diretorio>`
- verificacao final: `sema verificar <arquivo-ou-pasta> [--json]`

Antes de editar:
1. leia README, docs de IA e um exemplo oficial parecido
2. se a IA for pequena, rode `sema resumo <arquivo> --micro` e leia `briefing.min.json`
3. se a IA aguentar mais, rode `sema drift` para medir impls, vinculos, rotas, score e lacunas
4. se a tarefa for pesada, rode `sema contexto-ia` e leia `briefing.json`
5. consulte AST e IR do modulo alvo so quando a capacidade realmente aguentar

Depois de editar:
1. rode `sema formatar`
2. rode `sema validar --json`
3. se houver falha, use `diagnosticos --json`
4. rode `sema drift` de novo quando mexer em codigo vivo
5. se a tarefa pedir codigo derivado, rode `sema compilar`
6. feche com `sema verificar <arquivo-ou-pasta> --json`

Priorize sempre:
- exemplos oficiais
- JSON da CLI
- o menor artefato que resolva a tarefa da IA atual
- score, confianca e lacunas do `drift`
- `briefing.json` como guia de mudanca
- consistencia semantica

Superficies que a IA deve enxergar como first-class:
- `route`
- `worker`
- `evento`
- `fila`
- `cron`
- `webhook`
- `cache`
- `storage`
- `policy`

Nao improvise quando faltar contexto.
```

## Prompt IA

```txt
Prompt-base de IA da Sema

Modo IA-first da instalacao atual
- Use `sema` como interface publica principal.
- A Sema entra em projeto novo, projeto ja semantizado e adocao incremental em legado sem contrato inicial.
- Nao assuma monorepo, `node pacotes/cli/dist/index.js`, `npm run project:check` ou uma pasta `exemplos` externa ao projeto atual.
- Se a IA tiver contexto curto, comece por `sema resumo` e `sema prompt-curto`.
- Se a IA aguentar mais contexto, suba para `sema drift --json` e `sema contexto-ia`.
- So leia `ast.json` e `ir.json` completos quando a capacidade da IA realmente aguentar esse volume.
- Documentos locais empacotados: `AGENT_STARTER.md`, `como-ensinar-a-sema-para-ia.md`, `prompt-base-ia-sema.md`, `fluxo-pratico-ia-sema.md`.

Voce esta trabalhando com Sema, um Protocolo de Governanca de Intencao orientado a contrato, desenhado para operacao por IA.

Trate a Sema como camada semantica e linguagem de especificacao executavel feita para IA, nao para leitura humana confortavel. Nao invente sintaxe, palavras-chave ou blocos fora da gramatica e dos exemplos oficiais.

Fontes de verdade, em ordem:
1. SEMA_CONTEXT.md
2. SEMA_BRIEF.md
3. SEMA_INDEX.json
4. README do projeto
5. gramatica e documentacao de sintaxe da Sema
6. especificacao semantica da linguagem
7. exemplos oficiais, com prioridade para o vertical de pagamento
8. `sema resumo` e `briefing.min.json` quando a IA for pequena
9. AST, IR e diagnosticos exportados pela CLI em JSON quando a capacidade aguentar

Regras de operacao:
- preserve o significado semantico
- use o formatador oficial da Sema como fonte unica de estilo
- use diagnosticos estruturados como contrato de correcao
- use a IR como fonte de verdade semantica quando houver duvida
- nao conclua uma alteracao sem validar e verificar o modulo
- comece pelo menor artefato semantico que resolva a tarefa

Antes de editar `.sema`, entenda:
- o module alvo
- os contratos de task, route, error, effects, guarantees, state e flow
- os exemplos oficiais relacionados

Depois de editar `.sema`, execute este fluxo:
1. formatar
2. validar
3. diagnosticar, se houver falha
4. verificar

Se houver conflito entre texto livre e IR/diagnosticos, priorize a IR e os diagnosticos da CLI.

Se algo nao estiver claro, siga a forma ja usada nos exemplos oficiais. Nao improvise sem base.
```

## Como regenerar

- Rode `Preparar Contexto IA do Projeto` na extensao para atualizar este arquivo e os entrypoints da raiz.
- Opcionalmente, com um arquivo `.sema` aberto, a preparacao incorpora contexto detalhado do modulo ativo.
- `sema sync-ai-entrypoints --json`

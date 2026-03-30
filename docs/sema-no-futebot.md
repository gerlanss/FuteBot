# Sema no FuteBot

Este projeto agora passa a usar a Sema como camada de modelagem semantica para fluxos operacionais e contratos que merecem clareza antes de virar Python.

O objetivo nao e reescrever o bot inteiro em cima da Sema de uma vez, igual um animal com pressa. O objetivo e usar a Sema onde ela mais ajuda:

- contratos de operacao
- fluxos de notificacao
- estados de oportunidades
- manutencao de garantias entre scanner, scheduler e Telegram
- geracao futura de codigo derivado quando isso fizer sentido

## Onde isso entra hoje

O primeiro modulo modelado esta em [operacao_futebot.sema](/C:/GitHub/FuteBot/sema/operacao_futebot.sema).

O segundo modulo modelado esta em [quarentena_retreino_focal.sema](/C:/GitHub/FuteBot/sema/quarentena_retreino_focal.sema).

O terceiro modulo modelado esta em [telegram_operacao.sema](/C:/GitHub/FuteBot/sema/telegram_operacao.sema).

Ele descreve quatro tarefas centrais:

- gerar radar pre-live
- liberar mercados na janela T-30
- monitorar jogos ao vivo
- emitir relatorio diario e saude do modelo

Tambem existe um `flow operacao_diaria_futebot` para deixar explicito o encadeamento operacional.

No fluxo de manutencao, o modulo de quarentena cobre:

- deteccao de degradacao do modelo ativo
- quarentena seletiva de slices ruins
- priorizacao de ligas para manutencao
- retreino focal automatico

Na interface operacional, o modulo de Telegram cobre:

- onboarding e registro de chat
- menu principal e callbacks inline
- comandos publicos vs administrativos
- publicacao de mensagens automaticas do scheduler

## Regra pratica para usar daqui pra frente

Use Sema antes de editar Python quando a mudanca envolver:

- novo fluxo operacional
- estados ou transicoes de previsoes
- novos contratos de notificacao
- comandos mais complexos do Telegram
- garantias que hoje estao espalhadas em if solto no codigo
- geracao de borda publica ou codigo derivado

Pode ir direto no Python quando for:

- bug pequeno e localizado
- ajuste de threshold
- mudanca de texto
- fix pontual sem impacto semantico maior

## Fluxo recomendado

Antes de editar um `.sema`:

1. Rode `sema contexto-ia sema/operacao_futebot.sema`
2. Rode `sema ast sema/operacao_futebot.sema --json`
3. Rode `sema ir sema/operacao_futebot.sema --json`

Se a mudanca for em manutencao/quarentena:

1. Rode `sema contexto-ia sema/quarentena_retreino_focal.sema`
2. Rode `sema ast sema/quarentena_retreino_focal.sema --json`
3. Rode `sema ir sema/quarentena_retreino_focal.sema --json`

Se a mudanca for em comandos, menu ou callbacks do bot:

1. Rode `sema contexto-ia sema/telegram_operacao.sema`
2. Rode `sema ast sema/telegram_operacao.sema --json`
3. Rode `sema ir sema/telegram_operacao.sema --json`

Depois de editar:

1. Rode `sema formatar sema/operacao_futebot.sema`
2. Rode `sema validar sema/operacao_futebot.sema --json`
3. Se der ruim, rode `sema diagnosticos sema/operacao_futebot.sema --json`
4. Feche com `sema verificar sema`

Atalho pelo projeto:

- `npm run sema:contexto`
- `npm run sema:validar`
- `npm run sema:verificar`

Se a tarefa pedir codigo derivado:

1. Rode `sema compilar sema --alvo python --saida gerado/sema-python`
2. Compare o contrato compilado com a implementacao real antes de incorporar

## Convencao para novos modulos

Preferencia de pasta:

- `sema/operacao_*.sema` para fluxos do bot
- `sema/dominio_*.sema` para entidades e regras mais estaveis
- `sema/integracao_*.sema` para contratos com APIs externas

## O que nao fazer

- nao inventar sintaxe fora dos exemplos oficiais
- nao tratar Sema como gerador magico de UI
- nao sair gerando codigo derivado sem validar IR e diagnosticos
- nao deixar contrato `.sema` velho e codigo Python seguindo outra verdade

## Proximo passo natural

Se essa integracao se provar util, os melhores proximos candidatos para modelagem sao:

- contratos do Telegram e callbacks principais
- ciclo de vida de uma previsao
- fluxo de retreino focal e quarentena
- discovery semanal de strategies

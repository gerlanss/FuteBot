# FuteBot

FuteBot e um bot de analise de futebol orientado por dados, com pipeline automatizado para selecao pre-live, revisao contextual perto do horario da partida, monitoramento live e manutencao continua dos modelos.

O projeto combina:

- modelos por liga e por grupo de mercado
- scanner diario com filtro estatistico
- strategy gate por slice
- revisao final perto do kickoff
- automacao via scheduler e Telegram
- retreino e discovery de strategies

## Visao geral da arquitetura

O fluxo principal do sistema passa por quatro blocos:

1. coleta e persistencia de dados
2. previsao e filtro de mercados
3. orquestracao operacional
4. manutencao de modelos e strategies

### 1. Dados e persistencia

Responsavel por baixar fixtures, estatisticas e armazenar o estado operacional do bot.

Arquivos principais:

- `data/database.py`: camada SQLite com fixtures, stats, predictions, strategies, auditoria e estado operacional
- `data/bulk_download.py`: carga inicial e atualizacao incremental de dados
- `services/apifootball.py`: integracao com API-Football
- `services/odds_api.py`: integracao de odds e contexto de mercado

### 2. Predicao e selecao

Aqui vive o nucleo do bot.

O `Predictor` carrega modelos por liga e gera probabilidades para diferentes mercados. O `Scanner` transforma essas previsoes em candidatos operacionais, aplica thresholds, strategy gate, filtros anti-conflito e monta o radar do dia.

Arquivos principais:

- `models/predictor.py`: inferencia por liga
- `models/features.py`: extracao de features base
- `models/feature_factory.py`: compatibilidade e composicao de features
- `pipeline/scanner.py`: selecao de jogos e mercados

### 3. Orquestracao

Coordena a operacao automatica do produto.

O scheduler executa:

- radar diario
- liberacao final em janela proxima ao jogo
- check ao vivo
- coleta de resultados
- relatorios operacionais
- rotinas de manutencao

Arquivos principais:

- `pipeline/scheduler.py`: agendamentos e jobs recorrentes
- `services/telegram_bot.py`: comandos, mensagens e interface operacional via Telegram
- `bot.py`: entrypoint do processo principal

### 4. Aprendizado e manutencao

O bot nao depende apenas de um treino unico. Ele possui rotinas de evolucao e manutencao para manter strategies e modelos utilizaveis ao longo do tempo.

Arquivos principais:

- `models/trainer.py`: treino base dos modelos
- `models/autotuner.py`: retreino focal e ajuste por liga
- `models/market_discovery.py`: discovery de slices e strategies por mercado
- `models/learner.py`: saude do modelo, degradacao e quarentena de slices

## Como o fluxo funciona

### 1. Radar pre-live

O scanner busca jogos elegiveis nas ligas configuradas e gera mercados candidatos com base nas probabilidades dos modelos.

Depois disso ele aplica:

- confianca minima
- strategy gate
- deduplicacao por categoria
- remocao de conflitos

O resultado vira o radar inicial do dia.

### 2. Revisao final perto do jogo

Os candidatos aprovados no radar nao sao necessariamente liberados imediatamente.

Na janela proxima ao kickoff, o sistema revisita os jogos, enriquece o contexto e decide se cada mercado:

- continua de pe
- deve ser barrado
- entra em combinacoes validas

### 3. Operacao live

Depois da liberacao, o scheduler acompanha os jogos ativos, monitora sinais live e atualiza o status operacional das entradas.

Essa camada inclui:

- checagem ao vivo
- cancelamento de leitura quando o contexto degrada
- deduplicacao de sinais live
- resolucao de resultados

### 4. Manutencao continua

Quando slices perdem desempenho, o sistema pode:

- colocar slices em quarentena
- manter o restante do scanner operando
- disparar retreino focal
- executar discovery por liga e mercado

Isso evita retrabalhar tudo quando o problema esta concentrado em poucos segmentos.

## Estrutura do repositorio

- `bot.py`: inicializacao do bot e do scheduler
- `config.py`: configuracoes por ambiente e parametros de operacao
- `data/`: persistencia e carga de dados
- `deploy/`: exemplos de servico e deploy
- `docs/`: documentacao complementar
- `models/`: treino, predictor, tuner, discovery e aprendizado
- `pipeline/`: scanner, scheduler e coleta operacional
- `scripts/`: utilitarios tecnicos de apoio
- `services/`: integracoes externas e interface Telegram

## Componentes mais importantes

- `pipeline/scanner.py`: coracao do produto
- `pipeline/scheduler.py`: automacao e jobs recorrentes
- `models/predictor.py`: inferencia por liga
- `models/autotuner.py`: retreino focal
- `models/market_discovery.py`: descoberta de strategies
- `services/telegram_bot.py`: superficie operacional do bot
- `data/database.py`: estado e auditoria

## Requisitos

- Python 3.11+
- SQLite
- credenciais validas para as APIs configuradas
- bot do Telegram configurado

Dependencias Python estao em `requirements.txt`.

## Configuracao rapida

1. Crie um ambiente virtual.
2. Instale as dependencias com `pip install -r requirements.txt`.
3. Copie `.env.example` para `.env`.
4. Preencha as credenciais necessarias.
5. Execute `python bot.py`.

## Documentacao complementar

- `docs/como-o-futebot-funciona.md`: explicacao detalhada do pipeline e da logica operacional
- `docs/sema-no-futebot.md`: como usar Sema no projeto e quando modelar antes de editar Python

## Sema no projeto

O FuteBot agora tambem tem uma camada inicial de modelagem com Sema para fluxos operacionais mais sensiveis.

Arquivo inicial:

- `sema/operacao_futebot.sema`
- `sema/quarentena_retreino_focal.sema`
- `sema/telegram_operacao.sema`
- `sema/integracoes_externas.sema`

Use Sema principalmente quando a mudanca envolver:

- contratos de operacao
- fluxo entre scanner, liberacao T-30, live e relatorios
- estados de previsoes
- garantias entre notificacao, persistencia e auditoria

Fluxo curto recomendado:

1. `sema contexto-ia sema/operacao_futebot.sema`
2. `sema ir sema/operacao_futebot.sema --json`
3. edite o modulo
4. `sema formatar sema/operacao_futebot.sema`
5. `sema validar sema/operacao_futebot.sema --json`
6. `sema verificar sema`

Atalhos disponiveis:

- `npm run sema:contexto`
- `npm run sema:validar`
- `npm run sema:verificar`

## Declaracao de uso

Este projeto foi construido como sistema de apoio analitico e automacao operacional.

Ele nao deve ser interpretado como:

- garantia de resultado
- recomendacao financeira
- promessa de lucro
- substituto de validacao humana

Qualquer uso em ambiente real exige:

- configuracao responsavel
- validacao propria das entradas
- controle de risco
- acompanhamento continuo de desempenho

## Resumo

FuteBot nao e apenas um bot que envia tips.

Ele foi estruturado como um fluxo completo de decisao:

- coleta dados
- gera previsoes
- filtra por strategy
- revisa contexto
- opera via scheduler
- monitora saude do modelo
- reajusta slices degradados ao longo do tempo

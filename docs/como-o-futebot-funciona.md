# Como o FuteBot Funciona

## Visao geral

O FuteBot hoje opera em 3 camadas:

1. selecao estatistica por liga e mercado
2. revisao contextual perto do horario do jogo
3. operacao automatica via scheduler e Telegram

O fluxo principal passa por:

- `pipeline/scanner.py`
- `pipeline/scheduler.py`
- `data/database.py`
- `models/predictor.py`
- `services/llm_validator.py`
- `services/telegram_bot.py`


## Arquitetura geral

### 1. Modelos por liga

O bot nao usa um modelo global unico.

Cada liga possui seus proprios modelos, carregados pelo `Predictor`, com base em:

- historico de jogos
- forma recente
- mando
- medias ofensivas e defensivas
- xG
- chutes
- posse
- escanteios
- clean sheet
- failed to score
- features derivadas

Arquivo principal:

- `models/predictor.py`


### 2. Scanner diario

O scanner e o pipeline central do bot.

Arquivo:

- `pipeline/scanner.py`

Responsabilidades:

1. buscar jogos do dia nas ligas configuradas
2. gerar previsoes por mercado
3. expandir mercados possiveis
4. filtrar por confianca minima
5. aplicar strategy gate
6. remover conflitos
7. montar radar da manha
8. liberar mercados perto do jogo
9. salvar auditoria, previsoes e combos


### 3. Scheduler

O scheduler cuida da automacao.

Arquivo:

- `pipeline/scheduler.py`

Hoje ele agenda:

- bulk incremental
- coleta de resultados
- scanner diario
- liberacao final T-30
- check ao vivo
- relatorio diario
- retreino focal de quarentena
- discovery semanal
- retreino mensal


### 4. Banco

O banco SQLite centraliza estado operacional.

Arquivo:

- `data/database.py`

Tabelas principais:

- `fixtures`
- `fixture_stats`
- `team_stats`
- `predictions`
- `strategies`
- `combos`
- `scan_candidates`
- `scan_audit`
- `telegram_chats`


## Ligas e mercados

As ligas operacionais ficam em:

- `config.py`

O bot hoje trabalha com estes grupos de mercado:

- resultado FT
- over/under FT
- over/under 1T
- over/under 2T
- resultado HT
- escanteios

O mercado `BTTS` foi removido do produto.

Observacao:

- features historicas como `home_btts_pct` e `away_btts_pct` ainda podem existir como insumo estatistico para outros mercados
- isso nao significa que o mercado BTTS ainda esteja ativo


## Pipeline completo

## Etapa 1: busca de fixtures

O scanner busca apenas jogos:

- das ligas configuradas
- do dia
- ainda nao iniciados

Resultado:

- lista de fixtures elegiveis para analise


## Etapa 2: previsao por mercado

Para cada fixture, o `Predictor` gera probabilidades para os mercados daquela liga.

Exemplos:

- `h2h_home`
- `over15`
- `under35`
- `over05_ht`
- `under15_2t`
- `corners_under_105`


## Etapa 3: expansao de mercados

Cada previsao vira varios mercados candidatos.

O scanner so considera mercados com confianca minima global.

Regua atual:

- confianca minima base: `65%`

Constantes em:

- `pipeline/scanner.py`


## Etapa 4: strategy gate

O `strategy gate` e o filtro estatistico mais importante.

Ele consulta a tabela `strategies` e so deixa passar um mercado se existir historico validado para:

- aquela liga
- aquele mercado
- aquela faixa de confianca
- e, quando existir, as condicoes de features da strategy

Exemplos de condicoes:

- media ofensiva minima
- clean sheet pct
- failed to score pct
- xG
- chutes no alvo
- model_prob

O gate hoje tem:

- compatibilidade entre nomes de features do treino e do runtime
- tolerancia numerica
- pequena folga para evitar bloqueio por borda muito fina


## Etapa 5: anti-conflito

O bot nao manda mercados contraditorios no mesmo jogo.

Exemplos de conflito:

- `over 1.5` com `under 3.5`
- `casa` com `fora`
- mercados duplicados da mesma categoria

O scanner organiza mercados por categoria e mantem apenas os mais fortes.


## Etapa 6: radar do dia

As `07:00`, o bot roda o scanner em modo de pre-selecao.

Nesse momento:

- os mercados nao sao revelados
- so os jogos promissores aparecem
- esses jogos vao para a tabela `scan_candidates`

O usuario recebe:

- quantidade de jogos analisados
- quantidade de mercados candidatos
- quantidade de jogos pre-selecionados
- lista de partidas em observacao


## Etapa 7: liberacao final T-30

Essa e a segunda fase do produto.

O scheduler roda uma liberacao automatica a cada `5` minutos.

Janela atual:

- `30 +- 5 min` antes do jogo

O que acontece nessa fase:

1. pega os candidatos pendentes do dia
2. filtra os que estao dentro da janela
3. enriquece o contexto do jogo
4. roda a revisao final do FuteBot
5. decide liberar ou barrar
6. salva auditoria
7. envia o resultado no Telegram


## Revisao final do FuteBot

Na pratica, a revisao final usa:

- contexto estatistico do proprio bot
- contexto esportivo da API-Football
- contexto externo consultado perto do jogo

Ela nao aparece para o usuario com nome de fornecedor de IA.

No produto, tudo e tratado como:

- revisao final do FuteBot

Entram nessa revisao:

- lesoes
- ausencias
- classificacao
- forma
- contexto competitivo
- clima
- gramado
- risco de rotacao
- noticias recentes
- disponibilidade de mercado em outras casas


## O que o bot explica

Depois da revisao final, o bot separa:

- mercados liberados
- mercados barrados

E explica os dois lados:

- `Motivo da liberacao`
- `Motivo do bloqueio`

Isso deixa o comportamento auditavel logo no chat.


## Odds, mercado e contexto externo

O bot nao inventa preco.

Se existir linha exata de odd:

- usa a odd real
- calcula EV

Se nao existir:

- nao fabrica um preco artificial
- segue usando o contexto do jogo para a decisao

O contexto externo ajuda a responder coisas como:

- o mercado existe em outras casas?
- o clima esta ruim?
- o gramado pode atrapalhar?
- existe risco de time misto?
- ha noticia recente relevante?


## Combos

Combos nao nascem mais cedo no dia.

Hoje eles so podem nascer depois da liberacao final.

Regras atuais:

- maximo de 3 combos
- confianca composta minima: `50%`
- confianca minima individual por tip: `65%`
- jogos diferentes
- horarios proximos

Ou seja:

- primeiro a tip simples precisa ser liberada
- so depois o combo pode existir


## Treino e manutencao

## 1. Retreino mensal

O bot ainda tem retreino mensal per-league.

Objetivo:

- atualizar os modelos salvos por liga


## 2. Retreino focal

Quando slices entram em quarentena, o sistema pode retreinar focado em ligas especificas.

Isso evita:

- gastar recursos com tudo
- mexer em ligas saudaveis

Arquivos relevantes:

- `models/autotuner.py`
- `pipeline/scheduler.py`


## 3. Discovery semanal

O bot agora tambem faz discovery semanal por:

- liga
- mercado

Objetivo:

- encontrar cobertura para mercados sem strategy ativa
- revisar slices degradados
- promover novas strategies quando houver edge suficiente

Arquivo:

- `models/market_discovery.py`

O discovery e sequencial, leve e pensado para VPS pequena.


## Quarentena por slice

O sistema nao pausa mais o bot inteiro por causa de poucos mercados ruins.

Hoje ele:

- detecta slices degradados
- coloca esses slices em quarentena
- mantem o resto do scanner funcionando

Relatorio e saude passam por:

- `models/learner.py`


## Telegram

Arquivo principal:

- `services/telegram_bot.py`

Hoje o bot separa:

### Vai para todos

- radar do dia
- liberacao final T-30
- tips liberadas
- mercados barrados com motivo
- combos
- resultados

### Vai so para admin

- erros operacionais
- saude do modelo
- retreinos
- discovery
- bulk
- alertas de manutencao


## Formato visual

Os mercados hoje usam icones por categoria:

- `⚽` gols
- `⛳` escanteios
- `🎯` resultado
- `📌` fallback

E os blocos de revisao final separam:

- `✅ Liberados`
- `🚫 Barrados na revisao`


## Estado atual do produto

Hoje o FuteBot funciona assim:

1. escaneia os jogos do dia
2. escolhe os melhores jogos para o radar
3. segura o mercado pela manha
4. revisa o contexto perto do horario real
5. libera ou barra o mercado com explicacao
6. salva tudo para auditoria
7. aprende por liga e mercado
8. mantem manutencao automatica sem derrubar o bot inteiro


## Resumo pratico

O FuteBot nao e mais apenas um bot que "manda tips".

Hoje ele e um fluxo completo de decisao:

- modelo por liga
- strategy gate por slice
- radar do dia
- revisao final T-30
- explicacao do motivo
- auditoria
- manutencao automatica

Isso deixa o sistema:

- mais rigido onde precisa
- mais transparente
- mais auditavel
- mais adaptavel ao longo do tempo

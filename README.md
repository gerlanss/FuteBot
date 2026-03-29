# FuteBot

FuteBot e um bot de analise de futebol com pipeline automatizado para:

- buscar fixtures e estatisticas
- gerar previsoes por liga e mercado
- aplicar filtros de qualidade e strategies
- revisar entradas perto do horario do jogo
- operar via Telegram com scheduler

O repositorio publico foi enxugado para expor a arquitetura e o fluxo principal sem levar junto credenciais, dados operacionais ou artefatos privados de runtime.

## O que o repositorio contem

- codigo principal do bot e do scheduler
- pipeline de scanner e liberacao
- treino por liga e autotuning
- servicos de integracao com APIs externas
- documentacao funcional do fluxo

## O que nao vai para o repositorio publico

- `.env` e credenciais
- bancos SQLite locais
- modelos gerados em runtime
- logs, caches e auditorias locais
- testes internos e scripts temporarios de operacao

## Estrutura

- `bot.py`: entrypoint da aplicacao
- `config.py`: configuracoes por ambiente
- `pipeline/scanner.py`: radar diario, selecao e liberacao
- `pipeline/scheduler.py`: automacao recorrente
- `models/`: treino, predictor, discovery e autotuner
- `services/`: integracoes externas e Telegram
- `data/database.py`: persistencia SQLite
- `docs/como-o-futebot-funciona.md`: explicacao mais detalhada do fluxo

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
4. Preencha as credenciais e IDs necessarios.
5. Execute `python bot.py`.

## Fluxo resumido

1. O scanner busca jogos elegiveis das ligas configuradas.
2. O predictor gera probabilidades por mercado.
3. O strategy gate filtra o que tem historico valido.
4. O scheduler roda janelas de revisao e liberacao.
5. O Telegram entrega radar, entradas e alertas operacionais.

## Observacoes

- A execucao real depende de chaves e dados locais que nao sao distribuidos aqui.
- Alguns scripts em `scripts/` existem para analise e manutencao interna do pipeline.

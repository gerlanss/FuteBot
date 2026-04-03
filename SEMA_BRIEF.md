# SEMA_BRIEF

Sema e IA-first. Este arquivo existe para IA achar o ponto de entrada do projeto sem ter que catar o repo inteiro feito barata tonta.

- Gerado em: `2026-04-03T18:08:11.787Z`
- Modulos: `5`

## Entrada canonica para IA

- Ordem minima: llms.txt -> SEMA_BRIEF.md -> SEMA_INDEX.json -> AGENTS.md -> README.md -> llms-full.txt
- IA pequena: llms.txt -> SEMA_BRIEF.micro.txt -> SEMA_INDEX.json -> AGENTS.md
- IA media: llms.txt -> SEMA_BRIEF.curto.txt -> SEMA_INDEX.json -> AGENTS.md -> README.md
- IA grande: llms-full.txt -> SEMA_BRIEF.md -> SEMA_INDEX.json -> AGENTS.md -> README.md

## Guia por capacidade

- pequena: IA gratuita ou com contexto curto. Leia so o cartao semantico e o briefing minimo. Artefatos: resumo.micro.txt, briefing.min.json, prompt-curto.txt.
- media: IA com contexto medio. Aguenta resumo expandido, briefing minimo e drift. Artefatos: resumo.curto.txt, briefing.min.json, drift.json, prompt-curto.txt.
- grande: IA com contexto grande ou tool use. Pode consumir o pacote completo. Artefatos: README.md, resumo.md, briefing.json, drift.json, ir.json, ast.json.

## Modulos

### futebot.previsao
- Faz: governa 5 task(s) com foco em registrar candidatos radar
- Publico: nenhum
- Tocar: c:\GitHub\FuteBot\data\database.py, c:\GitHub\FuteBot\models\learner.py, c:\GitHub\FuteBot\pipeline\scheduler.py
- Score: 75 | Confianca: alta | Risco: alto
- Lacunas: audit_ausente, authz_frouxa, dados_nao_classificados, execucao_critica_sem_bloco (+2)

### futebot.integracoes
- Faz: governa 5 task(s) com foco em consultar status api football
- Publico: nenhum
- Tocar: c:\GitHub\FuteBot\data\bulk_download.py, c:\GitHub\FuteBot\services\apifootball.py, c:\GitHub\FuteBot\services\odds_api.py
- Score: 75 | Confianca: alta | Risco: alto
- Lacunas: audit_ausente, authz_frouxa, dados_nao_classificados, execucao_critica_sem_bloco (+2)

### futebot.operacao
- Faz: governa 4 task(s) com foco em gerar radar pre live
- Publico: nenhum
- Tocar: c:\GitHub\FuteBot\pipeline\scheduler.py
- Score: 75 | Confianca: alta | Risco: alto
- Lacunas: audit_ausente, authz_frouxa, dados_nao_classificados, execucao_critica_sem_bloco (+2)

### futebot.quarentena
- Faz: governa 4 task(s) com foco em avaliar degradacao modelo
- Publico: nenhum
- Tocar: c:\GitHub\FuteBot\models\learner.py, c:\GitHub\FuteBot\pipeline\scanner.py, c:\GitHub\FuteBot\pipeline\scheduler.py
- Score: 75 | Confianca: alta | Risco: alto
- Lacunas: audit_ausente, authz_frouxa, dados_nao_classificados, execucao_critica_sem_bloco (+2)

### futebot.telegram
- Faz: governa 4 task(s) com foco em registrar chat e menu
- Publico: nenhum
- Tocar: c:\GitHub\FuteBot\services\telegram_bot.py
- Score: 75 | Confianca: alta | Risco: alto
- Lacunas: audit_ausente, authz_frouxa, dados_nao_classificados, execucao_critica_sem_bloco (+2)

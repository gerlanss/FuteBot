"""
Validador LLM - "segundo par de olhos" para tips do XGBoost.

Usa o DeepSeek (API compativel com OpenAI) para analisar contexto
qualitativo que o modelo estatistico nao captura:
  - Lesoes/desfalques de jogadores-chave
  - Importancia do jogo (posicao na classificacao)
  - Padroes de confronto direto
  - Forma recente contextualizada
  - Relacao entre odd de referencia e EV estimado

Fluxo:
  Scanner -> XGBoost -> Strategy Gate -> Odds/EV -> LLM Validator -> Telegram
  O LLM recebe confianca do modelo + contexto + odd/EV de referencia.

Configuracao (.env):
  DEEPSEEK_API_KEY=sk-xxx
  USE_LLM_VALIDATION=True

Custo estimado: ~$0.50-2/mes (DeepSeek e ~10x mais barato que GPT-4o-mini).
"""

import json
import requests
import time
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    USE_LLM_VALIDATION,
)
from services.apifootball import (
    lesoes_fixture,
    previsao_api,
    classificacao,
)
from services.gemini_lookup import GeminiMarketLookup

# Timeout para chamada à API do DeepSeek (segundos)
_TIMEOUT = 30

# ══════════════════════════════════════════════
#  PROMPT DO SISTEMA — instruções fixas para o LLM
# ══════════════════════════════════════════════

SYSTEM_PROMPT = """Voce e um analista de futebol especializado em previsoes esportivas.
Seu trabalho e validar tips geradas por um modelo estatistico XGBoost.

Contexto:
- O modelo XGBoost foi treinado com dados historicos e gera probabilidades calibradas.
- As tips ja passaram por um Strategy Gate (filtro de accuracy historica por liga x mercado).
- Voce e o ultimo filtro antes do usuario receber a tip.
- Tambem recebera odd de referencia e EV estimado. Use isso como sinal de valor.

Regras:
1. Analise: confianca do modelo, odd de referencia, EV estimado, lesoes, classificacao, forma recente e importancia do jogo.
2. Responda SEMPRE em JSON puro (sem markdown, sem ```), neste formato exato:
   {"decisao": "APPROVE" ou "REJECT", "confianca": 0.0 a 1.0, "motivo": "explicacao em PT-BR (2-3 frases)"}
3. APPROVE = a tip faz sentido dado o contexto e o preco. REJECT = ha fatores que invalidam a tip.
4. Seja criterioso e conservador quando houver sinais de armadilha - o Strategy Gate nao substitui contexto humano.
5. Foque em fatores que o modelo NAO ve: lesoes de titulares, jogos sem importancia,
   times poupando, contexto de classificacao, derbi/classico.
6. Ausencia de informacao NAO e motivo suficiente para REJECT. So barre a tip quando houver fator negativo concreto.
7. Nunca invente dados - use apenas o que foi fornecido.
8. ATENCAO ao CONTEXTO TEMPORAL: a data do jogo sera informada. Use o numero de jogos
   disputados na classificacao para entender a fase da temporada.
9. ATENCAO ao FORMATO DA COMPETICAO: sera informado se e liga, copa ou grupos.
10. O campo 'motivo' deve ter 2-3 frases completas explicando sua analise. Nao corte.
11. ADVOGADO DO DIABO: Se a confianca do modelo for ACIMA DE 80% em mercados de gols
    (BTTS, Over 1.5, Over 2.5, Over 3.5), desconfie e procure motivos para under/sem gols.
12. CONFIANCA BAIXA: Se a confianca do modelo for ABAIXO DE 55%, REJECT automaticamente.
13. QUALIDADE > QUANTIDADE: Em caso de duvida, so rejeite se voce conseguir apontar um fator esportivo concreto contra a tip.
14. EV E PRECO SAO PARAMETROS, NAO VETO ISOLADO: use EV e odd como sinais auxiliares de valor, sem transformar sua ausencia em motivo automatico de rejeicao.
15. Se o EV estimado for menor ou igual a 0%, trate isso como sinal negativo importante, mas ainda julgue o contexto geral antes de decidir.
16. Se odd/EV nao estiverem disponiveis, NAO use isso como razao principal para REJECT. Julgue a tip principalmente por confianca do modelo e contexto esportivo.
17. DESFALQUES PESAM: Se houver ausencias importantes no time favorecido pela tip, procure ativamente motivos para REJECT.
18. ARMADILHA DE FAVORITO: Em mercado de vencedor/casa/fora, desfalques relevantes no favorito, forma inconsistente ou odd baixa exigem REJECT por padrao.
19. MERCADO DE GOLS: So prefira REJECT para overs quando houver sinal concreto de criacao comprometida, desfalque ofensivo relevante ou contexto realmente truncado.
20. Sua funcao e cortar armadilhas, nao justificar favoritismo.
21. O motivo precisa citar fatores concretos do jogo. Evite frases vagas como "sem contexto externo forte", "mercado sensivel" ou "cheiro de armadilha" sem explicar o porque.
22. Se rejeitar, explique de forma objetiva qual fator esportivo pesou mais: desfalques, forma, tabela, estilo esperado do jogo, importancia do confronto ou perfil ofensivo/defensivo.
23. Se o contexto esportivo estiver bom, nao reprove so porque falta odd/EV.
24. Clima neutro, ausencia de noticia, rotacao desconhecida, campo sem confirmacao ou texto vago NAO sao motivos suficientes para REJECT."""


class LLMValidator:
    """
    Valida tips do XGBoost usando DeepSeek como segundo par de olhos.

    Uso:
        validator = LLMValidator()
        resultado = validator.validar_tip(oportunidade, contexto_extra)
        if resultado["decisao"] == "APPROVE":
            # Salvar tip
    """

    def __init__(self):
        self.ativo = USE_LLM_VALIDATION and bool(DEEPSEEK_API_KEY)
        self.gemini_lookup = GeminiMarketLookup()
        if self.ativo:
            print("[LLMValidator] ✅ Validação LLM ativa (DeepSeek)")
        else:
            razao = "key ausente" if not DEEPSEEK_API_KEY else "desativado no config"
            print(f"[LLMValidator] ⚠️ Desativado ({razao}) — tips passam direto")

    def validar_tip(self, oportunidade: dict, contexto: dict = None) -> dict:
        """
        Valida uma tip via LLM.

        Parâmetros:
            oportunidade: dict com dados da tip (mercado, prob_modelo, times, etc.)
            contexto: dict opcional com dados extras (lesões, classificação, etc.)

        Retorna:
            {"decisao": "APPROVE"|"REJECT"|"SKIP",
             "confianca": float 0-1,
             "motivo": str}
        """
        # Se validação desativada, aprova tudo (bypass)
        if not self.ativo:
            return {"decisao": "APPROVE", "confianca": 1.0, "motivo": "LLM desativado"}

        # Montar prompt com dados do jogo
        prompt = self._montar_prompt(oportunidade, contexto or {})

        # Chamar DeepSeek
        try:
            resposta = self._chamar_deepseek(prompt)
            return self._parsear_resposta(resposta)
        except Exception as e:
            print(f"[LLMValidator] ❌ Erro na chamada: {e}")
            # Em caso de erro, não bloqueia — aprova com aviso
            return {"decisao": "APPROVE", "confianca": 0.3,
                    "motivo": f"Erro no LLM ({e}) — aprovado por segurança"}

    def enriquecer_contexto(self, oportunidade: dict) -> dict:
        """
        Busca dados extras da API-Football para enriquecer o contexto da tip.
        Consome 2-3 requests da API-Football por jogo.

        Parâmetros:
            oportunidade: dict com fixture_id, league_id, season, round, home_name, away_name

        Retorna dict com lesões, classificação, previsão da API e round.
        """
        ctx = {}
        fixture_id = oportunidade.get("fixture_id")
        league_id = oportunidade.get("league_id")
        season = oportunidade.get("season")  # Season do fixture (ex: 2026)
        rnd = oportunidade.get("round", "")  # Round do fixture (ex: "Regular Season - 4")

        # Propagar round para o prompt
        if rnd:
            ctx["round"] = rnd

        # 1. Lesões/desfalques (1 request)
        if fixture_id:
            try:
                lesoes = lesoes_fixture(fixture_id)
                if lesoes:
                    ctx["lesoes"] = []
                    for l in lesoes[:10]:  # Limitar a 10 jogadores
                        player = l.get("player", {})
                        team = l.get("team", {})
                        ctx["lesoes"].append({
                            "jogador": player.get("name", "?"),
                            "time": team.get("name", "?"),
                            "tipo": player.get("type", "?"),
                            "motivo": player.get("reason", "?"),
                        })
                    print(f"[LLMValidator] 🏥 {len(ctx['lesoes'])} lesões encontradas")
            except Exception as e:
                print(f"[LLMValidator] ⚠️ Erro buscando lesões: {e}")

        # 2. Classificação (1 request) — usar season do fixture, não DEFAULT_SEASON
        if league_id:
            try:
                tabela = classificacao(league_id, season=season) if season else classificacao(league_id)
                if tabela:
                    # Extrair posição dos dois times
                    home_name = oportunidade.get("home_name", "").lower()
                    away_name = oportunidade.get("away_name", "").lower()
                    ctx["classificacao"] = []
                    for pos in tabela:
                        team_name = pos.get("team", {}).get("name", "").lower()
                        if home_name in team_name or team_name in home_name or \
                           away_name in team_name or team_name in away_name:
                            ctx["classificacao"].append({
                                "time": pos.get("team", {}).get("name", "?"),
                                "posicao": pos.get("rank", "?"),
                                "pontos": pos.get("points", 0),
                                "jogos": pos.get("all", {}).get("played", 0),
                                "forma": pos.get("form", ""),
                                "gd": pos.get("goalsDiff", 0),
                            })
            except Exception as e:
                print(f"[LLMValidator] ⚠️ Erro buscando classificação: {e}")

        # 3. Previsão da API-Football (1 request)
        if fixture_id:
            try:
                pred_api = previsao_api(fixture_id)
                if pred_api:
                    predictions = pred_api.get("predictions", {})
                    comparison = pred_api.get("comparison", {})
                    ctx["api_prediction"] = {
                        "vencedor": predictions.get("winner", {}).get("name", "?"),
                        "conselho": predictions.get("advice", ""),
                        "goals_home": predictions.get("goals", {}).get("home", "?"),
                        "goals_away": predictions.get("goals", {}).get("away", "?"),
                    }
                    if comparison:
                        ctx["api_comparison"] = {
                            "forma": comparison.get("form", {}),
                            "ataque": comparison.get("att", {}),
                            "defesa": comparison.get("def", {}),
                            "total": comparison.get("total", {}),
                        }
            except Exception as e:
                print(f"[LLMValidator] ⚠️ Erro buscando previsão API: {e}")

        try:
            gemini_data = self.gemini_lookup.lookup_market(oportunidade)
            if gemini_data.get("enabled"):
                ctx["market_lookup"] = gemini_data
        except Exception as e:
            print(f"[LLMValidator] ⚠️ Erro no Gemini lookup: {e}")

        return ctx

    # Mapeamento league_id → formato da competição (para contexto do LLM)
    _FORMATOS_LIGA = {
        # Brasil
        71:  ("Liga", "Brasileirão Série A", "Pontos corridos, 38 rodadas, abr-dez"),
        72:  ("Liga", "Brasileirão Série B", "Pontos corridos, 38 rodadas, abr-dez"),
        73:  ("Copa", "Copa do Brasil", "Mata-mata com ida e volta"),
        475: ("Liga", "Campeonato Paulista", "Fase de grupos + mata-mata, jan-abr"),
        479: ("Liga", "Campeonato Carioca", "Fase de grupos + mata-mata, jan-abr"),
        476: ("Liga", "Campeonato Mineiro", "Fase de grupos + mata-mata, jan-abr"),
        478: ("Liga", "Campeonato Gaúcho", "Fase de grupos + mata-mata, jan-abr"),
        # Europa — Top 5
        39:  ("Liga", "Premier League", "Pontos corridos, 38 rodadas, ago-mai"),
        140: ("Liga", "La Liga", "Pontos corridos, 38 rodadas, ago-mai"),
        135: ("Liga", "Serie A (Itália)", "Pontos corridos, 38 rodadas, ago-mai"),
        78:  ("Liga", "Bundesliga", "Pontos corridos, 34 rodadas, ago-mai"),
        61:  ("Liga", "Ligue 1", "Pontos corridos, 34 rodadas, ago-mai"),
        # Europa — Copas
        2:   ("Copa/Grupos", "Champions League", "Fase de liga + mata-mata"),
        3:   ("Copa/Grupos", "Europa League", "Fase de liga + mata-mata"),
        848: ("Copa/Grupos", "Conference League", "Fase de liga + mata-mata"),
        # América do Sul
        13:  ("Copa/Grupos", "Copa Libertadores", "Fase de grupos + mata-mata"),
        11:  ("Copa/Grupos", "Copa Sul-Americana", "Fase de grupos + mata-mata"),
        128: ("Liga", "Liga Profesional (Argentina)", "Pontos corridos, ~27 rodadas, fev-dez"),
        239: ("Liga", "Liga BetPlay (Colômbia)", "Apertura/Finalización + playoffs, fev-dez"),
        265: ("Liga", "Primera División (Chile)", "Pontos corridos, 30 rodadas, fev-dez"),
        157: ("Liga", "Primera División (Paraguai)", "Apertura/Clausura, fev-dez"),
        # América do Norte
        253: ("Liga", "MLS", "Conferências + playoffs, fev-dez"),
        262: ("Liga", "Liga MX", "Apertura/Clausura + liguilla, jul-mai"),
        # Ásia / Oceania
        98:  ("Liga", "J1 League (Japão)", "Pontos corridos, 34 rodadas, fev-dez"),
        292: ("Liga", "K League 1 (Coreia)", "Pontos corridos + split, fev-dez"),
        307: ("Liga", "Saudi Pro League", "Pontos corridos, 30 rodadas, ago-mai"),
        188: ("Liga", "A-League (Austrália)", "Pontos corridos + playoffs, out-mai"),
        # Seleções
        1:   ("Copa", "Copa do Mundo FIFA", "Fase de grupos + mata-mata, a cada 4 anos"),
    }

    def _montar_prompt(self, oportunidade: dict, contexto: dict) -> str:
        """
        Monta o prompt do usuário com todos os dados disponíveis do jogo.
        Estruturado para ser claro e parseable pelo LLM.
        Opera sem odds/EV - foca em confianca do modelo + contexto.
        Inclui data do jogo, formato da competição e fase da temporada.
        """
        from datetime import datetime as dt

        home = oportunidade.get("home_name", "?")
        away = oportunidade.get("away_name", "?")
        mercado = oportunidade.get("mercado", "?")
        descricao = oportunidade.get("descricao", mercado)
        prob = oportunidade.get("prob_modelo", 0)
        league_id = oportunidade.get("league_id")
        odd_ref = oportunidade.get("odd_pinnacle")
        ev_ref = oportunidade.get("ev_percent")
        casa_ref = oportunidade.get("odd_fonte", "")

        # Data do jogo formatada
        date_str = oportunidade.get("date", "")
        data_jogo = ""
        if date_str:
            try:
                data_jogo = dt.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%d/%m/%Y")
            except Exception:
                data_jogo = date_str[:10]

        # Formato e nome da competição
        formato_info = self._FORMATOS_LIGA.get(league_id, ("Desconhecido", "Liga desconhecida", ""))
        tipo_comp, nome_comp, desc_formato = formato_info

        # Probabilidades completas do modelo
        prob_lines = []
        for key, label in [
            ("prob_home", "Vitória casa"), ("prob_draw", "Empate"), ("prob_away", "Vitória fora"),
            ("prob_over25", "Over 2.5"), ("prob_btts_yes", "BTTS Sim"),
            ("prob_ht_home", "1T Casa"), ("prob_ht_draw", "1T Empate"), ("prob_ht_away", "1T Fora"),
        ]:
            val = oportunidade.get(key)
            if val and val > 0:
                prob_lines.append(f"  {label}: {val:.1%}")

        # Rodada / fase do torneio
        rodada = oportunidade.get("round", "") or contexto.get("round", "")

        prompt = f"""JOGO: {home} vs {away}
DATA DO JOGO: {data_jogo}
COMPETI??O: {nome_comp} ({tipo_comp}) - {desc_formato}
RODADA: {rodada if rodada else "N/D"}
TIP DO MODELO: {descricao}
CONFIAN?A DO MODELO: {prob:.1%}
ODD DE REFERENCIA: {f"{odd_ref:.2f} ({casa_ref or 'referencia'})" if odd_ref else "N/D"}
EV ESTIMADO: {f"{ev_ref:+.1f}%" if ev_ref is not None else "N/D"}

PROBABILIDADES COMPLETAS DO XGBOOST:
{chr(10).join(prob_lines)}

NOTA: Use odd/EV como sinal de valor junto com o contexto abaixo.
"""

        # Adicionar lesões ao prompt
        lesoes = contexto.get("lesoes", [])
        if lesoes:
            prompt += "\nLESÕES/DESFALQUES:\n"
            for l in lesoes:
                prompt += f"  🏥 {l['jogador']} ({l['time']}) — {l['tipo']}: {l['motivo']}\n"
        else:
            prompt += "\nLESÕES: Nenhuma informação disponível\n"

        # Adicionar classificação com contexto de fase da temporada
        classif = contexto.get("classificacao", [])
        if classif:
            # Inferir fase da temporada pelo nº de jogos disputados
            max_jogos = max((c.get('jogos', 0) for c in classif), default=0)
            if max_jogos <= 5:
                fase = f"INÍCIO DE TEMPORADA ({max_jogos} jogos disputados — classificação NÃO é representativa)"
            elif max_jogos <= 15:
                fase = f"MEIO DA PRIMEIRA FASE ({max_jogos} jogos disputados)"
            elif max_jogos <= 30:
                fase = f"SEGUNDA METADE DA TEMPORADA ({max_jogos} jogos disputados)"
            else:
                fase = f"RETA FINAL ({max_jogos} jogos disputados)"

            prompt += f"\nCLASSIFICAÇÃO (FASE: {fase}):\n"
            for c in classif:
                prompt += (f"  {c['time']}: {c['posicao']}º lugar, "
                          f"{c['pontos']}pts, {c['jogos']}J, "
                          f"GD {c['gd']:+d}, forma: {c['forma']}\n")

        # Adicionar previsão da API-Football
        api_pred = contexto.get("api_prediction")
        if api_pred:
            prompt += (f"\nPREVISÃO API-FOOTBALL:\n"
                      f"  Favorito: {api_pred['vencedor']}\n"
                      f"  Conselho: {api_pred['conselho']}\n"
                      f"  Gols esperados: {api_pred['goals_home']} - {api_pred['goals_away']}\n")

        api_comp = contexto.get("api_comparison")
        if api_comp:
            total = api_comp.get("total", {})
            if total:
                prompt += f"  Força geral: Casa {total.get('home', '?')} vs Fora {total.get('away', '?')}\n"

        market_lookup = contexto.get("market_lookup")
        if market_lookup and market_lookup.get("enabled"):
            casas = market_lookup.get("bookmakers") or []
            flags = market_lookup.get("risk_flags") or []
            prompt += "\nCONTEXTO EXTERNO (Gemini + Google Search):\n"
            prompt += f"  Mercado encontrado fora da fonte principal: {'SIM' if market_lookup.get('market_found') else 'NAO'}\n"
            if casas:
                prompt += f"  Casas citadas: {', '.join(casas[:8])}\n"
            if market_lookup.get("market_summary"):
                prompt += f"  Mercado: {market_lookup['market_summary']}\n"
            if market_lookup.get("weather_summary"):
                prompt += f"  Clima: {market_lookup['weather_summary']}\n"
            if market_lookup.get("field_conditions"):
                prompt += f"  Campo/gramado: {market_lookup['field_conditions']}\n"
            if market_lookup.get("rotation_risk"):
                prompt += f"  Risco de rotacao: {market_lookup['rotation_risk']}\n"
            if market_lookup.get("motivation_context"):
                prompt += f"  Motivacao/competicao: {market_lookup['motivation_context']}\n"
            if market_lookup.get("news_summary"):
                prompt += f"  Noticias recentes: {market_lookup['news_summary']}\n"
            if flags:
                prompt += f"  Flags de risco: {', '.join(flags[:8])}\n"
            if market_lookup.get("context_summary"):
                prompt += f"  Resumo externo: {market_lookup['context_summary']}\n"
            fontes = market_lookup.get("sources") or []
            if fontes:
                prompt += "  Fontes:\n"
                for item in fontes[:5]:
                    prompt += f"    - {item.get('title', item.get('url', '?'))}: {item.get('url', '?')}\n"

        prompt += "\nDecida: APPROVE ou REJECT? Responda em JSON."
        return prompt

    def _chamar_deepseek(self, prompt: str) -> str:
        """
        Faz chamada à API do DeepSeek (compatível com OpenAI).
        Retorna o texto da resposta do modelo.
        """
        url = f"{DEEPSEEK_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,      # Baixa — queremos respostas consistentes
            "max_tokens": 500,       # Espaço para motivo detalhado sem cortar
            "stream": False,
        }

        inicio = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - inicio

        # Extrair texto da resposta
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Resposta vazia do DeepSeek")

        texto = choices[0].get("message", {}).get("content", "").strip()
        tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
        tokens_out = data.get("usage", {}).get("completion_tokens", 0)

        print(f"[LLMValidator] 🤖 DeepSeek respondeu em {elapsed:.1f}s "
              f"({tokens_in}+{tokens_out} tokens)")

        return texto

    def _parsear_resposta(self, texto: str) -> dict:
        """
        Parseia a resposta JSON do LLM.
        Robusto a variações de formato (markdown blocks, espaços, etc.).
        """
        # Limpar possíveis blocos de código markdown
        texto = texto.strip()
        if texto.startswith("```"):
            # Remove ```json e ``` de volta
            lines = texto.split("\n")
            texto = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        try:
            result = json.loads(texto)
        except json.JSONDecodeError:
            # Tentar encontrar JSON dentro do texto
            import re
            match = re.search(r'\{[^}]+\}', texto, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    return {"decisao": "APPROVE", "confianca": 0.3,
                            "motivo": f"Resposta não-JSON do LLM: {texto[:100]}"}
            else:
                return {"decisao": "APPROVE", "confianca": 0.3,
                        "motivo": f"Resposta inválida do LLM: {texto[:100]}"}

        # Normalizar campos
        decisao = result.get("decisao", "APPROVE").upper().strip()
        if decisao not in ("APPROVE", "REJECT"):
            decisao = "APPROVE"

        confianca = result.get("confianca", 0.5)
        try:
            confianca = float(confianca)
            confianca = max(0.0, min(1.0, confianca))
        except (ValueError, TypeError):
            confianca = 0.5

        motivo = result.get("motivo", "Sem motivo informado")

        return {
            "decisao": decisao,
            "confianca": round(confianca, 2),
            "motivo": str(motivo),
        }

    def validar_lote(self, oportunidades: list[dict]) -> list[dict]:
        """
        Valida um lote de oportunidades, enriquecendo contexto e chamando o LLM.

        Parâmetros:
            oportunidades: lista de dicts com dados de cada tip

        Retorna lista de dicts com campo extra 'llm_validacao' adicionado.
        """
        if not self.ativo:
            # Bypass — adiciona flag e retorna todas
            for op in oportunidades:
                op["llm_validacao"] = {
                    "decisao": "APPROVE", "confianca": 1.0,
                    "motivo": "LLM desativado"
                }
            return oportunidades

        aprovadas = []
        rejeitadas = 0

        for op in oportunidades:
            # Enriquecer com dados da API-Football
            ctx = self.enriquecer_contexto(op)
            op["llm_contexto"] = ctx

            # Validar via LLM
            resultado = self.validar_tip(op, ctx)
            op["llm_validacao"] = resultado

            if resultado["decisao"] == "APPROVE":
                aprovadas.append(op)
                emoji = "✅"
            else:
                rejeitadas += 1
                emoji = "❌"

            print(f"[LLMValidator] {emoji} {op.get('home_name', '?')} vs "
                  f"{op.get('away_name', '?')} | {op.get('mercado', '?')} | "
                  f"{resultado['decisao']} ({resultado['confianca']:.0%}) — "
                  f"{resultado['motivo'][:60]}")

            # Pequena pausa entre chamadas para não sobrecarregar a API
            time.sleep(0.5)

        if rejeitadas:
            print(f"[LLMValidator] 📊 Resultado: {len(aprovadas)} aprovadas, "
                  f"{rejeitadas} rejeitadas pelo LLM")

        return aprovadas

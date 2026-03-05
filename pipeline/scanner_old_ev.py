"""
Scanner diário — orquestra o pipeline completo de análise.

Execução diária (automática via scheduler ou manual):
  1. Buscar fixtures do dia/próximos nas ligas configuradas
  2. Gerar previsões com o modelo XGBoost
  3. Filtrar oportunidades com maior edge
  4. Buscar odds reais dos jogos filtrados
  5. Calcular EV e registrar previsões no banco

Economia de créditos:
  - Etapas 1-3 usam apenas API-Football (sem custo de créditos Odds)
  - Etapa 4 (odds) só é chamada para os top jogos filtrados
  - Máximo de MAX_JOGOS_ODDS jogos buscam odds

Uso:
  from pipeline.scanner import Scanner
  scanner = Scanner()
  resultado = scanner.executar()           # Pipeline completo
  resultado = scanner.executar_dia('2026-02-25')  # Data específica
"""

from datetime import datetime, timedelta
from config import LEAGUES, MAX_JOGOS_ODDS, EV_THRESHOLD, ODDS_SPORTS_MAP
from config import ROI_PAUSE_THRESHOLD, ROI_PAUSE_MIN_BETS, TIMEZONE
from config import USE_LLM_VALIDATION
from data.database import Database
from services.apifootball import raw_request
from services.odds_api import buscar_odds_por_league_id, encontrar_odds_jogo, resumo_odds_jogo
from models.predictor import Predictor
from services.llm_validator import LLMValidator


class Scanner:
    """Pipeline diário: scan → predict → odds → EV → strategy gate."""

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.predictor = Predictor(self.db)
        self.llm = LLMValidator()
        # Carregar estratégias ativas do AutoTuner (se existirem)
        self._strategies = self.db.strategies_ativas()
        if self._strategies:
            print(f"[Scanner] 🎯 {len(self._strategies)} estratégias ativas carregadas")
        else:
            print("[Scanner] ⚠️ Sem estratégias — emitindo tips apenas com EV_THRESHOLD")

    def executar(self, data: str = None, dias_adiante: int = 2) -> dict:
        """
        Executa o pipeline completo para uma data (ou hoje + próximos dias).

        Parâmetros:
          data: data no formato YYYY-MM-DD (default: hoje)
          dias_adiante: quantos dias à frente buscar (default: 2)

        Retorna resumo com fixtures encontrados, previsões e oportunidades.
        """
        if data is None:
            data = datetime.now().strftime("%Y-%m-%d")

        print(f"🔍 Scanner iniciando para {data} (+{dias_adiante} dias)")

        # ─── GUARD RAIL: Auto-pause se modelo está degradando ───
        pausado, motivo_pausa = self._verificar_auto_pause()
        if pausado:
            print(f"\n⛔ SCANNER PAUSADO: {motivo_pausa}")
            print("   Retreine o modelo (/treinar) ou ajuste ROI_PAUSE_THRESHOLD no config.")
            return {
                "fixtures": 0, "previsoes": 0, "oportunidades": [],
                "pausado": True, "motivo_pausa": motivo_pausa,
            }

        # ─── ETAPA 1: Scan de fixtures ───
        print("\n📋 Etapa 1: Buscando fixtures...")
        fixtures = self._scan_fixtures(data, dias_adiante)
        print(f"   Encontrados: {len(fixtures)} jogos nas nossas ligas")

        if not fixtures:
            return {"fixtures": 0, "previsoes": 0, "oportunidades": []}

        # Salvar fixtures no banco
        for f in fixtures:
            self.db.salvar_fixture(f)

        # ─── ETAPA 2: Gerar previsões ───
        print("\n🤖 Etapa 2: Gerando previsões...")
        previsoes = []

        if not self.predictor.modelo_pronto():
            print("   ⚠️ Modelo não treinado. Usando predictions da API-Football.")
            previsoes = self._previsoes_api(fixtures)
        else:
            for f in fixtures:
                fix_dict = {
                    "fixture_id": f["fixture"]["id"],
                    "home_id": f["teams"]["home"]["id"],
                    "away_id": f["teams"]["away"]["id"],
                    "home_name": f["teams"]["home"]["name"],
                    "away_name": f["teams"]["away"]["name"],
                    "league_id": f["league"]["id"],
                    "season": f["league"]["season"],
                    "date": f["fixture"]["date"],
                }
                pred = self.predictor.prever_jogo(fix_dict)
                if pred:
                    previsoes.append(pred)
                    print(f"   ✅ {fix_dict['home_name']} vs {fix_dict['away_name']}: "
                          f"H={pred.get('prob_home', 0):.0%} D={pred.get('prob_draw', 0):.0%} "
                          f"A={pred.get('prob_away', 0):.0%}")
                else:
                    print(f"   ⚠️ {fix_dict['home_name']} vs {fix_dict['away_name']}: "
                          f"dados insuficientes")

        print(f"   Previsões geradas: {len(previsoes)}")

        if not previsoes:
            return {"fixtures": len(fixtures), "previsoes": 0, "oportunidades": []}

        # ─── ETAPA 3: Filtrar top jogos para odds ───
        print(f"\n🎯 Etapa 3: Filtrando top {MAX_JOGOS_ODDS} jogos...")
        top_jogos = self._filtrar_top(previsoes)
        print(f"   Selecionados: {len(top_jogos)} jogos")

        # ─── ETAPA 4: Buscar odds ───
        print("\n💰 Etapa 4: Buscando odds reais...")
        oportunidades = self._buscar_odds_e_ev(top_jogos)
        print(f"   Oportunidades com EV+: {len([o for o in oportunidades if o['ev_pct'] > 0])}")

        # ─── ETAPA 5: Filtrar por estratégias + Registrar ───
        print("\n🎯 Etapa 5: Filtrando por estratégias + salvando...")
        ev_positivas_raw = [o for o in oportunidades if o["ev_pct"] >= EV_THRESHOLD * 100]

        # Se há estratégias ativas, só emite tips aprovadas pelo strategy gate
        if self._strategies:
            ev_positivas = [o for o in ev_positivas_raw if self._strategy_check(o)]
            bloqueadas = len(ev_positivas_raw) - len(ev_positivas)
            if bloqueadas > 0:
                print(f"   🚫 {bloqueadas} tips bloqueadas pelo strategy gate")
        else:
            ev_positivas = ev_positivas_raw

        # ─── ETAPA 6: Validação LLM (segundo par de olhos) ───
        if ev_positivas and self.llm.ativo:
            print(f"\n🤖 Etapa 6: Validação LLM ({len(ev_positivas)} tips)...")
            ev_positivas = self.llm.validar_lote(ev_positivas)
            print(f"   Aprovadas pelo LLM: {len(ev_positivas)}")

        # Obter versão do modelo ativo uma vez (fora do loop)
        _treino = self.db.ultimo_treino()
        versao_atual = _treino["modelo_versao"] if _treino else "v1"

        for op in ev_positivas:
            self.db.salvar_prediction({
                "fixture_id": op["fixture_id"],
                "date": op["date"],
                "league_id": op.get("league_id"),
                "home_name": op.get("home_name"),
                "away_name": op.get("away_name"),
                "prob_home": op.get("prob_home"),
                "prob_draw": op.get("prob_draw"),
                "prob_away": op.get("prob_away"),
                "prob_over25": op.get("prob_over25"),
                "prob_btts": op.get("prob_btts"),
                "mercado": op["mercado"],
                "odd_usada": op["odd"],
                "ev_percent": op["ev_pct"],
                "bookmaker": op.get("casa", ""),
                "modelo_versao": versao_atual,
                "features": op.get("features", {}),
            })
        print(f"   Salvas: {len(ev_positivas)} previsões com EV >= {EV_THRESHOLD*100}%")

        return {
            "fixtures": len(fixtures),
            "previsoes": len(previsoes),
            "oportunidades": oportunidades,
            "ev_positivas": ev_positivas,
            "data": data,
        }

    def _scan_fixtures(self, data: str, dias_adiante: int) -> list[dict]:
        """Busca fixtures do dia e dos próximos dias nas ligas configuradas."""
        league_ids = {l["id"] for l in LEAGUES.values()}
        all_fixtures = []

        for delta in range(dias_adiante + 1):
            dia = (datetime.strptime(data, "%Y-%m-%d") + timedelta(days=delta)).strftime("%Y-%m-%d")
            r = raw_request("fixtures", {"date": dia})
            fixtures = r.get("response", [])

            # Filtrar apenas nossas ligas
            nossas = [f for f in fixtures if f.get("league", {}).get("id") in league_ids]

            # Filtrar apenas jogos não iniciados
            nossas = [f for f in nossas if f.get("fixture", {}).get("status", {}).get("short") == "NS"]

            if nossas:
                print(f"   {dia}: {len(nossas)} jogos")
            all_fixtures.extend(nossas)

        return all_fixtures

    def _previsoes_api(self, fixtures: list[dict]) -> list[dict]:
        """
        Fallback: usa endpoint predictions da API-Football quando modelo
        XGBoost ainda não foi treinado.
        """
        previsoes = []
        for f in fixtures:
            fix_id = f["fixture"]["id"]
            r = raw_request("predictions", {"fixture": fix_id})
            resp = r.get("response", [])
            if not resp:
                continue

            pred_data = resp[0]
            preds = pred_data.get("predictions", {})
            pct = preds.get("percent", {})
            comparison = pred_data.get("comparison", {})
            teams = pred_data.get("teams", {})

            # Converter percentuais string para float
            def _pct(s):
                try:
                    return float(s.replace("%", "")) / 100
                except (ValueError, AttributeError):
                    return 0.33

            prob_home = _pct(pct.get("home", "33%"))
            prob_draw = _pct(pct.get("draw", "33%"))
            prob_away = _pct(pct.get("away", "33%"))

            # Extrair médias de gols para estimar over/under
            home_team = teams.get("home", {})
            away_team = teams.get("away", {})
            h_gf = float((home_team.get("last_5", {}).get("goals", {}).get("for", {}).get("average")) or 0)
            h_ga = float((home_team.get("last_5", {}).get("goals", {}).get("against", {}).get("average")) or 0)
            a_gf = float((away_team.get("last_5", {}).get("goals", {}).get("for", {}).get("average")) or 0)
            a_ga = float((away_team.get("last_5", {}).get("goals", {}).get("against", {}).get("average")) or 0)

            # Estimativa simples de over 2.5 baseada em médias
            total_esperado = (h_gf + a_gf) / 2 + (h_ga + a_ga) / 2 if (h_gf + a_gf) > 0 else 2.5
            prob_over25 = min(0.85, max(0.15, (total_esperado - 1.5) / 3))

            previsoes.append({
                "fixture_id": fix_id,
                "home_name": f["teams"]["home"]["name"],
                "away_name": f["teams"]["away"]["name"],
                "league_id": f["league"]["id"],
                "date": f["fixture"]["date"],
                "prob_home": round(prob_home, 4),
                "prob_draw": round(prob_draw, 4),
                "prob_away": round(prob_away, 4),
                "prob_over25": round(prob_over25, 4),
                "prob_under25": round(1 - prob_over25, 4),
                "winner_pred": preds.get("winner", {}).get("comment", ""),
                "advice": preds.get("advice", ""),
                "comparison": comparison,
                "features": {
                    "home_form": comparison.get("form", {}).get("home", "0%"),
                    "away_form": comparison.get("form", {}).get("away", "0%"),
                    "home_att": comparison.get("att", {}).get("home", "0%"),
                    "away_att": comparison.get("att", {}).get("away", "0%"),
                    "h2h_home": comparison.get("h2h", {}).get("home", "0%"),
                    "h2h_away": comparison.get("h2h", {}).get("away", "0%"),
                    "source": "api_predictions",
                },
            })

            print(f"   📡 {f['teams']['home']['name']} vs {f['teams']['away']['name']}: "
                  f"H={prob_home:.0%} D={prob_draw:.0%} A={prob_away:.0%} | "
                  f"Advice: {preds.get('advice', 'N/A')}")

        return previsoes

    def _filtrar_top(self, previsoes: list[dict]) -> list[dict]:
        """
        Filtra os jogos com maior potencial de edge.
        Critérios: jogos onde o modelo tem alta confiança (>50% em qualquer outcome).
        """
        scored = []
        for p in previsoes:
            # Score de confiança = máxima probabilidade entre os outcomes
            max_prob = max(
                p.get("prob_home", 0),
                p.get("prob_draw", 0),
                p.get("prob_away", 0)
            )
            # Bonus para over/under se muito desbalanceado
            ou_edge = abs(p.get("prob_over25", 0.5) - 0.5)

            score = max_prob + ou_edge
            scored.append((score, p))

        # Ordenar por score e pegar top N
        scored.sort(key=lambda x: -x[0])
        top = [p for _, p in scored[:MAX_JOGOS_ODDS]]
        return top

    def _strategy_check(self, oportunidade: dict) -> bool:
        """
        Verifica se uma oportunidade passa no strategy gate.

        Consulta a tabela de estratégias ativas para ver se existe
        um slice (mercado × liga × confiança) que aprova esta tip.

        Estratégias são sempre por liga específica — não existe global.
        Se não houver estratégia ativa para aquele mercado+liga+confiança,
        a tip é bloqueada (princípio conservador).

        Retorna True se a tip deve ser emitida, False se bloqueada.
        """
        mercado = oportunidade.get("mercado", "")
        league_id = oportunidade.get("league_id")
        prob = oportunidade.get("prob_modelo", 0)

        # Filtrar estratégias para este mercado + liga específica
        relevantes = [
            s for s in self._strategies
            if s["mercado"] == mercado
            and s["league_id"] == league_id
        ]

        # Sem estratégia ativa para esta liga+mercado → bloquear
        if not relevantes:
            return False

        # Verificar se alguma estratégia cobre a faixa de confiança
        for s in relevantes:
            if s["conf_min"] <= prob < s["conf_max"]:
                return True  # Estratégia ativa encontrada — tip aprovada

        # Confiança fora de qualquer faixa ativa — bloquear
        return False

    def _buscar_odds_e_ev(self, previsoes: list[dict]) -> list[dict]:
        """
        Busca odds reais e calcula EV para cada previsão filtrada.
        Agrupa por liga para minimizar chamadas à API de odds.
        """
        # Agrupar previsões por liga
        por_liga = {}
        for p in previsoes:
            lid = p.get("league_id")
            if lid not in por_liga:
                por_liga[lid] = []
            por_liga[lid].append(p)

        todas_oportunidades = []

        # Buscar odds liga por liga
        for league_id, preds in por_liga.items():
            sport_key = ODDS_SPORTS_MAP.get(league_id)
            if not sport_key:
                print(f"   ⚠️ Liga {league_id} sem mapeamento de odds")
                # Sem odds, registra só a previsão sem EV
                for p in preds:
                    todas_oportunidades.append({
                        **p,
                        "mercado": "sem_odds",
                        "odd": 0,
                        "ev_pct": 0,
                        "casa": "",
                    })
                continue

            odds_lista = buscar_odds_por_league_id(league_id)
            if not odds_lista:
                print(f"   ⚠️ Nenhuma odd encontrada para liga {league_id}")
                continue

            for p in preds:
                game_odds = encontrar_odds_jogo(
                    odds_lista,
                    p.get("home_name", ""),
                    p.get("away_name", ""),
                )

                if not game_odds:
                    print(f"   ⚠️ Odds não encontradas: {p['home_name']} vs {p['away_name']}")
                    continue

                odds_resumo = resumo_odds_jogo(game_odds)
                evs = self.predictor.calcular_ev(p, odds_resumo)

                for ev in evs:
                    ev["fixture_id"] = p["fixture_id"]
                    ev["date"] = p["date"]
                    ev["league_id"] = p.get("league_id")
                    ev["home_name"] = p.get("home_name")
                    ev["away_name"] = p.get("away_name")
                    # Propagar todas as probabilidades do modelo para rastreabilidade
                    ev["prob_home"] = p.get("prob_home")
                    ev["prob_draw"] = p.get("prob_draw")
                    ev["prob_away"] = p.get("prob_away")
                    ev["prob_over25"] = p.get("prob_over25")
                    ev["prob_btts"] = p.get("prob_btts_yes")
                    ev["prob_ht_home"] = p.get("prob_ht_home")
                    ev["prob_ht_draw"] = p.get("prob_ht_draw")
                    ev["prob_ht_away"] = p.get("prob_ht_away")
                    ev["features"] = p.get("features", {})
                    todas_oportunidades.append(ev)

        # Ordenar por EV decrescente
        todas_oportunidades.sort(key=lambda x: x.get("ev_pct", 0), reverse=True)
        return todas_oportunidades

    # ══════════════════════════════════════════════
    #  GUARD RAIL: Auto-pause se modelo está degradando
    # ══════════════════════════════════════════════

    def _verificar_auto_pause(self) -> tuple[bool, str]:
        """
        Verifica se o scanner deve ser pausado por performance ruim.

        Filtra apenas previsões do modelo ATIVO (versão corrente),
        ignorando histórico de modelos anteriores. Isso garante que
        após retreino, o scanner "reseta" sem arrastar ruído antigo.

        Critérios de pausa (precisa de dados suficientes):
          1. ROI acumulado < ROI_PAUSE_THRESHOLD (-15%)
          2. Mínimo de ROI_PAUSE_MIN_BETS previsões resolvidas

        Retorna (pausado: bool, motivo: str).
        """
        # Obter versão do modelo ativo para filtrar
        treino = self.db.ultimo_treino()
        versao = treino["modelo_versao"] if treino else None

        metricas = self.db.metricas_modelo(modelo_versao=versao)
        total = metricas.get("total", 0)

        # Não pausar se não tiver dados suficientes para julgar
        if total < ROI_PAUSE_MIN_BETS:
            return False, ""

        roi = metricas.get("roi", 0)
        accuracy = metricas.get("accuracy", 0)

        # Critério principal: ROI muito negativo
        if roi < ROI_PAUSE_THRESHOLD:
            return True, (
                f"ROI acumulado ({roi:+.1f}%) abaixo do limiar ({ROI_PAUSE_THRESHOLD}%). "
                f"Accuracy: {accuracy}% em {total} previsões (modelo {versao or 'v1'})."
            )

        # Critério secundário: accuracy abaixo de chute aleatório (33% para 1x2)
        if accuracy < 30 and total >= ROI_PAUSE_MIN_BETS:
            return True, (
                f"Accuracy ({accuracy}%) abaixo de chute aleatório (33%). "
                f"ROI: {roi:+.1f}% em {total} previsões (modelo {versao or 'v1'})."
            )

        return False, ""

    def formatar_relatorio(self, resultado: dict) -> str:
        """Formata resultado do scanner para envio no Telegram."""
        # Verificar se scanner estava pausado
        if resultado.get("pausado"):
            return (
                "⛔ *FuteBot — Scanner PAUSADO*\n\n"
                f"Motivo: {resultado.get('motivo_pausa', 'modelo degradado')}\n\n"
                "O scanner detectou que o modelo está com performance abaixo do aceitável.\n"
                "Use /treinar para retreinar ou /metricas para ver detalhes.\n"
                "O scanner volta a funcionar automaticamente após retreino aprovado."
            )

        ops = resultado.get("ev_positivas", [])
        data = resultado.get("data", "hoje")

        lines = [
            f"🤖 *FuteBot — Oportunidades {data}*",
            f"📋 {resultado['fixtures']} jogos analisados | "
            f"{resultado['previsoes']} previsões geradas",
            "",
        ]

        if not ops:
            lines.append("📭 Nenhuma oportunidade com EV+ encontrada hoje.")
            return "\n".join(lines)

        for op in ops[:10]:  # Máximo 10 oportunidades
            emoji = "🔥" if op["ev_pct"] > 10 else "⚡" if op["ev_pct"] > 5 else "📊"
            # Indicar se a odd é da casa preferida (Bet365) ou alternativa
            casa_nome = op.get('casa', '?')
            is_pref = op.get('is_preferida', True)
            casa_label = casa_nome if is_pref else f"{casa_nome} ⚠️"

            # Extrair horário do jogo (campo date vem como ISO: 2026-02-24T19:00:00+00:00)
            horario = ""
            date_str = op.get("date", "")
            if date_str and "T" in date_str:
                try:
                    from datetime import datetime as dt
                    from zoneinfo import ZoneInfo
                    dt_obj = dt.fromisoformat(date_str.replace("Z", "+00:00"))
                    dt_local = dt_obj.astimezone(ZoneInfo(TIMEZONE))
                    horario = f" ⏰ {dt_local.strftime('%H:%M')}"
                except Exception:
                    horario = ""

            lines.append(
                f"{emoji} *{op.get('home_name', '?')} vs {op.get('away_name', '?')}*{horario}\n"
                f"   {op.get('descricao', op.get('mercado', '?'))}\n"
                f"   Prob: {op.get('prob_modelo', 0):.1%} | Odd: {op.get('odd', 0):.2f} ({casa_label})\n"
                f"   EV: {op['ev_pct']:+.1f}%"
            )

            # Adicionar parecer do LLM se disponível
            llm = op.get("llm_validacao")
            if llm and llm.get("decisao") != "APPROVE" or \
               (llm and llm.get("motivo") and "desativado" not in llm.get("motivo", "")):
                lines[-1] += f"\n   🤖 IA: {llm['motivo'][:60]}"
            lines[-1] += "\n"

        metricas = self.db.metricas_modelo()
        if metricas["total"] > 0:
            lines.extend([
                f"📈 *Performance acumulada:*",
                f"   Accuracy: {metricas['accuracy']}% ({metricas['acertos']}/{metricas['total']})",
                f"   ROI: {metricas['roi']:+.1f}%",
            ])

        return "\n".join(lines)


if __name__ == "__main__":
    # Execução manual para teste
    scanner = Scanner()
    resultado = scanner.executar()
    print("\n" + scanner.formatar_relatorio(resultado))

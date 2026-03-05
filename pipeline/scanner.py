"""
Scanner diário — orquestra o pipeline completo de análise.

Execução diária (automática via scheduler ou manual):
  1. Buscar fixtures do dia/próximos nas ligas configuradas
  2. Gerar previsões com o modelo XGBoost
  3. Expandir mercados (cada previsão gera N tips por mercado)
  4. Filtrar pelo Strategy Gate (liga × mercado × confiança)
  5. Validar via DeepSeek LLM (segundo par de olhos)
  6. Registrar tips aprovadas no banco
  6b. Enriquecer tips com odds Pinnacle + calcular EV
  7. Enviar Telegram com odds de referência e EV

Fluxo de EV (Pinnacle como referência):
  Após aprovar tips, busca odds Pinnacle apenas das ligas envolvidas.
  EV = (prob_modelo × odd_pinnacle) - 1. Exibido como informação, não filtra.
  Custo: ~2 créditos/liga (h2h+totals) × ~3-6 ligas/dia = 6-12 créditos/dia.

Uso:
  from pipeline.scanner import Scanner
  scanner = Scanner()
  resultado = scanner.executar()           # Pipeline completo
"""

from datetime import datetime, timedelta
from config import LEAGUES, TIMEZONE
from config import ROI_PAUSE_THRESHOLD, ROI_PAUSE_MIN_BETS
from config import ODDS_SPORTS_MAP, PREFERRED_BOOKMAKER, PREFERRED_BOOKMAKER_LABEL
from data.database import Database
from services.apifootball import raw_request
from models.predictor import Predictor
from services.llm_validator import LLMValidator


# ──────────────────────────────────────────────
# Mapeamento: mercado_id → (prob_key, descrição template)
# Cada entrada gera uma tip se a probabilidade do modelo estiver
# acima do limiar mínimo (40%). {home}/{away} são substituídos.
# ──────────────────────────────────────────────
MERCADOS = [
    # ── 1x2 Full Time ──
    ("h2h_home",  "prob_home",      "Vitória {home}"),
    ("h2h_draw",  "prob_draw",      "Empate"),
    ("h2h_away",  "prob_away",      "Vitória {away}"),
    # ── Over/Under (jogo todo) ──
    ("over25",    "prob_over25",    "Over 2.5 gols"),
    ("under25",   "prob_under25",   "Under 2.5 gols"),
    ("over15",    "prob_over15",    "Over 1.5 gols"),
    ("under15",   "prob_under15",   "Under 1.5 gols"),
    ("over35",    "prob_over35",    "Over 3.5 gols"),
    ("under35",   "prob_under35",   "Under 3.5 gols"),
    # ── Over/Under 1º Tempo ──
    ("over05_ht", "prob_over05_ht", "1T Over 0.5 gols"),
    ("under05_ht","prob_under05_ht","1T Under 0.5 gols"),
    ("over15_ht", "prob_over15_ht", "1T Over 1.5 gols"),
    ("under15_ht","prob_under15_ht","1T Under 1.5 gols"),
    # ── Over/Under 2º Tempo ──
    ("over05_2t", "prob_over05_2t", "2T Over 0.5 gols"),
    ("under05_2t","prob_under05_2t","2T Under 0.5 gols"),
    ("over15_2t", "prob_over15_2t", "2T Over 1.5 gols"),
    ("under15_2t","prob_under15_2t","2T Under 1.5 gols"),
    # ── BTTS ──
    ("btts_yes",  "prob_btts_yes",  "Ambos Marcam — Sim"),
    ("btts_no",   "prob_btts_no",   "Ambos Marcam — Não"),
    # ── 1x2 Primeiro Tempo ──
    ("ht_home",   "prob_ht_home",   "1T Vitória {home}"),
    ("ht_draw",   "prob_ht_draw",   "1T Empate"),
    ("ht_away",   "prob_ht_away",   "1T Vitória {away}"),
    # ── Escanteios ──
    ("corners_over_85",  "prob_corners_over_85",  "⛳ Over 8.5 escanteios"),
    ("corners_under_85", "prob_corners_under_85", "⛳ Under 8.5 escanteios"),
    ("corners_over_95",  "prob_corners_over_95",  "⛳ Over 9.5 escanteios"),
    ("corners_under_95", "prob_corners_under_95", "⛳ Under 9.5 escanteios"),
    ("corners_over_105", "prob_corners_over_105", "⛳ Over 10.5 escanteios"),
    ("corners_under_105","prob_corners_under_105","⛳ Under 10.5 escanteios"),
]

# Probabilidade mínima para considerar uma tip (filtra ruído)
# Elevado de 0.40 → 0.60 para evitar tips de baixa confiança
PROB_MIN = 0.60

# Confiança mínima absoluta — mesmo com strategy gate, bloqueia abaixo disso
CONF_MIN_ABSOLUTA = 0.60

# Limite de tips por fixture (evita spam no mesmo jogo)
MAX_TIPS_POR_JOGO = 2

# Limite total de tips por execução do scanner (qualidade > quantidade)
MAX_TIPS_DIA = 15

# Categorias de conflito — no máximo 1 tip por categoria por jogo.
# Se múltiplos mercados da mesma categoria passam (ex: Over 1.5 + Under 3.5),
# apenas o de maior confiança sobrevive. Isso elimina qualquer combinação
# Over+Under, resultado contraditório, etc., independente de threshold.
CATEGORIAS_CONFLITO = {
    "gols":      {"over15", "under15", "over25", "under25", "over35", "under35"},
    "gols_1t":   {"over05_ht", "under05_ht", "over15_ht", "under15_ht"},
    "gols_2t":   {"over05_2t", "under05_2t", "over15_2t", "under15_2t"},
    "resultado": {"h2h_home", "h2h_draw", "h2h_away"},
    "btts":      {"btts_yes", "btts_no"},
    "ht":        {"ht_home", "ht_draw", "ht_away"},
    "escanteios": {"corners_over_85", "corners_under_85",
                   "corners_over_95", "corners_under_95",
                   "corners_over_105", "corners_under_105"},
}

# Mapa invertido: mercado → categoria (gerado automaticamente)
_MERCADO_CATEGORIA = {}
for _cat, _mercs in CATEGORIAS_CONFLITO.items():
    for _m in _mercs:
        _MERCADO_CATEGORIA[_m] = _cat

# Mapa league_id → nome amigável (para headers no Telegram)
_LEAGUE_NOME = {v["id"]: v["nome"] for v in LEAGUES.values()}

# ──────────────────────────────────────────────
# Configuração de combos (acumuladas)
# Combina tips de jogos DIFERENTES para multiplicar odds.
# Usa odds Pinnacle reais quando disponíveis.
# ──────────────────────────────────────────────
COMBO_MAX_TOTAL = 5           # Máximo de combos total (duplas + triplas)
COMBO_DUPLAS_MAX = 4          # Teto de duplas (limitado por COMBO_MAX_TOTAL)
COMBO_TRIPLAS_MAX = 3         # Teto de triplas (limitado por COMBO_MAX_TOTAL)
COMBO_PROB_MIN = 0.45         # Confiança composta mínima (produto das probs)
COMBO_TIP_PROB_MIN = 0.62     # Prob mínima individual para entrar em combo


# ──────────────────────────────────────────────
# Mapeamento mercado_id → parâmetros da Odds API
# Permite traduzir cada tip do FuteBot para extrair a odd correta
# da Pinnacle (ou qualquer bookmaker) via The Odds API.
# Formato: mercado_id → (market_key, outcome, point)
# ──────────────────────────────────────────────
_MERCADO_ODDS_MAP = {
    # 1x2 Full Time — market_key='h2h', outcome=nome do time ou 'Draw'
    "h2h_home":  ("h2h", "HOME", None),
    "h2h_draw":  ("h2h", "Draw", None),
    "h2h_away":  ("h2h", "AWAY", None),
    # Over/Under — market_key='totals', outcome='Over'/'Under', point=linha
    "over15":    ("totals", "Over",  1.5),
    "under15":   ("totals", "Under", 1.5),
    "over25":    ("totals", "Over",  2.5),
    "under25":   ("totals", "Under", 2.5),
    "over35":    ("totals", "Over",  3.5),
    "under35":   ("totals", "Under", 3.5),
    # BTTS — market_key='btts', outcome='Yes'/'No'
    "btts_yes":  ("btts", "Yes", None),
    "btts_no":   ("btts", "No",  None),
    # 1x2 Primeiro Tempo — market_key='h2h_h1'
    "ht_home":   ("h2h_h1", "HOME", None),
    "ht_draw":   ("h2h_h1", "Draw", None),
    "ht_away":   ("h2h_h1", "AWAY", None),
}


class Scanner:
    """Pipeline diário: scan → predict → strategy gate → LLM → Telegram."""

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.predictor = Predictor(self.db)
        self.llm = LLMValidator()
        # Carregar estratégias ativas do AutoTuner (se existirem)
        self._strategies = self.db.strategies_ativas()
        if self._strategies:
            print(f"[Scanner] 🎯 {len(self._strategies)} estratégias ativas carregadas")
        else:
            print("[Scanner] ⚠️ Sem estratégias — emitindo tips por confiança do modelo")

    def executar(self, data: str = None, dias_adiante: int = 0) -> dict:
        """
        Executa o pipeline completo para uma data (ou apenas hoje).

        Parâmetros:
          data: data no formato YYYY-MM-DD (default: hoje)
          dias_adiante: quantos dias à frente buscar (default: 0 = somente hoje)

        Retorna resumo com fixtures encontrados, previsões e tips emitidas.
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

        # ─── ETAPA 2: Gerar previsões com XGBoost ───
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
                    "round": f["league"].get("round", ""),
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

        # ─── ETAPA 3: Expandir em tips por mercado ───
        print("\n🎯 Etapa 3: Expandindo mercados...")
        tips_raw = self._expandir_mercados(previsoes)
        print(f"   Tips brutas (prob >= {PROB_MIN:.0%}): {len(tips_raw)}")

        # ─── ETAPA 4: Strategy Gate ───
        print("\n🛡️ Etapa 4: Filtrando pelo Strategy Gate...")
        if self._strategies:
            tips_gate = [t for t in tips_raw if self._strategy_check(t)]
            bloqueadas = len(tips_raw) - len(tips_gate)
            print(f"   ✅ Aprovadas: {len(tips_gate)} | 🚫 Bloqueadas: {bloqueadas}")
        else:
            # Sem estratégias: usa limiar de confiança mais alto
            tips_gate = [t for t in tips_raw if t["prob_modelo"] >= CONF_MIN_ABSOLUTA]
            print(f"   ⚠️ Sem strategy gate — usando confiança >= {CONF_MIN_ABSOLUTA:.0%}: {len(tips_gate)} tips")

        # Confiança mínima absoluta (safety net mesmo com strategy gate)
        antes = len(tips_gate)
        tips_gate = [t for t in tips_gate if t["prob_modelo"] >= CONF_MIN_ABSOLUTA]
        if antes != len(tips_gate):
            print(f"   🔒 Removidas {antes - len(tips_gate)} tips abaixo de {CONF_MIN_ABSOLUTA:.0%}")

        # ─── ETAPA 4b: Resolver conflitos + limitar por jogo ───
        tips_gate = self._filtrar_conflitos_e_limites(tips_gate)

        if not tips_gate:
            print("   📭 Nenhuma tip passou nos filtros")
            return {
                "fixtures": len(fixtures), "previsoes": len(previsoes),
                "oportunidades": [], "ev_positivas": [], "data": data,
            }

        # ─── ETAPA 5: Validação LLM (DeepSeek) ───
        if tips_gate and self.llm.ativo:
            print(f"\n🤖 Etapa 5: Validação DeepSeek ({len(tips_gate)} tips)...")
            tips_aprovadas = self.llm.validar_lote(tips_gate)
            print(f"   Aprovadas pelo DeepSeek: {len(tips_aprovadas)}")
        else:
            tips_aprovadas = tips_gate

        # ─── ETAPA 5b: Limite total de tips por dia ───
        if len(tips_aprovadas) > MAX_TIPS_DIA:
            print(f"   ✂️ Cortando de {len(tips_aprovadas)} para {MAX_TIPS_DIA} tips (limite diário)")
            tips_aprovadas = tips_aprovadas[:MAX_TIPS_DIA]

        # ─── ETAPA 6: Enriquecer tips com odds Pinnacle + EV ───
        print("\n📊 Etapa 6: Buscando odds Pinnacle...")
        tips_aprovadas = self._enriquecer_odds(tips_aprovadas)

        # ─── ETAPA 6b: Salvar no banco ───
        print("\n💾 Etapa 6b: Salvando tips aprovadas...")
        _treino = self.db.ultimo_treino()
        versao_atual = _treino["modelo_versao"] if _treino else "v1"

        for tip in tips_aprovadas:
            self.db.salvar_prediction({
                "fixture_id": tip["fixture_id"],
                "date": tip["date"],
                "league_id": tip.get("league_id"),
                "home_name": tip.get("home_name"),
                "away_name": tip.get("away_name"),
                "prob_home": tip.get("prob_home"),
                "prob_draw": tip.get("prob_draw"),
                "prob_away": tip.get("prob_away"),
                "prob_over25": tip.get("prob_over25"),
                "prob_btts": tip.get("prob_btts_yes"),
                "mercado": tip["mercado"],
                "odd_usada": tip.get("odd_pinnacle"),
                "ev_percent": tip.get("ev_percent"),
                "bookmaker": tip.get("odd_fonte", ""),
                "modelo_versao": versao_atual,
                "features": tip.get("features", {}),
            })
        print(f"   ✅ {len(tips_aprovadas)} tips salvas (modelo {versao_atual})")

        # ─── ETAPA 7: Gerar combos (acumuladas sugeridas) ───
        combos = self._gerar_combos(tips_aprovadas)
        if combos:
            print(f"\n🎰 Etapa 7: {len(combos)} combos gerados")

        return {
            "fixtures": len(fixtures),
            "previsoes": len(previsoes),
            "oportunidades": tips_raw,
            "ev_positivas": tips_aprovadas,
            "combos": combos,
            "data": data,
        }

    # ══════════════════════════════════════════════
    #  ETAPA 1: Scan de fixtures
    # ══════════════════════════════════════════════

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

    # ══════════════════════════════════════════════
    #  ETAPA 2 (fallback): Previsões via API-Football
    # ══════════════════════════════════════════════

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

    # ══════════════════════════════════════════════
    #  ETAPA 3: Expandir previsões em tips por mercado
    # ══════════════════════════════════════════════

    def _expandir_mercados(self, previsoes: list[dict]) -> list[dict]:
        """
        Transforma cada previsão (1 jogo com N probabilidades) em
        N tips individuais (1 por mercado), filtrando por PROB_MIN.

        Isso substitui o antigo _buscar_odds_e_ev: em vez de cruzar
        prob × odd para calcular EV, simplesmente emite a tip se a
        probabilidade do modelo é alta o suficiente.
        """
        tips = []
        for pred in previsoes:
            home = pred.get("home_name", "Casa")
            away = pred.get("away_name", "Fora")

            for mercado_id, prob_key, desc_tpl in MERCADOS:
                prob = pred.get(prob_key, 0)
                if not prob or prob < PROB_MIN:
                    continue

                tips.append({
                    "fixture_id": pred["fixture_id"],
                    "date": pred["date"],
                    "league_id": pred.get("league_id"),
                    "season": pred.get("season"),
                    "round": pred.get("round", ""),
                    "home_name": home,
                    "away_name": away,
                    "mercado": mercado_id,
                    "descricao": desc_tpl.format(home=home, away=away),
                    "prob_modelo": prob,
                    # Propagar todas as probabilidades para rastreabilidade
                    "prob_home": pred.get("prob_home"),
                    "prob_draw": pred.get("prob_draw"),
                    "prob_away": pred.get("prob_away"),
                    "prob_over25": pred.get("prob_over25"),
                    "prob_btts_yes": pred.get("prob_btts_yes"),
                    "prob_ht_home": pred.get("prob_ht_home"),
                    "prob_ht_draw": pred.get("prob_ht_draw"),
                    "prob_ht_away": pred.get("prob_ht_away"),
                    "features": pred.get("features", {}),
                })

        # Ordenar por confiança decrescente
        tips.sort(key=lambda x: x.get("prob_modelo", 0), reverse=True)
        return tips

    # ══════════════════════════════════════════════
    #  ETAPA 4b: Filtro de conflitos e limites por jogo
    # ══════════════════════════════════════════════

    def _filtrar_conflitos_e_limites(self, tips: list[dict]) -> list[dict]:
        """
        Remove tips contraditórias por CATEGORIA e limita por fixture.

        Lógica por categorias (CATEGORIAS_CONFLITO):
          - Todos os mercados Over/Under pertencem à categoria 'gols'.
            Só 1 tip de gols por jogo (a de maior confiança).
          - Idem para 'resultado' (h2h), 'btts' e 'ht'.
          Isso garante que Over 1.5 + Under 3.5 NUNCA coexistam.

        Após resolver categorias, limita a MAX_TIPS_POR_JOGO por fixture.
        """
        from collections import defaultdict
        por_fixture = defaultdict(list)
        for t in tips:
            por_fixture[t["fixture_id"]].append(t)

        removidos_categoria = 0
        removidos_limite = 0
        resultado = []

        for fix_id, fix_tips in por_fixture.items():
            # Ordenar por confiança desc dentro do fixture
            fix_tips.sort(key=lambda x: x.get("prob_modelo", 0), reverse=True)

            # Passo 1: max 1 tip por categoria (a primeira = maior confiança)
            categorias_usadas = set()
            tips_filtradas = []
            for t in fix_tips:
                cat = _MERCADO_CATEGORIA.get(t["mercado"])
                if cat and cat in categorias_usadas:
                    # Já tem tip nesta categoria — descartar
                    removidos_categoria += 1
                    continue
                if cat:
                    categorias_usadas.add(cat)
                tips_filtradas.append(t)

            # Passo 2: limitar a MAX_TIPS_POR_JOGO por fixture
            if len(tips_filtradas) > MAX_TIPS_POR_JOGO:
                removidos_limite += len(tips_filtradas) - MAX_TIPS_POR_JOGO
                tips_filtradas = tips_filtradas[:MAX_TIPS_POR_JOGO]

            resultado.extend(tips_filtradas)

        # Reordenar resultado final por confiança desc
        resultado.sort(key=lambda x: x.get("prob_modelo", 0), reverse=True)

        if removidos_categoria or removidos_limite:
            print(f"   ⚔️ Conflitos por categoria: {removidos_categoria} | "
                  f"Cortados por limite/jogo: {removidos_limite}")
            print(f"   📊 Tips após filtros: {len(resultado)}")

        return resultado

    # ══════════════════════════════════════════════
    #  ETAPA 4: Strategy Gate
    # ══════════════════════════════════════════════

    def _strategy_check(self, tip: dict) -> bool:
        """
        Verifica se uma tip passa no strategy gate.

        Consulta a tabela de estratégias ativas para ver se existe
        um slice (mercado × liga × confiança) que aprova esta tip.

        Estratégias são sempre por liga específica — não existe global.
        Se não houver estratégia ativa para aquele mercado+liga+confiança,
        a tip é bloqueada (princípio conservador).

        Retorna True se a tip deve ser emitida, False se bloqueada.
        """
        mercado = tip.get("mercado", "")
        league_id = tip.get("league_id")
        prob = tip.get("prob_modelo", 0)

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

    # ══════════════════════════════════════════════
    #  ETAPA 6: Enriquecer tips com odds Pinnacle + EV
    # ══════════════════════════════════════════════

    def _enriquecer_odds(self, tips: list[dict]) -> list[dict]:
        """
        Busca odds da Pinnacle (via The Odds API) para as ligas das tips
        aprovadas e enriquece cada tip com odd de referência e EV.

        Só busca odds de ligas que têm mapeamento no ODDS_SPORTS_MAP.
        Custo: ~2 créditos por liga (h2h + totals).

        Campos adicionados em cada tip:
          - odd_pinnacle: odd decimal da Pinnacle (ou None se indisponível)
          - ev_percent: EV em % = (prob × odd - 1) × 100 (ou None)
          - odd_fonte: nome do bookmaker que forneceu a odd
        """
        from services.odds_api import buscar_odds_liga

        if not tips:
            return tips

        # 1. Identificar ligas únicas das tips aprovadas
        ligas_tips = {t.get("league_id") for t in tips if t.get("league_id")}
        ligas_com_odds = {lid for lid in ligas_tips if ODDS_SPORTS_MAP.get(lid)}
        ligas_sem_odds = ligas_tips - ligas_com_odds

        if ligas_sem_odds:
            nomes = [_LEAGUE_NOME.get(lid, str(lid)) for lid in ligas_sem_odds]
            print(f"   ⚠️ Sem cobertura Odds API: {', '.join(nomes)}")

        if not ligas_com_odds:
            print("   📭 Nenhuma liga com cobertura — tips sem odds")
            return tips

        # 2. Buscar odds de cada liga (h2h + totals, apenas eu para economizar)
        # Cache: {sport_key: [jogos_com_odds]}
        cache_odds = {}
        for lid in ligas_com_odds:
            sport_key = ODDS_SPORTS_MAP[lid]
            if sport_key in cache_odds:
                continue
            try:
                jogos = buscar_odds_liga(sport_key, markets="h2h,totals", regions="eu")
                cache_odds[sport_key] = jogos
                print(f"   ✅ {_LEAGUE_NOME.get(lid, sport_key)}: {len(jogos)} jogos com odds")
            except Exception as e:
                print(f"   ❌ {_LEAGUE_NOME.get(lid, sport_key)}: erro ao buscar odds — {e}")
                cache_odds[sport_key] = []

        # 3. Para cada tip, encontrar o jogo e extrair a odd
        enriquecidas = 0
        for tip in tips:
            lid = tip.get("league_id")
            sport_key = ODDS_SPORTS_MAP.get(lid)
            if not sport_key or sport_key not in cache_odds:
                continue

            jogos = cache_odds[sport_key]
            jogo = self._match_jogo_odds(
                jogos, tip.get("home_name", ""), tip.get("away_name", "")
            )
            if not jogo:
                continue

            # Extrair odd para este mercado específico
            odd, fonte = self._extrair_odd_mercado(jogo, tip)
            if odd and odd > 1.0:
                tip["odd_pinnacle"] = round(odd, 2)
                tip["odd_fonte"] = fonte
                # EV = (prob_modelo × odd) - 1, em %
                prob = tip.get("prob_modelo", 0)
                tip["ev_percent"] = round((prob * odd - 1) * 100, 1)
                enriquecidas += 1

        print(f"   📊 {enriquecidas}/{len(tips)} tips enriquecidas com odds")
        return tips

    @staticmethod
    def _match_jogo_odds(jogos: list[dict], home: str, away: str) -> dict | None:
        """
        Encontra um jogo na lista de odds por nome dos times (fuzzy match).
        Usa busca parcial case-insensitive com palavras >3 chars.
        """
        home_lower = home.lower()
        away_lower = away.lower()

        for jogo in jogos:
            jh = jogo.get("home_team", "").lower()
            ja = jogo.get("away_team", "").lower()

            # Match por palavras significativas (>3 chars)
            home_words = [w for w in home_lower.split() if len(w) > 3]
            away_words = [w for w in away_lower.split() if len(w) > 3]

            # Fallback: se não tem palavras >3, usa todas
            if not home_words:
                home_words = home_lower.split()
            if not away_words:
                away_words = away_lower.split()

            home_match = any(w in jh for w in home_words)
            away_match = any(w in ja for w in away_words)

            if home_match and away_match:
                return jogo

        return None

    @staticmethod
    def _extrair_odd_mercado(jogo: dict, tip: dict) -> tuple[float, str]:
        """
        Extrai a odd de um mercado específico do jogo, priorizando Pinnacle.

        Usa _MERCADO_ODDS_MAP para traduzir o mercado_id do FuteBot
        nos parâmetros da Odds API (market_key, outcome, point).

        Retorna (odd, nome_bookmaker) ou (0, '') se não encontrar.
        """
        mercado_id = tip.get("mercado", "")
        mapping = _MERCADO_ODDS_MAP.get(mercado_id)
        if not mapping:
            return 0, ""

        market_key, outcome, point = mapping
        home_team = jogo.get("home_team", "")
        away_team = jogo.get("away_team", "")

        # Resolver HOME/AWAY para nomes reais
        if outcome == "HOME":
            outcome = home_team
        elif outcome == "AWAY":
            outcome = away_team

        # Prioridade: Pinnacle → melhor alternativa
        melhor_odd = 0.0
        melhor_casa = ""
        odd_pinnacle = 0.0

        # Coleta todas as linhas disponíveis para fallback (totals)
        linhas_disponiveis = {}  # {bookmaker_key: {point: price}}

        for bk in jogo.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                for oc in mkt.get("outcomes", []):
                    nome = oc.get("name", "")
                    price = oc.get("price", 0)

                    if nome != outcome:
                        continue

                    oc_point = oc.get("point")

                    # Match exato de point
                    if point is not None and oc_point == point:
                        if bk.get("key") == "pinnacle":
                            odd_pinnacle = price
                        if price > melhor_odd:
                            melhor_odd = price
                            melhor_casa = bk.get("title", bk.get("key", ""))

                    # Sem point (h2h, btts) — match direto
                    elif point is None:
                        if bk.get("key") == "pinnacle":
                            odd_pinnacle = price
                        if price > melhor_odd:
                            melhor_odd = price
                            melhor_casa = bk.get("title", bk.get("key", ""))

                    # Guardar linhas para fallback (totals com point diferente)
                    if point is not None and oc_point is not None:
                        bk_key = bk.get("key", "")
                        if bk_key not in linhas_disponiveis:
                            linhas_disponiveis[bk_key] = {}
                        linhas_disponiveis[bk_key][oc_point] = (price, bk.get("title", bk_key))

        # Se não encontrou a linha exata, usar linha mais próxima da Pinnacle
        if odd_pinnacle == 0 and melhor_odd == 0 and linhas_disponiveis:
            # Priorizar Pinnacle
            for bk_key in ["pinnacle"] + list(linhas_disponiveis.keys()):
                if bk_key not in linhas_disponiveis:
                    continue
                pts = linhas_disponiveis[bk_key]
                # Encontrar a linha mais próxima do point desejado
                closest = min(pts.keys(), key=lambda p: abs(p - point))
                price, casa = pts[closest]
                if bk_key == "pinnacle":
                    odd_pinnacle = price
                    break
                elif price > melhor_odd:
                    melhor_odd = price
                    melhor_casa = casa

        # Retornar Pinnacle se disponível, senão melhor alternativa
        if odd_pinnacle > 0:
            return odd_pinnacle, "Pinnacle"
        return melhor_odd, melhor_casa

    # ══════════════════════════════════════════════
    #  ETAPA 7: Geração de combos (acumuladas)
    # ══════════════════════════════════════════════

    def _gerar_combos(self, tips: list[dict]) -> list[dict]:
        """
        Gera sugestões de apostas combinadas (duplas e triplas).

        Regras:
          - Cada tip no combo deve ser de um FIXTURE diferente.
          - Um fixture NUNCA aparece em mais de um combo (global).
          - Confiança composta (produto das probs) >= COMBO_PROB_MIN.
          - Só entra tip com prob >= COMBO_TIP_PROB_MIN.
          - Prioriza combos com maior confiança composta.
          - Máx COMBO_MAX_TOTAL combos totais (duplas + triplas).
          - Duplas e triplas compartilham o mesmo set de fixtures usadas.

        Retorna lista de dicts com 'tipo', 'tips', 'prob_composta'.
        """
        from itertools import combinations

        # Filtrar tips elegíveis para combo (prob alta + 1 por fixture)
        vistas = set()
        elegiveis = []
        for t in tips:
            fid = t.get("fixture_id")
            if fid in vistas:
                continue  # Já tem tip deste jogo — pular
            if t.get("prob_modelo", 0) < COMBO_TIP_PROB_MIN:
                continue
            vistas.add(fid)
            elegiveis.append(t)

        if len(elegiveis) < 2:
            return []

        # ─── Duplas ───
        candidatas_duplas = []
        for a, b in combinations(elegiveis, 2):
            prob = a["prob_modelo"] * b["prob_modelo"]
            if prob >= COMBO_PROB_MIN:
                candidatas_duplas.append({
                    "tipo": "dupla",
                    "tips": [a, b],
                    "prob_composta": round(prob, 4),
                })
        candidatas_duplas.sort(key=lambda x: x["prob_composta"], reverse=True)

        # Set GLOBAL de fixtures usadas — compartilhado entre duplas e triplas
        # Garante que um jogo NUNCA aparece em mais de um combo
        fixtures_usadas = set()
        combos_total = []

        # Selecionar duplas diversificadas
        duplas = []
        for c in candidatas_duplas:
            if len(combos_total) >= COMBO_MAX_TOTAL:
                break
            fids = {t["fixture_id"] for t in c["tips"]}
            if fids & fixtures_usadas:
                continue  # Fixture já usada em outro combo
            duplas.append(c)
            combos_total.append(c)
            fixtures_usadas.update(fids)
            if len(duplas) >= COMBO_DUPLAS_MAX:
                break

        # ─── Triplas (só se há pelo menos 3 elegíveis e ainda cabe) ───
        triplas = []
        if len(elegiveis) >= 3 and len(combos_total) < COMBO_MAX_TOTAL:
            candidatas_triplas = []
            for a, b, c in combinations(elegiveis, 3):
                prob = a["prob_modelo"] * b["prob_modelo"] * c["prob_modelo"]
                if prob >= COMBO_PROB_MIN:
                    candidatas_triplas.append({
                        "tipo": "tripla",
                        "tips": [a, b, c],
                        "prob_composta": round(prob, 4),
                    })
            candidatas_triplas.sort(key=lambda x: x["prob_composta"], reverse=True)

            for c in candidatas_triplas:
                if len(combos_total) >= COMBO_MAX_TOTAL:
                    break
                fids = {t["fixture_id"] for t in c["tips"]}
                if fids & fixtures_usadas:
                    continue  # Fixture já usada em outro combo
                triplas.append(c)
                combos_total.append(c)
                fixtures_usadas.update(fids)
                if len(triplas) >= COMBO_TRIPLAS_MAX:
                    break

        return combos_total

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

    # ══════════════════════════════════════════════
    #  FORMATAÇÃO DO RELATÓRIO TELEGRAM
    # ══════════════════════════════════════════════

    # ──────────────────────────────────────────────
    #  Helpers de formatação
    # ──────────────────────────────────────────────

    @staticmethod
    def _data_br(date_str: str) -> str:
        """Converte 'YYYY-MM-DD' ou ISO para 'DD/MM'."""
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%d/%m")
        except Exception:
            return date_str[:10] if date_str else "hoje"

    @staticmethod
    def _horario_local(date_str: str) -> str:
        """Extrai horário local (HH:MM) de uma data ISO."""
        if not date_str or "T" not in date_str:
            return ""
        try:
            from zoneinfo import ZoneInfo
            dt_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt_local = dt_obj.astimezone(ZoneInfo(TIMEZONE))
            return dt_local.strftime("%H:%M")
        except Exception:
            return ""

    @staticmethod
    def _data_local(date_str: str) -> str:
        """Extrai data local (DD/MM) de uma data ISO."""
        if not date_str:
            return ""
        try:
            from zoneinfo import ZoneInfo
            dt_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt_local = dt_obj.astimezone(ZoneInfo(TIMEZONE))
            return dt_local.strftime("%d/%m")
        except Exception:
            return Scanner._data_br(date_str)

    def formatar_relatorio(self, resultado: dict) -> list[str]:
        """
        Formata resultado do scanner para envio no Telegram (HTML).

        Retorna uma LISTA de mensagens:
          - Índice 0: cabeçalho com resumo geral
          - Índice 1..N: uma mensagem por LIGA (agrupa todos os jogos da liga)
          - Último: performance acumulada

        Agrupamento: Liga → Fixture → Tips do fixture.
        Tips do mesmo jogo ficam juntas no mesmo bloco visual.
        """
        from collections import defaultdict

        # ─── Scanner pausado ───
        if resultado.get("pausado"):
            return [
                "⛔ <b>FuteBot — Scanner PAUSADO</b>\n\n"
                f"Motivo: {resultado.get('motivo_pausa', 'modelo degradado')}\n\n"
                "Use /treinar para retreinar ou /metricas para ver detalhes.\n"
                "O scanner volta a funcionar automaticamente após retreino aprovado."
            ]

        tips = resultado.get("ev_positivas", [])
        data_raw = resultado.get("data", "hoje")
        data_br = self._data_br(data_raw)

        msgs = []

        # ─── Cabeçalho ───
        header = (
            f"🤖 <b>FuteBot — Tips {data_br}</b>\n"
            f"📋 {resultado['fixtures']} jogos analisados | "
            f"{resultado['previsoes']} previsões geradas"
        )

        if not tips:
            msgs.append(header + "\n\n📭 Nenhuma tip aprovada hoje.")
            return msgs

        msgs.append(header + f"\n🎯 {len(tips)} tips aprovadas")

        # ─── Agrupar tips: liga → fixture → lista de tips ───
        por_liga = defaultdict(lambda: defaultdict(list))
        for tip in tips:
            lid = tip.get("league_id", 0)
            fid = tip.get("fixture_id", 0)
            por_liga[lid][fid].append(tip)

        # Ordenar ligas pelo nome para consistência
        ligas_ordenadas = sorted(por_liga.keys(),
                                 key=lambda lid: _LEAGUE_NOME.get(lid, f"Liga {lid}"))

        # ─── Uma mensagem por liga (com todos os jogos agrupados) ───
        for lid in ligas_ordenadas:
            fixtures_da_liga = por_liga[lid]
            nome_liga = _LEAGUE_NOME.get(lid, f"Liga {lid}")

            # Header da liga
            bloco = f"🏆 <b>{nome_liga}</b>\n"
            bloco += "─" * 24 + "\n"

            # Ordenar fixtures por horário (primeiro jogo primeiro)
            fixtures_ordenados = sorted(
                fixtures_da_liga.items(),
                key=lambda item: item[1][0].get("date", "")
            )

            for i, (fid, fix_tips) in enumerate(fixtures_ordenados):
                # Ordenar tips do fixture por confiança desc
                fix_tips.sort(key=lambda x: x.get("prob_modelo", 0), reverse=True)

                primeira = fix_tips[0]
                horario = self._horario_local(primeira.get("date", ""))
                data_jogo = self._data_local(primeira.get("date", ""))

                # Mostra data se for diferente de hoje
                data_info = ""
                if data_jogo and data_jogo != data_br:
                    data_info = f" 📅 {data_jogo}"

                hora_str = f" ⏰ {horario}" if horario else ""

                # Nome do jogo (1 linha)
                bloco += (
                    f"\n⚽ <b>{primeira.get('home_name', '?')} vs "
                    f"{primeira.get('away_name', '?')}</b>"
                    f"{hora_str}{data_info}\n"
                )

                # Tips do jogo (cada uma com emoji, nomes copiáveis, odd Pinnacle e EV)
                for tip in fix_tips:
                    prob = tip.get("prob_modelo", 0)
                    emoji = "🔥" if prob > 0.70 else "⚡" if prob > 0.55 else "📊"
                    desc = tip.get("descricao", tip.get("mercado", "?"))
                    home = primeira.get("home_name", "?")
                    away = primeira.get("away_name", "?")

                    # Linha principal: nomes copiáveis + mercado + confiança
                    linha = f"   {emoji} <code>{home}</code> vs <code>{away}</code> → {desc} — {prob:.0%}"

                    # Odd Pinnacle + EV (se disponível)
                    odd_p = tip.get("odd_pinnacle")
                    ev = tip.get("ev_percent")
                    if odd_p:
                        ev_str = f" | EV: {ev:+.0f}%" if ev is not None else ""
                        fonte = tip.get("odd_fonte", "Pinnacle")
                        linha += f"\n   💰 Odd {fonte}: <b>{odd_p:.2f}</b>{ev_str}"

                    bloco += linha + "\n"

                    # Parecer LLM (curto, abaixo da tip)
                    llm = tip.get("llm_validacao")
                    if llm and llm.get("motivo") and "desativado" not in llm.get("motivo", ""):
                        bloco += f"   🤖 <i>{llm['motivo']}</i>\n"

                # Separador entre jogos da mesma liga (exceto último)
                if i < len(fixtures_ordenados) - 1:
                    bloco += "\n"

            msgs.append(bloco.rstrip())

        # ─── Combos sugeridos (acumuladas) ───
        combos = resultado.get("combos", [])
        if combos:
            bloco_combo = "🎰 <b>Combos sugeridos</b>\n"
            bloco_combo += "─" * 24 + "\n"

            for i, combo in enumerate(combos, 1):
                tipo_label = "Dupla" if combo["tipo"] == "dupla" else "Tripla"
                prob_c = combo["prob_composta"]
                emoji_c = "🔥" if prob_c > 0.55 else "⚡" if prob_c > 0.45 else "📊"

                # Odd composta real (produto das odds Pinnacle)
                odd_composta = 1.0
                todas_com_odd = True
                for t in combo["tips"]:
                    odd_t = t.get("odd_pinnacle")
                    if odd_t and odd_t > 1.0:
                        odd_composta *= odd_t
                    else:
                        todas_com_odd = False

                # Header do combo com odd composta
                if todas_com_odd and odd_composta > 1.0:
                    ev_combo = (prob_c * odd_composta - 1) * 100
                    bloco_combo += (
                        f"\n{emoji_c} <b>{tipo_label} #{i}</b> — "
                        f"Odd: <b>{odd_composta:.2f}</b> | "
                        f"Prob: {prob_c:.0%} | EV: {ev_combo:+.0f}%\n"
                    )
                else:
                    bloco_combo += f"\n{emoji_c} <b>{tipo_label} #{i}</b> — {prob_c:.0%}\n"

                for t in combo["tips"]:
                    home = t.get("home_name", "?")
                    away = t.get("away_name", "?")
                    desc = t.get("descricao", t.get("mercado", "?"))
                    prob_t = t.get("prob_modelo", 0)
                    odd_t = t.get("odd_pinnacle")

                    # Nomes copiáveis + mercado + prob + odd
                    linha_combo = f"   • <code>{home}</code> vs <code>{away}</code> → {desc} ({prob_t:.0%})"
                    if odd_t:
                        linha_combo += f" @ {odd_t:.2f}"
                    bloco_combo += linha_combo + "\n"

            # Link genérico Pinnacle para facilitar acesso
            bloco_combo += "\n🔗 <a href=\"https://www.pinnacle.com/pt/soccer\">Abrir Pinnacle → Futebol</a>"

            msgs.append(bloco_combo.rstrip())

        # ─── Performance acumulada (última mensagem) ───
        metricas = self.db.metricas_modelo()
        if metricas["total"] > 0:
            perf = (
                f"📈 <b>Performance acumulada:</b>\n"
                f"   Accuracy: {metricas['accuracy']}% ({metricas['acertos']}/{metricas['total']})\n"
                f"   ROI: {metricas['roi']:+.1f}%"
            )
            msgs.append(perf)

        return msgs


if __name__ == "__main__":
    # Execução manual para teste
    scanner = Scanner()
    resultado = scanner.executar()
    msgs = scanner.formatar_relatorio(resultado)
    for msg in msgs:
        print(msg)
        print()

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

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from zoneinfo import ZoneInfo
from config import LEAGUES, TIMEZONE, TELEGRAM_CHAT_ID
from config import ROI_PAUSE_THRESHOLD, ROI_PAUSE_MIN_BETS, DEGRADATION_ACC_MIN
from config import (
    BET365_URL,
    COMBO_TIP_CONFIDENCE_MIN,
    MODEL_CONFIDENCE_MIN,
    ODDSPAPI_BOOKMAKER_LABEL,
    ODDSPAPI_COMBO_LEG_MIN_EV_PERCENT,
    ODDSPAPI_COMBO_LEG_MIN_ODD,
    ODDSPAPI_COMBO_MIN_EV_PERCENT,
    ODDSPAPI_COMBO_MIN_ODD,
    ODDSPAPI_SIMPLE_MIN_EV_PERCENT,
    ODDSPAPI_SIMPLE_MIN_ODD,
    ODDSPAPI_USE_PRELIVE,
    PREFERRED_BOOKMAKER_LABEL,
)
from data.database import Database
from services.apifootball import raw_request
from services.oddspapi import enriquecer_tips_com_odds_oddspapi
from models.predictor import Predictor
from services.llm_validator import LLMValidator
from services.user_prefs import get_runtime_preferences


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

# Sem strategy ativa, ainda usamos um fallback conservador para não abrir ruído total.
NO_STRATEGY_PROB_MIN = 0.70

# Limite de tips por fixture (evita spam no mesmo jogo)
MAX_TIPS_POR_JOGO = None

# Categorias de conflito — no máximo 1 tip por categoria por jogo.
# Se múltiplos mercados da mesma categoria passam (ex: Over 1.5 + Under 3.5),
# apenas o de maior confiança sobrevive. Isso elimina qualquer combinação
# Over+Under, resultado contraditório, etc., independente de threshold.
CATEGORIAS_CONFLITO = {
    "gols":      {"over15", "under15", "over25", "under25", "over35", "under35"},
    "gols_1t":   {"over05_ht", "under05_ht", "over15_ht", "under15_ht"},
    "gols_2t":   {"over05_2t", "under05_2t", "over15_2t", "under15_2t"},
    "resultado": {"h2h_home", "h2h_draw", "h2h_away"},
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

_CATEGORIA_MERCADOS = {
    _cat: tuple(sorted(_mercs))
    for _cat, _mercs in CATEGORIAS_CONFLITO.items()
}

# Mapa league_id → nome amigável (para headers no Telegram)
_LEAGUE_NOME = {v["id"]: v["nome"] for v in LEAGUES.values()}

# ──────────────────────────────────────────────
# Configuração de combos (acumuladas)
# Combina tips de jogos DIFERENTES para multiplicar odds.
# Usa odds Pinnacle reais quando disponíveis.
# ──────────────────────────────────────────────
COMBO_PROB_MIN = 0.50         # Confiança composta mínima (produto das probs)
COMBO_TIP_PROB_MIN = 0.65     # Prob mínima individual para entrar em combo
PRESELECT_MAX_JOGOS = 0
RELEASE_LOOKAHEAD_MINUTES = 30
RELEASE_WINDOW_MINUTES = 5
STRATEGY_EPSILON = 0.001
STRATEGY_RELAX_PCT = 0.05

# Centraliza o piso operacional de confianca do modelo para pre-live e combos.
NO_STRATEGY_PROB_MIN = MODEL_CONFIDENCE_MIN
COMBO_TIP_PROB_MIN = COMBO_TIP_CONFIDENCE_MIN


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
    # 1x2 Primeiro Tempo — market_key='h2h_h1'
    "ht_home":   ("h2h_h1", "HOME", None),
    "ht_draw":   ("h2h_h1", "Draw", None),
    "ht_away":   ("h2h_h1", "AWAY", None),
}


_STRATEGY_FEATURE_ALIASES = {
    # Compatibilidade entre strategies do discovery (FeatureExtractor)
    # e runtime do Predictor (FeatureFactory com features dinâmicas).
    "home_goals_for_avg": "home_gf_5",
    "home_goals_against_avg": "home_ga_5",
    "away_goals_for_avg": "away_gf_5",
    "away_goals_against_avg": "away_ga_5",
    "home_goals_for_home": "home_gf_mando",
    "home_goals_against_home": "home_ga_mando",
    "away_goals_for_away": "away_gf_mando",
    "away_goals_against_away": "away_ga_mando",
    "home_clean_sheet_pct": "home_cs_5",
    "away_clean_sheet_pct": "away_cs_5",
    "home_failed_score_pct": "home_fts_5",
    "away_failed_score_pct": "away_fts_5",
    "home_btts_pct": "home_btts_5",
    "away_btts_pct": "away_btts_5",
    "home_over15_pct": "home_over15_5",
    "away_over15_pct": "away_over15_5",
    "home_over25_pct": "home_over25_5",
    "away_over25_pct": "away_over25_5",
    "home_over35_pct": "home_over35_5",
    "away_over35_pct": "away_over35_5",
    "home_xg_avg": "home_xg_5",
    "away_xg_avg": "away_xg_5",
    "home_shots_avg": "home_shots_5",
    "away_shots_avg": "away_shots_5",
    "home_shots_on_avg": "home_shots_on_5",
    "away_shots_on_avg": "away_shots_on_5",
    "home_possession_avg": "home_poss_5",
    "away_possession_avg": "away_poss_5",
    "home_corners_avg": "home_corners_5",
    "away_corners_avg": "away_corners_5",
    "home_yellows_avg": "home_yellows_5",
    "away_yellows_avg": "away_yellows_5",
    "home_fouls_avg": "home_fouls_5",
    "away_fouls_avg": "away_fouls_5",
    "home_goals_ht_avg": "home_goals_ht_5",
    "away_goals_ht_avg": "away_goals_ht_5",
    "h2h_total_jogos": "h2h_total",
    "ref_yellows_avg": "ref_yellows",
    "ref_fouls_avg": "ref_fouls",
    "ref_total_jogos": "ref_jogos",
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

    def executar(
        self,
        data: str = None,
        dias_adiante: int = 0,
        mode: str = "preselect",
        reference_time: datetime | None = None,
        lookahead_minutes: int | None = None,
    ) -> dict:
        """
        Executa o pipeline completo para uma data (ou apenas hoje).

        Parâmetros:
          data: data no formato YYYY-MM-DD (default: hoje)
          dias_adiante: quantos dias à frente buscar (default: 0 = somente hoje)

        Retorna resumo com fixtures encontrados, previsões e tips emitidas.
        """
        if data is None:
            data = datetime.now().strftime("%Y-%m-%d")

        print(f"🔍 Scanner iniciando para {data} (+{dias_adiante} dias) [{mode}]")

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
        fixtures = self._scan_fixtures(
            data,
            dias_adiante,
            reference_time=reference_time,
            lookahead_minutes=lookahead_minutes,
        )
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
            fixtures_por_liga = {}
            for f in fixtures:
                league_id = f["league"]["id"]
                fixtures_por_liga.setdefault(league_id, []).append(f)

            for league_id, fixtures_liga in fixtures_por_liga.items():
                for f in fixtures_liga:
                    fix_dict = {
                        "fixture_id": f["fixture"]["id"],
                        "home_id": f["teams"]["home"]["id"],
                        "away_id": f["teams"]["away"]["id"],
                        "home_name": f["teams"]["home"]["name"],
                        "away_name": f["teams"]["away"]["name"],
                        "league_id": league_id,
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
        min_prob = 0.0 if self._strategies else NO_STRATEGY_PROB_MIN
        tips_raw = self._expandir_mercados(previsoes, min_prob=min_prob)
        tips_brutas = len(tips_raw)
        if self._strategies:
            print(f"   Tips brutas: {tips_brutas}")
        else:
            print(f"   Tips brutas (fallback sem strategy >= {NO_STRATEGY_PROB_MIN:.0%}): {tips_brutas}")

        # ─── ETAPA 4: Strategy Gate ───
        print("\n🛡️ Etapa 4: Filtrando pelo Strategy Gate...")
        if self._strategies:
            tips_gate = [t for t in tips_raw if self._strategy_check(t)]
            bloqueadas = len(tips_raw) - len(tips_gate)
            print(f"   ✅ Aprovadas: {len(tips_gate)} | 🚫 Bloqueadas: {bloqueadas}")
        else:
            tips_gate = tips_raw
            print(f"   ⚠️ Sem strategy gate — mantendo fallback sem strategy: {len(tips_gate)} tips")

        # ─── ETAPA 4b: Resolver conflitos + limitar por jogo ───
        tips_gate = self._filtrar_conflitos_e_limites(tips_gate)
        tips_pos_filtros = len(tips_gate)

        if not tips_gate:
            print("   📭 Nenhuma tip passou nos filtros")
            return {
                "fixtures": len(fixtures), "previsoes": len(previsoes),
                "oportunidades": [], "ev_positivas": [], "data": data,
            }

        if mode == "preselect":
            return self._finalizar_preselecao(
                data=data,
                fixtures=fixtures,
                previsoes=previsoes,
                tips_raw=tips_raw,
                tips_gate=tips_gate,
                tips_brutas=tips_brutas,
                tips_pos_filtros=tips_pos_filtros,
                reference_time=reference_time,
                lookahead_minutes=lookahead_minutes,
            )

        return self.liberar_mercados(
            data=data,
            reference_time=None,
            tips_override=tips_gate,
            base_result={
                "fixtures": len(fixtures),
                "previsoes": len(previsoes),
                "tips_brutas": tips_brutas,
                "tips_pos_filtros": tips_pos_filtros,
            },
        )

    def liberar_mercados(
        self,
        data: str = None,
        reference_time: datetime = None,
        tips_override: list[dict] = None,
        base_result: dict = None,
        test_mode: bool = False,
    ) -> dict:
        """Libera mercados perto do jogo, após odds + Gemini + DeepSeek."""
        now_local = datetime.now(ZoneInfo(TIMEZONE))
        if data is None:
            data = now_local.strftime("%Y-%m-%d")
        reference_time = reference_time or now_local
        reference_time = reference_time or datetime.now(ZoneInfo(TIMEZONE))

        if tips_override is None:
            candidatos = self.db.candidatos_por_data(data, status="pending")
            tips_base = [item.get("payload", {}) for item in candidatos]
        else:
            candidatos = []
            tips_base = list(tips_override)

        if base_result:
            base_result = dict(base_result)
        else:
            fixture_ids = {
                tip.get("fixture_id")
                for tip in tips_base
                if tip.get("fixture_id") is not None
            }
            base_result = {
                "fixtures": len(fixture_ids),
                "previsoes": len(fixture_ids),
                "tips_brutas": len(tips_base),
                "tips_pos_filtros": len(tips_base),
            }

        tips_janela = self._filtrar_janela_liberacao(tips_base, reference_time, test_mode=test_mode)
        print(f"\n⏳ Etapa 5: Janela T-30 -> {len(tips_janela)} mercado(s) elegível(is)")

        if not tips_janela:
            return {
                **base_result,
                "tips_bloqueadas_ev": 0,
                "tips_enviadas_llm": 0,
                "tips_aprovadas_llm": 0,
                "tips_rejeitadas_llm": [],
                "tips_aprovadas": 0,
                "ev_positivas": [],
                "combos": [],
                "data": data,
                "mode": "release",
                "preselecionados": [],
                "release_reference": reference_time.isoformat(),
            }

        print(f"\n📊 Etapa 6: Odds de referência + EV ({len(tips_janela)} tips)...")
        tips_enriquecidas = self._enriquecer_odds(tips_janela)
        tips_revisao, bloqueadas_gate = self._aplicar_gate_odds_ev(tips_enriquecidas)
        bloqueadas_por_ev = len(bloqueadas_gate)

        if tips_revisao and self.llm.ativo:
            print(f"\n🤖 Etapa 7: Validação DeepSeek ({len(tips_revisao)} tips)...")
            tips_aprovadas = self.llm.validar_lote(tips_revisao)
            print(f"   Aprovadas pelo DeepSeek: {len(tips_aprovadas)}")
        else:
            tips_aprovadas = tips_revisao

        tips_rejeitadas_llm = list(bloqueadas_gate) + [
            tip for tip in tips_revisao
            if (tip.get("llm_validacao") or {}).get("decisao") == "REJECT"
        ]
        tips_aprovadas_llm = [
            tip for tip in tips_revisao
            if (tip.get("llm_validacao") or {}).get("decisao") == "APPROVE"
        ]

        tips_aprovadas = self._aplicar_preferencias_usuario(tips_aprovadas)
        tips_aprovadas.sort(
            key=lambda t: (
                t.get("ev_percent") is not None,
                t.get("ev_percent", -999),
                t.get("prob_modelo", 0),
            ),
            reverse=True,
        )
        print("\n💾 Etapa 8: Salvando tips liberadas...")
        _treino = self.db.ultimo_treino()
        versao_atual = _treino["modelo_versao"] if _treino else "v1"
        finais = {(tip.get("fixture_id"), tip.get("mercado")) for tip in tips_aprovadas}
        tips_auditaveis = list(tips_revisao) + list(bloqueadas_gate)
        for tip in tips_auditaveis:
            tip["approved_final"] = (tip.get("fixture_id"), tip.get("mercado")) in finais
        self.db.salvar_scan_audit(data, tips_auditaveis)
        self.db.salvar_live_watchlist(data, self._montar_itens_live_watch(tips_auditaveis))

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
                "prob_modelo": tip.get("prob_modelo"),
                "mercado": tip["mercado"],
                "odd_usada": tip.get("odd_usada"),
                "ev_percent": tip.get("ev_percent"),
                "bookmaker": tip.get("bookmaker") or tip.get("odd_fonte", ""),
                "modelo_versao": versao_atual,
                "features": tip.get("features", {}),
            })

        if candidatos:
            elegiveis_ids = {
                item["id"]
                for item in candidatos
                if (item.get("fixture_id"), item.get("mercado")) in {
                    (tip.get("fixture_id"), tip.get("mercado")) for tip in tips_janela
                }
            }
            status_map = {
                item["id"]: (
                    "released"
                    if (item.get("fixture_id"), item.get("mercado")) in finais else
                    "rejected"
                )
                for item in candidatos
                if item["id"] in elegiveis_ids
            }
            for status in ("released", "rejected"):
                ids = [cid for cid, value in status_map.items() if value == status]
                self.db.atualizar_status_candidatos(ids, status)

        combos = self._gerar_combos_por_janela(tips_aprovadas)
        if combos:
            print(f"\n🎰 Etapa 9: {len(combos)} combos gerados")
            for combo in combos:
                combo_payload = dict(combo)
                combo_payload["date"] = data
                self.db.salvar_combo(combo_payload)

        return {
            **base_result,
            "tips_bloqueadas_ev": bloqueadas_por_ev,
            "tips_enviadas_llm": len(tips_revisao),
            "tips_aprovadas_llm": len(tips_aprovadas_llm),
            "tips_rejeitadas_llm": tips_rejeitadas_llm,
            "tips_aprovadas": len(tips_aprovadas),
            "ev_positivas": tips_aprovadas,
            "combos": combos,
            "data": data,
            "mode": "release",
            "preselecionados": [],
            "release_reference": reference_time.isoformat(),
        }

    # ══════════════════════════════════════════════
    #  ETAPA 1: Scan de fixtures
    # ══════════════════════════════════════════════

    def _scan_fixtures(
        self,
        data: str,
        dias_adiante: int,
        *,
        reference_time: datetime | None = None,
        lookahead_minutes: int | None = None,
    ) -> list[dict]:
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

        if reference_time is not None and lookahead_minutes:
            limite = reference_time + timedelta(minutes=lookahead_minutes)
            filtradas = []
            for fixture in all_fixtures:
                kickoff = self._parse_fixture_datetime((fixture.get("fixture") or {}).get("date"))
                if kickoff is None:
                    continue
                if reference_time <= kickoff <= limite:
                    filtradas.append(fixture)
            return filtradas

        return all_fixtures

    def _finalizar_preselecao(
        self,
        data: str,
        fixtures: list[dict],
        previsoes: list[dict],
        tips_raw: list[dict],
        tips_gate: list[dict],
        tips_brutas: int,
        tips_pos_filtros: int,
        reference_time: datetime | None = None,
        lookahead_minutes: int | None = None,
    ) -> dict:
        """Salva candidatos da manhã e devolve só os jogos pré-selecionados."""
        candidatos = self._reduzir_para_jogos(tips_gate)
        fixtures_revisados = {
            int(item["fixture_id"])
            for item in self.db.scan_audit_por_data(data)
            if item.get("fixture_id") is not None
        }
        if fixtures_revisados:
            candidatos = [
                item for item in candidatos
                if item.get("fixture_id") not in fixtures_revisados
            ]
        self.db.limpar_scan_candidates(data)
        self.db.salvar_scan_candidates(data, candidatos)
        print(f"\n🗂️ Etapa 5: {len(candidatos)} jogo(s) pré-selecionado(s) para revisão T-30")
        return {
            "fixtures": len(fixtures),
            "previsoes": len(previsoes),
            "tips_brutas": tips_brutas,
            "tips_pos_filtros": tips_pos_filtros,
            "tips_bloqueadas_ev": 0,
            "tips_enviadas_llm": 0,
            "tips_aprovadas_llm": 0,
            "tips_rejeitadas_llm": [],
            "oportunidades": tips_raw,
            "ev_positivas": [],
            "combos": [],
            "data": data,
            "mode": "preselect",
            "preselecionados": candidatos,
            "reference_time": reference_time.isoformat() if reference_time else None,
            "lookahead_minutes": int(lookahead_minutes or 0),
        }

    def _reduzir_para_jogos(self, tips: list[dict]) -> list[dict]:
        """Escolhe os melhores candidatos por jogo para revisão posterior."""
        por_fixture: dict[int, list[dict]] = {}
        for tip in tips:
            por_fixture.setdefault(tip.get("fixture_id"), []).append(tip)

        candidatos = []
        for _, grupo in por_fixture.items():
            grupo.sort(key=lambda item: item.get("prob_modelo", 0), reverse=True)
            melhores = grupo[: min(3, len(grupo))]
            kickoff = melhores[0].get("date", "")
            release_group = self._release_group(kickoff)
            for tip in melhores:
                novo = dict(tip)
                novo["candidate_status"] = "pending"
                novo["release_group"] = release_group
                candidatos.append(novo)

        candidatos.sort(
            key=lambda item: (
                item.get("prob_modelo", 0),
                item.get("date", ""),
            ),
            reverse=True,
        )
        vistos = set()
        jogos = []
        for item in candidatos:
            fid = item.get("fixture_id")
            if fid in vistos:
                jogos.append(item)
                continue
            vistos.add(fid)
            jogos.append(item)
            if PRESELECT_MAX_JOGOS > 0 and len(vistos) >= PRESELECT_MAX_JOGOS:
                break
        allowed = {item.get("fixture_id") for item in jogos}
        return [item for item in candidatos if item.get("fixture_id") in allowed]

    def _filtrar_janela_liberacao(
        self,
        tips: list[dict],
        reference_time: datetime,
        test_mode: bool = False,
    ) -> list[dict]:
        """Seleciona candidatos cuja partida começa em cerca de 30 minutos."""
        if test_mode:
            return list(tips)
        escolhidas = []
        for tip in tips:
            kickoff = self._parse_fixture_datetime(tip.get("date"))
            if kickoff is None:
                continue
            delta = (kickoff - reference_time).total_seconds() / 60
            if RELEASE_LOOKAHEAD_MINUTES - RELEASE_WINDOW_MINUTES <= delta <= RELEASE_LOOKAHEAD_MINUTES + RELEASE_WINDOW_MINUTES:
                escolhidas.append(tip)
        return escolhidas

    def _gerar_combos_por_janela(self, tips: list[dict]) -> list[dict]:
        """Gera combos apenas entre jogos com horários próximos."""
        if not tips:
            return []
        tips_validas = []
        for tip in tips:
            kickoff = self._parse_fixture_datetime(tip.get("date"))
            if kickoff is None:
                continue
            novo = dict(tip)
            novo["_kickoff"] = kickoff
            tips_validas.append(novo)

        combos = self._gerar_combos(tips_validas)
        combos_validos = []
        for combo in combos:
            horarios = sorted(item["_kickoff"] for item in combo["tips"])
            if not horarios:
                continue
            janela = (horarios[-1] - horarios[0]).total_seconds() / 60
            if janela <= RELEASE_WINDOW_MINUTES:
                for item in combo["tips"]:
                    item.pop("_kickoff", None)
                combos_validos.append(combo)
        return combos_validos

    def _montar_itens_live_watch(self, tips: list[dict]) -> list[dict]:
        """Separa itens que merecem acompanhamento live após a revisão final."""
        itens = []
        for tip in tips:
            llm = tip.get("llm_validacao") or {}
            decisao = llm.get("decisao")
            payload = dict(tip)
            if tip.get("approved_final"):
                itens.append({
                    **payload,
                    "watch_type": "approved_prelive",
                    "status": "active",
                    "note": "Entrada liberada no pre-live e acompanhada ao vivo.",
                })
                continue

            if decisao == "REJECT":
                observacao = self._observacao_bloqueio_live(tip)
                if observacao:
                    itens.append({
                        **payload,
                        "watch_type": "blocked_recheck",
                        "status": "active",
                        "note": observacao,
                    })
        return itens

    @staticmethod
    def _parse_fixture_datetime(date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            dt_obj = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            return dt_obj.astimezone(ZoneInfo(TIMEZONE))
        except Exception:
            return None

    @staticmethod
    def _release_group(date_str: str) -> str:
        kickoff = Scanner._parse_fixture_datetime(date_str)
        if kickoff is None:
            return ""
        return kickoff.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _link_bet365_html(self, tip: dict | None = None) -> str:
        return f'<a href="{BET365_URL}">1xBet</a>'

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

    def _expandir_mercados(self, previsoes: list[dict], min_prob: float = 0.0) -> list[dict]:
        """
        Transforma cada previsão (1 jogo com N probabilidades) em
        N tips individuais (1 por mercado), filtrando pelo mínimo informado.

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
                if not prob or prob < min_prob:
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

        Após resolver categorias, aplica o teto por fixture somente se configurado.
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

            # Passo 2: limitar a MAX_TIPS_POR_JOGO por fixture (opcional)
            if MAX_TIPS_POR_JOGO and len(tips_filtradas) > MAX_TIPS_POR_JOGO:
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
            if s["conf_min"] <= prob < s["conf_max"] and self._strategy_rule_match(s, tip):
                return True  # Estratégia ativa encontrada — tip aprovada

        # Confiança fora de qualquer faixa ativa — bloquear
        return False

    def _strategy_rule_match(self, strategy: dict, tip: dict) -> bool:
        """Aplica condições extras da strategy sobre as features do jogo."""
        params_raw = strategy.get("params_json") or strategy.get("params")
        if not params_raw:
            return True

        if isinstance(params_raw, str):
            try:
                params = json.loads(params_raw)
            except json.JSONDecodeError:
                return True
        else:
            params = params_raw

        conditions = params.get("conditions") or []
        if not conditions:
            return True

        features = dict(tip.get("features") or {})
        features["model_prob"] = float(tip.get("prob_modelo", 0) or 0)

        for feature, op, threshold in conditions:
            runtime_feature = _STRATEGY_FEATURE_ALIASES.get(feature, feature)
            raw_value = features.get(runtime_feature)
            if raw_value is None and runtime_feature != feature:
                raw_value = features.get(feature)
            value = float(raw_value or 0)
            limit = float(threshold)
            if op == ">=" and value < self._strategy_relaxed_limit(limit, op, runtime_feature):
                return False
            if op == "<=" and value > self._strategy_relaxed_limit(limit, op, runtime_feature):
                return False
        return True

    def _strategy_relaxed_limit(self, threshold: float, op: str, feature: str) -> float:
        """Aplica tolerância numérica e uma folga leve nas condições das strategies."""
        if feature == "model_prob":
            relax = min(0.01, threshold * 0.01)
        else:
            relax = max(STRATEGY_EPSILON, abs(threshold) * STRATEGY_RELAX_PCT)
        if op == ">=":
            return threshold - relax
        if op == "<=":
            return threshold + relax
        return threshold

    # ══════════════════════════════════════════════
    #  ETAPA 6: Enriquecer tips com odds Pinnacle + EV
    # ══════════════════════════════════════════════

    def _enriquecer_odds(self, tips: list[dict]) -> list[dict]:
        """Consulta a OddsPapi/1xBet apenas para a shortlist final da janela."""
        if not tips:
            return tips
        if not ODDSPAPI_USE_PRELIVE:
            print("   ⚠️ OddsPapi pre-live desativada; todas as entradas serão barradas por falta de odd.")
            for tip in tips:
                tip["odd_status"] = "desativado"
                tip["odd_block_reason"] = "fixture_sem_odd"
            return tips

        enriquecidas = enriquecer_tips_com_odds_oddspapi(tips, phase_operacional="prelive_final")
        total_validas = sum(1 for tip in enriquecidas if (tip.get("odd_usada") or 0) >= 1.0)
        print(f"   📊 {total_validas}/{len(tips)} tips enriquecidas com OddsPapi/1xBet")
        return enriquecidas

    @staticmethod
    def _motivo_bloqueio_odds_ev(tip: dict) -> str:
        odd = tip.get("odd_usada")
        ev = tip.get("ev_percent")
        reason = tip.get("odd_block_reason")
        if reason == "acesso_live_restrito":
            return "A OddsPapi bloqueou o acesso live da 1xBet para este fixture."
        if reason == "fixture_sem_odd":
            return "A 1xBet não devolveu odd válida na OddsPapi para este mercado."
        if odd is None:
            return "A entrada ficou sem odd válida na etapa final."
        if odd < ODDSPAPI_SIMPLE_MIN_ODD:
            return f"Odd abaixo do mínimo operacional ({odd:.2f} < {ODDSPAPI_SIMPLE_MIN_ODD:.2f})."
        if ev is None:
            return "Não foi possível calcular o EV com a odd final."
        return f"EV abaixo do mínimo operacional ({ev:+.1f}% < +{ODDSPAPI_SIMPLE_MIN_EV_PERCENT:.1f}%)."

    def _aplicar_gate_odds_ev(self, tips: list[dict]) -> tuple[list[dict], list[dict]]:
        aprovadas: list[dict] = []
        bloqueadas: list[dict] = []
        for tip in tips:
            odd = tip.get("odd_usada")
            ev = tip.get("ev_percent")
            passou = (
                odd is not None
                and odd >= ODDSPAPI_SIMPLE_MIN_ODD
                and ev is not None
                and ev >= ODDSPAPI_SIMPLE_MIN_EV_PERCENT
            )
            if passou:
                tip["approved_odds_gate"] = True
                aprovadas.append(tip)
                continue

            tip["approved_odds_gate"] = False
            tip["approved_final"] = False
            tip.setdefault("odd_block_reason", "fixture_sem_odd" if odd is None else "gate_odds_ev")
            tip["llm_validacao"] = {
                "decisao": "REJECT",
                "confianca": 1.0,
                "motivo": self._motivo_bloqueio_odds_ev(tip),
            }
            bloqueadas.append(tip)
        return aprovadas, bloqueadas

    @staticmethod
    def _extrair_odd_mercado(jogo: dict, tip: dict) -> tuple[float, str]:
        """
        Compatibilidade com testes legados da extração por linha exata.
        """
        mercado_id = tip.get("mercado", "")
        mapping = _MERCADO_ODDS_MAP.get(mercado_id)
        if not mapping:
            return 0, ""

        market_key, outcome, point = mapping
        home_team = jogo.get("home_team", "")
        away_team = jogo.get("away_team", "")
        if outcome == "HOME":
            outcome = home_team
        elif outcome == "AWAY":
            outcome = away_team

        melhor_odd = 0.0
        melhor_casa = ""
        for bk in jogo.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                for oc in mkt.get("outcomes", []):
                    if oc.get("name") != outcome:
                        continue
                    if point is not None and oc.get("point") != point:
                        continue
                    price = oc.get("price", 0) or 0
                    if price > melhor_odd:
                        melhor_odd = float(price)
                        melhor_casa = bk.get("title", bk.get("key", ""))
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
            odd = t.get("odd_usada")
            ev = t.get("ev_percent")
            if odd is None or odd < ODDSPAPI_COMBO_LEG_MIN_ODD:
                continue
            if ev is None or ev < ODDSPAPI_COMBO_LEG_MIN_EV_PERCENT:
                continue
            vistas.add(fid)
            elegiveis.append(t)

        if len(elegiveis) < 2:
            return []

        candidatas = []

        for a, b in combinations(elegiveis, 2):
            prob = a["prob_modelo"] * b["prob_modelo"]
            odd_composta = (a.get("odd_usada") or 1.0) * (b.get("odd_usada") or 1.0)
            ev_composto = ((prob * odd_composta) - 1) * 100
            if prob >= COMBO_PROB_MIN and odd_composta >= ODDSPAPI_COMBO_MIN_ODD and ev_composto >= ODDSPAPI_COMBO_MIN_EV_PERCENT:
                candidatas.append({
                    "tipo": "dupla",
                    "tips": [a, b],
                    "prob_composta": round(prob, 4),
                    "odd_composta": round(odd_composta, 2),
                    "ev_composto_percent": round(ev_composto, 1),
                })

        if len(elegiveis) >= 3:
            for a, b, c in combinations(elegiveis, 3):
                prob = a["prob_modelo"] * b["prob_modelo"] * c["prob_modelo"]
                odd_composta = (
                    (a.get("odd_usada") or 1.0)
                    * (b.get("odd_usada") or 1.0)
                    * (c.get("odd_usada") or 1.0)
                )
                ev_composto = ((prob * odd_composta) - 1) * 100
                if prob >= COMBO_PROB_MIN and odd_composta >= ODDSPAPI_COMBO_MIN_ODD and ev_composto >= ODDSPAPI_COMBO_MIN_EV_PERCENT:
                    candidatas.append({
                        "tipo": "tripla",
                        "tips": [a, b, c],
                        "prob_composta": round(prob, 4),
                        "odd_composta": round(odd_composta, 2),
                        "ev_composto_percent": round(ev_composto, 1),
                    })

        candidatas.sort(
            key=lambda item: (
                item.get("ev_composto_percent", -999),
                item.get("odd_composta", 0),
                item["prob_composta"],
                len(item["tips"]),
            ),
            reverse=True,
        )

        fixtures_usadas = set()
        combos_total = []
        for combo in candidatas:
            fids = {t["fixture_id"] for t in combo["tips"]}
            if fids & fixtures_usadas:
                continue

            combos_total.append(combo)
            fixtures_usadas.update(fids)

        return combos_total

    # ══════════════════════════════════════════════
    #  Preferencias do usuario (Mini App)
    # ══════════════════════════════════════════════

    def _aplicar_preferencias_usuario(self, tips: list[dict]) -> list[dict]:
        """Aplica filtros runtime configurados na Mini App."""
        prefs = get_runtime_preferences(TELEGRAM_CHAT_ID)
        if not prefs:
            return tips

        filtradas = tips
        favoritas = set(prefs.get("favorite_leagues", []))
        min_ev = prefs.get("min_ev")

        if favoritas:
            antes = len(filtradas)
            filtradas = [t for t in filtradas if t.get("league_id") in favoritas]
            if antes != len(filtradas):
                print(f"   👤 Preferencias: {antes - len(filtradas)} tips fora das ligas favoritas")

        if min_ev is not None:
            antes = len(filtradas)
            filtradas = [
                t for t in filtradas
                if t.get("ev_percent") is None or t.get("ev_percent", -999) >= float(min_ev)
            ]
            if antes != len(filtradas):
                print(f"   👤 Preferencias: {antes - len(filtradas)} tips abaixo do EV minimo do usuario")

        return filtradas

    # ══════════════════════════════════════════════
    #  GUARD RAIL: Auto-pause se modelo está degradando
    # ══════════════════════════════════════════════

    def _verificar_auto_pause(self) -> tuple[bool, str]:
        """
        Verifica se o scanner deve ser pausado por performance ruim.

        Filtra apenas previsões do modelo ATIVO (versão corrente),
        ignorando histórico de modelos anteriores. Isso garante que
        após retreino, o scanner "reseta" sem arrastar ruído antigo.

        Em vez de pausar o bot inteiro, coloca em quarentena apenas
        slices liga × mercado degradados.

        Retorna (pausado: bool, motivo: str) por compatibilidade.
        """
        treino = self.db.ultimo_treino()
        versao = treino["modelo_versao"] if treino else None
        slices_ruins = self.db.slices_degradados(
            modelo_versao=versao,
            min_amostras=max(5, ROI_PAUSE_MIN_BETS // 4),
            roi_threshold=ROI_PAUSE_THRESHOLD,
            acc_threshold=DEGRADATION_ACC_MIN * 100,
        )
        if not slices_ruins:
            return False, ""

        desativados = []
        for item in slices_ruins:
            alteradas = self.db.desativar_strategia_slice(item["league_id"], item["mercado"])
            if alteradas > 0:
                desativados.append(item)

        if desativados:
            self._strategies = self.db.strategies_ativas()
            resumo = ", ".join(
                f"L{item['league_id']}:{item['mercado']}"
                for item in desativados[:4]
            )
            extra = "..." if len(desativados) > 4 else ""
            print(f"   🛡️ Quarentena seletiva: {len(desativados)} slice(s) desativados ({resumo}{extra})")
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
    def _emoji_mercado(mercado: str) -> str:
        if mercado in {"over15", "under15", "over25", "under25", "over35", "under35",
                       "over05_ht", "under05_ht", "over15_ht", "under15_ht",
                       "over05_2t", "under05_2t", "over15_2t", "under15_2t"}:
            return "⚽"
        if mercado in {"corners_over_85", "corners_under_85", "corners_over_95",
                       "corners_under_95", "corners_over_105", "corners_under_105"}:
            return "⛳"
        if mercado in {"h2h_home", "h2h_draw", "h2h_away", "ht_home", "ht_draw", "ht_away"}:
            return "🎯"
        return "📌"

    def _descricao_mercado(self, tip: dict) -> str:
        desc = tip.get("descricao", tip.get("mercado", "?"))
        emoji = self._emoji_mercado(tip.get("mercado", ""))
        if desc.startswith(("⚽", "⛳", "🎯", "🤝", "📌")):
            return desc
        return f"{emoji} {desc}"

    @staticmethod
    def _quebrar_motivo_curto(texto: str) -> list[str]:
        texto = (texto or "").strip()
        if not texto:
            return []
        partes = [
            parte.strip(" .")
            for parte in texto.replace(":", ".").split(".")
            if parte.strip(" .")
        ]
        return partes

    @staticmethod
    def _resumo_lesoes(ctx: dict) -> str | None:
        lesoes = ctx.get("lesoes") or []
        if not lesoes:
            return None
        nomes = []
        for item in lesoes[:3]:
            jogador = (item.get("jogador") or "").strip()
            if jogador and jogador != "?":
                nomes.append(jogador)
        if not nomes:
            return None
        return f"Desfalques relevantes: {', '.join(nomes)}"

    @staticmethod
    def _resumo_externo(ctx: dict) -> list[str]:
        lookup = ctx.get("market_lookup") or {}
        fatores = []
        for bruto in (
            lookup.get("weather_summary"),
            f"Gramado: {str(lookup['field_conditions']).strip(' .')}" if lookup.get("field_conditions") else None,
            f"Rotacao: {str(lookup['rotation_risk']).strip(' .')}" if lookup.get("rotation_risk") else None,
            lookup.get("motivation_context"),
            lookup.get("news_summary"),
        ):
            if not bruto:
                continue
            fator = str(bruto).strip(" .")
            fator_low = fator.lower()
            if any(
                token in fator_low
                for token in (
                    "sem inform",
                    "desconhecid",
                    "podem nao ser",
                    "pode nao ser",
                    "nao confirmado",
                    "sem previs",
                    "incerto",
                )
            ):
                continue
            if any(
                token in fator_low
                for token in (
                    "buscam lider",
                    "parte alta da tabela",
                    "parte baixa da tabela",
                    "jogo crucial",
                    "confronto direto",
                )
            ):
                continue
            fatores.append(fator)
        return [f for f in fatores if f]

    @staticmethod
    def _fmt_pct(valor: float | None) -> str | None:
        if valor is None:
            return None
        try:
            return f"{round(float(valor) * 100)}%"
        except Exception:
            return None

    @staticmethod
    def _fmt_num(valor: float | None) -> str | None:
        if valor is None:
            return None
        try:
            return f"{float(valor):.1f}"
        except Exception:
            return None

    def _fatores_mercado_especificos(self, tip: dict | None, bloqueado: bool = False) -> list[str]:
        if not tip:
            return []

        mercado = (tip.get("mercado") or "").lower()
        features = tip.get("features") or {}
        fatores = []
        prob_modelo = self._fmt_pct(tip.get("prob_modelo"))
        prob_over25 = self._fmt_pct(tip.get("prob_over25"))
        prob_home = self._fmt_pct(tip.get("prob_home"))
        prob_draw = self._fmt_pct(tip.get("prob_draw"))
        prob_away = self._fmt_pct(tip.get("prob_away"))

        total_xg = self._fmt_num(features.get("total_xg_5"))
        total_shots_on = self._fmt_num(features.get("total_shots_on_5"))
        shots_diff = self._fmt_num(features.get("shots_on_diff_5"))
        home_cs = self._fmt_pct(features.get("home_cs_5"))
        away_cs = self._fmt_pct(features.get("away_cs_5"))
        home_fts = self._fmt_pct(features.get("home_fts_5"))
        away_fts = self._fmt_pct(features.get("away_fts_5"))
        home_corners = self._fmt_num(features.get("home_corners_5"))
        away_corners = self._fmt_num(features.get("away_corners_5"))
        home_goals_ht = self._fmt_num(features.get("home_goals_ht_5"))
        away_goals_ht = self._fmt_num(features.get("away_goals_ht_5"))

        if bloqueado:
            return []

        if mercado.startswith("over") and "corners" not in mercado:
            if prob_modelo:
                fatores.append(f"Leitura do modelo ainda fica forte para esse over em {prob_modelo}")
            if total_xg and float(total_xg) >= 2.4:
                fatores.append(f"Os dois lados chegam criando volume para jogo mais aberto ({total_xg} xG combinados)")
            if total_shots_on and float(total_shots_on) >= 9.0:
                fatores.append(f"As duas equipes vem finalizando bem no alvo ({total_shots_on} no recorte recente)")
            if "ht" in mercado and (home_goals_ht or away_goals_ht):
                fatores.append(
                    f"O 1T recente dos dois lados nao costuma passar em branco: casa {home_goals_ht or '?'} | fora {away_goals_ht or '?'} gol(s)"
                )
            if "2t" in mercado and shots_diff and abs(float(shots_diff)) >= 2.0:
                fatores.append("O perfil recente aponta para jogo que cresce de volume depois do intervalo")

        elif mercado.startswith("under") and "corners" not in mercado:
            if prob_modelo:
                fatores.append(f"O modelo ainda sustenta esse under em {prob_modelo}")
            if prob_over25 and float(prob_over25.strip('%')) <= 45:
                fatores.append(f"O modelo nao ve forca suficiente para um jogo acima de 2.5 gols ({prob_over25})")
            if home_cs or away_cs:
                fatores.append(f"Os dois lados chegam cedendo pouco espaco: clean sheets casa {home_cs or '?'} | fora {away_cs or '?'}")
            if home_fts or away_fts:
                fatores.append(f"Ha sinal recente de ataque travando em alguns jogos: casa {home_fts or '?'} | fora {away_fts or '?'} sem marcar")
            if "ht" in mercado and (home_goals_ht or away_goals_ht):
                fatores.append(
                    f"O 1T recente dos dois times tem cara mais controlada: casa {home_goals_ht or '?'} | fora {away_goals_ht or '?'} gol(s)"
                )

        elif mercado.startswith("h2h") or mercado.startswith("ht_"):
            if mercado.endswith("home") and prob_home:
                fatores.append(f"O mandante segue na frente na leitura do modelo ({prob_home})")
            elif mercado.endswith("draw") and prob_draw:
                fatores.append(f"O jogo aparece equilibrado o suficiente para empate em {prob_draw}")
            elif mercado.endswith("away") and prob_away:
                fatores.append(f"O visitante ainda aparece competitivo na leitura do modelo ({prob_away})")
            if shots_diff and abs(float(shots_diff)) >= 2.0:
                lado = "mandante" if float(shots_diff) > 0 else "visitante"
                fatores.append(f"O volume recente de finalizacao pende para o {lado}")

        elif mercado.startswith("corners_"):
            if prob_modelo:
                fatores.append(f"O modelo ainda sustenta esse mercado de escanteios em {prob_modelo}")
            if home_corners or away_corners:
                fatores.append(f"Media recente de cantos: casa {home_corners or '?'} | fora {away_corners or '?'}")
            if total_shots_on and float(total_shots_on) >= 9.0 and "over" in mercado:
                fatores.append("O volume recente de finalizacao combina com jogo que empilha ataques e cantos")
            if total_shots_on and float(total_shots_on) <= 7.5 and "under" in mercado:
                fatores.append("O volume recente de finalizacao ajuda um jogo com menos cantos")

        return [f for f in fatores if f]

    @staticmethod
    def _tese_mercado(tip: dict | None, bloqueado: bool) -> str | None:
        mercado = ((tip or {}).get("mercado") or "").lower()
        if mercado.startswith("under") and "corners" not in mercado:
            return (
                "Minha leitura principal aqui ainda e de jogo mais controlado do que aberto."
                if not bloqueado else
                "O modelo gostou do under, mas o contexto final nao sustentou a entrada."
            )
        if mercado.startswith("over") and "corners" not in mercado:
            return (
                "Minha leitura principal aqui ainda e de jogo com espaco para gol."
                if not bloqueado else
                "O modelo gostou do over, mas o contexto final nao sustentou a entrada."
            )
        if mercado.startswith("corners_over"):
            return (
                "Minha leitura principal aqui ainda e de jogo com pressao suficiente pelos lados."
                if not bloqueado else
                "O modelo gostou desse mercado de cantos, mas o contexto final nao sustentou a entrada."
            )
        if mercado.startswith("corners_under"):
            return (
                "Minha leitura principal aqui ainda e de jogo com menos acao pelos lados."
                if not bloqueado else
                "O modelo gostou desse mercado de cantos, mas o contexto final nao sustentou a entrada."
            )
        if mercado.startswith("h2h") or mercado.startswith("ht_"):
            return (
                "Minha leitura principal aqui ainda sustenta esse lado."
                if not bloqueado else
                "O modelo ate enxergou valor nesse lado, mas o contexto final nao sustentou a entrada."
            )
        return None

    @staticmethod
    def _risco_mercado(tip: dict | None, bloqueado: bool) -> str | None:
        if bloqueado:
            return None
        mercado = ((tip or {}).get("mercado") or "").lower()
        if mercado.startswith("under") and "corners" not in mercado:
            return "Se sair gol cedo, essa leitura perde bastante forca."
        if mercado.startswith("over") and "corners" not in mercado:
            return "Se o ritmo cair depois do inicio, essa leitura perde valor rapido."
        if mercado.startswith("corners_over"):
            return "Se os ataques pelos lados sumirem, esse mercado esfria rapido."
        if mercado.startswith("corners_under"):
            return "Se o jogo abrir e empilhar cruzamentos, essa leitura perde valor."
        if mercado.startswith("h2h") or mercado.startswith("ht_"):
            return "Se o jogo equilibrar demais, esse lado perde sustentacao."
        return None

    @staticmethod
    def _conclusao_mercado(tip: dict | None, bloqueado: bool) -> str | None:
        mercado = ((tip or {}).get("mercado") or "").lower()
        if bloqueado:
            if mercado.startswith("under") and "corners" not in mercado:
                return "Por isso preferi ficar fora desse under por enquanto."
            if mercado.startswith("over") and "corners" not in mercado:
                return "Por isso preferi nao insistir nesse over agora."
            if mercado.startswith("corners_"):
                return "Por isso preferi nao liberar esse mercado de cantos agora."
            if mercado.startswith("h2h") or mercado.startswith("ht_"):
                return "Por isso preferi nao liberar esse lado agora."
            return "Por isso preferi ficar de fora nesta janela."
        else:
            if mercado.startswith("under") and "corners" not in mercado:
                return "Por isso esse under segue de pe."
            if mercado.startswith("over") and "corners" not in mercado:
                return "Por isso esse over segue vivo."
            if mercado.startswith("corners_"):
                return "Por isso esse mercado de cantos segue de pe."
            if mercado.startswith("h2h") or mercado.startswith("ht_"):
                return "Por isso esse lado segue sustentado."
            return "Por isso a entrada segue mantida."

    @staticmethod
    def _observacao_bloqueio_live(tip: dict | None) -> str | None:
        mercado = (tip or {}).get("mercado", "")
        if mercado.startswith("over") and "corners" not in mercado:
            return "Se o ritmo subir e o volume ofensivo aparecer, isso pode virar leitura de live"
        if mercado.startswith("under") and "corners" not in mercado:
            return "Se o jogo continuar travado, isso reforca a leitura de nao entrar contra esse cenario"
        if mercado.startswith("corners_over"):
            return "So vale reavaliar se a pressao pelos lados e o volume de cruzamentos crescerem"
        if mercado.startswith("corners_under"):
            return "So muda se o jogo abrir e comecar a empilhar ataques pelos lados"
        if mercado.startswith("h2h") or mercado.startswith("ht_"):
            return "So reavalio se o favoritismo ficar claro dentro de campo"
        return "Prefiro esperar o jogo mostrar algo diferente antes de olhar de novo"

    def _formatar_resumo_revisao(self, motivo: str, bloqueado: bool, tip: dict | None = None) -> list[str]:
        partes = self._quebrar_motivo_curto(motivo)
        ctx = (tip or {}).get("llm_contexto") or {}
        candidatos = []

        for fator in self._fatores_mercado_especificos(tip, bloqueado=bloqueado):
            if fator not in candidatos:
                candidatos.append(fator)

        lesoes = self._resumo_lesoes(ctx)
        if lesoes:
            candidatos.append(lesoes)

        for fator in self._resumo_externo(ctx):
            if fator not in candidatos:
                candidatos.append(fator)

        for parte in partes:
            if parte not in candidatos:
                candidatos.append(parte)

        if not candidatos:
            return []

        linhas = []
        decisao = "Entrada cancelada nesta janela." if bloqueado else "Entrada mantida nesta janela."
        linhas.append(f"  <b>{decisao}</b>")

        tese = self._tese_mercado(tip, bloqueado)
        if tese:
            linhas.append(f"  {tese}")

        fatores = candidatos[:3]
        for fator in fatores:
            linhas.append(f"  • {fator}.")

        risco = self._risco_mercado(tip, bloqueado)
        if risco:
            linhas.append(f"  <i>{risco}</i>")

        conclusao = self._conclusao_mercado(tip, bloqueado)
        if conclusao:
            linhas.append(f"  <i>{conclusao}</i>")
        if bloqueado:
            observacao = self._observacao_bloqueio_live(tip)
            if observacao:
                linhas.append(f"  <i>{observacao}.</i>")
        else:
            resto = candidatos[3:]
            if resto:
                linhas.append(f"  <i>{resto[0]}.</i>")

        return linhas

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

    def formatar_relatorio(self, resultado: dict) -> list[tuple[str, list]]:
        """
        Formata resultado do scanner para envio no Telegram (HTML).

        Retorna uma lista de tuplas (texto_html, botoes).
        Cada bloco eh organizado por secao para ficar mais legivel no chat.
        """
        from collections import defaultdict

        if resultado.get("pausado"):
            return [(
                "<b>Scanner pausado</b>\n\n"
                f"Motivo: {resultado.get('motivo_pausa', 'modelo degradado')}\n\n"
                "Use /treinar para retreinar ou /metricas para ver detalhes.",
                [],
            )]

        tips = resultado.get("ev_positivas", [])
        data_raw = resultado.get("data", "hoje")
        data_br = self._data_br(data_raw)
        msgs: list[tuple[str, list]] = []
        mode = resultado.get("mode", "release")

        if mode == "preselect":
            candidatos = resultado.get("preselecionados", [])
            total_jogos = len({tip.get("fixture_id") for tip in candidatos})
            total_mercados = len(candidatos)
            lookahead_minutes = int(resultado.get("lookahead_minutes") or 0)
            lookahead_horas = max(1, lookahead_minutes // 60) if lookahead_minutes else 0
            header = "\n".join([
                f"<b>🗓️ Radar das proximas {lookahead_horas}h | {data_br}</b>" if lookahead_horas else f"<b>🗓️ Observação do dia {data_br}</b>",
                f"Agora separei <b>{total_jogos}</b> jogo(s) para observar." if lookahead_horas else f"Hoje separei <b>{total_jogos}</b> jogo(s) para observar.",
                f"• Jogos analisados: <b>{resultado['fixtures']}</b>",
                f"• Jogos com previsão: <b>{resultado['previsoes']}</b>",
                f"• Mercados candidatos: <b>{resultado.get('tips_brutas', 0)}</b>",
                f"• Após filtros internos: <b>{resultado.get('tips_pos_filtros', 0)}</b>",
                f"• No radar agora: <b>{total_mercados}</b> mercado(s) em <b>{total_jogos}</b> jogo(s)",
                "",
                "ℹ️ Por enquanto vou deixar só os jogos no radar.",
                "⏳ Os mercados saem na revisão final, 30 min antes de cada partida.",
            ])
            msgs.append((header, []))
            por_liga = defaultdict(lambda: defaultdict(list))
            for tip in candidatos:
                por_liga[tip.get("league_id", 0)][tip.get("fixture_id")].append(tip)
            for lid in sorted(por_liga.keys(), key=lambda item: _LEAGUE_NOME.get(item, f"Liga {item}")):
                linhas = [f"<b>👀 {_LEAGUE_NOME.get(lid, f'Liga {lid}')}</b>"]
                fixtures_ordenados = sorted(
                    por_liga[lid].values(),
                    key=lambda items: items[0].get("date", "") if items else "",
                )
                for tips_fixture in fixtures_ordenados:
                    tip = max(tips_fixture, key=lambda item: item.get("prob_modelo", 0))
                    horario = self._horario_local(tip.get("date", ""))
                    linhas.append(
                        f"• <code>{tip.get('home_name', '?')}</code> <b>x</b> <code>{tip.get('away_name', '?')}</code>"
                        f" <i>({horario})</i>"
                    )
                    if len(tips_fixture) > 1:
                        linhas.append(f"  <i>{len(tips_fixture)} mercados em observação nesta partida.</i>")
                msgs.append(("\n".join(linhas), []))
            return msgs

        aprovadas_revisao = int(resultado.get("tips_aprovadas_llm") or 0)
        rejeitadas_revisao = len(resultado.get("tips_rejeitadas_llm", []))
        mercados_revisados = int(resultado.get("tips_enviadas_llm") or 0)
        if mercados_revisados == 0:
            return []
        ref_raw = resultado.get("reference_time")
        janela_label = data_br
        if ref_raw:
            try:
                ref_dt = datetime.fromisoformat(str(ref_raw).replace("Z", "+00:00"))
                ref_local = ref_dt.astimezone(ZoneInfo(TIMEZONE))
                janela_label = ref_local.strftime("%d/%m %H:%M")
            except Exception:
                janela_label = data_br
        header_lines = [
            f"<b>🚨 Revisão T-30 | {janela_label}</b>",
            f"Revisei <b>{mercados_revisados}</b> mercado(s) nesta janela.",
            f"• Jogos analisados: <b>{resultado['fixtures']}</b>",
            f"• Jogos com previsão: <b>{resultado['previsoes']}</b>",
        ]
        if resultado.get("tips_brutas") is not None:
            header_lines.append(f"• Mercados candidatos: <b>{resultado['tips_brutas']}</b>")
        if resultado.get("tips_pos_filtros") is not None:
            header_lines.append(f"• Após filtros internos: <b>{resultado['tips_pos_filtros']}</b>")
        if resultado.get("tips_bloqueadas_ev") is not None:
            header_lines.append(f"• Bloqueadas por EV: <b>{resultado['tips_bloqueadas_ev']}</b>")
        if resultado.get("tips_enviadas_llm") is not None:
            header_lines.append(f"• Mercados revisados: <b>{resultado['tips_enviadas_llm']}</b>")
        if resultado.get("tips_aprovadas_llm") is not None:
            header_lines.append(
                f"• Resultado da revisão: <b>{aprovadas_revisao}</b> liberados | <b>{rejeitadas_revisao}</b> barrados"
            )
        header = "\n".join(header_lines)

        if not tips:
            resumo_sem_liberacao = (
                header
                + "\n\n<b>🚫 Nenhum mercado foi liberado nesta janela.</b>\n"
                + "<i>Olhei o contexto de perto, mas preferi não liberar entrada agora.</i>"
            )
            msgs.append((resumo_sem_liberacao, []))
        else:
            msgs.append((header + f"\n• Entradas liberadas: <b>{len(tips)}</b>", []))

        por_liga = defaultdict(lambda: defaultdict(list))
        for tip in tips:
            por_liga[tip.get("league_id", 0)][tip.get("fixture_id", 0)].append(tip)

        ligas_ordenadas = sorted(
            por_liga.keys(),
            key=lambda lid: _LEAGUE_NOME.get(lid, f"Liga {lid}"),
        )

        for lid in ligas_ordenadas:
            fixtures_da_liga = por_liga[lid]
            nome_liga = _LEAGUE_NOME.get(lid, f"Liga {lid}")
            linhas = [f"<b>✅ Entradas liberadas | {nome_liga}</b>"]
            fixtures_ordenados = sorted(
                fixtures_da_liga.items(),
                key=lambda item: item[1][0].get("date", ""),
            )

            for _, fix_tips in fixtures_ordenados:
                fix_tips.sort(key=lambda x: x.get("prob_modelo", 0), reverse=True)
                primeira = fix_tips[0]
                horario = self._horario_local(primeira.get("date", ""))
                data_jogo = self._data_local(primeira.get("date", ""))

                agenda = []
                if horario:
                    agenda.append(horario)
                if data_jogo and data_jogo != data_br:
                    agenda.append(data_jogo)
                agenda_txt = f" ({' | '.join(agenda)})" if agenda else ""

                linhas.append(
                    f"\n🏟️ <code>{primeira.get('home_name', '?')}</code> <b>x</b> "
                    f"<code>{primeira.get('away_name', '?')}</code>{agenda_txt}"
                )

                for tip in fix_tips:
                    prob = tip.get("prob_modelo", 0)
                    desc = self._descricao_mercado(tip)
                    odd = tip.get("odd_usada") or tip.get("odd_pinnacle") or tip.get("odd") or 0
                    ev = tip.get("ev_percent")
                    casa = tip.get("bookmaker") or tip.get("odd_fonte") or ODDSPAPI_BOOKMAKER_LABEL or PREFERRED_BOOKMAKER_LABEL

                    detalhes = [f"Conf {prob:.0%}"]
                    if odd and odd > 1:
                        detalhes.append(f"Odd {odd:.2f}")
                    if ev is not None:
                        detalhes.append(f"EV {ev:+.1f}%")
                    detalhes.append(casa)

                    linhas.append(f"• <b>{desc}</b>")
                    linhas.append(f"  <i>{' | '.join(detalhes)}</i>")
                    linhas.append(f"  🔗 {self._link_bet365_html(tip)}")

                    llm = tip.get("llm_validacao")
                    if llm and llm.get("motivo") and "desativado" not in llm.get("motivo", ""):
                        linhas.extend(self._formatar_resumo_revisao(llm["motivo"], bloqueado=False, tip=tip))

            msgs.append(("\n".join(linhas).rstrip(), []))

        combos = resultado.get("combos", [])
        if combos:
            linhas_combo = ["<b>🎰 Combos que mantive</b>"]
            for i, combo in enumerate(combos, 1):
                tipo_label = "Dupla" if combo["tipo"] == "dupla" else "Tripla"
                linhas_combo.append(f"\n<b>{tipo_label} #{i}</b> | 🔗 Conf composta {combo['prob_composta']:.0%}")
                for t in combo["tips"]:
                    desc = self._descricao_mercado(t)
                    linhas_combo.append(
                        f"• <code>{t.get('home_name', '?')}</code> <b>x</b> "
                        f"<code>{t.get('away_name', '?')}</code>"
                    )
                    linhas_combo.append(f"  <i>{desc} | {t.get('prob_modelo', 0):.0%}</i>")
            msgs.append(("\n".join(linhas_combo).rstrip(), []))

        metricas = self.db.metricas_modelo()
        if metricas["total"] > 0:
            perf = (
                "<b>📈 Performance acumulada</b>\n"
                f"• Accuracy: {metricas['accuracy']}% ({metricas['acertos']}/{metricas['total']})\n"
                f"• ROI: {metricas['roi']:+.1f}%"
            )
            msgs.append((perf, []))

        rejeitadas = resultado.get("tips_rejeitadas_llm", [])
        if rejeitadas:
            por_liga_rejeitada = defaultdict(list)
            for tip in rejeitadas:
                por_liga_rejeitada[tip.get("league_id", 0)].append(tip)

            for lid in sorted(por_liga_rejeitada.keys(), key=lambda item: _LEAGUE_NOME.get(item, f"Liga {item}")):
                nome_liga = _LEAGUE_NOME.get(lid, f"Liga {lid}")
                linhas = [f"<b>🚫 Entradas barradas | {nome_liga}</b>"]
                tips_liga = sorted(
                    por_liga_rejeitada[lid],
                    key=lambda item: (item.get("date", ""), -item.get("prob_modelo", 0)),
                )
                for tip in tips_liga:
                    llm = tip.get("llm_validacao") or {}
                    linhas.append(
                        f"\n<code>{tip.get('home_name', '?')}</code> <b>x</b> <code>{tip.get('away_name', '?')}</code>"
                    )
                    linhas.append(
                        f"• <b>{self._descricao_mercado(tip)}</b> | Conf {tip.get('prob_modelo', 0):.0%}"
                    )
                    linhas.append(f"  🔗 {self._link_bet365_html(tip)}")
                    if llm.get("motivo"):
                        linhas.extend(self._formatar_resumo_revisao(llm["motivo"], bloqueado=True, tip=tip))
                    observacao_live = self._observacao_bloqueio_live(tip)
                    if observacao_live:
                        linhas.append("  👀 <i>Segue em observação para live.</i>")
                        linhas.append(f"  • {observacao_live}.")
                msgs.append(("\n".join(linhas).rstrip(), []))

        return msgs

if __name__ == "__main__":
    # Execução manual para teste
    scanner = Scanner()
    resultado = scanner.executar()
    msgs = scanner.formatar_relatorio(resultado)
    for texto, botoes in msgs:
        print(texto)
        if botoes:
            print(f"  [{len(botoes)} botões ✏️ Odd]")
        print()

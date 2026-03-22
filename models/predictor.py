"""
Preditor de resultados e calculador de EV (Expected Value).

Usa os modelos XGBoost treinados para prever probabilidades de:
  - Resultado 1x2 (home/draw/away) — Full Time
  - Over/Under 1.5 / 2.5 / 3.5 gols
  - BTTS (ambos marcam)
  - Resultado 1T (home/draw/away) — Primeiro Tempo
  - HT/FT (combinado 1T + FT — 9 classes)

Depois compara com odds reais para calcular EV:
  EV = (probabilidade_modelo × odd) - 1

Um EV positivo indica valor esperado favorável ao apostador.

Uso:
  from models.predictor import Predictor
  pred = Predictor(db)
  resultado = pred.prever_jogo(fixture)
  oportunidades = pred.calcular_ev(resultado, odds)
"""

import gc
import os
import json
from collections import OrderedDict
import numpy as np
from config import PREDICTOR_MAX_LIGAS_CACHE
from data.database import Database
from models.features import FeatureExtractor
from models.feature_factory import FeatureFactory

try:
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")


class Predictor:
    """
    Gera previsões e calcula EV usando modelos XGBoost treinados.

    100% per-league: usa data/models/league_{id}/{nome}.json.
    Se a liga não tem modelo treinado, retorna None (sem fallback global).
    """

    # Lista completa de modelos: 14 modelos pré-live
    ALL_MODELS = [
        "resultado_1x2", "over_under_15", "over_under_25", "over_under_35",
        "btts", "resultado_ht", "htft",
        "over_under_05_ht", "over_under_15_ht",
        "over_under_05_2t", "over_under_15_2t",
        "corners_over_85", "corners_over_95", "corners_over_105",
    ]

    def __init__(self, db: Database):
        self.db = db
        self.fe = FeatureExtractor(db)
        self.ff = FeatureFactory(db)
        # Cache de modelos per-league — carregados sob demanda (lazy loading)
        self._modelos_liga = OrderedDict()  # {league_id: {nome: Booster}}
        # Cache de mapas de features por liga — {league_id: {modelo: [feat_names]}}
        self._feature_maps = {}
        self._max_ligas_cache = int(PREDICTOR_MAX_LIGAS_CACHE or 0)
        print(f"[Predictor] ✅ Inicializado (100% per-league, features dinâmicas)")

    def _carregar_modelos_liga(self, league_id: int) -> dict:
        """
        Carrega modelos de uma liga específica (lazy loading com cache).

        Também carrega feature_map.json se existir (features dinâmicas).
        Retorna dict {nome: Booster} dos modelos encontrados para a liga.
        Se a liga não tem modelos próprios, retorna dict vazio.
        """
        if league_id in self._modelos_liga:
            self._modelos_liga.move_to_end(league_id)
            return self._modelos_liga[league_id]

        modelos = {}
        liga_dir = os.path.join(MODELS_DIR, f"league_{league_id}")

        if os.path.isdir(liga_dir):
            for nome in self.ALL_MODELS:
                path = os.path.join(liga_dir, f"{nome}.json")
                if os.path.exists(path):
                    model = xgb.Booster()
                    model.load_model(path)
                    modelos[nome] = model

            # Carregar mapa de features dinâmicas (se existir)
            map_path = os.path.join(liga_dir, "feature_map.json")
            if os.path.exists(map_path):
                with open(map_path) as f:
                    self._feature_maps[league_id] = json.load(f)

        if modelos:
            tem_evo = league_id in self._feature_maps
            print(f"[Predictor] 🏟️ Liga {league_id}: {len(modelos)} modelos"
                  f"{' (features evoluídas)' if tem_evo else ''}")

        self._modelos_liga[league_id] = modelos
        self._modelos_liga.move_to_end(league_id)
        self._aplicar_limite_cache_modelos()
        return modelos

    def _aplicar_limite_cache_modelos(self):
        if self._max_ligas_cache <= 0:
            return
        while len(self._modelos_liga) > self._max_ligas_cache:
            league_id, _ = self._modelos_liga.popitem(last=False)
            self._feature_maps.pop(league_id, None)
            gc.collect()

    def descartar_modelos_liga(self, league_id: int | None):
        if league_id is None:
            return
        removidos = self._modelos_liga.pop(league_id, None)
        self._feature_maps.pop(league_id, None)
        if removidos is not None:
            gc.collect()

    def limpar_cache_modelos(self):
        if not self._modelos_liga and not self._feature_maps:
            return
        self._modelos_liga.clear()
        self._feature_maps.clear()
        gc.collect()

    def _get_modelo(self, nome: str, league_id: int = None):
        """
        Retorna o modelo per-league para um mercado + liga.

        Sem fallback global — se a liga não tem modelo, retorna None.
        """
        if league_id:
            modelos_liga = self._carregar_modelos_liga(league_id)
            if nome in modelos_liga:
                return modelos_liga[nome]
        return None

    def modelo_pronto(self, league_id: int = None) -> bool:
        """
        Verifica se existem modelos treinados.

        Se league_id informado: verifica se a liga tem pelo menos o modelo 1x2.
        Se não informado: verifica se ALGUMA liga tem modelos no disco.
        """
        if league_id:
            modelos = self._carregar_modelos_liga(league_id)
            return "resultado_1x2" in modelos

        # Verificar se há alguma pasta league_* com modelos
        if os.path.isdir(MODELS_DIR):
            for entry in os.listdir(MODELS_DIR):
                if entry.startswith("league_"):
                    liga_dir = os.path.join(MODELS_DIR, entry)
                    if os.path.exists(os.path.join(liga_dir, "resultado_1x2.json")):
                        return True
        return False

    def _dmat_para_modelo(self, feats: dict, nome_modelo: str,
                           league_id: int) -> "xgb.DMatrix":
        """
        Cria DMatrix com as features corretas para o modelo.

        Se feature_map.json existe para a liga, usa features evoluídas
        (subconjunto selecionado pelo genético). Caso contrário, usa
        as 51 features estáticas originais (backward compat).
        """
        feature_map = self._feature_maps.get(league_id, {})
        if nome_modelo in feature_map:
            fn = feature_map[nome_modelo]
        else:
            fn = FeatureExtractor.feature_names()
        X = np.array([[feats.get(n, 0) or 0 for n in fn]], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=fn)

    def prever_jogo(self, fixture: dict) -> dict | None:
        """
        Gera previsão completa para um jogo.

        Usa modelos per-league da liga do fixture.
        Se feature_map.json existir, usa features evoluídas por modelo.
        Se a liga não tem modelo treinado, retorna None.

        Parâmetros:
          fixture: dict com dados do jogo (fixture_id, home_id, away_id, league_id, etc.)

        Retorna dict com probabilidades de cada mercado, ou None sem modelo/dados.
        """
        league_id = fixture.get("league_id")

        # Garantir que modelos (e feature_map) estejam carregados
        self._carregar_modelos_liga(league_id)

        # Extrair features: usa pool completo se feature_map existe, senão estático
        if league_id in self._feature_maps:
            feats = self.ff.features_jogo(fixture)
        else:
            feats = self.fe.features_jogo(fixture)
        if feats is None:
            return None

        resultado = {
            "fixture_id": fixture["fixture_id"],
            "home_name": fixture.get("home_name", ""),
            "away_name": fixture.get("away_name", ""),
            "league_id": league_id,
            "season": fixture.get("season"),
            "round": fixture.get("round", ""),
            "date": fixture.get("date", ""),
            "features": feats,
        }

        # ─── Previsão 1x2 ───
        modelo = self._get_modelo("resultado_1x2", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "resultado_1x2", league_id)
            probs = modelo.predict(dmat)
            if probs.ndim > 1:
                probs = probs[0]

            resultado["prob_home"] = round(float(probs[0]), 4)
            resultado["prob_draw"] = round(float(probs[1]), 4)
            resultado["prob_away"] = round(float(probs[2]), 4)

            idx = int(np.argmax(probs))
            resultado["winner_pred"] = ["home", "draw", "away"][idx]
            resultado["winner_conf"] = round(float(probs[idx]), 4)

        # ─── Previsão Over/Under 2.5 ───
        modelo = self._get_modelo("over_under_25", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_25", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over25"] = round(prob, 4)
            resultado["prob_under25"] = round(1 - prob, 4)

        # ─── Previsão Over/Under 1.5 ───
        modelo = self._get_modelo("over_under_15", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_15", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over15"] = round(prob, 4)
            resultado["prob_under15"] = round(1 - prob, 4)

        # ─── Previsão Over/Under 3.5 ───
        modelo = self._get_modelo("over_under_35", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_35", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over35"] = round(prob, 4)
            resultado["prob_under35"] = round(1 - prob, 4)

        # ─── Previsão BTTS ───
        modelo = self._get_modelo("btts", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "btts", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_btts_yes"] = round(prob, 4)
            resultado["prob_btts_no"] = round(1 - prob, 4)

        # ─── Previsão Resultado 1T ───
        modelo = self._get_modelo("resultado_ht", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "resultado_ht", league_id)
            probs = modelo.predict(dmat)
            if probs.ndim > 1:
                probs = probs[0]
            resultado["prob_ht_home"] = round(float(probs[0]), 4)
            resultado["prob_ht_draw"] = round(float(probs[1]), 4)
            resultado["prob_ht_away"] = round(float(probs[2]), 4)
            idx = int(np.argmax(probs))
            resultado["ht_winner_pred"] = ["home", "draw", "away"][idx]
            resultado["ht_winner_conf"] = round(float(probs[idx]), 4)

        # ─── Previsão HT/FT (9 classes) ───
        modelo = self._get_modelo("htft", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "htft", league_id)
            probs = modelo.predict(dmat)
            if probs.ndim > 1:
                probs = probs[0]
            HTFT_LABELS = [
                "H/H", "H/D", "H/A",
                "D/H", "D/D", "D/A",
                "A/H", "A/D", "A/A",
            ]
            resultado["htft_probs"] = {
                HTFT_LABELS[i]: round(float(probs[i]), 4)
                for i in range(min(len(probs), 9))
            }
            idx = int(np.argmax(probs))
            resultado["htft_pred"] = HTFT_LABELS[idx] if idx < 9 else "?"
            resultado["htft_conf"] = round(float(probs[idx]), 4)

        # ─── Previsão Over/Under 0.5 gols 1T ───
        modelo = self._get_modelo("over_under_05_ht", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_05_ht", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over05_ht"] = round(prob, 4)
            resultado["prob_under05_ht"] = round(1 - prob, 4)

        # ─── Previsão Over/Under 1.5 gols 1T ───
        modelo = self._get_modelo("over_under_15_ht", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_15_ht", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over15_ht"] = round(prob, 4)
            resultado["prob_under15_ht"] = round(1 - prob, 4)

        # ─── Previsão Over/Under 0.5 gols 2T ───
        modelo = self._get_modelo("over_under_05_2t", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_05_2t", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over05_2t"] = round(prob, 4)
            resultado["prob_under05_2t"] = round(1 - prob, 4)

        # ─── Previsão Over/Under 1.5 gols 2T ───
        modelo = self._get_modelo("over_under_15_2t", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "over_under_15_2t", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_over15_2t"] = round(prob, 4)
            resultado["prob_under15_2t"] = round(1 - prob, 4)

        # ─── Previsão Escanteios Over/Under ───
        modelo = self._get_modelo("corners_over_85", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "corners_over_85", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_corners_over_85"] = round(prob, 4)
            resultado["prob_corners_under_85"] = round(1 - prob, 4)

        modelo = self._get_modelo("corners_over_95", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "corners_over_95", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_corners_over_95"] = round(prob, 4)
            resultado["prob_corners_under_95"] = round(1 - prob, 4)

        modelo = self._get_modelo("corners_over_105", league_id)
        if modelo:
            dmat = self._dmat_para_modelo(feats, "corners_over_105", league_id)
            prob = float(modelo.predict(dmat)[0])
            resultado["prob_corners_over_105"] = round(prob, 4)
            resultado["prob_corners_under_105"] = round(1 - prob, 4)

        return resultado

    # ══════════════════════════════════════════════
    #  MAPEAMENTO DECLARATIVO: Modelo → Odds → EV
    # ══════════════════════════════════════════════
    # Cada entrada conecta uma probabilidade do modelo a um bloco/outcome
    # no resumo de odds. Para adicionar um novo mercado, basta 1 linha.
    # Formato: (mercado_id, prob_key, odds_bloco, odds_outcome, descricao_template)
    #   - {home}/{away} no template são substituídos pelo nome real do time.
    EV_MAP = [
        # ── 1x2 Full Time (modelo: resultado_1x2) ──
        ("h2h_home",  "prob_home",      "h2h",       "home",  "Vitória {home}"),
        ("h2h_draw",  "prob_draw",      "h2h",       "draw",  "Empate"),
        ("h2h_away",  "prob_away",      "h2h",       "away",  "Vitória {away}"),
        # ── Over/Under (modelos: over_under_15/25/35) ──
        ("over25",    "prob_over25",    "totals",    "over",  "Over 2.5 gols"),
        ("under25",   "prob_under25",   "totals",    "under", "Under 2.5 gols"),
        ("over15",    "prob_over15",    "totals_15", "over",  "Over 1.5 gols"),
        ("under15",   "prob_under15",   "totals_15", "under", "Under 1.5 gols"),
        ("over35",    "prob_over35",    "totals_35", "over",  "Over 3.5 gols"),
        ("under35",   "prob_under35",   "totals_35", "under", "Under 3.5 gols"),
        # ── BTTS — Ambos Marcam (modelo: btts) ──
        ("btts_yes",  "prob_btts_yes",  "btts",      "yes",   "Ambos Marcam — Sim"),
        ("btts_no",   "prob_btts_no",   "btts",      "no",    "Ambos Marcam — Não"),
        # ── 1x2 Primeiro Tempo (modelo: resultado_ht) ──
        ("ht_home",   "prob_ht_home",   "h2h_h1",    "home",  "1T Vitória {home}"),
        ("ht_draw",   "prob_ht_draw",   "h2h_h1",    "draw",  "1T Empate"),
        ("ht_away",   "prob_ht_away",   "h2h_h1",    "away",  "1T Vitória {away}"),
    ]

    def calcular_ev(self, previsao: dict, odds_resumo: dict) -> list[dict]:
        """
        Calcula Expected Value (EV) para cada mercado usando mapeamento declarativo.

        EV = (probabilidade_modelo × odd) - 1
        Positivo = valor esperado favorável ao apostador.

        O loop genérico cruza automaticamente TODA probabilidade do modelo
        com TODA odd disponível. Para adicionar um novo mercado, basta
        incluir 1 linha em EV_MAP — zero código novo.

        Parâmetros:
          previsao: dict retornado por prever_jogo()
          odds_resumo: dict retornado por odds_api.resumo_odds_jogo()

        Retorna lista de oportunidades (mercados com EV calculado),
        ordenada por EV decrescente.
        """
        oportunidades = []
        home = previsao.get("home_name", "Casa")
        away = previsao.get("away_name", "Fora")

        # Loop genérico: cruza modelo × odds para cada combinação do EV_MAP
        for mercado_id, prob_key, odds_bloco, odds_outcome, desc_tpl in self.EV_MAP:
            # Verificar se o modelo gerou essa probabilidade
            prob = previsao.get(prob_key, 0)
            if not prob:
                continue

            # Verificar se há odds disponíveis para esse mercado
            bloco = odds_resumo.get(odds_bloco, {})
            dados = bloco.get(odds_outcome, {})
            odd = dados.get("odd", 0)
            if odd <= 1:
                continue

            # Calcular EV e montar oportunidade
            ev = round((prob * odd) - 1, 4)
            oportunidades.append({
                "mercado": mercado_id,
                "descricao": desc_tpl.format(home=home, away=away),
                "prob_modelo": prob,
                "odd": odd,
                "casa": dados.get("casa", ""),
                "ev": ev,
                "ev_pct": round(ev * 100, 1),
                "is_preferida": dados.get("preferida", True),
            })

        # Ordenar por EV decrescente
        oportunidades.sort(key=lambda x: x["ev"], reverse=True)
        return oportunidades

    @staticmethod
    def formatar_previsao(previsao: dict) -> str:
        """Formata previsão completa para exibição legível."""
        lines = [
            f"⚽ {previsao.get('home_name', '?')} vs {previsao.get('away_name', '?')}",
            f"📅 {previsao.get('date', '?')[:10]}",
            "",
        ]

        # Resultado FT
        if "prob_home" in previsao:
            lines.append("── Resultado (FT) ──")
            lines.append(f"🏠 Casa: {previsao['prob_home']:.1%}")
            lines.append(f"🤝 Empate: {previsao['prob_draw']:.1%}")
            lines.append(f"✈️ Fora: {previsao['prob_away']:.1%}")
            lines.append(f"→ {previsao.get('winner_pred', '?')} ({previsao.get('winner_conf', 0):.1%})")

        # Resultado 1T
        if "prob_ht_home" in previsao:
            lines.append("\n── Resultado (1T) ──")
            lines.append(f"🏠 Casa: {previsao['prob_ht_home']:.1%}")
            lines.append(f"🤝 Empate: {previsao['prob_ht_draw']:.1%}")
            lines.append(f"✈️ Fora: {previsao['prob_ht_away']:.1%}")

        # Over/Under
        ou_lines = []
        if "prob_over15" in previsao:
            ou_lines.append(f"O/U 1.5: {previsao['prob_over15']:.0%}/{previsao['prob_under15']:.0%}")
        if "prob_over25" in previsao:
            ou_lines.append(f"O/U 2.5: {previsao['prob_over25']:.0%}/{previsao['prob_under25']:.0%}")
        if "prob_over35" in previsao:
            ou_lines.append(f"O/U 3.5: {previsao['prob_over35']:.0%}/{previsao['prob_under35']:.0%}")
        if ou_lines:
            lines.append("\n── Gols ──")
            lines.extend(ou_lines)

        # BTTS
        if "prob_btts_yes" in previsao:
            lines.append(f"\n⚽⚽ BTTS Sim: {previsao['prob_btts_yes']:.1%}")

        # HT/FT
        if "htft_pred" in previsao:
            lines.append(f"\n── HT/FT ──")
            lines.append(f"→ {previsao['htft_pred']} ({previsao.get('htft_conf', 0):.1%})")
            # Top 3 combinações
            htft_probs = previsao.get("htft_probs", {})
            top3 = sorted(htft_probs.items(), key=lambda x: -x[1])[:3]
            for label, prob in top3:
                lines.append(f"   {label}: {prob:.1%}")

        return "\n".join(lines)

    @staticmethod
    def formatar_oportunidade(op: dict) -> str:
        """Formata uma oportunidade de EV para exibição."""
        emoji = "🔥" if op["ev_pct"] > 10 else "⚡" if op["ev_pct"] > 5 else "📊"
        return (
            f"{emoji} {op['descricao']}\n"
            f"   Prob: {op['prob_modelo']:.1%} | Odd: {op['odd']:.2f} ({op['casa']})\n"
            f"   EV: {op['ev_pct']:+.1f}%"
        )

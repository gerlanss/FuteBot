"""
Feature engineering para o modelo de previsão.

Extrai features numéricas de cada jogo a partir dos dados do SQLite
para alimentar o XGBoost. Cada fixture vira uma linha com ~45 features.

Features extraídas:
  - Posicionais: posição na tabela (home/away), diferença de posição
  - Forma recente: % vitórias últimos 5/10 jogos, sequências
  - Gols: média gols marcados/sofridos (home/away), total
  - Gols 1T: média gols no primeiro tempo (derivado de score_ht_h/a)
  - Defensivas: clean sheets %, failed to score %
  - Confronto direto (H2H): % vitórias de cada lado
  - Estatísticas de partida: média xG, chutes, posse, escanteios, cartões
  - Temporais: cartões/gols antes dos 30', depois dos 75' (via eventos)
  - Árbitro: média de cartões por jogo do árbitro
  - Contextuais: é derby?, diferença de gols/jogo

Uso:
  from models.features import FeatureExtractor
  fe = FeatureExtractor(db)
  X, y = fe.build_dataset(league_id=71, seasons=[2024, 2025])
"""

import json
import numpy as np
from data.database import Database


class FeatureExtractor:
    """Extrai features de jogos do banco de dados para treino e previsão."""

    def __init__(self, db: Database):
        self.db = db

    def features_jogo(self, fixture: dict) -> dict | None:
        """
        Calcula features para um único jogo.

        Parâmetros:
          fixture: dict de um fixture do banco (com fixture_id, home_id, etc.)

        Retorna dict de features numéricas ou None se dados insuficientes.

        Estratégia de busca (fallback progressivo):
          1. Mesma liga + mesma season (ideal)
          2. Qualquer liga + mesma season (cross-league)
          3. Qualquer liga + season anterior (cross-season)
        Isso garante features mesmo para times no início de temporada
        ou em copas com poucos jogos.
        """
        home_id = fixture["home_id"]
        away_id = fixture["away_id"]
        league_id = fixture["league_id"]
        season = fixture["season"]
        fixture_date = fixture["date"]

        conn = self.db._conn()

        # ─── Busca com fallback progressivo ───
        home_prev = self._buscar_jogos_anteriores(
            conn, home_id, league_id, season, fixture_date
        )
        away_prev = self._buscar_jogos_anteriores(
            conn, away_id, league_id, season, fixture_date
        )

        conn.close()

        # Precisamos de pelo menos 3 jogos de cada time
        if len(home_prev) < 3 or len(away_prev) < 3:
            return None

        # ─── Calcular features ───

        feats = {}
        # Flag: indica se usou dados cross-league (útil para debug)
        feats["_cross_league"] = any(
            r.get("league_id") != league_id for r in home_prev + away_prev
        )

        # 1. FORMA RECENTE (últimos 5 jogos)
        feats["home_form_5"] = self._calc_form(home_prev[:5], home_id)
        feats["away_form_5"] = self._calc_form(away_prev[:5], away_id)
        feats["form_diff"] = feats["home_form_5"] - feats["away_form_5"]

        # 2. GOLS MARCADOS/SOFRIDOS (média últimos jogos)
        h_gf, h_ga = self._calc_gols(home_prev[:5], home_id)
        a_gf, a_ga = self._calc_gols(away_prev[:5], away_id)
        feats["home_goals_for_avg"] = h_gf
        feats["home_goals_against_avg"] = h_ga
        feats["away_goals_for_avg"] = a_gf
        feats["away_goals_against_avg"] = a_ga
        feats["goals_diff"] = (h_gf - h_ga) - (a_gf - a_ga)

        # 3. GOLS EM CASA / FORA (específico do mando)
        h_gf_home, h_ga_home = self._calc_gols_mando(home_prev, home_id, mando="home")
        a_gf_away, a_ga_away = self._calc_gols_mando(away_prev, away_id, mando="away")
        feats["home_goals_for_home"] = h_gf_home
        feats["home_goals_against_home"] = h_ga_home
        feats["away_goals_for_away"] = a_gf_away
        feats["away_goals_against_away"] = a_ga_away

        # 4. CLEAN SHEETS E FAILED TO SCORE
        feats["home_clean_sheet_pct"] = self._calc_clean_sheet(home_prev[:5], home_id)
        feats["away_clean_sheet_pct"] = self._calc_clean_sheet(away_prev[:5], away_id)
        feats["home_failed_score_pct"] = self._calc_failed_score(home_prev[:5], home_id)
        feats["away_failed_score_pct"] = self._calc_failed_score(away_prev[:5], away_id)

        # 5. OVER/UNDER 2.5 RATE
        feats["home_over25_pct"] = self._calc_over25(home_prev[:5])
        feats["away_over25_pct"] = self._calc_over25(away_prev[:5])

        # 6. BTTS RATE (ambos marcam)
        feats["home_btts_pct"] = self._calc_btts(home_prev[:5])
        feats["away_btts_pct"] = self._calc_btts(away_prev[:5])

        # 7. SEQUÊNCIAS
        feats["home_win_streak"] = self._calc_streak(home_prev, home_id, "win")
        feats["away_win_streak"] = self._calc_streak(away_prev, away_id, "win")
        feats["home_loss_streak"] = self._calc_streak(home_prev, home_id, "loss")
        feats["away_loss_streak"] = self._calc_streak(away_prev, away_id, "loss")

        # 8. XG MÉDIO (se disponível no fixture_stats)
        feats["home_xg_avg"] = self._calc_xg_avg(home_prev[:5], home_id)
        feats["away_xg_avg"] = self._calc_xg_avg(away_prev[:5], away_id)

        # 9. CONFRONTO DIRETO (H2H nos últimos jogos disponíveis)
        h2h_pct = self._calc_h2h(home_id, away_id, league_id, fixture_date)
        feats["h2h_home_pct"] = h2h_pct[0]
        feats["h2h_draw_pct"] = h2h_pct[1]
        feats["h2h_away_pct"] = h2h_pct[2]
        feats["h2h_total_jogos"] = h2h_pct[3]

        # ─── NOVAS FEATURES (stats de partida) ───

        # 10. CHUTES (média por jogo dos últimos 5)
        feats["home_shots_avg"] = self._calc_stat_avg(home_prev[:5], home_id, "shots_total")
        feats["away_shots_avg"] = self._calc_stat_avg(away_prev[:5], away_id, "shots_total")
        feats["home_shots_on_avg"] = self._calc_stat_avg(home_prev[:5], home_id, "shots_on_target")
        feats["away_shots_on_avg"] = self._calc_stat_avg(away_prev[:5], away_id, "shots_on_target")

        # 11. POSSE DE BOLA (média %)
        feats["home_possession_avg"] = self._calc_stat_avg(home_prev[:5], home_id, "possession")
        feats["away_possession_avg"] = self._calc_stat_avg(away_prev[:5], away_id, "possession")

        # 12. ESCANTEIOS (média por jogo)
        feats["home_corners_avg"] = self._calc_stat_avg(home_prev[:5], home_id, "corners")
        feats["away_corners_avg"] = self._calc_stat_avg(away_prev[:5], away_id, "corners")

        # 13. CARTÕES AMARELOS (média por jogo)
        feats["home_yellows_avg"] = self._calc_stat_avg(home_prev[:5], home_id, "yellow_cards")
        feats["away_yellows_avg"] = self._calc_stat_avg(away_prev[:5], away_id, "yellow_cards")

        # 14. FALTAS (média por jogo)
        feats["home_fouls_avg"] = self._calc_stat_avg(home_prev[:5], home_id, "fouls")
        feats["away_fouls_avg"] = self._calc_stat_avg(away_prev[:5], away_id, "fouls")

        # 15. GOLS NO PRIMEIRO TEMPO (derivado de score_ht_h/a dos fixtures)
        feats["home_goals_ht_avg"] = self._calc_gols_ht(home_prev[:5], home_id)
        feats["away_goals_ht_avg"] = self._calc_gols_ht(away_prev[:5], away_id)

        # 16. OVER/UNDER 1.5 e 3.5 RATE (ampliando mercados)
        feats["home_over15_pct"] = self._calc_over_n(home_prev[:5], 1.5)
        feats["away_over15_pct"] = self._calc_over_n(away_prev[:5], 1.5)
        feats["home_over35_pct"] = self._calc_over_n(home_prev[:5], 3.5)
        feats["away_over35_pct"] = self._calc_over_n(away_prev[:5], 3.5)

        # 17. PERFIL DO ÁRBITRO (média de cartões e faltas nos jogos dele)
        referee = fixture.get("referee", "")
        arb = self._calc_perfil_arbitro(referee, fixture_date)
        feats["ref_yellows_avg"] = arb[0]
        feats["ref_fouls_avg"] = arb[1]
        feats["ref_total_jogos"] = arb[2]

        # 18. FEATURES DERIVADAS DE PRESSAO/QUALIDADE
        feats["total_xg_avg"] = round(feats["home_xg_avg"] + feats["away_xg_avg"], 2)
        feats["total_shots_on_avg"] = round(feats["home_shots_on_avg"] + feats["away_shots_on_avg"], 2)
        feats["shots_on_diff"] = round(feats["home_shots_on_avg"] - feats["away_shots_on_avg"], 2)
        feats["home_xg_per_shot_on"] = round(
            feats["home_xg_avg"] / max(feats["home_shots_on_avg"], 0.1), 3
        )
        feats["away_xg_per_shot_on"] = round(
            feats["away_xg_avg"] / max(feats["away_shots_on_avg"], 0.1), 3
        )
        feats["home_attack_vs_away_def"] = round(
            feats["home_goals_for_home"] * max(feats["away_goals_against_away"], 0.1), 3
        )
        feats["away_attack_vs_home_def"] = round(
            feats["away_goals_for_away"] * max(feats["home_goals_against_home"], 0.1), 3
        )
        feats["over15_env"] = round(
            feats["home_over15_pct"] * feats["away_over15_pct"], 3
        )

        return feats

    def build_dataset(self, league_id: int = None,
                      seasons: list[int] = None) -> tuple[list[dict], list[dict]]:
        """
        Constrói dataset completo para treinamento.

        Retorna:
          features: lista de dicts com features de cada jogo
          labels: lista de dicts com targets (resultado, total_gols, btts)
        """
        fixtures = self.db.fixtures_finalizados(league_id=league_id)

        if seasons:
            fixtures = [f for f in fixtures if f["season"] in seasons]

        features = []
        labels = []

        for fix in fixtures:
            feats = self.features_jogo(fix)
            if feats is None:
                continue

            # Adicionar metadados (não são features, mas úteis para análise)
            feats["_fixture_id"] = fix["fixture_id"]
            feats["_league_id"] = fix["league_id"]
            feats["_season"] = fix["season"]
            feats["_date"] = fix["date"]

            # Labels (targets) — expande para todos os mercados
            gh = fix["goals_home"] or 0
            ga = fix["goals_away"] or 0
            ht_h = fix.get("score_ht_h") or 0
            ht_a = fix.get("score_ht_a") or 0

            # Gols no 2º tempo (derivado: total - 1T)
            g2t_h = gh - ht_h
            g2t_a = ga - ht_a
            gols_ht = ht_h + ht_a
            gols_2t = g2t_h + g2t_a

            # Total de escanteios da partida (soma dos 2 times)
            stats_fix = self.db.stats_partida(fix["fixture_id"])
            total_corners = sum(
                (s.get("corners") or 0) for s in stats_fix
            )

            label = {
                "fixture_id": fix["fixture_id"],
                # Resultado 1x2 (0=home, 1=draw, 2=away)
                "resultado": 0 if gh > ga else (1 if gh == ga else 2),
                "resultado_str": "home" if gh > ga else ("draw" if gh == ga else "away"),
                # Over/Under 1.5 / 2.5 / 3.5 (jogo todo)
                "over15": 1 if (gh + ga) > 1 else 0,
                "over25": 1 if (gh + ga) > 2 else 0,
                "over35": 1 if (gh + ga) > 3 else 0,
                # BTTS (ambos marcam)
                "btts": 1 if gh > 0 and ga > 0 else 0,
                # Resultado 1T (0=home, 1=draw, 2=away)
                "resultado_ht": 0 if ht_h > ht_a else (1 if ht_h == ht_a else 2),
                # HT/FT combinado (9 possibilidades: 0-8)
                "htft": (0 if ht_h > ht_a else (1 if ht_h == ht_a else 2)) * 3 +
                        (0 if gh > ga else (1 if gh == ga else 2)),
                # Over/Under 1º Tempo (gols no 1T)
                "over05_ht": 1 if gols_ht > 0 else 0,
                "over15_ht": 1 if gols_ht > 1 else 0,
                # Over/Under 2º Tempo (gols no 2T)
                "over05_2t": 1 if gols_2t > 0 else 0,
                "over15_2t": 1 if gols_2t > 1 else 0,
                # Escanteios Over/Under (thresholds comuns em casas)
                "corners_over_85": 1 if total_corners > 8 else 0,
                "corners_over_95": 1 if total_corners > 9 else 0,
                "corners_over_105": 1 if total_corners > 10 else 0,
                # Gols totais (para análise)
                "total_gols": gh + ga,
                "gols_home": gh,
                "gols_away": ga,
                "gols_ht_home": ht_h,
                "gols_ht_away": ht_a,
                "total_corners": total_corners,
            }

            features.append(feats)
            labels.append(label)

        print(f"[Features] Dataset: {len(features)} jogos com features "
              f"(de {len(fixtures)} fixtures totais)")
        return features, labels

    # ══════════════════════════════════════════════
    #  BUSCA DE JOGOS ANTERIORES (com fallback)
    # ══════════════════════════════════════════════

    def _buscar_jogos_anteriores(self, conn, team_id: int,
                                  league_id: int, season: int,
                                  fixture_date: str) -> list[dict]:
        """
        Busca até 10 jogos anteriores de um time com fallback progressivo.

        Ordem de busca:
          1. Mesma liga + mesma season (ideal — contexto tático igual)
          2. Qualquer liga + mesma season (cross-league — forma recente)
          3. Qualquer liga + season anterior (cross-season — garante mínimo)

        Retorna lista de dicts (já convertidos de sqlite3.Row).
        """
        # Passo 1: mesma liga + mesma season
        rows = [dict(r) for r in conn.execute("""
            SELECT * FROM fixtures
            WHERE (home_id=? OR away_id=?) AND league_id=? AND season=?
            AND date < ? AND status='FT'
            ORDER BY date DESC LIMIT 10
        """, (team_id, team_id, league_id, season, fixture_date)).fetchall()]

        if len(rows) >= 3:
            return rows

        # Passo 2: qualquer liga, mesma season (amplia busca)
        rows = [dict(r) for r in conn.execute("""
            SELECT * FROM fixtures
            WHERE (home_id=? OR away_id=?) AND season=?
            AND date < ? AND status='FT'
            ORDER BY date DESC LIMIT 10
        """, (team_id, team_id, season, fixture_date)).fetchall()]

        if len(rows) >= 3:
            return rows

        # Passo 3: qualquer liga, season anterior (último recurso)
        rows = [dict(r) for r in conn.execute("""
            SELECT * FROM fixtures
            WHERE (home_id=? OR away_id=?)
            AND date < ? AND status='FT'
            ORDER BY date DESC LIMIT 10
        """, (team_id, team_id, fixture_date)).fetchall()]

        return rows

    # ══════════════════════════════════════════════
    #  HELPERS DE CÁLCULO
    # ══════════════════════════════════════════════

    def _calc_form(self, jogos: list, team_id: int) -> float:
        """Calcula forma: % de pontos ganhos (V=3, E=1, D=0) / máximo possível."""
        if not jogos:
            return 0.5
        pontos = 0
        for j in jogos:
            resultado = self._resultado_time(j, team_id)
            if resultado == "win":
                pontos += 3
            elif resultado == "draw":
                pontos += 1
        return pontos / (len(jogos) * 3)

    def _calc_gols(self, jogos: list, team_id: int) -> tuple[float, float]:
        """Média de gols marcados e sofridos."""
        if not jogos:
            return 0.0, 0.0
        gf, ga = 0, 0
        for j in jogos:
            if j["home_id"] == team_id:
                gf += j["goals_home"] or 0
                ga += j["goals_away"] or 0
            else:
                gf += j["goals_away"] or 0
                ga += j["goals_home"] or 0
        n = len(jogos)
        return round(gf / n, 2), round(ga / n, 2)

    def _calc_gols_mando(self, jogos: list, team_id: int,
                         mando: str = "home") -> tuple[float, float]:
        """Média de gols só nos jogos em casa (mando='home') ou fora (mando='away')."""
        filtrados = []
        for j in jogos:
            if mando == "home" and j["home_id"] == team_id:
                filtrados.append(j)
            elif mando == "away" and j["away_id"] == team_id:
                filtrados.append(j)

        if not filtrados:
            return 0.0, 0.0

        gf, ga = 0, 0
        for j in filtrados:
            if j["home_id"] == team_id:
                gf += j["goals_home"] or 0
                ga += j["goals_away"] or 0
            else:
                gf += j["goals_away"] or 0
                ga += j["goals_home"] or 0
        n = len(filtrados)
        return round(gf / n, 2), round(ga / n, 2)

    def _calc_clean_sheet(self, jogos: list, team_id: int) -> float:
        """% de jogos sem sofrer gol."""
        if not jogos:
            return 0.0
        cs = 0
        for j in jogos:
            if j["home_id"] == team_id:
                if (j["goals_away"] or 0) == 0:
                    cs += 1
            else:
                if (j["goals_home"] or 0) == 0:
                    cs += 1
        return round(cs / len(jogos), 2)

    def _calc_failed_score(self, jogos: list, team_id: int) -> float:
        """% de jogos sem marcar gol."""
        if not jogos:
            return 0.0
        fs = 0
        for j in jogos:
            if j["home_id"] == team_id:
                if (j["goals_home"] or 0) == 0:
                    fs += 1
            else:
                if (j["goals_away"] or 0) == 0:
                    fs += 1
        return round(fs / len(jogos), 2)

    def _calc_over25(self, jogos: list) -> float:
        """% de jogos com mais de 2.5 gols."""
        if not jogos:
            return 0.5
        over = sum(1 for j in jogos if ((j["goals_home"] or 0) + (j["goals_away"] or 0)) > 2)
        return round(over / len(jogos), 2)

    def _calc_btts(self, jogos: list) -> float:
        """% de jogos onde ambos marcaram."""
        if not jogos:
            return 0.5
        btts = sum(1 for j in jogos if (j["goals_home"] or 0) > 0 and (j["goals_away"] or 0) > 0)
        return round(btts / len(jogos), 2)

    def _calc_streak(self, jogos: list, team_id: int, tipo: str) -> int:
        """Conta sequência atual (win/draw/loss streak)."""
        streak = 0
        for j in jogos:
            resultado = self._resultado_time(j, team_id)
            if resultado == tipo:
                streak += 1
            else:
                break
        return streak

    def _calc_xg_avg(self, jogos: list, team_id: int) -> float:
        """Média de xG dos últimos jogos (se disponível)."""
        xg_total = 0.0
        count = 0
        for j in jogos:
            stats = self.db.stats_partida(j["fixture_id"])
            for s in stats:
                if s["team_id"] == team_id and s["expected_goals"] is not None:
                    xg_total += s["expected_goals"]
                    count += 1
        return round(xg_total / count, 2) if count else 0.0

    def _calc_stat_avg(self, jogos: list, team_id: int, stat_col: str) -> float:
        """
        Média de uma estatística específica (chutes, posse, escanteios, cartões, etc.)
        dos últimos jogos do time. Consulta fixture_stats por coluna extraída.
        """
        total = 0.0
        count = 0
        for j in jogos:
            stats = self.db.stats_partida(j["fixture_id"])
            for s in stats:
                if s["team_id"] == team_id and s.get(stat_col) is not None:
                    total += float(s[stat_col])
                    count += 1
        return round(total / count, 2) if count else 0.0

    def _calc_gols_ht(self, jogos: list, team_id: int) -> float:
        """
        Média de gols no primeiro tempo (usando score_ht_h/a dos fixtures).
        Útil para prever mercados de 1T e HT/FT.
        """
        total = 0.0
        count = 0
        for j in jogos:
            ht_h = j.get("score_ht_h")
            ht_a = j.get("score_ht_a")
            if ht_h is not None and ht_a is not None:
                if j["home_id"] == team_id:
                    total += ht_h
                else:
                    total += ht_a
                count += 1
        return round(total / count, 2) if count else 0.0

    def _calc_over_n(self, jogos: list, threshold: float) -> float:
        """% de jogos com total de gols acima de N (1.5, 2.5, 3.5)."""
        if not jogos:
            return 0.5
        over = sum(1 for j in jogos
                   if ((j["goals_home"] or 0) + (j["goals_away"] or 0)) > threshold)
        return round(over / len(jogos), 2)

    def _calc_perfil_arbitro(self, referee: str, before_date: str) -> tuple[float, float, int]:
        """
        Calcula perfil do árbitro: média de cartões amarelos e faltas nos jogos que ele apitou.
        Retorna (media_amarelos, media_faltas, total_jogos).
        Só considera jogos ANTES da data informada (não vazar dados).
        """
        if not referee or not referee.strip():
            return 0.0, 0.0, 0

        conn = self.db._conn()
        # Buscar fixtures deste árbitro antes da data
        rows = conn.execute("""
            SELECT fixture_id FROM fixtures
            WHERE referee = ? AND status = 'FT' AND date < ?
            ORDER BY date DESC LIMIT 20
        """, (referee, before_date)).fetchall()

        if not rows:
            conn.close()
            return 0.0, 0.0, 0

        total_yellows = 0.0
        total_fouls = 0.0
        count = 0

        for r in rows:
            fid = r["fixture_id"]
            stats = conn.execute(
                "SELECT yellow_cards, fouls FROM fixture_stats WHERE fixture_id=?",
                (fid,)
            ).fetchall()
            for s in stats:
                yc = s["yellow_cards"]
                fo = s["fouls"]
                if yc is not None:
                    total_yellows += yc
                if fo is not None:
                    total_fouls += fo
            if stats:
                count += 1

        conn.close()

        if count == 0:
            return 0.0, 0.0, 0

        # Média POR JOGO (soma dos 2 times / jogo)
        return (
            round(total_yellows / count, 2),
            round(total_fouls / count, 2),
            count,
        )

    def _calc_h2h(self, home_id: int, away_id: int,
                  league_id: int, before_date: str) -> tuple[float, float, float, int]:
        """
        Calcula % de vitórias no confronto direto (H2H).
        Retorna (home_pct, draw_pct, away_pct, total_jogos).
        """
        conn = self.db._conn()
        rows = conn.execute("""
            SELECT * FROM fixtures
            WHERE ((home_id=? AND away_id=?) OR (home_id=? AND away_id=?))
            AND status='FT' AND date < ?
            ORDER BY date DESC LIMIT 10
        """, (home_id, away_id, away_id, home_id, before_date)).fetchall()
        conn.close()

        if not rows:
            return 0.33, 0.34, 0.33, 0

        hw, dw, aw = 0, 0, 0
        for j in rows:
            gh = j["goals_home"] or 0
            ga = j["goals_away"] or 0
            if gh > ga:
                if j["home_id"] == home_id:
                    hw += 1
                else:
                    aw += 1
            elif gh == ga:
                dw += 1
            else:
                if j["away_id"] == home_id:
                    hw += 1
                else:
                    aw += 1

        n = len(rows)
        return round(hw/n, 2), round(dw/n, 2), round(aw/n, 2), n

    def _resultado_time(self, jogo: dict, team_id: int) -> str:
        """Retorna 'win', 'draw' ou 'loss' para o time neste jogo."""
        gh = jogo["goals_home"] or 0
        ga = jogo["goals_away"] or 0
        if gh == ga:
            return "draw"
        if jogo["home_id"] == team_id:
            return "win" if gh > ga else "loss"
        else:
            return "win" if ga > gh else "loss"

    @staticmethod
    def feature_names() -> list[str]:
        """Lista de nomes das features (excluindo metadados _*)."""
        return [
            # Forma e gols (originais)
            "home_form_5", "away_form_5", "form_diff",
            "home_goals_for_avg", "home_goals_against_avg",
            "away_goals_for_avg", "away_goals_against_avg", "goals_diff",
            "home_goals_for_home", "home_goals_against_home",
            "away_goals_for_away", "away_goals_against_away",
            "home_clean_sheet_pct", "away_clean_sheet_pct",
            "home_failed_score_pct", "away_failed_score_pct",
            "home_over25_pct", "away_over25_pct",
            "home_btts_pct", "away_btts_pct",
            "home_win_streak", "away_win_streak",
            "home_loss_streak", "away_loss_streak",
            "home_xg_avg", "away_xg_avg",
            "h2h_home_pct", "h2h_draw_pct", "h2h_away_pct", "h2h_total_jogos",
            # Chutes
            "home_shots_avg", "away_shots_avg",
            "home_shots_on_avg", "away_shots_on_avg",
            # Posse
            "home_possession_avg", "away_possession_avg",
            # Escanteios
            "home_corners_avg", "away_corners_avg",
            # Cartões
            "home_yellows_avg", "away_yellows_avg",
            # Faltas
            "home_fouls_avg", "away_fouls_avg",
            # Gols 1T
            "home_goals_ht_avg", "away_goals_ht_avg",
            # Over/Under rates extras
            "home_over15_pct", "away_over15_pct",
            "home_over35_pct", "away_over35_pct",
            # Árbitro
            "ref_yellows_avg", "ref_fouls_avg", "ref_total_jogos",
            # Pressao/qualidade derivados
            "total_xg_avg", "total_shots_on_avg", "shots_on_diff",
            "home_xg_per_shot_on", "away_xg_per_shot_on",
            "home_attack_vs_away_def", "away_attack_vs_home_def",
            "over15_env",
        ]

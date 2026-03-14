"""
Feature Factory — Pool expandido de ~217 features candidatas para evolução genética.

Herda todo o cálculo do FeatureExtractor original e expande com:
  - Janelas variáveis [3, 5, 8, 10] para cada métrica base
  - Ratios derivados: ataque/defesa, precisão de chute, eficiência de finalização
  - Diffs entre times: chutes, posse, escanteios, xG
  - Tendências (momentum): janela curta vs longa
  - Interações cruzadas: forma × sequência adversária, over × over

O FeatureEvolution seleciona o subconjunto ótimo para cada mercado × liga.
O Predictor carrega o mapa de features selecionadas (feature_map.json) por liga.

Uso:
  from models.feature_factory import FeatureFactory
  ff = FeatureFactory(db)
  X, y = ff.build_dataset(league_id=71, seasons=[2024, 2025])
  nomes = FeatureFactory.feature_names_full()
"""

from models.features import FeatureExtractor


# Janelas de jogos anteriores para cálculo de features
JANELAS = [3, 5, 8, 10]


class FeatureFactory(FeatureExtractor):
    """
    Pool expandido de ~217 features com janelas variáveis,
    ratios, diffs, tendências e interações.

    Herda de FeatureExtractor para reutilizar todos os helpers
    de cálculo (_calc_form, _calc_gols, etc.) e o build_dataset.

    Override de features_jogo: retorna ~217 features em vez de 51.
    O build_dataset herdado chama self.features_jogo() polimorficamente,
    então automaticamente gera datasets com o pool completo.
    """

    def features_jogo(self, fixture: dict) -> dict | None:
        """
        Calcula todas as ~217 features candidatas para um jogo.

        Estratégia:
          1. Busca até 10 jogos anteriores de cada time (fallback progressivo)
          2. Para cada janela [3,5,8,10], calcula métricas base + ratios + diffs
          3. Adiciona features não-janeladas (streaks, H2H, árbitro, mando)
          4. Calcula tendências (momentum curto vs longo)
          5. Calcula interações cruzadas

        Retorna dict de features numéricas ou None se dados insuficientes.
        """
        home_id = fixture["home_id"]
        away_id = fixture["away_id"]
        league_id = fixture["league_id"]
        season = fixture["season"]
        fixture_date = fixture["date"]

        conn = self.db._conn()
        home_prev = self._buscar_jogos_anteriores(
            conn, home_id, league_id, season, fixture_date
        )
        away_prev = self._buscar_jogos_anteriores(
            conn, away_id, league_id, season, fixture_date
        )
        conn.close()

        # Mínimo de 3 jogos por time para calcular features
        if len(home_prev) < 3 or len(away_prev) < 3:
            return None

        feats = {}

        # Metadata (prefixo _ = não é feature, usado para debug/split)
        feats["_cross_league"] = any(
            r.get("league_id") != league_id for r in home_prev + away_prev
        )

        # ════════════════════════════════════════
        #  FEATURES POR JANELA [3, 5, 8, 10]
        # ════════════════════════════════════════

        for w in JANELAS:
            hp = home_prev[:w]
            ap = away_prev[:w]

            # ── 1. Forma recente (% pontos ganhos) ──
            feats[f"home_form_{w}"] = self._calc_form(hp, home_id)
            feats[f"away_form_{w}"] = self._calc_form(ap, away_id)
            feats[f"form_diff_{w}"] = feats[f"home_form_{w}"] - feats[f"away_form_{w}"]

            # ── 2. Gols marcados/sofridos (média) ──
            h_gf, h_ga = self._calc_gols(hp, home_id)
            a_gf, a_ga = self._calc_gols(ap, away_id)
            feats[f"home_gf_{w}"] = h_gf
            feats[f"home_ga_{w}"] = h_ga
            feats[f"away_gf_{w}"] = a_gf
            feats[f"away_ga_{w}"] = a_ga
            feats[f"goals_diff_{w}"] = (h_gf - h_ga) - (a_gf - a_ga)

            # ── 3. Defensivas ──
            feats[f"home_cs_{w}"] = self._calc_clean_sheet(hp, home_id)
            feats[f"away_cs_{w}"] = self._calc_clean_sheet(ap, away_id)
            feats[f"home_fts_{w}"] = self._calc_failed_score(hp, home_id)
            feats[f"away_fts_{w}"] = self._calc_failed_score(ap, away_id)

            # ── 4. Over/Under e BTTS (taxas) ──
            feats[f"home_over25_{w}"] = self._calc_over25(hp)
            feats[f"away_over25_{w}"] = self._calc_over25(ap)
            feats[f"home_over15_{w}"] = self._calc_over_n(hp, 1.5)
            feats[f"away_over15_{w}"] = self._calc_over_n(ap, 1.5)
            feats[f"home_over35_{w}"] = self._calc_over_n(hp, 3.5)
            feats[f"away_over35_{w}"] = self._calc_over_n(ap, 3.5)
            feats[f"home_btts_{w}"] = self._calc_btts(hp)
            feats[f"away_btts_{w}"] = self._calc_btts(ap)

            # ── 5. xG esperado ──
            h_xg = self._calc_xg_avg(hp, home_id)
            a_xg = self._calc_xg_avg(ap, away_id)
            feats[f"home_xg_{w}"] = h_xg
            feats[f"away_xg_{w}"] = a_xg

            # ── 6. Chutes (total e no gol) ──
            h_shots = self._calc_stat_avg(hp, home_id, "shots_total")
            a_shots = self._calc_stat_avg(ap, away_id, "shots_total")
            h_shots_on = self._calc_stat_avg(hp, home_id, "shots_on_target")
            a_shots_on = self._calc_stat_avg(ap, away_id, "shots_on_target")
            feats[f"home_shots_{w}"] = h_shots
            feats[f"away_shots_{w}"] = a_shots
            feats[f"home_shots_on_{w}"] = h_shots_on
            feats[f"away_shots_on_{w}"] = a_shots_on

            # ── 7. Posse, escanteios, cartões, faltas ──
            feats[f"home_poss_{w}"] = self._calc_stat_avg(hp, home_id, "possession")
            feats[f"away_poss_{w}"] = self._calc_stat_avg(ap, away_id, "possession")
            feats[f"home_corners_{w}"] = self._calc_stat_avg(hp, home_id, "corners")
            feats[f"away_corners_{w}"] = self._calc_stat_avg(ap, away_id, "corners")
            feats[f"home_yellows_{w}"] = self._calc_stat_avg(hp, home_id, "yellow_cards")
            feats[f"away_yellows_{w}"] = self._calc_stat_avg(ap, away_id, "yellow_cards")
            feats[f"home_fouls_{w}"] = self._calc_stat_avg(hp, home_id, "fouls")
            feats[f"away_fouls_{w}"] = self._calc_stat_avg(ap, away_id, "fouls")

            # ── 8. Gols no 1º tempo ──
            feats[f"home_goals_ht_{w}"] = self._calc_gols_ht(hp, home_id)
            feats[f"away_goals_ht_{w}"] = self._calc_gols_ht(ap, away_id)

            # ── 9. Ratios derivados ──
            # Ataque vs defesa: gols marcados / gols sofridos
            feats[f"home_atk_def_ratio_{w}"] = round(h_gf / max(h_ga, 0.1), 3)
            feats[f"away_atk_def_ratio_{w}"] = round(a_gf / max(a_ga, 0.1), 3)
            # Precisão de chute: chutes no gol / chutes total
            feats[f"home_shot_acc_{w}"] = round(h_shots_on / max(h_shots, 0.1), 3)
            feats[f"away_shot_acc_{w}"] = round(a_shots_on / max(a_shots, 0.1), 3)
            # Eficiência de finalização: gols / xG (>1 = sobre-performando)
            feats[f"home_finish_eff_{w}"] = round(h_gf / max(h_xg, 0.1), 3)
            feats[f"away_finish_eff_{w}"] = round(a_gf / max(a_xg, 0.1), 3)

            # ── 10. Diffs entre times ──
            feats[f"shots_on_diff_{w}"] = round(h_shots_on - a_shots_on, 2)
            feats[f"total_shots_on_{w}"] = round(h_shots_on + a_shots_on, 2)
            feats[f"total_xg_{w}"] = round(h_xg + a_xg, 2)
            feats[f"shots_diff_{w}"] = round(h_shots - a_shots, 2)
            feats[f"poss_diff_{w}"] = round(
                feats[f"home_poss_{w}"] - feats[f"away_poss_{w}"], 2
            )
            feats[f"corners_diff_{w}"] = round(
                feats[f"home_corners_{w}"] - feats[f"away_corners_{w}"], 2
            )
            feats[f"xg_diff_{w}"] = round(h_xg - a_xg, 2)
            feats[f"home_xg_per_shot_on_{w}"] = round(h_xg / max(h_shots_on, 0.1), 3)
            feats[f"away_xg_per_shot_on_{w}"] = round(a_xg / max(a_shots_on, 0.1), 3)
            feats[f"home_attack_vs_away_def_{w}"] = round(h_gf * max(a_ga, 0.1), 3)
            feats[f"away_attack_vs_home_def_{w}"] = round(a_gf * max(h_ga, 0.1), 3)

        # ════════════════════════════════════════
        #  FEATURES NÃO-JANELADAS (fixas)
        # ════════════════════════════════════════

        # Gols por mando de campo (todos os jogos em casa/fora)
        h_gf_home, h_ga_home = self._calc_gols_mando(home_prev, home_id, "home")
        a_gf_away, a_ga_away = self._calc_gols_mando(away_prev, away_id, "away")
        feats["home_gf_mando"] = h_gf_home
        feats["home_ga_mando"] = h_ga_home
        feats["away_gf_mando"] = a_gf_away
        feats["away_ga_mando"] = a_ga_away

        # Sequências atuais (win/loss streaks)
        feats["home_win_streak"] = self._calc_streak(home_prev, home_id, "win")
        feats["away_win_streak"] = self._calc_streak(away_prev, away_id, "win")
        feats["home_loss_streak"] = self._calc_streak(home_prev, home_id, "loss")
        feats["away_loss_streak"] = self._calc_streak(away_prev, away_id, "loss")

        # Confronto direto (H2H)
        h2h = self._calc_h2h(home_id, away_id, league_id, fixture_date)
        feats["h2h_home_pct"] = h2h[0]
        feats["h2h_draw_pct"] = h2h[1]
        feats["h2h_away_pct"] = h2h[2]
        feats["h2h_total"] = h2h[3]
        h2h_market = self._calc_h2h_market_rates(home_id, away_id, fixture_date, limit=5)
        feats["h2h_over15_5"] = h2h_market["over15"]
        feats["h2h_over25_5"] = h2h_market["over25"]
        feats["h2h_under35_5"] = h2h_market["under35"]
        feats["h2h_btts_5"] = h2h_market["btts"]
        feats["h2h_corners_over_85_5"] = h2h_market["corners_over_85"]
        feats["h2h_goals_ht_over05_5"] = h2h_market["over05_ht"]

        # Perfil do árbitro
        referee = fixture.get("referee", "")
        arb = self._calc_perfil_arbitro(referee, fixture_date)
        feats["ref_yellows"] = arb[0]
        feats["ref_fouls"] = arb[1]
        feats["ref_jogos"] = arb[2]

        # ════════════════════════════════════════
        #  TENDÊNCIAS (momentum curto vs longo)
        # ════════════════════════════════════════
        # Delta positivo = time melhorando nos últimos jogos

        feats["home_form_trend"] = round(
            feats["home_form_3"] - feats["home_form_8"], 3
        )
        feats["away_form_trend"] = round(
            feats["away_form_3"] - feats["away_form_8"], 3
        )
        feats["home_gf_trend"] = round(
            feats["home_gf_3"] - feats["home_gf_8"], 3
        )
        feats["away_gf_trend"] = round(
            feats["away_gf_3"] - feats["away_gf_8"], 3
        )
        feats["home_ga_trend"] = round(
            feats["home_ga_3"] - feats["home_ga_8"], 3
        )
        feats["away_ga_trend"] = round(
            feats["away_ga_3"] - feats["away_ga_8"], 3
        )
        feats["home_xg_trend"] = round(
            feats["home_xg_3"] - feats["home_xg_8"], 3
        )
        feats["away_xg_trend"] = round(
            feats["away_xg_3"] - feats["away_xg_8"], 3
        )

        # ════════════════════════════════════════
        #  INTERAÇÕES CRUZADAS
        # ════════════════════════════════════════
        # Capturam sinergia ou conflito entre métricas dos 2 times

        # Forma do mandante × sequência de derrotas do visitante
        feats["int_home_form_x_away_loss"] = round(
            feats["home_form_5"] * feats["away_loss_streak"], 3
        )
        feats["int_away_form_x_home_loss"] = round(
            feats["away_form_5"] * feats["home_loss_streak"], 3
        )

        # Over de ambos os times (se ambos tendem a over, jogo provavelmente tem gols)
        for w in JANELAS:
            feats[f"int_over25_{w}"] = round(
                feats[f"home_over25_{w}"] * feats[f"away_over25_{w}"], 3
            )
            feats[f"int_over15_{w}"] = round(
                feats[f"home_over15_{w}"] * feats[f"away_over15_{w}"], 3
            )

        # BTTS de ambos (se ambos marcam frequentemente)
        for w in JANELAS:
            feats[f"int_btts_{w}"] = round(
                feats[f"home_btts_{w}"] * feats[f"away_btts_{w}"], 3
            )

        return feats

    @staticmethod
    def feature_names_full() -> list[str]:
        """
        Lista determinística de todos os ~217 nomes de features do pool.

        A ordem DEVE ser idêntica à ordem em que features_jogo() gera as features.
        O FeatureEvolution e o autotuner usam esta lista como referência
        para mapear índices do cromossomo → nomes de features.
        """
        nomes = []

        # ── Features por janela ──
        for w in JANELAS:
            # Base por time (17 métricas × 2 lados)
            for lado in ("home", "away"):
                nomes.append(f"{lado}_form_{w}")
                nomes.append(f"{lado}_gf_{w}")
                nomes.append(f"{lado}_ga_{w}")
                nomes.append(f"{lado}_cs_{w}")
                nomes.append(f"{lado}_fts_{w}")
                nomes.append(f"{lado}_over25_{w}")
                nomes.append(f"{lado}_over15_{w}")
                nomes.append(f"{lado}_over35_{w}")
                nomes.append(f"{lado}_btts_{w}")
                nomes.append(f"{lado}_xg_{w}")
                nomes.append(f"{lado}_shots_{w}")
                nomes.append(f"{lado}_shots_on_{w}")
                nomes.append(f"{lado}_poss_{w}")
                nomes.append(f"{lado}_corners_{w}")
                nomes.append(f"{lado}_yellows_{w}")
                nomes.append(f"{lado}_fouls_{w}")
                nomes.append(f"{lado}_goals_ht_{w}")
            # Diffs entre times
            nomes.append(f"form_diff_{w}")
            nomes.append(f"goals_diff_{w}")
            # Ratios derivados (3 por lado × 2 lados)
            for lado in ("home", "away"):
                nomes.append(f"{lado}_atk_def_ratio_{w}")
                nomes.append(f"{lado}_shot_acc_{w}")
                nomes.append(f"{lado}_finish_eff_{w}")
            # Diffs derivados
            nomes.append(f"shots_on_diff_{w}")
            nomes.append(f"total_shots_on_{w}")
            nomes.append(f"total_xg_{w}")
            nomes.append(f"shots_diff_{w}")
            nomes.append(f"poss_diff_{w}")
            nomes.append(f"corners_diff_{w}")
            nomes.append(f"xg_diff_{w}")
            nomes.append(f"home_xg_per_shot_on_{w}")
            nomes.append(f"away_xg_per_shot_on_{w}")
            nomes.append(f"home_attack_vs_away_def_{w}")
            nomes.append(f"away_attack_vs_home_def_{w}")

        # ── Features não-janeladas ──
        nomes.extend([
            "home_gf_mando", "home_ga_mando",
            "away_gf_mando", "away_ga_mando",
            "home_win_streak", "away_win_streak",
            "home_loss_streak", "away_loss_streak",
            "h2h_home_pct", "h2h_draw_pct", "h2h_away_pct", "h2h_total",
            "h2h_over15_5", "h2h_over25_5", "h2h_under35_5",
            "h2h_btts_5", "h2h_corners_over_85_5", "h2h_goals_ht_over05_5",
            "ref_yellows", "ref_fouls", "ref_jogos",
        ])

        # ── Tendências (momentum curto vs longo) ──
        nomes.extend([
            "home_form_trend", "away_form_trend",
            "home_gf_trend", "away_gf_trend",
            "home_ga_trend", "away_ga_trend",
            "home_xg_trend", "away_xg_trend",
        ])

        # ── Interações cruzadas ──
        nomes.extend([
            "int_home_form_x_away_loss",
            "int_away_form_x_home_loss",
        ])
        for w in JANELAS:
            nomes.append(f"int_over25_{w}")
        for w in JANELAS:
            nomes.append(f"int_over15_{w}")
        for w in JANELAS:
            nomes.append(f"int_btts_{w}")

        return nomes

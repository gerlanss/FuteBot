"""Leitura live enxuta, especifica por mercado e por tempo."""

from __future__ import annotations

from typing import Any


class LiveIntelligence:
    """Analisa um jogo ao vivo com heuristicas baratas e por familia de mercado."""

    @staticmethod
    def _is_first_half_market(mercado: str) -> bool:
        return "_ht" in mercado or mercado.startswith("ht_")

    @staticmethod
    def _is_second_half_market(mercado: str) -> bool:
        return "_2t" in mercado

    def analisar(self, item: dict, fixture: dict, stats: list[dict] | None = None) -> dict:
        status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "NS")
        elapsed = ((fixture.get("fixture") or {}).get("status") or {}).get("elapsed") or 0
        goals = fixture.get("goals") or {}
        score = fixture.get("score") or {}
        halftime = score.get("halftime") or {}
        gols_home = int(goals.get("home") or 0)
        gols_away = int(goals.get("away") or 0)
        total_gols = gols_home + gols_away
        gols_ht_home = int(halftime.get("home") or 0)
        gols_ht_away = int(halftime.get("away") or 0)
        gols_ht = gols_ht_home + gols_ht_away
        gols_2t = max(0, total_gols - gols_ht)
        metricas = self._stats_totais(stats or [])
        mercado = item.get("mercado", "")
        watch_type = item.get("watch_type", "approved_prelive")

        base = {
            "status": status,
            "elapsed": elapsed,
            "mercado": mercado,
            "watch_type": watch_type,
            "gols_home": gols_home,
            "gols_away": gols_away,
            "gols_ht": gols_ht,
            "gols_2t": gols_2t,
            "metricas": metricas,
            "veredito": "monitorar",
            "mensagem": "Sigo acompanhando sem mudanca clara.",
        }

        if status not in ("1H", "2H", "HT", "LIVE", "ET"):
            return base

        analise = (
            self._analisar_mercado_gols(
                mercado=mercado,
                watch_type=watch_type,
                elapsed=elapsed,
                total_gols=total_gols,
                gols_ht=gols_ht,
                gols_2t=gols_2t,
                metricas=metricas,
            )
            or self._analisar_mercado_escanteios(
                mercado=mercado,
                watch_type=watch_type,
                elapsed=elapsed,
                metricas=metricas,
            )
            or self._analisar_resultado(
                mercado=mercado,
                watch_type=watch_type,
                elapsed=elapsed,
                gols_home=gols_home,
                gols_away=gols_away,
                metricas=metricas,
            )
        )
        if analise:
            base.update(analise)
        return base

    @staticmethod
    def _stats_totais(stats: list[dict]) -> dict[str, float]:
        total = {
            "shots_total": 0.0,
            "shots_on": 0.0,
            "corners": 0.0,
            "xg": 0.0,
            "red_cards": 0.0,
            "yellow_cards": 0.0,
        }
        teams = []
        for item in stats:
            parsed = {
                "shots_total": 0.0,
                "shots_on": 0.0,
                "corners": 0.0,
                "xg": 0.0,
                "red_cards": 0.0,
                "yellow_cards": 0.0,
            }
            for stat in item.get("statistics", []):
                name = stat.get("type")
                value = stat.get("value")
                parsed_value = LiveIntelligence._num(value)
                if name == "Total Shots":
                    parsed["shots_total"] = parsed_value
                elif name == "Shots on Goal":
                    parsed["shots_on"] = parsed_value
                elif name == "Corner Kicks":
                    parsed["corners"] = parsed_value
                elif name == "expected_goals":
                    parsed["xg"] = parsed_value
                elif name == "Red Cards":
                    parsed["red_cards"] = parsed_value
                elif name == "Yellow Cards":
                    parsed["yellow_cards"] = parsed_value
            teams.append(parsed)
            for key, value in parsed.items():
                total[key] += value
        total["teams"] = teams
        return total

    @staticmethod
    def _num(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        value = str(value).replace("%", "").strip()
        try:
            return float(value)
        except ValueError:
            return 0.0

    @staticmethod
    def _linha_mercado(mercado: str, default: float = 0.5) -> float:
        linha = default
        if "15" in mercado:
            linha = 1.5
        elif "25" in mercado:
            linha = 2.5
        elif "35" in mercado:
            linha = 3.5
        elif "85" in mercado:
            linha = 8.5
        elif "95" in mercado:
            linha = 9.5
        elif "105" in mercado:
            linha = 10.5
        return linha

    @staticmethod
    def _team_metric(metricas: dict, idx: int, key: str) -> float:
        teams = metricas.get("teams") or []
        if len(teams) <= idx:
            return 0.0
        return float(teams[idx].get(key) or 0.0)

    @staticmethod
    def _cancel_text(watch_type: str, approved_text: str, recheck_text: str) -> str:
        return approved_text if watch_type in {"approved_prelive", "live_opportunity"} else recheck_text

    def _analisar_mercado_gols(
        self,
        *,
        mercado: str,
        watch_type: str,
        elapsed: int,
        total_gols: int,
        gols_ht: int,
        gols_2t: int,
        metricas: dict[str, float],
    ) -> dict | None:
        if "corners" in mercado or not ("over" in mercado or "under" in mercado):
            return None
        if "over" in mercado:
            if self._is_first_half_market(mercado):
                return self._analisar_over_ht(watch_type, elapsed, gols_ht, metricas)
            if self._is_second_half_market(mercado):
                return self._analisar_over_2t(watch_type, elapsed, gols_2t, metricas)
            return self._analisar_over_ft(watch_type, elapsed, total_gols, metricas, mercado)
        if self._is_first_half_market(mercado):
            return self._analisar_under_ht(watch_type, elapsed, gols_ht, metricas, mercado)
        if self._is_second_half_market(mercado):
            return self._analisar_under_2t(watch_type, elapsed, gols_2t, metricas, mercado)
        return self._analisar_under_ft(watch_type, elapsed, total_gols, metricas, mercado)

    def _analisar_over_ht(
        self,
        watch_type: str,
        elapsed: int,
        gols_ht: int,
        metricas: dict[str, float],
    ) -> dict:
        shots_on = metricas["shots_on"]
        shots_total = metricas["shots_total"]
        xg = metricas["xg"]
        if gols_ht > 0:
            return {}
        if 24 <= elapsed <= 36 and (
            (shots_on >= 3 and shots_total >= 7 and xg >= 0.55)
            or (shots_on >= 4 and xg >= 0.45)
        ):
            return {
                "veredito": "sinal_live",
                "mensagem": "O 1T ja ganhou volume real e esse over ficou executavel ao vivo.",
            }
        if 10 <= elapsed <= 28 and shots_on <= 1 and shots_total <= 4 and xg <= 0.15:
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O 1T segue frio demais para sustentar esse over.",
                    "Nao apareceu gatilho real para reavaliar esse over no 1T.",
                ),
            }
        return {}

    def _analisar_under_ht(
        self,
        watch_type: str,
        elapsed: int,
        gols_ht: int,
        metricas: dict[str, float],
        mercado: str,
    ) -> dict:
        linha = self._linha_mercado(mercado, default=1.5)
        shots_on = metricas["shots_on"]
        shots_total = metricas["shots_total"]
        xg = metricas["xg"]
        red_cards = metricas["red_cards"]
        if gols_ht >= linha:
            return {}
        if 26 <= elapsed <= 36 and shots_on <= 3 and shots_total <= 9 and xg <= 0.65 and red_cards == 0:
            return {
                "veredito": "sinal_live",
                "mensagem": "O 1T segue controlado e esse under continua defendivel ao vivo.",
            }
        if 10 <= elapsed <= 28 and (shots_on >= 5 or xg >= 1.1 or (red_cards > 0 and xg >= 0.8)):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O 1T abriu cedo demais para sustentar esse under.",
                    "O 1T ficou aberto demais para insistir nesse under.",
                ),
            }
        return {}

    def _analisar_over_ft(
        self,
        watch_type: str,
        elapsed: int,
        total_gols: int,
        metricas: dict[str, float],
        mercado: str,
    ) -> dict:
        linha = self._linha_mercado(mercado, default=0.5)
        shots_on = metricas["shots_on"]
        shots_total = metricas["shots_total"]
        xg = metricas["xg"]
        if total_gols >= linha:
            return {}
        if elapsed >= 55 and shots_on >= 5 and shots_total >= 11 and xg >= 0.85:
            return {
                "veredito": "sinal_live",
                "mensagem": "O jogo ganhou ritmo de gol e esse over ficou bem sustentado ao vivo.",
            }
        if elapsed >= 72 and total_gols == 0 and shots_on < 4 and xg < 0.75:
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O jogo nao entrou em ritmo de gol e a leitura perdeu forca.",
                    "Ainda nao apareceu gatilho real para esse over ao vivo.",
                ),
            }
        return {}

    def _analisar_under_ft(
        self,
        watch_type: str,
        elapsed: int,
        total_gols: int,
        metricas: dict[str, float],
        mercado: str,
    ) -> dict:
        linha = self._linha_mercado(mercado, default=1.5)
        shots_on = metricas["shots_on"]
        shots_total = metricas["shots_total"]
        xg = metricas["xg"]
        red_cards = metricas["red_cards"]
        if total_gols >= linha:
            return {}
        if elapsed >= 55 and shots_on <= 6 and shots_total <= 16 and xg <= 1.15 and red_cards == 0:
            return {
                "veredito": "sinal_live",
                "mensagem": "O jogo segue mais controlado do que aberto e o under continua de pe.",
            }
        if 10 <= elapsed <= 50 and (shots_on >= 7 and xg >= 1.6):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O jogo abriu cedo demais para sustentar esse under.",
                    "O cenario abriu demais para insistir nesse under.",
                ),
            }
        if elapsed >= 70 and (shots_on >= 8 or xg >= 1.55 or red_cards > 0):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O volume ofensivo ficou alto demais para sustentar esse under.",
                    "O live abriu demais para sustentar essa reavaliacao de under.",
                ),
            }
        return {}

    def _analisar_over_2t(
        self,
        watch_type: str,
        elapsed: int,
        gols_2t: int,
        metricas: dict[str, float],
    ) -> dict:
        shots_on = metricas["shots_on"]
        shots_total = metricas["shots_total"]
        xg = metricas["xg"]
        if gols_2t > 0:
            return {}
        if elapsed >= 55 and shots_on >= 5 and shots_total >= 11 and xg >= 0.75:
            return {
                "veredito": "sinal_live",
                "mensagem": "O 2T ganhou ritmo e esse over ficou executavel no live.",
            }
        if elapsed >= 82 and gols_2t == 0 and shots_on <= 3 and xg <= 0.65:
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O 2T nao deu volume suficiente para sustentar esse over.",
                    "Nao apareceu gatilho forte para esse over no 2T.",
                ),
            }
        return {}

    def _analisar_under_2t(
        self,
        watch_type: str,
        elapsed: int,
        gols_2t: int,
        metricas: dict[str, float],
        mercado: str,
    ) -> dict:
        linha = self._linha_mercado(mercado, default=1.5)
        shots_on = metricas["shots_on"]
        shots_total = metricas["shots_total"]
        xg = metricas["xg"]
        red_cards = metricas["red_cards"]
        if gols_2t >= linha:
            return {}
        if elapsed >= 65 and shots_on <= 5 and shots_total <= 13 and xg <= 1.05 and red_cards == 0:
            return {
                "veredito": "sinal_live",
                "mensagem": "O 2T segue controlado e esse under continua bem defendido ao vivo.",
            }
        if elapsed >= 78 and (shots_on >= 7 or xg >= 1.5 or red_cards > 0):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O 2T abriu demais para sustentar esse under.",
                    "O 2T ja passou do ponto para insistir nesse under.",
                ),
            }
        return {}

    def _analisar_mercado_escanteios(
        self,
        *,
        mercado: str,
        watch_type: str,
        elapsed: int,
        metricas: dict[str, float],
    ) -> dict | None:
        if "corners" not in mercado:
            return None

        linha = self._linha_mercado(mercado, default=8.5)
        corners = metricas["corners"]
        shots_total = metricas["shots_total"]
        shots_on = metricas["shots_on"]
        xg = metricas["xg"]
        ht_market = self._is_first_half_market(mercado)

        if "over" in mercado:
            if ht_market:
                if 24 <= elapsed <= 36 and corners >= 3 and shots_total >= 9 and (shots_on >= 3 or xg >= 0.65):
                    return {
                        "veredito": "sinal_live",
                        "mensagem": "A pressao ofensiva ja virou cantos suficientes para esse over.",
                    }
                if 10 <= elapsed <= 28 and corners <= 1 and shots_total <= 5 and xg <= 0.25:
                    return {
                        "veredito": "cancelar",
                        "mensagem": self._cancel_text(
                            watch_type,
                            "O 1T nao gerou pressao lateral suficiente para esse over de cantos.",
                            "Nao apareceu pressao real para buscar esse over de cantos no 1T.",
                        ),
                    }
                return {}
            if elapsed >= 55 and corners >= max(4, linha - 5) and shots_total >= 11 and (shots_on >= 4 or xg >= 0.9):
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O jogo segue empurrando pelos lados e esse over de cantos ganhou sustentacao.",
                }
            if elapsed >= 72 and corners < max(3, linha - 6) and shots_total < 9 and xg < 0.75:
                return {
                    "veredito": "cancelar",
                    "mensagem": self._cancel_text(
                        watch_type,
                        "Os cantos nao aceleraram o bastante para sustentar essa leitura.",
                        "Nao apareceu pressao suficiente para esse over de cantos.",
                    ),
                }
            return {}

        if ht_market:
            if 26 <= elapsed <= 36 and corners <= 3 and shots_total <= 9 and shots_on <= 3 and xg <= 0.75:
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O 1T ainda nao criou pressao lateral suficiente para estourar essa linha de cantos.",
                }
            if 10 <= elapsed <= 28 and corners >= 5 and (shots_total >= 10 or xg >= 0.9):
                return {
                    "veredito": "cancelar",
                    "mensagem": self._cancel_text(
                        watch_type,
                        "O jogo abriu pelos lados cedo demais para esse under de cantos.",
                        "A pressao lateral ja passou do ponto para esse under de cantos.",
                    ),
                }
            return {}

        if elapsed >= 55 and corners <= max(5, linha - 3) and shots_total <= 15 and shots_on <= 5 and xg <= 1.25:
            return {
                "veredito": "sinal_live",
                "mensagem": "O ritmo dos lados segue baixo e esse under de cantos continua saudavel.",
            }
        if elapsed >= 68 and corners >= max(7, linha - 1) and (shots_total >= 16 or shots_on >= 6 or xg >= 1.45):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "A pressao lateral ficou forte demais para sustentar esse under de cantos.",
                    "O jogo passou do ponto para esse under de cantos.",
                ),
            }
        return {}

    def _analisar_resultado(
        self,
        *,
        mercado: str,
        watch_type: str,
        elapsed: int,
        gols_home: int,
        gols_away: int,
        metricas: dict[str, float],
    ) -> dict | None:
        if not (mercado.startswith("h2h") or mercado.startswith("ht_")):
            return None
        if self._is_first_half_market(mercado):
            return self._analisar_resultado_ht(
                mercado=mercado,
                watch_type=watch_type,
                elapsed=elapsed,
                gols_home=gols_home,
                gols_away=gols_away,
                metricas=metricas,
            )
        return self._analisar_resultado_ft(
            mercado=mercado,
            watch_type=watch_type,
            elapsed=elapsed,
            gols_home=gols_home,
            gols_away=gols_away,
            metricas=metricas,
        )

    def _analisar_resultado_ht(
        self,
        *,
        mercado: str,
        watch_type: str,
        elapsed: int,
        gols_home: int,
        gols_away: int,
        metricas: dict[str, float],
    ) -> dict:
        home_on = self._team_metric(metricas, 0, "shots_on")
        away_on = self._team_metric(metricas, 1, "shots_on")
        home_xg = self._team_metric(metricas, 0, "xg")
        away_xg = self._team_metric(metricas, 1, "xg")
        home_shots = self._team_metric(metricas, 0, "shots_total")
        away_shots = self._team_metric(metricas, 1, "shots_total")

        if mercado.endswith("home"):
            if 25 <= elapsed <= 36 and gols_home >= gols_away and home_on >= away_on + 2 and home_xg >= away_xg + 0.25:
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O mandante ja construiu superioridade suficiente para esse lado no 1T.",
                }
            if 10 <= elapsed <= 30 and gols_home < gols_away and away_xg >= home_xg + 0.35:
                return {
                    "veredito": "cancelar",
                    "mensagem": self._cancel_text(
                        watch_type,
                        "O 1T foi para o lado oposto e o mandante nao confirmou superioridade suficiente.",
                        "O 1T caminhou para o lado oposto e nao abriu gatilho para o mandante.",
                    ),
                }
            return {}

        if mercado.endswith("away"):
            if 25 <= elapsed <= 36 and gols_away >= gols_home and away_on >= home_on + 2 and away_xg >= home_xg + 0.25:
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O visitante ja construiu superioridade suficiente para esse lado no 1T.",
                }
            if 10 <= elapsed <= 30 and gols_away < gols_home and home_xg >= away_xg + 0.35:
                return {
                    "veredito": "cancelar",
                    "mensagem": self._cancel_text(
                        watch_type,
                        "O 1T foi para o lado oposto e o visitante nao confirmou superioridade suficiente.",
                        "O 1T caminhou para o lado oposto e nao abriu gatilho para o visitante.",
                    ),
                }
            return {}

        if 28 <= elapsed <= 36 and gols_home == gols_away and abs(home_on - away_on) <= 1 and abs(home_xg - away_xg) <= 0.18 and abs(home_shots - away_shots) <= 2:
            return {
                "veredito": "sinal_live",
                "mensagem": "O 1T segue equilibrado de verdade e o empate continua bem defendido.",
            }
        if 10 <= elapsed <= 30 and (gols_home != gols_away or abs(home_xg - away_xg) >= 0.45 or abs(home_on - away_on) >= 2):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O 1T perdeu o equilibrio necessario para sustentar esse empate.",
                    "O 1T ja nao mostra o equilibrio necessario para buscar esse empate.",
                ),
            }
        return {}

    def _analisar_resultado_ft(
        self,
        *,
        mercado: str,
        watch_type: str,
        elapsed: int,
        gols_home: int,
        gols_away: int,
        metricas: dict[str, float],
    ) -> dict:
        home_on = self._team_metric(metricas, 0, "shots_on")
        away_on = self._team_metric(metricas, 1, "shots_on")
        home_xg = self._team_metric(metricas, 0, "xg")
        away_xg = self._team_metric(metricas, 1, "xg")
        home_shots = self._team_metric(metricas, 0, "shots_total")
        away_shots = self._team_metric(metricas, 1, "shots_total")

        if mercado.endswith("home"):
            if elapsed >= 55 and gols_home >= gols_away and home_on >= away_on + 2 and home_xg >= away_xg + 0.3:
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O mandante esta confirmando superioridade real dentro do jogo.",
                }
            if elapsed >= 60 and gols_home < gols_away and away_xg >= home_xg + 0.4:
                return {
                    "veredito": "cancelar",
                    "mensagem": self._cancel_text(
                        watch_type,
                        "O jogo foi para o lado oposto e o mandante nao sustentou o favoritismo.",
                        "O jogo caminhou para o lado oposto e nao abriu gatilho para o mandante.",
                    ),
                }
            return {}

        if mercado.endswith("away"):
            if elapsed >= 55 and gols_away >= gols_home and away_on >= home_on + 2 and away_xg >= home_xg + 0.3:
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O visitante esta confirmando superioridade real dentro do jogo.",
                }
            if elapsed >= 60 and gols_away < gols_home and home_xg >= away_xg + 0.4:
                return {
                    "veredito": "cancelar",
                    "mensagem": self._cancel_text(
                        watch_type,
                        "O jogo foi para o lado oposto e o visitante nao sustentou a leitura.",
                        "O jogo caminhou para o lado oposto e nao abriu gatilho para o visitante.",
                    ),
                }
            return {}

        if elapsed >= 58 and gols_home == gols_away and abs(home_on - away_on) <= 1 and abs(home_xg - away_xg) <= 0.22 and abs(home_shots - away_shots) <= 2:
            return {
                "veredito": "sinal_live",
                "mensagem": "O jogo segue equilibrado de verdade e o empate continua bem sustentado.",
            }
        if elapsed >= 65 and (gols_home != gols_away or abs(home_xg - away_xg) >= 0.35 or abs(home_on - away_on) >= 2):
            return {
                "veredito": "cancelar",
                "mensagem": self._cancel_text(
                    watch_type,
                    "O jogo perdeu o equilibrio necessario para sustentar essa leitura de empate.",
                    "O jogo ja nao mostra o equilibrio necessario para buscar esse empate.",
                ),
            }
        return {}

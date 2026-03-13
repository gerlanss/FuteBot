"""Leitura live enxuta para jogos monitorados pelo pre-live."""

from __future__ import annotations

from typing import Any


class LiveIntelligence:
    """Analisa um jogo ao vivo com heurísticas simples e baratas."""

    def analisar(self, item: dict, fixture: dict, stats: list[dict] | None = None) -> dict:
        status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "NS")
        elapsed = ((fixture.get("fixture") or {}).get("status") or {}).get("elapsed") or 0
        goals = fixture.get("goals") or {}
        gols_home = int(goals.get("home") or 0)
        gols_away = int(goals.get("away") or 0)
        total_gols = gols_home + gols_away
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
            "metricas": metricas,
            "veredito": "monitorar",
            "mensagem": "Sigo observando sem mudança clara.",
        }

        if status not in ("1H", "2H", "HT", "LIVE", "ET"):
            return base

        analise = (
            self._analisar_mercado_gols(mercado, watch_type, elapsed, total_gols, metricas)
            or self._analisar_mercado_escanteios(mercado, watch_type, elapsed, metricas)
            or self._analisar_resultado(mercado, watch_type, elapsed, gols_home, gols_away, metricas)
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
        }
        teams = []
        for item in stats:
            parsed = {"shots_total": 0.0, "shots_on": 0.0, "corners": 0.0, "xg": 0.0}
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

    def _analisar_mercado_gols(
        self,
        mercado: str,
        watch_type: str,
        elapsed: int,
        total_gols: int,
        metricas: dict[str, float],
    ) -> dict | None:
        if "over" in mercado and "corners" not in mercado:
            return self._analisar_over_gols(mercado, watch_type, elapsed, total_gols, metricas)
        if "under" in mercado and "corners" not in mercado:
            return self._analisar_under_gols(mercado, watch_type, elapsed, total_gols, metricas)
        return None

    def _analisar_over_gols(self, mercado: str, watch_type: str, elapsed: int, total_gols: int, metricas: dict) -> dict:
        linha = 0.5
        if "15" in mercado:
            linha = 1.5
        elif "25" in mercado:
            linha = 2.5
        elif "35" in mercado:
            linha = 3.5

        shots_on = metricas["shots_on"]
        xg = metricas["xg"]

        if elapsed >= 55 and total_gols < linha and shots_on >= 6 and xg >= 1.4:
            return {
                "veredito": "sinal_live",
                "mensagem": "O jogo ganhou volume e o over ainda respira bem no live.",
            }
        if elapsed >= 60 and total_gols == 0 and shots_on < 4 and xg < 0.9:
            texto = "A leitura perdeu força no live." if watch_type == "approved_prelive" else "Ainda não vejo gatilho para reavaliar esse over ao vivo."
            return {
                "veredito": "cancelar",
                "mensagem": texto,
            }
        if elapsed >= 30 and shots_on >= 4 and xg >= 0.9:
            return {
                "veredito": "monitorar_forte",
                "mensagem": "O volume ofensivo está saudável e vale continuar acompanhando.",
            }
        return {}

    def _analisar_under_gols(self, mercado: str, watch_type: str, elapsed: int, total_gols: int, metricas: dict) -> dict:
        linha = 1.5
        if "25" in mercado:
            linha = 2.5
        elif "35" in mercado:
            linha = 3.5

        shots_on = metricas["shots_on"]
        xg = metricas["xg"]

        if elapsed >= 55 and total_gols < linha and shots_on <= 5 and xg <= 1.1:
            return {
                "veredito": "sinal_live",
                "mensagem": "O jogo segue travado e o under continua combinando com o live.",
            }
        if elapsed <= 35 and shots_on >= 6 and xg >= 1.5:
            texto = "O jogo abriu cedo demais e essa leitura perdeu força." if watch_type == "approved_prelive" else "O cenário ficou aberto demais para insistir nesse under."
            return {
                "veredito": "cancelar",
                "mensagem": texto,
            }
        return {}

    def _analisar_mercado_escanteios(self, mercado: str, watch_type: str, elapsed: int, metricas: dict) -> dict | None:
        if "corners" not in mercado:
            return None
        corners = metricas["corners"]

        if "over" in mercado:
            if elapsed >= 35 and corners >= 5:
                return {
                    "veredito": "sinal_live",
                    "mensagem": "O volume pelos lados está forte e os escanteios seguem vivos.",
                }
            if elapsed >= 60 and corners <= 4:
                texto = "Os cantos não aceleraram e a leitura perdeu força." if watch_type == "approved_prelive" else "Ainda não vejo pressão suficiente para procurar esse over de escanteios."
                return {
                    "veredito": "cancelar",
                    "mensagem": texto,
                }
            return {}

        if elapsed >= 55 and corners <= 6:
            return {
                "veredito": "sinal_live",
                "mensagem": "O ritmo dos lados segue baixo e o under de escanteios continua saudável.",
            }
        if elapsed <= 35 and corners >= 7:
            texto = "O jogo abriu pelos lados e essa leitura perdeu força." if watch_type == "approved_prelive" else "A pressão lateral já passou do ponto para esse under."
            return {
                "veredito": "cancelar",
                "mensagem": texto,
            }
        return {}

    def _analisar_resultado(
        self,
        mercado: str,
        watch_type: str,
        elapsed: int,
        gols_home: int,
        gols_away: int,
        metricas: dict,
    ) -> dict | None:
        if not (mercado.startswith("h2h") or mercado.startswith("ht_")):
            return None

        teams = metricas.get("teams") or [{}, {}]
        home_on = teams[0].get("shots_on", 0.0) if len(teams) > 0 else 0.0
        away_on = teams[1].get("shots_on", 0.0) if len(teams) > 1 else 0.0
        home_xg = teams[0].get("xg", 0.0) if len(teams) > 0 else 0.0
        away_xg = teams[1].get("xg", 0.0) if len(teams) > 1 else 0.0

        if mercado.endswith("home") and elapsed >= 55 and gols_home >= gols_away and home_on >= away_on + 2 and home_xg >= away_xg + 0.4:
            return {
                "veredito": "sinal_live",
                "mensagem": "O mandante está confirmando o favoritismo dentro de campo.",
            }
        if mercado.endswith("away") and elapsed >= 55 and gols_away >= gols_home and away_on >= home_on + 2 and away_xg >= home_xg + 0.4:
            return {
                "veredito": "sinal_live",
                "mensagem": "O visitante está confirmando a superioridade no live.",
            }
        if elapsed >= 60 and abs(home_on - away_on) <= 1 and abs(home_xg - away_xg) <= 0.2:
            texto = "O jogo está equilibrado demais para sustentar essa leitura." if watch_type == "approved_prelive" else "Ainda não vejo superioridade clara para procurar esse lado ao vivo."
            return {
                "veredito": "cancelar",
                "mensagem": texto,
            }
        return {}

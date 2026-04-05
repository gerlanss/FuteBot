import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bulk_download import (
    LIMITE_SEGURANCA,
    baixar_contexto_operacional_fixtures,
    baixar_eventos_live_treino,
    baixar_team_stats_contexto,
)
from data.database import Database
from services.apifootball import status_conta


def _remaining_safe_requests() -> int:
    status = status_conta()
    req = (status or {}).get("requests", {})
    current = int(req.get("current") or 0)
    return max(LIMITE_SEGURANCA - current, 0)


def main():
    db = Database()
    safe_remaining = _remaining_safe_requests()

    summary = {
        "generated_at": datetime.now().isoformat(),
        "safe_remaining_before": safe_remaining,
        "steps": [],
    }

    # 1) Eventos dos fixtures live primeiro: maior retorno por request para o treino live.
    eventos_live = 0
    if safe_remaining > 80:
        max_eventos_live = max(min(safe_remaining - 60, 120), 0)
        eventos_live = baixar_eventos_live_treino(db, max_items=max_eventos_live)
        summary["steps"].append({
            "step": "eventos_live",
            "max_items": max_eventos_live,
            "result": eventos_live,
        })

    safe_remaining = _remaining_safe_requests()

    # 2) Contexto operacional barato para os jogos próximos.
    contexto = {}
    if safe_remaining > 300:
        contexto = baixar_contexto_operacional_fixtures(db, lookahead_days=3, include_players=False)
        summary["steps"].append({
            "step": "contexto_operacional",
            "result": contexto,
        })

    safe_remaining = _remaining_safe_requests()

    # 3) Consumir o resto com team_stats faltantes, mas sem ficar abraçando o capeta.
    team_stats = 0
    if safe_remaining > 180:
        budget_team_stats = max(min(safe_remaining - 120, 300), 0)
        if budget_team_stats > 0:
            team_stats = baixar_team_stats_contexto(db, max_items=budget_team_stats)
            summary["steps"].append({
                "step": "team_stats_missing",
                "max_items": budget_team_stats,
                "result": team_stats,
            })

    summary["safe_remaining_after"] = _remaining_safe_requests()
    summary["db_counts"] = db.resumo()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

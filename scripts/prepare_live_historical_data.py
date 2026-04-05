import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bulk_download import baixar_eventos_historicos_live
from data.database import Database
from services.apifootball import status_conta


def _remaining_requests(reserve: int) -> tuple[int, int, int]:
    status = status_conta()
    req = (status or {}).get("requests", {})
    current = int(req.get("current") or 0)
    limit_day = int(req.get("limit_day") or 7500)
    available = max(limit_day - current - reserve, 0)
    return current, limit_day, available


def main():
    parser = argparse.ArgumentParser(description="Prepara base histórica do live sem torrar a API feito um animal.")
    parser.add_argument("--reserve", type=int, default=150, help="Reserva de requests para não encostar no limite diário.")
    parser.add_argument("--max-items", type=int, default=0, help="Máximo de fixtures para backfill nesta rodada.")
    args = parser.parse_args()

    current, limit_day, available = _remaining_requests(args.reserve)
    db = Database()

    budget = args.max_items if args.max_items > 0 else available
    budget = max(min(budget, available), 0)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "source": "api_historical",
        "requests_current": current,
        "requests_limit_day": limit_day,
        "reserve": args.reserve,
        "available_for_backfill": available,
        "budget_used": budget,
    }

    if budget <= 0:
        summary["status"] = "blocked"
        summary["reason"] = "sem_orcamento_api"
        summary["db_counts"] = db.resumo()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    result = baixar_eventos_historicos_live(db, max_items=budget, reserve=args.reserve)
    summary["status"] = "ok"
    summary["result"] = result
    summary["db_counts"] = db.resumo()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

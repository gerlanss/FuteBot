import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from services.apifootball import status_conta


def main():
    db = Database()
    conn = db._conn()

    counts = db.resumo()

    missing_events = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM fixtures f
        LEFT JOIN fixture_events fe ON fe.fixture_id = f.fixture_id
        WHERE f.status = 'FT' AND fe.fixture_id IS NULL
        """
    ).fetchone()["n"]

    team_pairs = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM (
            SELECT DISTINCT home_id AS team_id, league_id, season
            FROM fixtures
            UNION
            SELECT DISTINCT away_id AS team_id, league_id, season
            FROM fixtures
        )
        """
    ).fetchone()["n"]

    upcoming_fixtures = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM fixtures
        WHERE datetime(date) >= datetime('now', '-1 day')
          AND datetime(date) <= datetime('now', '+3 day')
        """
    ).fetchone()["n"]
    conn.close()

    status = status_conta()
    requests_info = (status or {}).get("requests", {})
    current = int(requests_info.get("current") or 0)
    limit_day = int(requests_info.get("limit_day") or 0)
    remaining = max(limit_day - current, 0)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "db_counts": counts,
        "backfill_plan": {
            "fixture_events_missing": int(missing_events or 0),
            "team_stats_pairs": int(team_pairs or 0),
            "upcoming_operational_fixtures": int(upcoming_fixtures or 0),
            "estimated_requests": {
                "events_historicos": int(missing_events or 0),
                "team_stats_agregadas": int(team_pairs or 0),
                "contexto_operacional_sem_players": int(upcoming_fixtures or 0) * 2,
                "contexto_operacional_com_players": int(upcoming_fixtures or 0) * 3,
            },
        },
        "api_status": {
            "current": current,
            "limit_day": limit_day,
            "remaining_today": remaining,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

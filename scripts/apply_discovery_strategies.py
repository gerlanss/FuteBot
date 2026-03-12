import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database


def _latest_run(root: Path) -> Path:
    dirs = [item for item in root.iterdir() if item.is_dir()]
    if not dirs:
        raise FileNotFoundError("nenhum run de discovery encontrado")
    return sorted(dirs, key=lambda item: item.name)[-1]


def _infer_conf_band(best_rule: dict | None) -> tuple[float, float]:
    conf_min = 0.65
    conf_max = 1.01
    if not best_rule:
        return conf_min, conf_max
    for feature, op, threshold in best_rule.get("conditions", []):
        if feature != "model_prob":
            continue
        if op == ">=":
            conf_min = max(conf_min, float(threshold))
        elif op == "<=":
            conf_max = min(conf_max, float(threshold))
    if conf_max <= conf_min:
        conf_max = min(1.01, conf_min + 0.05)
    return round(conf_min, 4), round(conf_max, 4)


def main():
    parser = argparse.ArgumentParser(description="Converte discovery em strategies operacionais do scanner.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "futebot.db"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_dir) if args.run_dir else _latest_run(ROOT / "data" / "discovery_runs")
    summary_path = run_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json não encontrado em {run_root}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    strategies = []
    touched_leagues = set()

    for league in summary.get("leagues", []):
        if league.get("status") == "error":
            continue
        lid = int(league["league_id"])
        touched_leagues.add(lid)
        for market in league.get("markets", []):
            if market.get("status") != "accepted":
                continue
            if market["market"] in {"btts_yes", "btts_no"}:
                continue
            best = market.get("best_rule")
            conf_min, conf_max = _infer_conf_band(best)
            test = best.get("test", {}) if best else {}
            strategies.append({
                "mercado": market["market"],
                "league_id": lid,
                "conf_min": conf_min,
                "conf_max": conf_max,
                "accuracy": float(test.get("precision") or market.get("test_base_rate") or 0),
                "n_samples": int(test.get("samples") or market.get("test_base_samples") or 0),
                "ev_medio": 0,
                "ativo": 1,
                "params": {
                    "source": "market_discovery",
                    "rule": best.get("rule") if best else "",
                    "conditions": best.get("conditions", []) if best else [],
                    "train": best.get("train") if best else {},
                    "test": best.get("test") if best else {},
                    "run_id": summary.get("run_id"),
                },
                "modelo_versao": f"discovery_{summary.get('run_id')}",
            })

    db = Database(args.db_path)
    db.salvar_strategies(
        strategies,
        replace=args.replace,
        league_ids=None if args.replace else sorted(touched_leagues),
    )
    print(json.dumps({
        "run_dir": str(run_root),
        "strategies_saved": len(strategies),
        "leagues": sorted(touched_leagues),
        "replace": bool(args.replace),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.market_discovery import DEFAULT_OUTPUT_DIR, MarketDiscoveryTrainer


def main():
    parser = argparse.ArgumentParser(description="Treino sequencial por liga e mercado com descoberta de padrões.")
    parser.add_argument("--league-ids", default="")
    parser.add_argument("--markets", default="")
    parser.add_argument("--target-precision", type=float, default=0.65)
    parser.add_argument("--min-train-samples", type=int, default=30)
    parser.add_argument("--min-test-samples", type=int, default=10)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    league_ids = [int(item.strip()) for item in args.league_ids.split(",") if item.strip()]
    markets = [item.strip() for item in args.markets.split(",") if item.strip()]
    trainer = MarketDiscoveryTrainer()

    def on_progress(event: dict):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if event["type"] == "market":
            item = event["summary"]
            best = item.get("best_rule")
            if best:
                print(
                    f"[{now}] liga={event['league_id']} mercado={item['market']} status={item['status']} "
                    f"test={best['test']['precision']:.3f} ({best['test']['samples']}) "
                    f"rule={best['rule']}"
                )
            else:
                print(
                    f"[{now}] liga={event['league_id']} mercado={item['market']} "
                    f"status={item['status']} sem regra robusta"
                )
        elif event["type"] == "league":
            summary = event["summary"]
            if summary.get("status") == "error":
                print(f"[{now}] liga={event['league_id']} erro={summary['error']}")
            else:
                print(
                    f"[{now}] liga={event['league_id']} concluída "
                    f"aceitos={summary['accepted_markets']}/{len(summary['markets'])} "
                    f"tempo={summary['duration_seconds']:.1f}s"
                )

    result = trainer.run(
        league_ids=league_ids or None,
        markets=markets or None,
        target_precision=args.target_precision,
        min_train_samples=args.min_train_samples,
        min_test_samples=args.min_test_samples,
        optuna_trials=args.optuna_trials,
        output_dir=args.output_dir,
        progress_callback=on_progress,
    )
    output_root = Path(args.output_dir) / result["run_id"]
    print(json.dumps({
        "run_id": result["run_id"],
        "accepted_markets": result["accepted_markets"],
        "duration_seconds": result["duration_seconds"],
        "output_dir": str(output_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

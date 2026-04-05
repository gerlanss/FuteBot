import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DISCOVERY_MIN_TEST_SAMPLES,
    DISCOVERY_MIN_TEST_SAMPLES_COPA,
    DISCOVERY_MIN_TRAIN_SAMPLES,
    DISCOVERY_MIN_TRAIN_SAMPLES_COPA,
    DISCOVERY_OPTUNA_TRIALS,
    DISCOVERY_TARGET_PRECISION,
    DISCOVERY_TARGET_PRECISION_COPA,
    LEAGUES,
)
from data.database import Database
from models.autotuner import AutoTuner
from models.live_trainer import LiveTrainer
from models.market_discovery import MarketDiscoveryTrainer


def _infer_seasons(db: Database) -> tuple[list[int], int]:
    fixtures = db.fixtures_finalizados()
    seasons = sorted({int(item["season"]) for item in fixtures if item.get("season")})
    if len(seasons) < 2:
        raise RuntimeError("Nao ha seasons suficientes para treino pre-live.")
    test_season = seasons[-1]
    train_seasons = seasons[:-1]
    if len(train_seasons) > 3:
        train_seasons = train_seasons[-3:]
    return train_seasons, test_season


def main():
    parser = argparse.ArgumentParser(description="Treino combinado do FuteBot: pre-live + live bootstrap")
    parser.add_argument("--league-ids", default="")
    parser.add_argument("--prelive-trials", type=int, default=20)
    parser.add_argument("--live-min-samples", type=int, default=12)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    db = Database()
    league_ids = [int(item.strip()) for item in args.league_ids.split(",") if item.strip()]
    train_seasons, test_season = _infer_seasons(db)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "train_seasons": train_seasons,
        "test_season": test_season,
    }

    tuner = AutoTuner(db)
    summary["prelive_autotuner"] = tuner.executar(
        train_seasons=train_seasons,
        test_season=test_season,
        n_trials=args.prelive_trials,
        league_ids=league_ids or None,
    )

    if not args.skip_discovery:
        trainer = MarketDiscoveryTrainer(db)
        discovery_results = []
        target_leagues = league_ids or None
        if target_leagues:
            selected = target_leagues
        else:
            selected = [info["id"] for info in sorted(LEAGUES.values(), key=lambda x: x["nome"])]
        for lid in selected:
            target_precision = DISCOVERY_TARGET_PRECISION_COPA if trainer._is_cup(lid) else DISCOVERY_TARGET_PRECISION
            min_train = DISCOVERY_MIN_TRAIN_SAMPLES_COPA if trainer._is_cup(lid) else DISCOVERY_MIN_TRAIN_SAMPLES
            min_test = DISCOVERY_MIN_TEST_SAMPLES_COPA if trainer._is_cup(lid) else DISCOVERY_MIN_TEST_SAMPLES
            discovery_results.append(
                trainer.run(
                    league_ids=[lid],
                    target_precision=target_precision,
                    min_train_samples=min_train,
                    min_test_samples=min_test,
                    optuna_trials=DISCOVERY_OPTUNA_TRIALS,
                )
            )
        summary["prelive_discovery"] = discovery_results

    if not args.skip_live:
        live_trainer = LiveTrainer(db)
        summary["live_bootstrap"] = live_trainer.treinar(min_amostras_mercado=args.live_min_samples)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

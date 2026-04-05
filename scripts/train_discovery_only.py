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
from models.market_discovery import MarketDiscoveryTrainer


def main():
    parser = argparse.ArgumentParser(description="Executa apenas o discovery pre-live.")
    parser.add_argument("--league-ids", default="")
    args = parser.parse_args()

    db = Database()
    trainer = MarketDiscoveryTrainer(db)
    league_ids = [int(item.strip()) for item in args.league_ids.split(",") if item.strip()]
    selected = league_ids or [info["id"] for info in sorted(LEAGUES.values(), key=lambda x: x["nome"])]

    results = []
    for lid in selected:
        target_precision = DISCOVERY_TARGET_PRECISION_COPA if trainer._is_cup(lid) else DISCOVERY_TARGET_PRECISION
        min_train = DISCOVERY_MIN_TRAIN_SAMPLES_COPA if trainer._is_cup(lid) else DISCOVERY_MIN_TRAIN_SAMPLES
        min_test = DISCOVERY_MIN_TEST_SAMPLES_COPA if trainer._is_cup(lid) else DISCOVERY_MIN_TEST_SAMPLES
        results.append(
            trainer.run(
                league_ids=[lid],
                target_precision=target_precision,
                min_train_samples=min_train,
                min_test_samples=min_test,
                optuna_trials=DISCOVERY_OPTUNA_TRIALS,
            )
        )

    print(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "league_ids": selected,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

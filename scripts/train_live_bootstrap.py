import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from models.live_trainer import LiveTrainer


def main():
    parser = argparse.ArgumentParser(description="Treino live real por liga x mercado, sem global e sem odds.")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-samples-corners", type=int, default=5)
    args = parser.parse_args()

    db = Database()
    trainer = LiveTrainer(db)
    result = trainer.treinar(
        min_amostras_mercado=args.min_samples,
        min_amostras_cantos=args.min_samples_corners,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

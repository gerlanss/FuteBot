import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from models.features import FeatureExtractor
from models.feature_factory import FeatureFactory


MARKETS = {
    "over15": {"label_key": "over15", "positive_name": "over15", "negative_name": "under15"},
    "over35": {"label_key": "over35", "positive_name": "over35", "negative_name": "under35"},
}


def _load_split(db_path: str, league_id: int, train_seasons: list[int],
                val_season: int, test_season: int, feature_set: str):
    db = Database(db_path)
    extractor = FeatureFactory(db) if feature_set == "full" else FeatureExtractor(db)
    feat_names = (
        FeatureFactory.feature_names_full()
        if feature_set == "full"
        else FeatureExtractor.feature_names()
    )
    feats, labels = extractor.build_dataset(
        league_id=league_id,
        seasons=train_seasons + [val_season, test_season],
    )

    buckets = {
        "train": {"x": [], "labels": []},
        "val": {"x": [], "labels": []},
        "test": {"x": [], "labels": []},
    }
    for feat, label in zip(feats, labels):
        row = [feat.get(name, 0) or 0 for name in feat_names]
        season = feat.get("_season")
        if season in train_seasons:
            buckets["train"]["x"].append(row)
            buckets["train"]["labels"].append(label)
        elif season == val_season:
            buckets["val"]["x"].append(row)
            buckets["val"]["labels"].append(label)
        elif season == test_season:
            buckets["test"]["x"].append(row)
            buckets["test"]["labels"].append(label)

    payload = {}
    for key in ("train", "val", "test"):
        payload[key] = {
            "x": np.asarray(buckets[key]["x"], dtype=np.float32),
            "labels": buckets[key]["labels"],
        }
    payload["feat_names"] = feat_names
    return payload


def _extract_y(labels: list[dict], label_key: str) -> np.ndarray:
    return np.asarray([row[label_key] for row in labels], dtype=np.int64)


def _suggest_params(trial: optuna.Trial, y_train: np.ndarray) -> dict:
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
        "nthread": os.cpu_count() or 4,
    }
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos > 0 and neg > 0:
        ratio = neg / pos
        low = max(0.1, min(ratio * 0.5, ratio * 2.0))
        high = max(low + 0.01, max(ratio * 0.5, ratio * 2.0))
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", low, high)
    return params


def _fit_predict(params: dict, x_train: np.ndarray, y_train: np.ndarray,
                 x_pred: np.ndarray, feat_names: list[str]) -> np.ndarray:
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feat_names)
    dpred = xgb.DMatrix(x_pred, feature_names=feat_names)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        verbose_eval=False,
    )
    pred = model.predict(dpred)
    del dtrain, dpred, model
    gc.collect()
    return np.asarray(pred, dtype=np.float32)


def _optimize(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
              y_val: np.ndarray, feat_names: list[str], n_trials: int) -> tuple[dict, float]:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
    )

    def objective(trial: optuna.Trial):
        params = _suggest_params(trial, y_train)
        pred = _fit_predict(params, x_train, y_train, x_val, feat_names)
        return log_loss(y_val, np.clip(pred, 1e-7, 1 - 1e-7))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def _search_thresholds(p_yes: np.ndarray, y_true: np.ndarray,
                       positive_name: str, negative_name: str,
                       min_samples: int):
    out = []
    for thr in np.arange(0.55, 0.981, 0.025):
        idx = p_yes >= thr
        total = int(idx.sum())
        if total >= min_samples:
            out.append({
                "side": positive_name,
                "threshold": round(float(thr), 3),
                "samples": total,
                "accuracy": float(np.mean(y_true[idx] == 1)),
            })
        idx = p_yes <= (1.0 - thr)
        total = int(idx.sum())
        if total >= min_samples:
            out.append({
                "side": negative_name,
                "threshold": round(float(thr), 3),
                "samples": total,
                "accuracy": float(np.mean(y_true[idx] == 0)),
            })
    out.sort(key=lambda item: (item["accuracy"], item["samples"]), reverse=True)
    return out


def _apply_threshold(strategy: dict, p_yes: np.ndarray, y_true: np.ndarray,
                     positive_name: str):
    if strategy["side"] == positive_name:
        idx = p_yes >= strategy["threshold"]
        wins = y_true[idx] == 1
    else:
        idx = p_yes <= (1.0 - strategy["threshold"])
        wins = y_true[idx] == 0
    total = int(idx.sum())
    return {
        "samples": total,
        "accuracy": float(np.mean(wins)) if total else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Tuning XGBoost+Optuna por liga/mercado com busca de slices.")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "futebot.db"))
    parser.add_argument("--league-ids", default="128,253")
    parser.add_argument("--markets", default="over15,over35")
    parser.add_argument("--train-seasons", default="2024")
    parser.add_argument("--val-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--feature-sets", default="static,full")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--min-val-samples", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default=str(ROOT / "data" / "optuna_xgb_market_slices.json"))
    args = parser.parse_args()

    league_ids = [int(x.strip()) for x in args.league_ids.split(",") if x.strip()]
    markets = [x.strip() for x in args.markets.split(",") if x.strip()]
    train_seasons = [int(x.strip()) for x in args.train_seasons.split(",") if x.strip()]
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seasons": train_seasons,
        "val_season": args.val_season,
        "test_season": args.test_season,
        "trials": args.trials,
        "runs": [],
    }

    for league_id in league_ids:
        for market in markets:
            cfg = MARKETS[market]
            for feature_set in feature_sets:
                print(f"[optuna-xgb] league={league_id} market={market} feature_set={feature_set}")
                split = _load_split(
                    args.db_path, league_id, train_seasons, args.val_season, args.test_season, feature_set
                )
                x_train = split["train"]["x"]
                x_val = split["val"]["x"]
                x_test = split["test"]["x"]
                feat_names = split["feat_names"]
                y_train = _extract_y(split["train"]["labels"], cfg["label_key"])
                y_val = _extract_y(split["val"]["labels"], cfg["label_key"])
                y_test = _extract_y(split["test"]["labels"], cfg["label_key"])

                best_params, best_val_logloss = _optimize(
                    x_train, y_train, x_val, y_val, feat_names, n_trials=args.trials
                )
                full_params = {
                    **best_params,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "seed": 42,
                    "verbosity": 0,
                    "nthread": os.cpu_count() or 4,
                }
                val_pred = _fit_predict(full_params, x_train, y_train, x_val, feat_names)
                test_pred = _fit_predict(full_params, x_train, y_train, x_test, feat_names)

                overall = {
                    "val_accuracy": float(accuracy_score(y_val, (val_pred >= 0.5).astype(int))),
                    "test_accuracy": float(accuracy_score(y_test, (test_pred >= 0.5).astype(int))),
                    "val_log_loss": float(log_loss(y_val, np.clip(val_pred, 1e-7, 1 - 1e-7))),
                    "test_log_loss": float(log_loss(y_test, np.clip(test_pred, 1e-7, 1 - 1e-7))),
                    "val_samples": int(len(y_val)),
                    "test_samples": int(len(y_test)),
                }
                strategies = _search_thresholds(
                    val_pred, y_val, cfg["positive_name"], cfg["negative_name"], args.min_val_samples
                )
                top = []
                for item in strategies[:args.top_k]:
                    applied = _apply_threshold(item, test_pred, y_test, cfg["positive_name"])
                    top.append(item | {
                        "test_samples": applied["samples"],
                        "test_accuracy": applied["accuracy"],
                    })

                results["runs"].append({
                    "league_id": league_id,
                    "market": market,
                    "feature_set": feature_set,
                    "best_params": best_params,
                    "best_val_logloss": best_val_logloss,
                    "overall": overall,
                    "top_strategies": top,
                })

                if top:
                    best = top[0]
                    print(
                        f"  best={best['side']} thr={best['threshold']:.3f} "
                        f"| val={best['accuracy']:.3f} ({best['samples']}) "
                        f"| test={best['test_accuracy']} ({best['test_samples']})"
                    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[optuna-xgb] resultado salvo em {out}")


if __name__ == "__main__":
    main()

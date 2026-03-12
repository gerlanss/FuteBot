import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from models.features import FeatureExtractor
from models.feature_factory import FeatureFactory


def _build_split(db_path: str, league_id: int, train_seasons: list[int],
                 val_season: int, test_season: int, feature_set: str):
    db = Database(db_path)
    extractor = FeatureFactory(db) if feature_set == "full" else FeatureExtractor(db)
    feat_names = (
        FeatureFactory.feature_names_full()
        if feature_set == "full"
        else FeatureExtractor.feature_names()
    )
    features, labels = extractor.build_dataset(
        league_id=league_id,
        seasons=train_seasons + [val_season, test_season],
    )

    buckets = {
        "train": {"x": [], "y": []},
        "val": {"x": [], "y": []},
        "test": {"x": [], "y": []},
    }
    for feat, label in zip(features, labels):
        row = [feat.get(name, 0) or 0 for name in feat_names]
        season = feat.get("_season")
        if season in train_seasons:
            buckets["train"]["x"].append(row)
            buckets["train"]["y"].append(label)
        elif season == val_season:
            buckets["val"]["x"].append(row)
            buckets["val"]["y"].append(label)
        elif season == test_season:
            buckets["test"]["x"].append(row)
            buckets["test"]["y"].append(label)

    payload = {}
    for key in ("train", "val", "test"):
        payload[key] = {
            "x": np.asarray(buckets[key]["x"], dtype=np.float32),
            "y": buckets[key]["y"],
        }
    return payload


def _fit_xgb_binary(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray):
    import xgboost as xgb

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "gamma": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "verbosity": 0,
    }
    model = xgb.train(
        params,
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=300,
        verbose_eval=False,
    )
    pred = model.predict(xgb.DMatrix(x_pred))
    return np.asarray(pred, dtype=np.float32)


def _fit_tabpfn_binary(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version = os.environ.get("TABPFN_VERSION", "v2").strip().lower()
    model_version = ModelVersion.V2_5 if version in {"v2.5", "2.5", "v25"} else ModelVersion.V2

    clf = TabPFNClassifier.create_default_for_version(
        model_version,
        device="cpu",
        fit_mode="low_memory",
        memory_saving_mode="auto",
        n_estimators=16,
        balance_probabilities=False,
        random_state=42,
        n_preprocessing_jobs=1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_pred)[:, 1]
    return np.asarray(pred, dtype=np.float32)


def _predict(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray):
    if model_name == "xgboost":
        return _fit_xgb_binary(x_train, y_train, x_pred)
    if model_name == "tabpfn":
        return _fit_tabpfn_binary(x_train, y_train, x_pred)
    raise ValueError(model_name)


def _search_strategies(p_yes: np.ndarray, y_true: np.ndarray, min_samples: int):
    positive_name = os.environ.get("SLICE_POSITIVE_NAME", "mercado_yes")
    negative_name = os.environ.get("SLICE_NEGATIVE_NAME", "mercado_no")
    candidates = []
    thresholds = np.arange(0.55, 0.981, 0.025)

    for thr in thresholds:
        idx = p_yes >= thr
        total = int(np.sum(idx))
        if total < min_samples:
            continue
        acc = float(np.mean(y_true[idx] == 1))
        candidates.append({
            "side": positive_name,
            "threshold": round(float(thr), 3),
            "samples": total,
            "accuracy": acc,
        })

    for thr in thresholds:
        idx = p_yes <= (1.0 - thr)
        total = int(np.sum(idx))
        if total < min_samples:
            continue
        acc = float(np.mean(y_true[idx] == 0))
        candidates.append({
            "side": negative_name,
            "threshold": round(float(thr), 3),
            "samples": total,
            "accuracy": acc,
        })

    candidates.sort(key=lambda item: (item["accuracy"], item["samples"], item["threshold"]), reverse=True)
    return candidates


def _apply_strategy(strategy: dict, p_yes: np.ndarray, y_true: np.ndarray):
    positive_name = os.environ.get("SLICE_POSITIVE_NAME", "mercado_yes")
    if strategy["side"] == positive_name:
        idx = p_yes >= strategy["threshold"]
        wins = y_true[idx] == 1
    else:
        idx = p_yes <= (1.0 - strategy["threshold"])
        wins = y_true[idx] == 0

    total = int(np.sum(idx))
    if total == 0:
        return {"samples": 0, "accuracy": None}
    return {
        "samples": total,
        "accuracy": float(np.mean(wins)),
    }


def main():
    parser = argparse.ArgumentParser(description="Busca slices binários de alta confiança para mercados fracos.")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "futebot.db"))
    parser.add_argument("--league-ids", default="71,239")
    parser.add_argument("--market", default="btts")
    parser.add_argument("--train-seasons", default="2024")
    parser.add_argument("--val-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--feature-sets", default="static,full")
    parser.add_argument("--models", default="xgboost,tabpfn")
    parser.add_argument("--min-val-samples", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "data" / "binary_slice_search.json"))
    args = parser.parse_args()

    league_ids = [int(x.strip()) for x in args.league_ids.split(",") if x.strip()]
    train_seasons = [int(x.strip()) for x in args.train_seasons.split(",") if x.strip()]
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": args.market,
        "train_seasons": train_seasons,
        "val_season": args.val_season,
        "test_season": args.test_season,
        "runs": [],
    }

    for league_id in league_ids:
        for feature_set in feature_sets:
            split = _build_split(
                args.db_path,
                league_id,
                train_seasons,
                args.val_season,
                args.test_season,
                feature_set,
            )
            x_train = split["train"]["x"]
            x_val = split["val"]["x"]
            x_test = split["test"]["x"]
            y_train = np.asarray([row[args.market] for row in split["train"]["y"]], dtype=np.int64)
            y_val = np.asarray([row[args.market] for row in split["val"]["y"]], dtype=np.int64)
            y_test = np.asarray([row[args.market] for row in split["test"]["y"]], dtype=np.int64)

            for model_name in models:
                print(f"[slice-search] league={league_id} feature_set={feature_set} model={model_name}")
                val_pred = _predict(model_name, x_train, y_train, x_val)
                test_pred = _predict(model_name, x_train, y_train, x_test)
                overall = {
                    "val_accuracy": float(accuracy_score(y_val, (val_pred >= 0.5).astype(int))),
                    "test_accuracy": float(accuracy_score(y_test, (test_pred >= 0.5).astype(int))),
                    "val_log_loss": float(log_loss(y_val, np.column_stack([1 - val_pred, val_pred]), labels=[0, 1])),
                    "test_log_loss": float(log_loss(y_test, np.column_stack([1 - test_pred, test_pred]), labels=[0, 1])),
                    "val_samples": int(len(y_val)),
                    "test_samples": int(len(y_test)),
                }
                candidates = _search_strategies(val_pred, y_val, min_samples=args.min_val_samples)
                top = []
                for candidate in candidates[:args.top_k]:
                    applied = _apply_strategy(candidate, test_pred, y_test)
                    top.append(candidate | {
                        "test_samples": applied["samples"],
                        "test_accuracy": applied["accuracy"],
                    })

                results["runs"].append({
                    "league_id": league_id,
                    "feature_set": feature_set,
                    "model": model_name,
                    "overall": overall,
                    "top_strategies": top,
                })

                if top:
                    best = top[0]
                    print(
                        f"  best={best['side']} thr={best['threshold']:.3f} "
                        f"| val={best['accuracy']:.3f} ({best['samples']}) "
                        f"| test={best['test_accuracy'] if best['test_accuracy'] is not None else 'n/a'} "
                        f"({best['test_samples']})"
                    )
                else:
                    print("  no strategy found with min sample requirement")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[slice-search] resultado salvo em {output_path}")


if __name__ == "__main__":
    main()

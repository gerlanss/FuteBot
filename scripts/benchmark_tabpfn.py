import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from models.features import FeatureExtractor

MARKETS = {
    "resultado": {"label_key": "resultado", "num_class": 3},
    "over15": {"label_key": "over15", "num_class": 2},
    "over35": {"label_key": "over35", "num_class": 2},
    "btts": {"label_key": "btts", "num_class": 2},
    "over05_ht": {"label_key": "over05_ht", "num_class": 2},
}


def _build_dataset(db_path: str, league_id: int, train_seasons: list[int], test_season: int):
    db = Database(db_path)
    fe = FeatureExtractor(db)
    features, labels = fe.build_dataset(league_id=league_id, seasons=train_seasons + [test_season])
    feat_names = FeatureExtractor.feature_names()

    x_train, x_test, y_train_raw, y_test_raw = [], [], [], []
    for feat, label in zip(features, labels):
        row = [feat.get(name, 0) or 0 for name in feat_names]
        season = feat.get("_season")
        if season in train_seasons:
            x_train.append(row)
            y_train_raw.append(label)
        elif season == test_season:
            x_test.append(row)
            y_test_raw.append(label)

    if not x_test:
        from sklearn.model_selection import train_test_split

        idx = list(range(len(x_train)))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
        all_x = x_train
        all_y = y_train_raw
        x_train = [all_x[i] for i in idx_train]
        x_test = [all_x[i] for i in idx_test]
        y_train_raw = [all_y[i] for i in idx_train]
        y_test_raw = [all_y[i] for i in idx_test]

    return (
        np.asarray(x_train, dtype=np.float32),
        np.asarray(x_test, dtype=np.float32),
        y_train_raw,
        y_test_raw,
        feat_names,
    )


def _metrics(y_true: np.ndarray, probs: np.ndarray, num_class: int) -> dict:
    if num_class == 2:
        pos = probs[:, 1] if probs.ndim == 2 else probs
        pred = (pos >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "log_loss": float(log_loss(y_true, np.column_stack([1 - pos, pos]), labels=[0, 1])),
            "brier": float(brier_score_loss(y_true, pos)),
        }

    pred = np.argmax(probs, axis=1)
    y_one_hot = np.eye(num_class, dtype=np.float32)[y_true]
    brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(num_class)))),
        "brier": brier,
    }


def _run_xgboost(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, num_class: int):
    import xgboost as xgb

    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
        "objective": "multi:softprob" if num_class > 2 else "binary:logistic",
        "eval_metric": "mlogloss" if num_class > 2 else "logloss",
    }
    if num_class > 2:
        params["num_class"] = num_class

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    raw = model.predict(dtest)
    if num_class == 2:
        raw = np.asarray(raw, dtype=np.float32)
        probs = np.column_stack([1 - raw, raw])
    else:
        probs = np.asarray(raw, dtype=np.float32)
    return probs


def _run_tabpfn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version = os.environ.get("TABPFN_VERSION", "v2").strip().lower()
    model_version = ModelVersion.V2_5 if version in {"v2.5", "2.5", "v25"} else ModelVersion.V2

    clf = TabPFNClassifier.create_default_for_version(
        model_version,
        device="cpu",
        fit_mode="low_memory",
        memory_saving_mode="auto",
        n_estimators=8,
        random_state=42,
        inference_precision="auto",
        n_preprocessing_jobs=1,
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)
    return np.asarray(probs, dtype=np.float32)


def _worker_main(args: argparse.Namespace):
    train_seasons = (
        _parse_int_list(args.train_seasons)
        if isinstance(args.train_seasons, str)
        else list(args.train_seasons)
    )
    x_train, x_test, y_train_raw, y_test_raw, _ = _build_dataset(
        args.db_path,
        args.league_id,
        train_seasons,
        args.test_season,
    )
    cfg = MARKETS[args.market]
    y_train = np.asarray([row[cfg["label_key"]] for row in y_train_raw], dtype=np.int64)
    y_test = np.asarray([row[cfg["label_key"]] for row in y_test_raw], dtype=np.int64)

    start = time.perf_counter()
    if args.model == "xgboost":
        probs = _run_xgboost(x_train, y_train, x_test, cfg["num_class"])
    else:
        probs = _run_tabpfn(x_train, y_train, x_test)
    elapsed = time.perf_counter() - start

    result = {
        "model": args.model,
        "league_id": args.league_id,
        "market": args.market,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "n_features": int(x_train.shape[1]),
        "seconds": round(elapsed, 3),
        **_metrics(y_test, probs, cfg["num_class"]),
    }
    print(json.dumps(result))


def _run_with_monitor(base_args: list[str], memory_limit_mb: int) -> dict:
    proc = subprocess.Popen(
        [sys.executable, __file__, "--worker", *base_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ps = psutil.Process(proc.pid)
    peak_rss = 0
    killed_for_memory = False

    while proc.poll() is None:
        try:
            rss = ps.memory_info().rss
            for child in ps.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak_rss = max(peak_rss, rss)
            if rss > memory_limit_mb * 1024 * 1024:
                killed_for_memory = True
                proc.kill()
                break
        except psutil.Error:
            pass
        time.sleep(0.2)

    stdout, stderr = proc.communicate()
    if killed_for_memory:
        return {
            "status": "memory_limit_exceeded",
            "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
            "stderr": stderr.strip(),
        }
    if proc.returncode != 0:
        return {
            "status": "error",
            "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
            "stderr": stderr.strip(),
            "stdout": stdout.strip(),
        }
    payload = json.loads(stdout.strip().splitlines()[-1])
    payload["status"] = "ok"
    payload["peak_rss_mb"] = round(peak_rss / (1024 * 1024), 1)
    return payload


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark XGBoost vs TabPFN em datasets do FuteBot.")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "futebot.db"))
    parser.add_argument("--league-ids", default="71,128,239")
    parser.add_argument("--markets", default="resultado,over15,over35,btts")
    parser.add_argument("--train-seasons", default="2024,2025")
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--memory-limit-mb", type=int, default=2500)
    parser.add_argument("--output", default=str(ROOT / "data" / "tabpfn_benchmark.json"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=["xgboost", "tabpfn"])
    parser.add_argument("--league-id", type=int)
    parser.add_argument("--market", choices=sorted(MARKETS))
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    league_ids = _parse_int_list(args.league_ids)
    train_seasons = _parse_int_list(args.train_seasons)
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory_limit_mb": args.memory_limit_mb,
        "train_seasons": train_seasons,
        "test_season": args.test_season,
        "runs": [],
    }

    for league_id in league_ids:
        for market in markets:
            print(f"[benchmark] league={league_id} market={market}")
            for model in ("xgboost", "tabpfn"):
                base_args = [
                    "--db-path", args.db_path,
                    "--league-id", str(league_id),
                    "--market", market,
                    "--model", model,
                    "--train-seasons", ",".join(map(str, train_seasons)),
                    "--test-season", str(args.test_season),
                ]
                result = _run_with_monitor(base_args, memory_limit_mb=args.memory_limit_mb)
                results["runs"].append(result | {"league_id": league_id, "market": market, "model": model})
                status = result.get("status")
                print(
                    f"  - {model}: {status} | acc={result.get('accuracy')} "
                    f"| logloss={result.get('log_loss')} | peak_rss={result.get('peak_rss_mb')} MB"
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[benchmark] resultado salvo em {output_path}")


if __name__ == "__main__":
    main()

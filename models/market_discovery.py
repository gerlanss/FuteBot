import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from config import LEAGUES
from data.database import Database
from models.features import FeatureExtractor

try:
    import optuna
except ImportError:  # pragma: no cover - optional at runtime
    optuna = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "discovery_runs"

KEY_FEATURES = [
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "home_goals_for_home",
    "home_goals_against_home",
    "away_goals_for_away",
    "away_goals_against_away",
    "home_btts_pct",
    "away_btts_pct",
    "home_over15_pct",
    "away_over15_pct",
    "home_failed_score_pct",
    "away_failed_score_pct",
    "home_clean_sheet_pct",
    "away_clean_sheet_pct",
    "home_xg_avg",
    "away_xg_avg",
    "home_shots_on_avg",
    "away_shots_on_avg",
    "model_prob",
]


@dataclass(frozen=True)
class MarketSpec:
    market_id: str
    label_key: str
    green_value: int
    title: str


MARKET_SPECS = [
    MarketSpec("h2h_home", "resultado", 0, "Vitória Casa"),
    MarketSpec("h2h_draw", "resultado", 1, "Empate"),
    MarketSpec("h2h_away", "resultado", 2, "Vitória Fora"),
    MarketSpec("over15", "over15", 1, "Over 1.5"),
    MarketSpec("under15", "over15", 0, "Under 1.5"),
    MarketSpec("over25", "over25", 1, "Over 2.5"),
    MarketSpec("under25", "over25", 0, "Under 2.5"),
    MarketSpec("over35", "over35", 1, "Over 3.5"),
    MarketSpec("under35", "over35", 0, "Under 3.5"),
    MarketSpec("btts_yes", "btts", 1, "BTTS Sim"),
    MarketSpec("btts_no", "btts", 0, "BTTS Não"),
    MarketSpec("ht_home", "resultado_ht", 0, "Vitória Casa HT"),
    MarketSpec("ht_draw", "resultado_ht", 1, "Empate HT"),
    MarketSpec("ht_away", "resultado_ht", 2, "Vitória Fora HT"),
    MarketSpec("over05_ht", "over05_ht", 1, "1T Over 0.5"),
    MarketSpec("under05_ht", "over05_ht", 0, "1T Under 0.5"),
    MarketSpec("over15_ht", "over15_ht", 1, "1T Over 1.5"),
    MarketSpec("under15_ht", "over15_ht", 0, "1T Under 1.5"),
    MarketSpec("over05_2t", "over05_2t", 1, "2T Over 0.5"),
    MarketSpec("under05_2t", "over05_2t", 0, "2T Under 0.5"),
    MarketSpec("over15_2t", "over15_2t", 1, "2T Over 1.5"),
    MarketSpec("under15_2t", "over15_2t", 0, "2T Under 1.5"),
    MarketSpec("corners_over_85", "corners_over_85", 1, "Escanteios Over 8.5"),
    MarketSpec("corners_under_85", "corners_over_85", 0, "Escanteios Under 8.5"),
    MarketSpec("corners_over_95", "corners_over_95", 1, "Escanteios Over 9.5"),
    MarketSpec("corners_under_95", "corners_over_95", 0, "Escanteios Under 9.5"),
    MarketSpec("corners_over_105", "corners_over_105", 1, "Escanteios Over 10.5"),
    MarketSpec("corners_under_105", "corners_over_105", 0, "Escanteios Under 10.5"),
]
MARKET_SPEC_MAP = {item.market_id: item for item in MARKET_SPECS}


class MarketDiscoveryTrainer:
    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.extractor = FeatureExtractor(self.db)
        self.feature_names = FeatureExtractor.feature_names()

    def run(
        self,
        league_ids: list[int] | None = None,
        markets: list[str] | None = None,
        target_precision: float = 0.70,
        min_train_samples: int = 30,
        min_test_samples: int = 10,
        optuna_trials: int = 20,
        output_dir: str | os.PathLike | None = None,
        progress_callback=None,
    ) -> dict:
        started = time.time()
        run_id = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_markets = self._resolve_markets(markets)
        selected_leagues = self._resolve_leagues(league_ids)
        run_summary = {
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_precision": target_precision,
            "min_train_samples": min_train_samples,
            "min_test_samples": min_test_samples,
            "optuna_trials": optuna_trials,
            "markets": [spec.market_id for spec in selected_markets],
            "leagues": [],
        }

        for idx, (league_id, league_name) in enumerate(selected_leagues, 1):
            league_started = time.time()
            league_data = self._load_league_data(league_id)
            if "error" in league_data:
                league_result = {
                    "league_id": league_id,
                    "league_name": league_name,
                    "status": "error",
                    "error": league_data["error"],
                }
                run_summary["leagues"].append(league_result)
                self._write_json(out_dir / f"league_{league_id}.json", league_result)
                self._emit(progress_callback, {
                    "type": "league",
                    "league_id": league_id,
                    "league_name": league_name,
                    "summary": league_result,
                })
                continue

            league_result = {
                "league_id": league_id,
                "league_name": league_name,
                "train_seasons": league_data["train_seasons"],
                "test_season": league_data["test_season"],
                "rows": len(league_data["rows"]),
                "markets": [],
            }

            for market_spec in selected_markets:
                result = self._run_market(
                    league_data=league_data,
                    market_spec=market_spec,
                    target_precision=target_precision,
                    min_train_samples=min_train_samples,
                    min_test_samples=min_test_samples,
                optuna_trials=optuna_trials,
                )
                league_result["markets"].append(result)
                self._emit(progress_callback, {
                    "type": "market",
                    "league_id": league_id,
                    "league_name": league_name,
                    "summary": result,
                })
                self._write_json(out_dir / f"league_{league_id}_{market_spec.market_id}.json", result)

            league_result["duration_seconds"] = round(time.time() - league_started, 1)
            league_result["accepted_markets"] = sum(
                1 for item in league_result["markets"] if item.get("status") == "accepted"
            )
            run_summary["leagues"].append(league_result)
            self._write_json(out_dir / f"league_{league_id}.json", league_result)
            self._write_text(out_dir / f"league_{league_id}.md", self._format_league_markdown(idx, len(selected_leagues), league_result))
            self._emit(progress_callback, {
                "type": "league",
                "league_id": league_id,
                "league_name": league_name,
                "summary": league_result,
            })

        run_summary["duration_seconds"] = round(time.time() - started, 1)
        run_summary["accepted_markets"] = sum(
            league.get("accepted_markets", 0) for league in run_summary["leagues"]
        )
        self._write_json(out_dir / "summary.json", run_summary)
        self._write_text(out_dir / "summary.md", self._format_run_markdown(run_summary))
        return run_summary

    def _resolve_markets(self, markets: list[str] | None) -> list[MarketSpec]:
        if not markets:
            return list(MARKET_SPECS)
        return [MARKET_SPEC_MAP[item] for item in markets if item in MARKET_SPEC_MAP]

    def _resolve_leagues(self, league_ids: list[int] | None) -> list[tuple[int, str]]:
        all_leagues = sorted(
            ((info["id"], info["nome"]) for info in LEAGUES.values()),
            key=lambda item: item[1],
        )
        if not league_ids:
            return all_leagues
        wanted = set(league_ids)
        return [item for item in all_leagues if item[0] in wanted]

    def _load_league_data(self, league_id: int) -> dict:
        fixtures = self.db.fixtures_finalizados(league_id=league_id)
        seasons = sorted({int(item["season"]) for item in fixtures if item.get("season")})
        if len(seasons) < 2:
            return {"error": "liga sem seasons suficientes para split temporal"}

        test_season = seasons[-1]
        train_seasons = seasons[:-1]
        features, labels = self.extractor.build_dataset(
            league_id=league_id,
            seasons=train_seasons + [test_season],
        )
        if len(features) < 50:
            return {"error": f"dados insuficientes ({len(features)} jogos com features)"}

        rows = []
        matrix = []
        for feat, label in zip(features, labels):
            row = {name: float(feat.get(name, 0) or 0) for name in self.feature_names}
            row["_season"] = int(feat.get("_season") or 0)
            row["_fixture_id"] = int(label["fixture_id"])
            row["_labels"] = label
            rows.append(row)
            matrix.append([row[name] for name in self.feature_names])

        return {
            "league_id": league_id,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "rows": rows,
            "matrix": np.asarray(matrix, dtype=np.float32),
        }

    def _run_market(
        self,
        league_data: dict,
        market_spec: MarketSpec,
        target_precision: float,
        min_train_samples: int,
        min_test_samples: int,
        optuna_trials: int,
    ) -> dict:
        started = time.time()
        rows = league_data["rows"]
        X = league_data["matrix"]
        train_mask = np.asarray([row["_season"] in league_data["train_seasons"] for row in rows], dtype=bool)
        test_mask = np.asarray([row["_season"] == league_data["test_season"] for row in rows], dtype=bool)
        y = np.asarray(
            [1 if row["_labels"][market_spec.label_key] == market_spec.green_value else 0 for row in rows],
            dtype=np.int64,
        )

        if len(np.unique(y[train_mask])) < 2:
            return self._market_result_stub(market_spec, "sem variância suficiente no treino", started, rows, train_mask, test_mask, y)

        base_params = self._default_xgb_params(y[train_mask])
        train_probs, test_probs = self._fit_predict(base_params, X[train_mask], y[train_mask], X[train_mask], X[test_mask])
        baseline = self._search_best_rule(
            rows, y, train_mask, test_mask, train_probs, test_probs, min_train_samples, min_test_samples
        )

        best = baseline
        optuna_used = False
        optuna_best_params = None

        if self._should_try_optuna(baseline, target_precision, optuna_trials):
            tuned_params = self._optimize_params(
                X[train_mask], y[train_mask], optuna_trials
            )
            if tuned_params:
                optuna_used = True
                optuna_best_params = tuned_params
                tuned_train_probs, tuned_test_probs = self._fit_predict(
                    {**base_params, **tuned_params},
                    X[train_mask], y[train_mask], X[train_mask], X[test_mask]
                )
                tuned = self._search_best_rule(
                    rows, y, train_mask, test_mask, tuned_train_probs, tuned_test_probs,
                    min_train_samples, min_test_samples
                )
                if self._rule_score(tuned) > self._rule_score(best):
                    best = tuned

        accepted = bool(
            best
            and best["train"]["precision"] >= target_precision
            and best["test"]["precision"] is not None
            and best["test"]["precision"] >= target_precision
        )
        train_stats = self._mask_stats(y[train_mask], np.ones(int(train_mask.sum()), dtype=bool))
        test_stats = self._mask_stats(y[test_mask], np.ones(int(test_mask.sum()), dtype=bool))
        return {
            "market": market_spec.market_id,
            "title": market_spec.title,
            "label_key": market_spec.label_key,
            "green_value": market_spec.green_value,
            "status": "accepted" if accepted else "needs_work",
            "train_base_rate": train_stats["precision"],
            "train_base_samples": train_stats["samples"],
            "test_base_rate": test_stats["precision"],
            "test_base_samples": test_stats["samples"],
            "best_rule": best,
            "optuna_used": optuna_used,
            "optuna_best_params": optuna_best_params,
            "duration_seconds": round(time.time() - started, 1),
        }

    def _market_result_stub(self, market_spec, reason, started, rows, train_mask, test_mask, y):
        return {
            "market": market_spec.market_id,
            "title": market_spec.title,
            "status": "skipped",
            "reason": reason,
            "train_base_rate": self._mask_stats(y[train_mask], np.ones(int(train_mask.sum()), dtype=bool))["precision"],
            "train_base_samples": int(train_mask.sum()),
            "test_base_rate": self._mask_stats(y[test_mask], np.ones(int(test_mask.sum()), dtype=bool))["precision"],
            "test_base_samples": int(test_mask.sum()),
            "best_rule": None,
            "duration_seconds": round(time.time() - started, 1),
        }

    def _default_xgb_params(self, y_train: np.ndarray) -> dict:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": 42,
            "verbosity": 0,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "nthread": os.cpu_count() or 4,
        }
        if pos > 0 and neg > 0:
            params["scale_pos_weight"] = neg / pos
        return params

    def _fit_predict(self, params: dict, X_train: np.ndarray, y_train: np.ndarray,
                     X_pred_train: np.ndarray, X_pred_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dpred_train = xgb.DMatrix(X_pred_train, feature_names=self.feature_names)
        dpred_test = xgb.DMatrix(X_pred_test, feature_names=self.feature_names)
        model = xgb.train(params, dtrain, num_boost_round=160, verbose_eval=False)
        train_probs = np.asarray(model.predict(dpred_train), dtype=np.float32)
        test_probs = np.asarray(model.predict(dpred_test), dtype=np.float32)
        return train_probs, test_probs

    def _should_try_optuna(self, baseline: dict | None, target_precision: float, optuna_trials: int) -> bool:
        if optuna_trials <= 0 or optuna is None or not baseline:
            return False
        test_precision = baseline["test"]["precision"]
        if test_precision is None:
            return False
        return baseline["train"]["precision"] >= target_precision and test_precision < target_precision

    def _optimize_params(self, X_train: np.ndarray, y_train: np.ndarray,
                         trials: int) -> dict | None:
        if optuna is None:
            return None
        if len(X_train) < 40:
            return None

        split_idx = max(20, int(len(X_train) * 0.8))
        if split_idx >= len(X_train):
            return None

        X_subtrain = X_train[:split_idx]
        y_subtrain = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        if len(np.unique(y_subtrain)) < 2 or len(np.unique(y_val)) < 2:
            return None

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            _, probs_val = self._fit_predict(
                {**self._default_xgb_params(y_subtrain), **params},
                X_subtrain,
                y_subtrain,
                X_subtrain,
                X_val,
            )
            eps = np.clip(probs_val, 1e-7, 1 - 1e-7)
            return float(-np.mean(y_val * np.log(eps) + (1 - y_val) * np.log(1 - eps)))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
        )
        try:
            study.optimize(objective, n_trials=trials, show_progress_bar=False)
        except Exception:
            return None
        return study.best_params if study.best_trials else None

    def _search_best_rule(self, rows, y, train_mask, test_mask, train_probs, test_probs,
                          min_train_samples: int, min_test_samples: int) -> dict | None:
        train_rows = []
        test_rows = []
        train_y = y[train_mask]
        test_y = y[test_mask]
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        for pos, idx in enumerate(train_indices):
            row = {key: rows[idx][key] for key in rows[idx] if not key.startswith("_")}
            row["model_prob"] = float(train_probs[pos])
            train_rows.append(row)
        for pos, idx in enumerate(test_indices):
            row = {key: rows[idx][key] for key in rows[idx] if not key.startswith("_")}
            row["model_prob"] = float(test_probs[pos])
            test_rows.append(row)

        candidates = []
        candidates.extend(self._discover_rules(train_rows, train_y, min_train_samples, pair=False))
        candidates.extend(self._discover_rules(train_rows, train_y, min_train_samples, pair=True))
        if not candidates:
            return None

        best = None
        for candidate in candidates:
            train_eval = self._evaluate_rule(train_rows, train_y, candidate["conditions"])
            test_eval = self._evaluate_rule(test_rows, test_y, candidate["conditions"])
            if test_eval["samples"] < min_test_samples:
                continue
            item = {
                "rule": candidate["rule"],
                "conditions": candidate["conditions"],
                "train": train_eval,
                "test": test_eval,
            }
            if self._rule_score(item) > self._rule_score(best):
                best = item
        return best

    def _discover_rules(self, train_rows, train_y, min_train_samples: int, pair: bool) -> list[dict]:
        green_rows = [row for row, label in zip(train_rows, train_y) if label == 1]
        if not green_rows:
            return []
        features = KEY_FEATURES if not pair else KEY_FEATURES[:12] + ["model_prob"]
        prepared = {}
        for feature in features:
            values = np.asarray([row[feature] for row in green_rows], dtype=np.float32)
            thresholds = sorted(set(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)))
            prepared[feature] = []
            for threshold in thresholds:
                prepared[feature].append((feature, ">=", threshold))
                prepared[feature].append((feature, "<=", threshold))

        rules = []
        if pair:
            for i, feature_a in enumerate(prepared.keys()):
                for feature_b in list(prepared.keys())[i + 1:]:
                    for cond_a in prepared[feature_a]:
                        for cond_b in prepared[feature_b]:
                            conditions = [cond_a, cond_b]
                            stats = self._evaluate_rule(train_rows, train_y, conditions)
                            if stats["samples"] < min_train_samples:
                                continue
                            rules.append({
                                "rule": f"{self._condition_text(*cond_a)} AND {self._condition_text(*cond_b)}",
                                "conditions": conditions,
                                "precision": stats["precision"],
                                "samples": stats["samples"],
                            })
        else:
            for feature, conditions in prepared.items():
                for condition in conditions:
                    stats = self._evaluate_rule(train_rows, train_y, [condition])
                    if stats["samples"] < min_train_samples:
                        continue
                    rules.append({
                        "rule": self._condition_text(*condition),
                        "conditions": [condition],
                        "precision": stats["precision"],
                        "samples": stats["samples"],
                    })
        rules.sort(key=lambda item: (item["precision"], item["samples"]), reverse=True)
        return rules[:50]

    def _evaluate_rule(self, rows, y_true: np.ndarray, conditions: list[tuple[str, str, float]]) -> dict:
        mask = np.ones(len(rows), dtype=bool)
        for feature, op, threshold in conditions:
            values = np.asarray([row[feature] for row in rows], dtype=np.float32)
            if op == ">=":
                mask &= values >= threshold
            else:
                mask &= values <= threshold
        return self._mask_stats(y_true, mask)

    def _mask_stats(self, y_true: np.ndarray, mask: np.ndarray) -> dict:
        samples = int(mask.sum())
        if samples == 0:
            return {"samples": 0, "wins": 0, "losses": 0, "precision": None}
        wins = int(np.sum(y_true[mask] == 1))
        return {
            "samples": samples,
            "wins": wins,
            "losses": samples - wins,
            "precision": wins / samples,
        }

    def _rule_score(self, item: dict | None) -> tuple:
        if not item:
            return (-1.0, -1, -1.0, -1)
        test_precision = item["test"]["precision"] if item["test"]["precision"] is not None else -1.0
        return (
            test_precision,
            item["test"]["samples"],
            item["train"]["precision"],
            item["train"]["samples"],
        )

    def _condition_text(self, feature: str, op: str, threshold: float) -> str:
        return f"{feature} {op} {threshold:.3f}"

    def _format_league_markdown(self, idx: int, total: int, league_result: dict) -> str:
        lines = [
            f"# Liga {idx}/{total} - {league_result['league_name']} ({league_result['league_id']})",
            f"- Seasons treino: {', '.join(str(x) for x in league_result.get('train_seasons', []))}",
            f"- Season teste: {league_result.get('test_season')}",
            f"- Jogos com features: {league_result.get('rows', 0)}",
            f"- Mercados aceitos: {league_result.get('accepted_markets', 0)}/{len(league_result.get('markets', []))}",
            "",
        ]
        for item in league_result.get("markets", []):
            best = item.get("best_rule")
            if best:
                lines.extend([
                    f"## {item['title']} [{item['status']}]",
                    f"- Base treino: {item['train_base_rate']*100:.1f}% ({item['train_base_samples']})",
                    f"- Base teste: {item['test_base_rate']*100:.1f}% ({item['test_base_samples']})",
                    f"- Melhor regra: {best['rule']}",
                    f"- Train: {best['train']['precision']*100:.1f}% ({best['train']['wins']}/{best['train']['samples']})",
                    f"- Test: {best['test']['precision']*100:.1f}% ({best['test']['wins']}/{best['test']['samples']})",
                    f"- Optuna: {'sim' if item.get('optuna_used') else 'não'}",
                    "",
                ])
            else:
                lines.extend([
                    f"## {item['title']} [{item['status']}]",
                    f"- Base treino: {item['train_base_rate']*100:.1f}% ({item['train_base_samples']})",
                    f"- Base teste: {item['test_base_rate']*100:.1f}% ({item['test_base_samples']})",
                    f"- Sem regra robusta encontrada",
                    "",
                ])
        return "\n".join(lines).strip() + "\n"

    def _format_run_markdown(self, run_summary: dict) -> str:
        lines = [
            f"# Treino por Descoberta - {run_summary['run_id']}",
            f"- Mercados aceitos: {run_summary['accepted_markets']}",
            f"- Duração: {run_summary['duration_seconds']/60:.1f} min",
            "",
        ]
        for league in run_summary["leagues"]:
            if league.get("status") == "error":
                lines.append(f"- {league['league_name']} ({league['league_id']}): erro - {league['error']}")
            else:
                lines.append(
                    f"- {league['league_name']} ({league['league_id']}): "
                    f"{league['accepted_markets']}/{len(league['markets'])} mercados aceitos"
                )
        return "\n".join(lines).strip() + "\n"

    def _write_json(self, path: Path, payload: dict):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_text(self, path: Path, text: str):
        path.write_text(text, encoding="utf-8")

    def _emit(self, callback, payload: dict):
        if callback:
            callback(payload)

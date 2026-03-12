import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from models.features import FeatureExtractor


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
]

MARKETS = {
    "over15": {"label_key": "over15", "green_value": 1, "green_name": "Over 1.5"},
    "under35": {"label_key": "over35", "green_value": 0, "green_name": "Under 3.5"},
}


def _load_rows(db_path: str, league_id: int, seasons: list[int]):
    db = Database(db_path)
    fe = FeatureExtractor(db)
    feats, labels = fe.build_dataset(league_id=league_id, seasons=seasons)
    rows = []
    for feat, label in zip(feats, labels):
        row = {name: float(feat.get(name, 0) or 0) for name in FeatureExtractor.feature_names()}
        row["fixture_id"] = label["fixture_id"]
        row["season"] = feat.get("_season")
        row["over15"] = int(label["over15"])
        row["over35"] = int(label["over35"])
        rows.append(row)
    return rows


def _condition_text(feature: str, op: str, threshold: float):
    return f"{feature} {op} {threshold:.3f}"


def _mask_for_rule(rows: list[dict], conditions: list[tuple[str, str, float]]):
    mask = np.ones(len(rows), dtype=bool)
    for feature, op, threshold in conditions:
        values = np.asarray([r[feature] for r in rows], dtype=np.float32)
        if op == ">=":
            mask &= values >= threshold
        else:
            mask &= values <= threshold
    return mask


def _rule_stats(rows: list[dict], mask: np.ndarray, market_key: str):
    cfg = MARKETS[market_key]
    y = np.asarray([r[cfg["label_key"]] for r in rows], dtype=np.int64)
    selected = int(mask.sum())
    if selected == 0:
        return None
    greens = int(np.sum(y[mask] == cfg["green_value"]))
    reds = selected - greens
    precision = greens / selected
    support = selected / len(rows)
    return {
        "samples": selected,
        "greens": greens,
        "reds": reds,
        "precision": precision,
        "support": support,
    }


def _discover_single_rules(train_rows: list[dict], market_key: str, min_samples: int):
    cfg = MARKETS[market_key]
    green_rows = [r for r in train_rows if r[cfg["label_key"]] == cfg["green_value"]]
    rules = []
    for feature in KEY_FEATURES:
        values = np.asarray([r[feature] for r in green_rows], dtype=np.float32)
        quantiles = sorted(set(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)))
        for threshold in quantiles:
            for op in (">=", "<="):
                conditions = [(feature, op, threshold)]
                mask_train = _mask_for_rule(train_rows, conditions)
                stats = _rule_stats(train_rows, mask_train, market_key)
                if not stats or stats["samples"] < min_samples:
                    continue
                rules.append({
                    "rule": _condition_text(feature, op, threshold),
                    "conditions": conditions,
                    **stats,
                })
    rules.sort(key=lambda item: (item["precision"], item["samples"]), reverse=True)
    return rules


def _discover_pair_rules(train_rows: list[dict], market_key: str, min_samples: int):
    cfg = MARKETS[market_key]
    green_rows = [r for r in train_rows if r[cfg["label_key"]] == cfg["green_value"]]
    prepared = {}
    for feature in KEY_FEATURES[:12]:
        values = np.asarray([r[feature] for r in green_rows], dtype=np.float32)
        quantiles = sorted(set(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)))
        conds = []
        for threshold in quantiles:
            conds.append((feature, ">=", threshold))
            conds.append((feature, "<=", threshold))
        prepared[feature] = conds

    rules = []
    for fa, fb in itertools.combinations(prepared.keys(), 2):
        for ca in prepared[fa]:
            for cb in prepared[fb]:
                conditions = [ca, cb]
                mask_train = _mask_for_rule(train_rows, conditions)
                stats = _rule_stats(train_rows, mask_train, market_key)
                if not stats or stats["samples"] < min_samples:
                    continue
                rules.append({
                    "rule": f"{_condition_text(*ca)} AND {_condition_text(*cb)}",
                    "conditions": conditions,
                    **stats,
                })
    rules.sort(key=lambda item: (item["precision"], item["samples"]), reverse=True)
    return rules


def _fmt_rule(item: dict, test_stats: dict):
    return (
        f"- Train {item['precision']*100:.1f}% ({item['greens']}/{item['samples']})"
        f" | Test {test_stats['precision']*100:.1f}% ({test_stats['greens']}/{test_stats['samples']})\n"
        f"  {item['rule']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Descobre coincidencias entre greens de um mercado.")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "futebot.db"))
    parser.add_argument("--league-id", type=int, default=128)
    parser.add_argument("--markets", default="under35,over15")
    parser.add_argument("--train-seasons", default="2024,2025")
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--min-single-samples", type=int, default=30)
    parser.add_argument("--min-pair-samples", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default=str(ROOT / "data" / "market_coincidences_128.md"))
    args = parser.parse_args()

    train_seasons = [int(x.strip()) for x in args.train_seasons.split(",") if x.strip()]
    markets = [x.strip() for x in args.markets.split(",") if x.strip()]
    rows = _load_rows(args.db_path, args.league_id, train_seasons + [args.test_season])
    train_rows = [r for r in rows if r["season"] in train_seasons]
    test_rows = [r for r in rows if r["season"] == args.test_season]

    blocks = [
        f"# Coincidencias de Greens - Liga {args.league_id}",
        f"- Amostra treino: {len(train_rows)} jogos",
        f"- Amostra teste: {len(test_rows)} jogos",
        "",
    ]

    for market_key in markets:
        cfg = MARKETS[market_key]
        base_train = _rule_stats(train_rows, np.ones(len(train_rows), dtype=bool), market_key)
        base_test = _rule_stats(test_rows, np.ones(len(test_rows), dtype=bool), market_key)
        singles = _discover_single_rules(train_rows, market_key, args.min_single_samples)
        pairs = _discover_pair_rules(train_rows, market_key, args.min_pair_samples)

        blocks.extend([
            f"## {cfg['green_name']}",
            f"- Base treino: {base_train['precision']*100:.1f}% ({base_train['greens']}/{base_train['samples']})",
            f"- Base teste: {base_test['precision']*100:.1f}% ({base_test['greens']}/{base_test['samples']})",
            "",
            "### Regras simples encontradas nos greens",
        ])

        for item in singles[:args.top_k]:
            test_mask = _mask_for_rule(test_rows, item["conditions"])
            test_stats = _rule_stats(test_rows, test_mask, market_key)
            if not test_stats or test_stats["samples"] == 0:
                continue
            blocks.append(_fmt_rule(item, test_stats))

        blocks.extend(["", "### Regras duplas encontradas nos greens"])
        for item in pairs[:args.top_k]:
            test_mask = _mask_for_rule(test_rows, item["conditions"])
            test_stats = _rule_stats(test_rows, test_mask, market_key)
            if not test_stats or test_stats["samples"] == 0:
                continue
            blocks.append(_fmt_rule(item, test_stats))

        blocks.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"relatorio salvo em {output_path}")


if __name__ == "__main__":
    main()

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

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


def _load_rows(db_path: str, league_id: int, seasons: list[int] | None):
    db = Database(db_path)
    fe = FeatureExtractor(db)
    feats, labels = fe.build_dataset(league_id=league_id, seasons=seasons)
    rows = []
    for f, l in zip(feats, labels):
        row = {name: float(f.get(name, 0) or 0) for name in FeatureExtractor.feature_names()}
        row["btts"] = int(l["btts"])
        row["fixture_id"] = l["fixture_id"]
        row["season"] = f.get("_season")
        rows.append(row)
    return rows


def _rule_metrics(mask: np.ndarray, y: np.ndarray, target: int, base_rate: float):
    total = int(mask.sum())
    if total == 0:
        return None
    precision = float(np.mean(y[mask] == target))
    support = float(total / len(y))
    if target == 1:
        lift = precision - base_rate
    else:
        lift = precision - (1.0 - base_rate)
    return {
        "samples": total,
        "precision": precision,
        "support": support,
        "lift": lift,
    }


def _condition_text(feature: str, op: str, threshold: float):
    return f"{feature} {op} {threshold:.3f}"


def _single_rules(rows: list[dict], min_samples: int):
    y = np.asarray([r["btts"] for r in rows], dtype=np.int64)
    base_yes = float(np.mean(y == 1))
    rules = []
    for feat in KEY_FEATURES:
        values = np.asarray([r[feat] for r in rows], dtype=np.float32)
        quantiles = sorted(set(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)))
        for thr in quantiles:
            masks = [
                (values >= thr, ">="),
                (values <= thr, "<="),
            ]
            for mask, op in masks:
                for target, label in ((1, "BTTS sim"), (0, "BTTS nao")):
                    metrics = _rule_metrics(mask, y, target, base_yes)
                    if not metrics or metrics["samples"] < min_samples:
                        continue
                    rules.append({
                        "target": label,
                        "rule": _condition_text(feat, op, thr),
                        **metrics,
                    })
    rules.sort(key=lambda r: (r["precision"], r["lift"], r["samples"]), reverse=True)
    return base_yes, rules


def _pair_rules(rows: list[dict], min_samples: int):
    y = np.asarray([r["btts"] for r in rows], dtype=np.int64)
    base_yes = float(np.mean(y == 1))
    rules = []
    feats = KEY_FEATURES[:12]
    prepared = {}
    for feat in feats:
        values = np.asarray([r[feat] for r in rows], dtype=np.float32)
        quantiles = sorted(set(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)))
        conds = []
        for thr in quantiles:
            conds.append((values >= thr, _condition_text(feat, ">=", thr)))
            conds.append((values <= thr, _condition_text(feat, "<=", thr)))
        prepared[feat] = conds

    for feat_a, feat_b in itertools.combinations(feats, 2):
        for mask_a, text_a in prepared[feat_a]:
            for mask_b, text_b in prepared[feat_b]:
                combo = mask_a & mask_b
                for target, label in ((1, "BTTS sim"), (0, "BTTS nao")):
                    metrics = _rule_metrics(combo, y, target, base_yes)
                    if not metrics or metrics["samples"] < min_samples:
                        continue
                    rules.append({
                        "target": label,
                        "rule": f"{text_a} AND {text_b}",
                        **metrics,
                    })
    rules.sort(key=lambda r: (r["precision"], r["lift"], r["samples"]), reverse=True)
    return rules


def _tree_rules(rows: list[dict], max_depth: int, min_samples_leaf: int):
    feature_names = KEY_FEATURES
    X = np.asarray([[r[f] for f in feature_names] for r in rows], dtype=np.float32)
    y = np.asarray([r["btts"] for r in rows], dtype=np.int64)
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X, y)
    tree = clf.tree_
    out = []

    def walk(node: int, clauses: list[str]):
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            probs = tree.value[node][0]
            total = int(tree.n_node_samples[node])
            yes = int(round(probs[1] * total)) if len(probs) > 1 else 0
            no = int(round(probs[0] * total)) if len(probs) > 0 else 0
            pred = "BTTS sim" if yes >= no else "BTTS nao"
            precision = max(yes, no) / total if total else 0
            out.append({
                "rule": " AND ".join(clauses) if clauses else "ROOT",
                "samples": total,
                "pred": pred,
                "precision": precision,
                "yes": yes,
                "no": no,
            })
            return

        feat = feature_names[tree.feature[node]]
        thr = float(tree.threshold[node])
        walk(tree.children_left[node], clauses + [f"{feat} <= {thr:.3f}"])
        walk(tree.children_right[node], clauses + [f"{feat} > {thr:.3f}"])

    walk(0, [])
    out.sort(key=lambda r: (r["precision"], r["samples"]), reverse=True)
    return out


def _fmt_rule(rule: dict):
    return (
        f"- {rule['target']} | {rule['precision']*100:.1f}% | "
        f"{rule['samples']} jogos | lift {rule['lift']*100:+.1f}pp\n"
        f"  {rule['rule']}"
    )


def _fmt_leaf(leaf: dict):
    return (
        f"- {leaf['pred']} | {leaf['precision']*100:.1f}% | "
        f"{leaf['samples']} jogos | sim={leaf['yes']} nao={leaf['no']}\n"
        f"  {leaf['rule']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Analise interpretavel de padroes BTTS por liga.")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "futebot.db"))
    parser.add_argument("--league-ids", default="71,128,239")
    parser.add_argument("--seasons", default="2024,2025,2026")
    parser.add_argument("--min-single-samples", type=int, default=25)
    parser.add_argument("--min-pair-samples", type=int, default=20)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--tree-min-leaf", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default=str(ROOT / "data" / "btts_pattern_report.md"))
    args = parser.parse_args()

    league_ids = [int(x.strip()) for x in args.league_ids.split(",") if x.strip()]
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]

    blocks = ["# Analise Interpretavel de BTTS", ""]
    for league_id in league_ids:
        rows = _load_rows(args.db_path, league_id, seasons)
        if not rows:
            continue
        base_yes, single_rules = _single_rules(rows, args.min_single_samples)
        pair_rules = _pair_rules(rows, args.min_pair_samples)
        tree_rules = _tree_rules(rows, args.tree_depth, args.tree_min_leaf)

        blocks.extend([
            f"## Liga {league_id}",
            f"- Amostra: {len(rows)} jogos",
            f"- Taxa base BTTS sim: {base_yes*100:.1f}%",
            f"- Taxa base BTTS nao: {(1-base_yes)*100:.1f}%",
            "",
            "### Melhores regras simples",
        ])
        for rule in single_rules[:args.top_k]:
            blocks.append(_fmt_rule(rule))
        blocks.extend(["", "### Melhores regras duplas"])
        for rule in pair_rules[:args.top_k]:
            blocks.append(_fmt_rule(rule))
        blocks.extend(["", "### Folhas da arvore rasa"])
        for leaf in tree_rules[:args.top_k]:
            blocks.append(_fmt_leaf(leaf))
        blocks.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"relatorio salvo em {output_path}")


if __name__ == "__main__":
    main()

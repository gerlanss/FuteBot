from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from config import LEAGUES, TRAIN_SEASONS
from data.database import Database
from models.features import FeatureExtractor
from services.live_market_windows import dentro_janela_operacional_live, janela_operacional_live

try:
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, log_loss

    HAS_ML = True
except ImportError:
    HAS_ML = False


MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "live"
HISTORICAL_LIVE_MARKETS = [
    "h2h_home",
    "h2h_draw",
    "h2h_away",
    "ht_home",
    "ht_draw",
    "ht_away",
    "over15",
    "under15",
    "over25",
    "under25",
    "over35",
    "under35",
    "over05_ht",
    "under05_ht",
    "over15_ht",
    "under15_ht",
    "over05_2t",
    "under05_2t",
    "over15_2t",
    "under15_2t",
]
CORNER_SNAPSHOT_MARKETS = [
    "corners_over_85",
    "corners_under_85",
    "corners_over_95",
    "corners_under_95",
    "corners_over_105",
    "corners_under_105",
]
LIVE_MARKETS = HISTORICAL_LIVE_MARKETS + CORNER_SNAPSHOT_MARKETS


@dataclass
class LiveSample:
    fixture_id: int
    fixture_date: str
    league_id: int
    market: str
    signal_minute: int
    label: int
    features: dict[str, float]
    source: str


class LiveTrainer:
    """Treina o live por liga x mercado, sem fallback global."""

    MIN_FIXTURES_WITH_EVENTS = 25
    MIN_SAMPLES_TOTAL = 120
    MIN_TEST_SAMPLES = 8
    MIN_CORNER_SNAPSHOT_SAMPLES = 5

    def __init__(self, db: Database):
        self.db = db
        self.fe = FeatureExtractor(db)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._feature_cache: dict[int, dict[str, float] | None] = {}

    def readiness(self) -> dict:
        conn = self.db._conn()
        placeholders = ",".join("?" for _ in TRAIN_SEASONS)
        league_ids = [info["id"] for info in LEAGUES.values()]
        league_placeholders = ",".join("?" for _ in league_ids)
        params = tuple(TRAIN_SEASONS) + tuple(league_ids)
        total_ft = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM fixtures
            WHERE status='FT'
              AND season IN ({placeholders})
              AND league_id IN ({league_placeholders})
            """,
            params,
        ).fetchone()["n"]
        total_event_fixtures = conn.execute(
            f"""
            SELECT COUNT(DISTINCT f.fixture_id) AS n
            FROM fixtures f
            JOIN fixture_events fe ON fe.fixture_id = f.fixture_id
            WHERE f.status='FT'
              AND f.season IN ({placeholders})
              AND f.league_id IN ({league_placeholders})
            """,
            params,
        ).fetchone()["n"]
        corners_snapshots = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM live_watchlist lw
            JOIN live_results lr ON lr.live_watch_id = lw.id
            WHERE lr.acertou IS NOT NULL
              AND lw.mercado LIKE 'corners_%'
              AND (
                lr.snapshot_json IS NOT NULL
                OR lr.snapshot_json LIKE '%elapsed%'
                OR lw.payload_json LIKE '%training_snapshot%'
                OR lw.payload_json LIKE '%metricas_live%'
                OR lw.payload_json LIKE '%live_reading%'
              )
            """
        ).fetchone()["n"]
        conn.close()

        pronto = (
            total_event_fixtures >= self.MIN_FIXTURES_WITH_EVENTS
            and total_ft >= self.MIN_FIXTURES_WITH_EVENTS
        )
        return {
            "pronto": pronto,
            "layout": "league_market_only",
            "source_historical": "api_historical",
            "source_corners": "bot_snapshots",
            "fixtures_ft_total": int(total_ft or 0),
            "fixtures_com_eventos": int(total_event_fixtures or 0),
            "corner_snapshots_resolvidos": int(corners_snapshots or 0),
            "min_fixtures_with_events": self.MIN_FIXTURES_WITH_EVENTS,
        }

    def _eligible_fixtures(self) -> list[dict]:
        conn = self.db._conn()
        placeholders = ",".join("?" for _ in TRAIN_SEASONS)
        league_ids = [info["id"] for info in LEAGUES.values()]
        league_placeholders = ",".join("?" for _ in league_ids)
        params = tuple(TRAIN_SEASONS) + tuple(league_ids)
        rows = conn.execute(
            f"""
            SELECT f.*
            FROM fixtures f
            JOIN (
                SELECT fixture_id, COUNT(*) AS n
                FROM fixture_events
                GROUP BY fixture_id
            ) fe ON fe.fixture_id = f.fixture_id
            WHERE f.status='FT'
              AND f.season IN ({placeholders})
              AND f.league_id IN ({league_placeholders})
            ORDER BY f.date
            """,
            params,
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _fixture_events(self, fixture_id: int) -> list[dict]:
        conn = self.db._conn()
        rows = conn.execute(
            """
            SELECT fixture_id, team_id, event_type, event_detail, minute, extra_minute
            FROM fixture_events
            WHERE fixture_id=?
            ORDER BY COALESCE(minute, 0), COALESCE(extra_minute, 0), id
            """,
            (fixture_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _fixture_static_features(self, fixture: dict) -> dict[str, float] | None:
        fixture_id = int(fixture["fixture_id"])
        if fixture_id not in self._feature_cache:
            feats = self.fe.features_jogo(fixture)
            if feats is not None:
                feats = {
                    name: float(feats.get(name, 0.0) or 0.0)
                    for name in self.fe.feature_names()
                }
            self._feature_cache[fixture_id] = feats
        return self._feature_cache[fixture_id]

    @staticmethod
    def _event_is_goal(event: dict) -> bool:
        return (event.get("event_type") or "").lower() == "goal"

    @staticmethod
    def _event_is_yellow(event: dict) -> bool:
        detail = (event.get("event_detail") or "").lower()
        return (event.get("event_type") or "").lower() == "card" and "yellow" in detail and "red" not in detail

    @staticmethod
    def _event_is_red(event: dict) -> bool:
        detail = (event.get("event_detail") or "").lower()
        return (event.get("event_type") or "").lower() == "card" and "red" in detail

    @staticmethod
    def _event_is_sub(event: dict) -> bool:
        return (event.get("event_type") or "").lower() == "subst"

    @staticmethod
    def _event_before_minute(event: dict, minute: int) -> bool:
        try:
            elapsed = int(event.get("minute") or 0)
            extra = int(event.get("extra_minute") or 0)
        except Exception:
            return False
        if elapsed < minute:
            return True
        if elapsed == minute and extra <= 0:
            return True
        return False

    def _has_consistent_goal_timeline(self, fixture: dict, events: list[dict]) -> bool:
        final_goals = int(fixture.get("goals_home") or 0) + int(fixture.get("goals_away") or 0)
        parsed_goals = sum(1 for event in events if self._event_is_goal(event))
        return parsed_goals >= final_goals

    def _state_until_minute(self, fixture: dict, events: list[dict], minute: int) -> dict[str, float]:
        home_id = int(fixture["home_id"])
        away_id = int(fixture["away_id"])
        state = {
            "goals_total": 0,
            "goals_home": 0,
            "goals_away": 0,
            "goals_ht": 0,
            "goals_2t": 0,
            "yellows_total": 0,
            "reds_total": 0,
            "home_reds": 0,
            "away_reds": 0,
            "subs_total": 0,
            "time_since_last_goal": float(minute),
        }
        last_goal_minute: int | None = None
        for event in events:
            if not self._event_before_minute(event, minute):
                continue
            team_id = int(event.get("team_id") or 0)
            elapsed = int(event.get("minute") or 0)
            if self._event_is_goal(event):
                state["goals_total"] += 1
                if team_id == home_id:
                    state["goals_home"] += 1
                elif team_id == away_id:
                    state["goals_away"] += 1
                if elapsed <= 45:
                    state["goals_ht"] += 1
                else:
                    state["goals_2t"] += 1
                last_goal_minute = elapsed
            elif self._event_is_yellow(event):
                state["yellows_total"] += 1
            elif self._event_is_red(event):
                state["reds_total"] += 1
                if team_id == home_id:
                    state["home_reds"] += 1
                elif team_id == away_id:
                    state["away_reds"] += 1
            elif self._event_is_sub(event):
                state["subs_total"] += 1

        if last_goal_minute is not None:
            state["time_since_last_goal"] = float(max(minute - last_goal_minute, 0))
        return state

    @staticmethod
    def _line_value(market: str) -> float:
        market = (market or "").lower()
        token = ""
        if market.startswith("over"):
            token = market[len("over"):].split("_", 1)[0]
        elif market.startswith("under"):
            token = market[len("under"):].split("_", 1)[0]
        elif market.startswith("corners_over_"):
            token = market[len("corners_over_"):]
        elif market.startswith("corners_under_"):
            token = market[len("corners_under_"):]
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            return 0.0
        if digits.endswith("5"):
            return float(digits[:-1] or "0") + 0.5
        return float(digits)

    def _market_is_resolvable(self, market: str, minute: int, state: dict[str, float]) -> bool:
        if not dentro_janela_operacional_live(market, minute):
            return False
        line = self._line_value(market)
        if market.startswith("over") and "_ht" not in market and "_2t" not in market:
            return state["goals_total"] <= line
        if market.startswith("under") and "_ht" not in market and "_2t" not in market:
            return state["goals_total"] <= line
        if market.endswith("_ht") and market.startswith("over"):
            return state["goals_ht"] <= line
        if market.endswith("_ht") and market.startswith("under"):
            return state["goals_ht"] <= line
        if market.endswith("_2t") and market.startswith("over"):
            return state["goals_2t"] <= line
        if market.endswith("_2t") and market.startswith("under"):
            return state["goals_2t"] <= line
        return True

    def _label_for_market(self, market: str, fixture: dict) -> int:
        gh = int(fixture.get("goals_home") or 0)
        ga = int(fixture.get("goals_away") or 0)
        total = gh + ga
        ht_h = int(fixture.get("score_ht_h") or 0)
        ht_a = int(fixture.get("score_ht_a") or 0)
        ht_total = ht_h + ht_a
        g2t_total = total - ht_total
        if market == "h2h_home":
            return 1 if gh > ga else 0
        if market == "h2h_draw":
            return 1 if gh == ga else 0
        if market == "h2h_away":
            return 1 if ga > gh else 0
        if market == "ht_home":
            return 1 if ht_h > ht_a else 0
        if market == "ht_draw":
            return 1 if ht_h == ht_a else 0
        if market == "ht_away":
            return 1 if ht_a > ht_h else 0
        line = self._line_value(market)
        if market.endswith("_ht"):
            if market.startswith("over"):
                return 1 if ht_total > line else 0
            return 1 if ht_total <= line else 0
        if market.endswith("_2t"):
            if market.startswith("over"):
                return 1 if g2t_total > line else 0
            return 1 if g2t_total <= line else 0
        if market.startswith("over"):
            return 1 if total > line else 0
        if market.startswith("under"):
            return 1 if total <= line else 0
        return 0

    def _minute_candidates(self, market: str) -> list[int]:
        start_minute, end_minute = janela_operacional_live(market)
        return list(range(start_minute, end_minute))

    def _base_market_features(self, fixture: dict, market: str, minute: int) -> dict[str, float]:
        return {
            "signal_minute": float(minute),
            "minute_bucket": float(minute // 5),
            "phase_first_half": 1.0 if minute <= 45 else 0.0,
            "phase_second_half": 1.0 if minute > 45 else 0.0,
            "league_id_numeric": float(fixture["league_id"]),
            "season_numeric": float(fixture.get("season") or 0),
            "market_is_ht": 1.0 if market.endswith("_ht") or market.startswith("ht_") else 0.0,
            "market_is_2t": 1.0 if market.endswith("_2t") else 0.0,
            "market_is_over": 1.0 if "over" in market else 0.0,
            "market_is_under": 1.0 if "under" in market else 0.0,
            "market_is_result": 1.0 if market.startswith("h2h_") or market.startswith("ht_") else 0.0,
            "market_is_draw": 1.0 if market.endswith("draw") else 0.0,
            "market_is_corners": 1.0 if "corners_" in market else 0.0,
        }

    def _sample_features(
        self,
        fixture: dict,
        market: str,
        minute: int,
        state: dict[str, float],
        static_features: dict[str, float],
    ) -> dict[str, float]:
        features = dict(static_features)
        features.update(self._base_market_features(fixture, market, minute))
        features.update({
            "goals_before_signal": float(state["goals_total"]),
            "home_goals_before_signal": float(state["goals_home"]),
            "away_goals_before_signal": float(state["goals_away"]),
            "goals_ht_before_signal": float(state["goals_ht"]),
            "goals_2t_before_signal": float(state["goals_2t"]),
            "score_diff_before_signal": float(state["goals_home"] - state["goals_away"]),
            "is_draw_before_signal": 1.0 if state["goals_home"] == state["goals_away"] else 0.0,
            "home_leading_before_signal": 1.0 if state["goals_home"] > state["goals_away"] else 0.0,
            "away_leading_before_signal": 1.0 if state["goals_away"] > state["goals_home"] else 0.0,
            "yellows_before_signal": float(state["yellows_total"]),
            "reds_before_signal": float(state["reds_total"]),
            "home_reds_before_signal": float(state["home_reds"]),
            "away_reds_before_signal": float(state["away_reds"]),
            "subs_before_signal": float(state["subs_total"]),
            "time_since_last_goal": float(state["time_since_last_goal"]),
        })
        return features

    @staticmethod
    def _coerce_float(value, default: float = 0.0) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return default

    @staticmethod
    def _payload_to_dict(raw_payload: str | None) -> dict:
        if not raw_payload:
            return {}
        try:
            data = json.loads(raw_payload)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _extract_snapshot_metrics(self, payload: dict) -> tuple[int | None, dict | None]:
        metricas = None
        for candidate in (
            payload.get("training_snapshot"),
            payload.get("metricas_live"),
            (payload.get("live_reading") or {}).get("metricas"),
        ):
            if isinstance(candidate, dict) and candidate:
                metricas = candidate
                break
        if not metricas:
            return None, None

        elapsed = (
            payload.get("last_live_minute")
            or payload.get("signal_minute")
            or (payload.get("live_reading") or {}).get("elapsed")
            or metricas.get("elapsed")
        )
        try:
            minute = int(elapsed or 0)
        except Exception:
            return None, None
        if minute <= 0:
            return None, None
        return minute, metricas

    def _corners_snapshot_features(
        self,
        market: str,
        league_id: int,
        minute: int,
        payload: dict,
        metricas: dict,
    ) -> dict[str, float]:
        features = {}
        payload_features = payload.get("features") or {}
        if isinstance(payload_features, dict):
            for name, value in payload_features.items():
                if isinstance(value, (int, float)):
                    features[str(name)] = float(value)

        fixture_stub = {"league_id": league_id, "season": payload.get("season") or 0}
        features.update(self._base_market_features(fixture_stub, market, minute))

        teams = metricas.get("teams") or []
        team0 = teams[0] if len(teams) > 0 and isinstance(teams[0], dict) else {}
        team1 = teams[1] if len(teams) > 1 and isinstance(teams[1], dict) else {}
        home_corners = self._coerce_float(team0.get("corners"))
        away_corners = self._coerce_float(team1.get("corners"))
        home_shots = self._coerce_float(team0.get("shots_total"))
        away_shots = self._coerce_float(team1.get("shots_total"))
        home_shots_on = self._coerce_float(team0.get("shots_on"))
        away_shots_on = self._coerce_float(team1.get("shots_on"))
        home_xg = self._coerce_float(team0.get("xg"))
        away_xg = self._coerce_float(team1.get("xg"))

        corners_total = self._coerce_float(metricas.get("corners"), home_corners + away_corners)
        shots_total = self._coerce_float(metricas.get("shots_total"), home_shots + away_shots)
        shots_on = self._coerce_float(metricas.get("shots_on"), home_shots_on + away_shots_on)
        xg_total = self._coerce_float(metricas.get("xg"), home_xg + away_xg)
        red_cards = self._coerce_float(metricas.get("red_cards"))
        yellow_cards = self._coerce_float(metricas.get("yellow_cards"))

        features.update({
            "corners_before_signal": corners_total,
            "corners_home_before_signal": home_corners,
            "corners_away_before_signal": away_corners,
            "corners_diff_before_signal": home_corners - away_corners,
            "corners_pace_per_minute": corners_total / max(float(minute), 1.0),
            "shots_before_signal": shots_total,
            "shots_on_before_signal": shots_on,
            "xg_before_signal": xg_total,
            "shots_home_before_signal": home_shots,
            "shots_away_before_signal": away_shots,
            "shots_on_home_before_signal": home_shots_on,
            "shots_on_away_before_signal": away_shots_on,
            "xg_home_before_signal": home_xg,
            "xg_away_before_signal": away_xg,
            "shots_diff_before_signal": home_shots - away_shots,
            "shots_on_diff_before_signal": home_shots_on - away_shots_on,
            "xg_diff_before_signal": home_xg - away_xg,
            "red_cards_before_signal": red_cards,
            "yellow_cards_before_signal": yellow_cards,
        })
        return features

    def _historical_samples_for_fixture(self, fixture: dict) -> list[LiveSample]:
        events = self._fixture_events(int(fixture["fixture_id"]))
        if not events or not self._has_consistent_goal_timeline(fixture, events):
            return []
        static_features = self._fixture_static_features(fixture)
        if static_features is None:
            return []

        samples: list[LiveSample] = []
        for market in HISTORICAL_LIVE_MARKETS:
            label = self._label_for_market(market, fixture)
            for minute in self._minute_candidates(market):
                state = self._state_until_minute(fixture, events, minute)
                if not self._market_is_resolvable(market, minute, state):
                    continue
                features = self._sample_features(fixture, market, minute, state, static_features)
                samples.append(
                    LiveSample(
                        fixture_id=int(fixture["fixture_id"]),
                        fixture_date=str(fixture["date"]),
                        league_id=int(fixture["league_id"]),
                        market=market,
                        signal_minute=minute,
                        label=label,
                        features=features,
                        source="api_historical",
                    )
                )
        return samples

    def _load_historical_samples(self) -> list[LiveSample]:
        samples: list[LiveSample] = []
        for fixture in self._eligible_fixtures():
            samples.extend(self._historical_samples_for_fixture(fixture))
        return samples

    def _load_corner_snapshot_samples(self) -> list[LiveSample]:
        conn = self.db._conn()
        rows = conn.execute(
            """
            SELECT
              lw.id,
              lw.fixture_id,
              lw.fixture_date,
              lw.league_id,
              lw.mercado,
              lw.payload_json,
              lr.snapshot_json,
              lr.acertou,
              lr.signal_minute
            FROM live_watchlist lw
            JOIN live_results lr ON lr.live_watch_id = lw.id
            WHERE lr.acertou IS NOT NULL
              AND lw.mercado LIKE 'corners_%'
              AND (lw.payload_json IS NOT NULL OR lr.snapshot_json IS NOT NULL)
            ORDER BY COALESCE(lw.fixture_date, ''), lw.id
            """
        ).fetchall()
        conn.close()

        samples: list[LiveSample] = []
        for row in rows:
            item = dict(row)
            market = str(item["mercado"])
            if market not in CORNER_SNAPSHOT_MARKETS:
                continue
            payload = self._payload_to_dict(item.get("payload_json"))
            snapshot = self._payload_to_dict(item.get("snapshot_json"))
            if snapshot and not payload.get("training_snapshot"):
                payload["training_snapshot"] = snapshot
            minute, metricas = self._extract_snapshot_metrics(payload)
            if minute is None or metricas is None:
                continue
            if not dentro_janela_operacional_live(market, minute):
                continue
            features = self._corners_snapshot_features(
                market=market,
                league_id=int(item["league_id"] or 0),
                minute=minute,
                payload=payload,
                metricas=metricas,
            )
            if not features:
                continue
            samples.append(
                LiveSample(
                    fixture_id=int(item["fixture_id"]),
                    fixture_date=str(item.get("fixture_date") or ""),
                    league_id=int(item["league_id"] or 0),
                    market=market,
                    signal_minute=minute,
                    label=int(item["acertou"]),
                    features=features,
                    source="bot_snapshot",
                )
            )
        return samples

    def _load_samples(self) -> list[LiveSample]:
        samples = self._load_historical_samples()
        samples.extend(self._load_corner_snapshot_samples())
        return samples

    @staticmethod
    def _split_fixture_ids(samples: list[LiveSample]) -> tuple[set[int], set[int]]:
        fixture_dates: dict[int, str] = {}
        for sample in samples:
            fixture_dates.setdefault(sample.fixture_id, sample.fixture_date)
        ordered = [fixture_id for fixture_id, _ in sorted(fixture_dates.items(), key=lambda item: item[1])]
        if len(ordered) < 4:
            return set(ordered), set()
        split_idx = max(int(len(ordered) * 0.7), 1)
        if split_idx >= len(ordered):
            split_idx = len(ordered) - 1
        return set(ordered[:split_idx]), set(ordered[split_idx:])

    @staticmethod
    def _xgb_params() -> dict:
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 4,
            "gamma": 0.1,
            "tree_method": "hist",
            "seed": 42,
            "verbosity": 0,
        }

    def _reset_output_dir(self):
        for child in MODELS_DIR.iterdir():
            if child.is_dir() and child.name.startswith("league_"):
                shutil.rmtree(child)
            elif child.is_file() and child.suffix == ".json":
                child.unlink()

    def treinar(self, min_amostras_mercado: int = 30, min_amostras_cantos: int | None = None) -> dict:
        readiness = self.readiness()
        if not HAS_ML:
            return {
                "status": "blocked",
                "reason": "ml_indisponivel",
                "readiness": readiness,
            }
        if not readiness.get("pronto"):
            return {
                "status": "blocked",
                "reason": "base_live_insuficiente",
                "readiness": readiness,
            }

        min_amostras_cantos = int(min_amostras_cantos or self.MIN_CORNER_SNAPSHOT_SAMPLES)
        samples = self._load_samples()
        if len(samples) < self.MIN_SAMPLES_TOTAL:
            return {
                "status": "blocked",
                "reason": "amostras_historicas_insuficientes",
                "readiness": readiness,
                "samples_total": len(samples),
            }

        self._reset_output_dir()

        samples_by_slice: dict[tuple[int, str], list[LiveSample]] = defaultdict(list)
        samples_by_league: dict[int, int] = Counter()
        fixtures_by_league: dict[int, set[int]] = defaultdict(set)
        source_counter: dict[str, int] = Counter()
        for sample in samples:
            samples_by_slice[(sample.league_id, sample.market)].append(sample)
            samples_by_league[sample.league_id] += 1
            fixtures_by_league[sample.league_id].add(sample.fixture_id)
            source_counter[sample.source] += 1

        summary: dict[str, object] = {
            "status": "ok",
            "layout": "league_market_only",
            "source_historical": "api_historical",
            "source_corners": "bot_snapshots",
            "readiness": readiness,
            "fixtures_usados": len({sample.fixture_id for sample in samples}),
            "samples_total": len(samples),
            "samples_by_source": dict(sorted(source_counter.items())),
            "samples_by_league": {
                str(league_id): {
                    "league_id": league_id,
                    "samples": count,
                    "fixtures": len(fixtures_by_league[league_id]),
                }
                for league_id, count in sorted(samples_by_league.items())
            },
            "slices": {},
            "slices_trained": 0,
            "slices_skipped": 0,
        }

        for league_id, market in sorted(samples_by_slice):
            slice_samples = samples_by_slice[(league_id, market)]
            source_name = slice_samples[0].source if slice_samples else "desconhecido"
            min_slice = min_amostras_cantos if market in CORNER_SNAPSHOT_MARKETS else min_amostras_mercado
            slice_key = f"{league_id}:{market}"
            feature_names = sorted({name for sample in slice_samples for name in sample.features.keys()})
            if len(slice_samples) < min_slice:
                summary["slices"][slice_key] = {
                    "status": "skipped",
                    "reason": "dados_insuficientes",
                    "league_id": league_id,
                    "market": market,
                    "source": source_name,
                    "samples": len(slice_samples),
                    "fixtures": len({sample.fixture_id for sample in slice_samples}),
                    "min_required": min_slice,
                }
                summary["slices_skipped"] += 1
                continue

            train_ids, test_ids = self._split_fixture_ids(slice_samples)
            if not train_ids or not test_ids:
                summary["slices"][slice_key] = {
                    "status": "skipped",
                    "reason": "split_historico_insuficiente",
                    "league_id": league_id,
                    "market": market,
                    "source": source_name,
                    "samples": len(slice_samples),
                    "fixtures": len({sample.fixture_id for sample in slice_samples}),
                }
                summary["slices_skipped"] += 1
                continue

            train_samples = [sample for sample in slice_samples if sample.fixture_id in train_ids]
            test_samples = [sample for sample in slice_samples if sample.fixture_id in test_ids]
            if len(train_samples) < min_slice or len(test_samples) < self.MIN_TEST_SAMPLES:
                summary["slices"][slice_key] = {
                    "status": "skipped",
                    "reason": "split_insuficiente",
                    "league_id": league_id,
                    "market": market,
                    "source": source_name,
                    "samples": len(slice_samples),
                    "train_samples": len(train_samples),
                    "test_samples": len(test_samples),
                    "fixtures": len({sample.fixture_id for sample in slice_samples}),
                }
                summary["slices_skipped"] += 1
                continue

            x_train = np.array(
                [[sample.features.get(name, 0.0) for name in feature_names] for sample in train_samples],
                dtype=np.float32,
            )
            y_train = np.array([sample.label for sample in train_samples], dtype=np.float32)
            x_test = np.array(
                [[sample.features.get(name, 0.0) for name in feature_names] for sample in test_samples],
                dtype=np.float32,
            )
            y_test = np.array([sample.label for sample in test_samples], dtype=np.float32)

            if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                summary["slices"][slice_key] = {
                    "status": "skipped",
                    "reason": "classe_unica",
                    "league_id": league_id,
                    "market": market,
                    "source": source_name,
                    "samples": len(slice_samples),
                    "train_samples": len(train_samples),
                    "test_samples": len(test_samples),
                }
                summary["slices_skipped"] += 1
                continue

            dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
            dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
            model = xgb.train(
                self._xgb_params(),
                dtrain,
                num_boost_round=400,
                evals=[(dtrain, "train"), (dtest, "test")],
                early_stopping_rounds=40,
                verbose_eval=False,
            )
            probs = model.predict(dtest)
            preds = (probs >= 0.5).astype(int)
            accuracy = float(accuracy_score(y_test, preds))

            league_dir = MODELS_DIR / f"league_{league_id}"
            league_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(league_dir / f"{market}.json"))
            with open(league_dir / f"{market}.meta.json", "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "league_id": league_id,
                        "market": market,
                        "source": source_name,
                        "feature_names": feature_names,
                    },
                    fp,
                    ensure_ascii=False,
                    indent=2,
                )

            summary["slices"][slice_key] = {
                "status": "trained",
                "league_id": league_id,
                "market": market,
                "source": source_name,
                "samples": len(slice_samples),
                "fixtures": len({sample.fixture_id for sample in slice_samples}),
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "positive_rate_train": round(float(np.mean(y_train)), 4),
                "positive_rate_test": round(float(np.mean(y_test)), 4),
                "accuracy": round(accuracy, 4),
                "logloss": round(float(log_loss(y_test, probs, labels=[0, 1])), 4),
                "best_iteration": int(model.best_iteration),
                "feature_count": len(feature_names),
            }
            summary["slices_trained"] += 1

        with open(MODELS_DIR / "summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        return summary

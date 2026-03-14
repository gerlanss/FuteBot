import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime


class ConfigTests(unittest.TestCase):
    def test_config_uses_env_without_secret_fallbacks(self):
        import config

        original = {
            "API_FOOTBALL_KEY": os.environ.get("API_FOOTBALL_KEY"),
            "ODDS_API_KEY": os.environ.get("ODDS_API_KEY"),
            "TIMEZONE": os.environ.get("TIMEZONE"),
        }

        try:
            os.environ["API_FOOTBALL_KEY"] = "api-key-test"
            os.environ["ODDS_API_KEY"] = "odds-key-test"
            os.environ["TIMEZONE"] = "America/Manaus"

            config = importlib.reload(config)

            self.assertEqual(config.API_FOOTBALL_KEY, "api-key-test")
            self.assertEqual(config.ODDS_API_KEY, "odds-key-test")
            self.assertEqual(config.TIMEZONE, "America/Manaus")
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(config)


class TrainerModelDiscoveryTests(unittest.TestCase):
    def test_modelo_existe_detects_per_league_layout(self):
        import models.trainer as trainer_module

        with tempfile.TemporaryDirectory() as tmpdir:
            league_dir = os.path.join(tmpdir, "league_71")
            os.makedirs(league_dir, exist_ok=True)

            with open(os.path.join(league_dir, "resultado_1x2.json"), "w", encoding="utf-8") as f:
                f.write("{}")

            with patch.object(trainer_module, "MODELS_DIR", tmpdir):
                self.assertTrue(trainer_module.Trainer.modelo_existe("resultado_1x2"))
                self.assertTrue(trainer_module.Trainer.modelo_existe("resultado_1x2", league_id=71))
                self.assertTrue(trainer_module.Trainer.ha_modelos_treinados())
                self.assertGreaterEqual(trainer_module.Trainer.contar_modelos_base(), 1)


class SchedulerTests(unittest.TestCase):
    def test_job_relatorio_runs_without_timedelta_name_error(self):
        from pipeline.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.db = MagicMock()
        sent_messages = []
        scheduler._enviar_telegram = sent_messages.append

        learner = MagicMock()
        learner.relatorio_resultado_dia.return_value = "Resultados OK"
        learner.relatorio_diario.return_value = "Diario OK"
        learner.relatorio_saude.return_value = "Saude OK"
        learner.verificar_degradacao.return_value = {
            "degradado": False,
            "pausado": False,
            "alertas": [],
        }

        with patch("pipeline.scheduler.Learner", return_value=learner), patch("builtins.print"):
            scheduler._job_relatorio()

        self.assertEqual(sent_messages, ["Resultados OK", "Diario OK", "Saude OK"])

    def test_priorizar_ligas_quarentena_keeps_working_same_league_first(self):
        from pipeline.scheduler import Scheduler

        ordem = Scheduler._priorizar_ligas_quarentena([
            {"league_id": 71, "roi": -12.0, "total": 5},
            {"league_id": 71, "roi": -30.0, "total": 8},
            {"league_id": 135, "roi": -40.0, "total": 4},
        ])

        self.assertEqual(ordem[0], 71)


class LearnerConfidenceTests(unittest.TestCase):
    @staticmethod
    def _insert_fixture(db, fixture_id, data):
        conn = db._conn()
        conn.execute("""
            INSERT INTO fixtures (
                fixture_id, league_id, league_name, season, round, date, status,
                home_id, home_name, away_id, away_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id, 71, "Liga Teste", 2026, "Round 1", f"{data} 19:00:00", "FT",
            fixture_id * 10 + 1, f"Time {fixture_id}A", fixture_id * 10 + 2, f"Time {fixture_id}B",
        ))
        conn.commit()
        conn.close()

    def test_relatorio_resultado_uses_prob_modelo_for_saved_tip(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 1, data)
            db.salvar_prediction({
                "fixture_id": 1,
                "date": f"{data} 19:00:00",
                "league_id": 71,
                "home_name": "Time A",
                "away_name": "Time B",
                "mercado": "under35",
                "prob_modelo": 0.82,
                "odd_usada": 1.9,
                "ev_percent": 12.3,
            })
            db.resolver_prediction(1, "away", 0, 2)

            relatorio = Learner(db).relatorio_resultado_dia(data)

            self.assertIn("Confiança: 82%", relatorio)

    def test_relatorio_resultado_shows_nd_for_legacy_market_without_prob_mapping(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 2, data)
            conn = db._conn()
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, gols_home, gols_away, lucro
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (2, f"{data} 19:00:00", 71, "Time C", "Time D", "under35", 1, 1, 1, 0.9))
            conn.commit()
            conn.close()

            relatorio = Learner(db).relatorio_resultado_dia(data)

            self.assertIn("Confiança: n/d", relatorio)

    def test_relatorio_resultado_includes_combo_results(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 11, data)
            self._insert_fixture(db, 12, data)

            db.salvar_prediction({
                "fixture_id": 11,
                "date": f"{data} 19:00:00",
                "league_id": 71,
                "home_name": "Time 11A",
                "away_name": "Time 11B",
                "mercado": "over15",
                "prob_modelo": 0.81,
                "odd_usada": 1.8,
            })
            db.salvar_prediction({
                "fixture_id": 12,
                "date": f"{data} 19:00:00",
                "league_id": 71,
                "home_name": "Time 12A",
                "away_name": "Time 12B",
                "mercado": "under35",
                "prob_modelo": 0.77,
                "odd_usada": 1.7,
            })
            db.resolver_prediction(11, "home", 2, 1)
            db.resolver_prediction(12, "away", 0, 1)

            db.salvar_combo({
                "date": data,
                "tipo": "dupla",
                "prob_composta": 0.62,
                "tips": [
                    {"fixture_id": 11, "mercado": "over15", "home_name": "Time 11A", "away_name": "Time 11B", "prob_modelo": 0.81},
                    {"fixture_id": 12, "mercado": "under35", "home_name": "Time 12A", "away_name": "Time 12B", "prob_modelo": 0.77},
                ],
            })

            relatorio = Learner(db).relatorio_resultado_dia(data)

            self.assertIn("<b>COMBOS</b>", relatorio)
            self.assertIn("Dupla #1", relatorio)

    def test_context_feedback_labels_successful_release_and_block(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 21, data)

            conn = db._conn()
            conn.execute("""
                UPDATE fixtures
                SET score_ht_h = 0, score_ht_a = 0
                WHERE fixture_id = 21
            """)
            conn.execute("""
                INSERT INTO scan_audit (
                    scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, llm_decisao, llm_motivo, approved_final, contexto_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data, 21, 71, "Time 21A", "Time 21B",
                "under35", "APPROVE", "Chuva forte favorece under", 1,
                '{"market_lookup":{"weather_summary":"chuva forte","risk_flags":["rain"]}}',
            ))
            conn.execute("""
                INSERT INTO scan_audit (
                    scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, llm_decisao, llm_motivo, approved_final, contexto_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data, 21, 71, "Time 21A", "Time 21B",
                "over15", "REJECT", "Tempo ruim atrapalha gols", 0,
                '{"market_lookup":{"weather_summary":"chuva forte","risk_flags":["rain"]}}',
            ))
            conn.commit()
            conn.close()

            learner = Learner(db)
            learner._registrar_feedback_contextual_fixture(21, 0, 0, db.fixture_por_id(21))

            resumo = {item["context_label"]: item["total"] for item in db.context_feedback_resumo()}
            self.assertEqual(resumo.get("good_release"), 1)
            self.assertEqual(resumo.get("good_block"), 1)

            conn = db._conn()
            rows = conn.execute("""
                SELECT context_label, weather_summary
                FROM context_feedback
                ORDER BY context_label ASC
            """).fetchall()
            conn.close()
            labels = {row["context_label"]: row["weather_summary"] for row in rows}
            self.assertEqual(labels["good_release"], "chuva forte")
            self.assertEqual(labels["good_block"], "chuva forte")

    def test_relatorio_resultado_includes_blocked_markets_without_counting_them_as_tips(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 31, data)
            self._insert_fixture(db, 32, data)

            conn = db._conn()
            conn.execute("""
                UPDATE fixtures SET goals_home = 0, goals_away = 0, score_ht_h = 0, score_ht_a = 0
                WHERE fixture_id = 31
            """)
            conn.execute("""
                UPDATE fixtures SET goals_home = 1, goals_away = 1, score_ht_h = 0, score_ht_a = 1
                WHERE fixture_id = 32
            """)
            conn.commit()
            conn.close()

            db.salvar_prediction({
                "fixture_id": 31,
                "date": f"{data} 19:00:00",
                "league_id": 71,
                "home_name": "Time 31A",
                "away_name": "Time 31B",
                "mercado": "under35",
                "prob_modelo": 0.82,
                "odd_usada": 1.8,
            })
            db.resolver_prediction(31, "draw", 0, 0)

            conn = db._conn()
            conn.execute("""
                INSERT INTO scan_audit (
                    scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, prob_modelo, llm_decisao, llm_motivo, approved_final, contexto_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data, 32, 71, "Time 32A", "Time 32B",
                "under35", 0.74, "REJECT", "Chuva forte e ritmo baixo esperado.", 0, "{}"
            ))
            conn.commit()
            conn.close()

            Learner(db).backfill_feedback_contextual()
            relatorio = Learner(db).relatorio_resultado_dia(data)

            self.assertIn("<b>BARRADAS NA REVISAO</b>", relatorio)
            self.assertIn("Bloqueio discutivel", relatorio)
            self.assertIn("Time 32A vs Time 32B", relatorio)
            self.assertIn("Tips: 1 | Acertos: 1/1", relatorio)


class StrategySliceTests(unittest.TestCase):
    @staticmethod
    def _insert_fixture(db, fixture_id, data):
        conn = db._conn()
        conn.execute("""
            INSERT INTO fixtures (
                fixture_id, league_id, league_name, season, round, date, status,
                home_id, home_name, away_id, away_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fixture_id, 71, "Liga Teste", 2026, "Round 1", f"{data} 19:00:00", "FT",
            fixture_id * 10 + 1, f"Time {fixture_id}A", fixture_id * 10 + 2, f"Time {fixture_id}B",
        ))
        conn.commit()
        conn.close()

    def test_slices_degradados_identifies_bad_market_by_league(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 100, data)
            self._insert_fixture(db, 101, data)
            self._insert_fixture(db, 102, data)
            self._insert_fixture(db, 103, data)
            self._insert_fixture(db, 104, data)

            conn = db._conn()
            for idx in range(5):
                conn.execute("""
                    INSERT INTO predictions (
                        fixture_id, date, league_id, home_name, away_name,
                        mercado, acertou, lucro, modelo_versao
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    100 + idx, f"{data} 19:00:00", 71, "Time A", "Time B",
                    "over15", 0, -1.0, "vtest",
                ))
            conn.commit()
            conn.close()

            ruins = db.slices_degradados(
                modelo_versao="vtest",
                min_amostras=5,
                roi_threshold=-15.0,
                acc_threshold=35.0,
            )

            self.assertEqual(len(ruins), 1)
            self.assertEqual(ruins[0]["league_id"], 71)
            self.assertEqual(ruins[0]["mercado"], "over15")

    def test_salvar_strategies_can_replace_single_league_only(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.salvar_strategies([
                {"mercado": "over15", "league_id": 71, "conf_min": 0.7, "conf_max": 1.0, "accuracy": 0.6, "n_samples": 12, "ativo": 1},
                {"mercado": "under35", "league_id": 78, "conf_min": 0.7, "conf_max": 1.0, "accuracy": 0.62, "n_samples": 12, "ativo": 1},
            ])
            db.salvar_strategies([
                {"mercado": "h2h_home", "league_id": 71, "conf_min": 0.7, "conf_max": 1.0, "accuracy": 0.65, "n_samples": 14, "ativo": 1},
            ], replace=False, league_ids=[71])

            rows = db.strategies_ativas()
            pares = {(r["league_id"], r["mercado"]) for r in rows}

            self.assertIn((71, "h2h_home"), pares)
            self.assertIn((78, "under35"), pares)
            self.assertNotIn((71, "over15"), pares)


class ScannerSelectionTests(unittest.TestCase):
    def test_scanner_limits_total_combos_and_keeps_tripla_possible(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        tips = [
            {"fixture_id": 1, "prob_modelo": 0.90},
            {"fixture_id": 2, "prob_modelo": 0.88},
            {"fixture_id": 3, "prob_modelo": 0.86},
            {"fixture_id": 4, "prob_modelo": 0.84},
            {"fixture_id": 5, "prob_modelo": 0.82},
            {"fixture_id": 6, "prob_modelo": 0.80},
        ]

        combos = scanner._gerar_combos(tips)

        self.assertLessEqual(len(combos), 3)
        self.assertTrue(all(len(c["tips"]) in (2, 3) for c in combos))
        self.assertEqual(len({t["fixture_id"] for c in combos for t in c["tips"]}),
                         sum(len(c["tips"]) for c in combos))

    def test_extrair_odd_mercado_requires_exact_total_line(self):
        from pipeline.scanner import Scanner

        jogo = {
            "home_team": "RB Leipzig",
            "away_team": "FC Augsburg",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.90, "point": 2.5},
                                {"name": "Under", "price": 1.95, "point": 2.5},
                            ],
                        }
                    ],
                }
            ],
        }

        odd, casa = Scanner._extrair_odd_mercado(jogo, {"mercado": "over15"})

        self.assertEqual((odd, casa), (0, ""))

    def test_formatar_relatorio_exibe_estagios_e_nomes_em_code(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        scanner.db = MagicMock()
        scanner.db.metricas_modelo.return_value = {"total": 0}

        msgs = scanner.formatar_relatorio({
            "fixtures": 29,
            "previsoes": 29,
            "tips_brutas": 187,
            "tips_pos_filtros": 119,
            "tips_bloqueadas_ev": 2,
            "tips_enviadas_llm": 117,
            "ev_positivas": [{
                "league_id": 78,
                "fixture_id": 1,
                "home_name": "RB Leipzig",
                "away_name": "FC Augsburg",
                "date": "2026-03-07T13:30:00+00:00",
                "prob_modelo": 0.822,
                "descricao": "Over 1.5 gols",
                "odd_pinnacle": 1.90,
                "ev_percent": 56.1,
                "odd_fonte": "Pinnacle",
            }],
            "combos": [],
            "data": "2026-03-07",
        })

        header, body = msgs[0][0], msgs[1][0]
        self.assertIn("Mercados candidatos: <b>187</b>", header)
        self.assertIn("Bloqueadas por EV: <b>2</b>", header)
        self.assertIn("Mercados revisados: <b>117</b>", header)
        self.assertIn("<code>RB Leipzig</code> <b>x</b> <code>FC Augsburg</code>", body)
        self.assertIn("Bet365", body)


class TelegramChatPersistenceTests(unittest.TestCase):
    def test_database_persists_public_and_admin_chats(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.salvar_telegram_chat(123, is_admin=False, username="user", first_name="User")
            db.salvar_telegram_chat(999, is_admin=True, username="admin", first_name="Admin")

            self.assertEqual(db.telegram_chat_ids(), [123, 999])
            self.assertEqual(db.telegram_chat_ids(apenas_admin=True), [999])

    def test_predictions_por_data_returns_saved_scan_batch(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")

            conn = db._conn()
            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, status,
                    home_id, home_name, away_id, away_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                501, 71, "Liga Teste", 2026, "Round 1", f"{data} 19:00:00", "NS",
                1, "Casa", 2, "Fora",
            ))
            conn.commit()
            conn.close()

            db.salvar_prediction({
                "fixture_id": 501,
                "date": f"{data} 19:00:00",
                "league_id": 71,
                "home_name": "Casa",
                "away_name": "Fora",
                "mercado": "under35",
                "prob_modelo": 0.74,
                "odd_usada": 1.88,
            })

            rows = db.predictions_por_data(data)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["league_name"], "Liga Teste")
            self.assertEqual(rows[0]["mercado"], "under35")

    def test_scan_audit_persists_llm_decisions(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.salvar_scan_audit("2026-03-12", [
                {
                    "fixture_id": 1,
                    "league_id": 71,
                    "home_name": "Casa",
                    "away_name": "Fora",
                    "mercado": "under35",
                    "descricao": "Under 3.5 gols",
                    "prob_modelo": 0.81,
                    "approved_final": True,
                    "llm_validacao": {
                        "decisao": "APPROVE",
                        "confianca": 0.75,
                        "motivo": "Jogo travado.",
                    },
                    "llm_contexto": {"classificacao": []},
                },
                {
                    "fixture_id": 2,
                    "league_id": 71,
                    "home_name": "Casa 2",
                    "away_name": "Fora 2",
                    "mercado": "over15",
                    "descricao": "Over 1.5 gols",
                    "prob_modelo": 0.74,
                    "approved_final": False,
                    "llm_validacao": {
                        "decisao": "REJECT",
                        "confianca": 0.65,
                        "motivo": "Ataques fracos.",
                    },
                },
            ])

            rows = db.scan_audit_por_data("2026-03-12")

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["llm_decisao"], "APPROVE")
            self.assertEqual(rows[1]["llm_decisao"], "REJECT")


class MarketDiscoveryTests(unittest.TestCase):
    def test_market_discovery_has_unique_market_ids(self):
        from models.market_discovery import MARKET_SPECS

        market_ids = [item.market_id for item in MARKET_SPECS]
        self.assertEqual(len(market_ids), len(set(market_ids)))

    def test_market_discovery_covers_result_and_corner_markets(self):
        from models.market_discovery import MARKET_SPEC_MAP

        self.assertIn("h2h_home", MARKET_SPEC_MAP)
        self.assertIn("under35", MARKET_SPEC_MAP)
        self.assertIn("over05_ht", MARKET_SPEC_MAP)
        self.assertIn("corners_under_105", MARKET_SPEC_MAP)

    def test_apply_discovery_infers_conf_band_from_model_prob_rule(self):
        from scripts.apply_discovery_strategies import _infer_conf_band

        conf_min, conf_max = _infer_conf_band({
            "conditions": [
                ["away_btts_pct", ">=", 0.4],
                ["model_prob", ">=", 0.654],
            ]
        })

        self.assertEqual(conf_min, 0.70)
        self.assertEqual(conf_max, 1.01)

    def test_strategy_rule_match_checks_tip_features(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        strategy = {
            "params_json": "{\"conditions\": [[\"away_btts_pct\", \">=\", 0.4], [\"model_prob\", \">=\", 0.75]]}"
        }

        self.assertTrue(scanner._strategy_rule_match(strategy, {
            "prob_modelo": 0.8,
            "features": {"away_btts_pct": 0.5},
        }))
        self.assertFalse(scanner._strategy_rule_match(strategy, {
            "prob_modelo": 0.8,
            "features": {"away_btts_pct": 0.2},
        }))

    def test_cup_split_uses_last_two_seasons_when_available(self):
        from models.market_discovery import MarketDiscoveryTrainer

        trainer = MarketDiscoveryTrainer.__new__(MarketDiscoveryTrainer)
        rows = [
            {"_season": 2022, "_date": "2022-06-01"},
            {"_season": 2023, "_date": "2023-06-01"},
            {"_season": 2024, "_date": "2024-06-01"},
            {"_season": 2025, "_date": "2025-06-01"},
        ]

        split = trainer._build_temporal_split(13, rows, [2022, 2023, 2024, 2025])

        self.assertEqual(split["competition_type"], "cup")
        self.assertEqual(split["split_mode"], "last_two_seasons")
        self.assertEqual(split["train_seasons"], [2022, 2023])
        self.assertEqual(split["test_seasons"], [2024, 2025])
        self.assertEqual(int(split["train_mask"].sum()), 2)
        self.assertEqual(int(split["test_mask"].sum()), 2)

    def test_cup_single_season_uses_chronological_split(self):
        from models.market_discovery import MarketDiscoveryTrainer

        trainer = MarketDiscoveryTrainer.__new__(MarketDiscoveryTrainer)
        rows = [
            {"_season": 2026, "_date": f"2026-06-{day:02d}"}
            for day in range(1, 11)
        ]

        split = trainer._build_temporal_split(1, rows, [2026])

        self.assertEqual(split["competition_type"], "cup")
        self.assertEqual(split["split_mode"], "chronological_single_season")
        self.assertEqual(int(split["train_mask"].sum()), 7)
        self.assertEqual(int(split["test_mask"].sum()), 3)


class GeminiLookupTests(unittest.TestCase):
    def test_gemini_extract_json_from_fenced_block(self):
        from services.gemini_lookup import GeminiMarketLookup

        lookup = GeminiMarketLookup()
        data = lookup._extract_json("""```json
{"market_found": true, "bookmakers": ["Bet365"], "summary": "ok", "confidence": 0.8}
```""")

        self.assertTrue(data["market_found"])
        self.assertEqual(data["bookmakers"], ["Bet365"])

    def test_gemini_normalize_keeps_external_context_fields(self):
        from services.gemini_lookup import GeminiMarketLookup

        lookup = GeminiMarketLookup()
        data = lookup._normalize({
            "market_found": True,
            "bookmakers": ["Bet365"],
            "market_summary": "Mercado encontrado.",
            "weather_summary": "Chuva leve prevista.",
            "field_conditions": "Gramado pesado.",
            "rotation_risk": "alto",
            "motivation_context": "Jogo decisivo por classificacao.",
            "news_summary": "Time da casa pode poupar titulares.",
            "risk_flags": ["chuva", "rotacao"],
            "context_summary": "Contexto externo relevante.",
            "confidence": 0.8,
        }, "", {})

        self.assertEqual(data["rotation_risk"], "alto")
        self.assertIn("chuva", data["risk_flags"])
        self.assertEqual(data["weather_summary"], "Chuva leve prevista.")


class ScannerAuditFormattingTests(unittest.TestCase):
    def test_formatar_relatorio_includes_llm_rejections_block(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        scanner.db = MagicMock()
        scanner.db.metricas_modelo.return_value = {"total": 0}

        msgs = scanner.formatar_relatorio({
            "fixtures": 10,
            "previsoes": 10,
            "tips_brutas": 20,
            "tips_pos_filtros": 5,
            "tips_bloqueadas_ev": 0,
            "tips_enviadas_llm": 5,
            "tips_aprovadas_llm": 2,
            "tips_rejeitadas_llm": [{
                "league_id": 71,
                "home_name": "Casa",
                "away_name": "Fora",
                "descricao": "Over 1.5 gols",
                "prob_modelo": 0.76,
                "llm_validacao": {"motivo": "Desfalques ofensivos importantes."},
            }],
            "ev_positivas": [{
                "league_id": 71,
                "fixture_id": 1,
                "home_name": "Casa",
                "away_name": "Fora",
                "date": "2026-03-07T13:30:00+00:00",
                "prob_modelo": 0.822,
                "descricao": "Under 3.5 gols",
                "llm_validacao": {"motivo": "Jogo travado."},
            }],
            "combos": [],
            "data": "2026-03-07",
        })

        joined = "\n\n".join(texto for texto, _ in msgs)
        self.assertIn("Resultado da revis", joined)
        self.assertIn("Entradas barradas", joined)
        self.assertIn("Entrada cancelada nesta janela.", joined)
        self.assertIn("• Desfalques ofensivos importantes.", joined)
        self.assertIn("Bet365", joined)

    def test_formatar_resumo_revisao_shortens_llm_text(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        linhas = scanner._formatar_resumo_revisao(
            "O contexto esportivo apresenta cautela. Desfalques ofensivos importantes. "
            "Chuva forte e gramado pesado. Favorito deve controlar o ritmo. "
            "Evite entrar agora.",
            bloqueado=True,
        )

        texto = "\n".join(linhas)
        self.assertIn("Entrada cancelada nesta janela.", texto)
        self.assertIn("O contexto esportivo apresenta cautela.", texto)
        self.assertIn("Desfalques ofensivos importantes.", texto)
        self.assertIn("Chuva forte e gramado pesado.", texto)
        self.assertIn("Por isso preferi ficar de fora", texto)

    def test_formatar_resumo_revisao_prioritizes_concrete_context(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        linhas = scanner._formatar_resumo_revisao(
            "O contexto esportivo apresenta cautela. Favorito deve controlar o ritmo.",
            bloqueado=True,
            tip={
                "llm_contexto": {
                    "lesoes": [
                        {"jogador": "Paulinho"},
                        {"jogador": "Vitor Roque"},
                    ],
                    "market_lookup": {
                        "weather_summary": "Chuva moderada durante o jogo",
                        "field_conditions": "Gramado pesado",
                    },
                }
            },
        )

        texto = "\n".join(linhas)
        self.assertIn("Desfalques relevantes: Paulinho, Vitor Roque.", texto)
        self.assertIn("Chuva moderada durante o jogo.", texto)
        self.assertIn("Gramado: Gramado pesado.", texto)
        self.assertIn("Por isso preferi ficar de fora", texto)

    def test_formatar_resumo_revisao_prioritizes_market_specific_reasoning(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        linhas = scanner._formatar_resumo_revisao(
            "Ambos buscam liderança na liga. Jogo crucial pela parte alta da tabela.",
            bloqueado=False,
            tip={
                "mercado": "under35",
                "prob_modelo": 0.79,
                "prob_over25": 0.43,
                "features": {
                    "home_cs_5": 0.4,
                    "away_cs_5": 0.6,
                    "home_fts_5": 0.2,
                    "away_fts_5": 0.2,
                },
                "llm_contexto": {
                    "market_lookup": {
                        "motivation_context": "Ambos buscam liderança na liga.",
                    }
                },
            },
        )

        texto = "\n".join(linhas)
        self.assertIn("O modelo ainda sustenta esse under em 79%.", texto)
        self.assertIn("O modelo nao ve forca suficiente para um jogo acima de 2.5 gols (43%).", texto)
        self.assertIn("Os dois lados chegam cedendo pouco espaco: clean sheets casa 40% | fora 60%.", texto)
        self.assertNotIn("liderança", texto.lower())


class LiveIntelligenceTests(unittest.TestCase):
    def test_live_intelligence_flags_goal_over_signal(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 62}},
            "goals": {"home": 0, "away": 0},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 4},
                    {"type": "Total Shots", "value": 10},
                    {"type": "expected_goals", "value": 0.9},
                    {"type": "Corner Kicks", "value": 4},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 7},
                    {"type": "expected_goals", "value": 0.7},
                    {"type": "Corner Kicks", "value": 2},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "over15", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "sinal_live")

    def test_live_intelligence_cancels_under_when_game_opens_early(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "1H", "elapsed": 30}},
            "goals": {"home": 1, "away": 0},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 4},
                    {"type": "Total Shots", "value": 9},
                    {"type": "expected_goals", "value": 0.8},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 6},
                    {"type": "expected_goals", "value": 0.9},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "under35", "watch_type": "blocked_recheck"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "cancelar")


if __name__ == "__main__":
    unittest.main()

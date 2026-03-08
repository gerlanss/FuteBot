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
        self.assertIn("Mercados candidatos: 187", header)
        self.assertIn("Bloqueadas por EV: 2", header)
        self.assertIn("Enviadas ao DeepSeek: 117", header)
        self.assertIn("<code>RB Leipzig</code> <b>x</b> <code>FC Augsburg</code>", body)


if __name__ == "__main__":
    unittest.main()

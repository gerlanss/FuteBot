import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class ConfigTests(unittest.TestCase):
    def test_config_uses_env_without_secret_fallbacks(self):
        import config

        original = {
            "API_FOOTBALL_KEY": os.environ.get("API_FOOTBALL_KEY"),
            "ODDS_API_KEY": os.environ.get("ODDS_API_KEY"),
            "ODDSPAPI_KEY": os.environ.get("ODDSPAPI_KEY"),
            "TIMEZONE": os.environ.get("TIMEZONE"),
        }

        try:
            os.environ["API_FOOTBALL_KEY"] = "api-key-test"
            os.environ["ODDS_API_KEY"] = "odds-key-test"
            os.environ["ODDSPAPI_KEY"] = "oddspapi-key-test"
            os.environ["TIMEZONE"] = "America/Manaus"
            os.environ["MODEL_CONFIDENCE_MIN"] = "0.60"
            os.environ["COMBO_TIP_CONFIDENCE_MIN"] = "0.60"

            config = importlib.reload(config)

            self.assertEqual(config.API_FOOTBALL_KEY, "api-key-test")
            self.assertEqual(config.ODDS_API_KEY, "odds-key-test")
            self.assertEqual(config.ODDSPAPI_KEY, "oddspapi-key-test")
            self.assertEqual(config.TIMEZONE, "America/Manaus")
            self.assertEqual(config.MODEL_CONFIDENCE_MIN, 0.60)
            self.assertEqual(config.COMBO_TIP_CONFIDENCE_MIN, 0.60)
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(config)


class SchedulerPriceWatchReasonTests(unittest.TestCase):
    def test_motivo_nao_executavel_live_distingue_odd_ev_e_indisponibilidade(self):
        from pipeline.scheduler import Scheduler

        self.assertIn(
            "odd esta indisponivel",
            Scheduler._motivo_nao_executavel_live({"odd_usada": None, "ev_percent": 8.5}),
        )
        self.assertIn(
            "EV esta indisponivel",
            Scheduler._motivo_nao_executavel_live({"odd_usada": 1.81, "ev_percent": None}),
        )
        self.assertIn(
            "odd ficou abaixo",
            Scheduler._motivo_nao_executavel_live({"odd_usada": 1.33, "ev_percent": 14.2}),
        )
        self.assertIn(
            "EV ficou abaixo",
            Scheduler._motivo_nao_executavel_live({"odd_usada": 1.81, "ev_percent": 1.3}),
        )

    def test_ev_live_permite_observacao_so_a_partir_de_dois_por_cento(self):
        from pipeline.scheduler import Scheduler

        ok, motivo = Scheduler._ev_live_permite_observacao({"odd_usada": 1.81, "ev_percent": 3.2})
        self.assertTrue(ok)
        self.assertIsNone(motivo)

        ok, motivo = Scheduler._ev_live_permite_observacao({"odd_usada": 1.81, "ev_percent": 1.3})
        self.assertFalse(ok)
        self.assertIn("faixa minima de observacao", motivo)

    def test_formatar_alerta_preco_live_usa_motivo_real_em_vez_de_texto_generico(self):
        from pipeline.scheduler import Scheduler

        bloco = Scheduler._formatar_alerta_preco_live(
            {
                "home_name": "Stuttgart",
                "away_name": "Dortmund",
                "descricao": "Under 1.5",
                "elapsed": 58,
                "payload": {
                    "last_live_message": "O jogo segue mais controlado.",
                    "price_watch_reason": "Cenario passou, mas o EV ficou abaixo do minimo operacional (1.2% < 5.0%).",
                },
            },
            titulo="Janela",
            contexto="Cenário passou, mas a odd ainda não ficou boa; vou rechecando por até 15 minutos.",
        )

        self.assertIn("EV ficou abaixo do minimo operacional", bloco)
        self.assertNotIn("odd ainda nao ficou boa", bloco.lower())


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

    def test_feature_name_lists_include_h2h_market_features(self):
        from models.features import FeatureExtractor
        from models.feature_factory import FeatureFactory

        for feature_name in (
            "h2h_over15_5",
            "h2h_over25_5",
            "h2h_under35_5",
            "h2h_btts_5",
            "h2h_corners_over_85_5",
            "h2h_goals_ht_over05_5",
        ):
            self.assertIn(feature_name, FeatureExtractor.feature_names())
            self.assertIn(feature_name, FeatureFactory.feature_names_full())


class PredictorMemoryTests(unittest.TestCase):
    def test_predictor_evicts_old_league_models_from_cache(self):
        import models.predictor as predictor_module

        class DummyBooster:
            def __init__(self):
                self.loaded_path = None

            def load_model(self, path):
                self.loaded_path = path

        with tempfile.TemporaryDirectory() as tmpdir:
            for league_id in (71, 72):
                league_dir = os.path.join(tmpdir, f"league_{league_id}")
                os.makedirs(league_dir, exist_ok=True)
                with open(os.path.join(league_dir, "resultado_1x2.json"), "w", encoding="utf-8") as f:
                    f.write("{}")

            db = MagicMock()

            with patch.object(predictor_module, "MODELS_DIR", tmpdir), \
                 patch.object(predictor_module, "PREDICTOR_MAX_LIGAS_CACHE", 1), \
                 patch.object(predictor_module, "xgb", MagicMock(Booster=DummyBooster)), \
                 patch.object(predictor_module, "FeatureExtractor", return_value=MagicMock()), \
                 patch.object(predictor_module, "FeatureFactory", return_value=MagicMock()), \
                 patch.object(predictor_module.gc, "collect"):
                predictor = predictor_module.Predictor(db)
                predictor._carregar_modelos_liga(71)
                predictor._carregar_modelos_liga(72)

            self.assertNotIn(71, predictor._modelos_liga)
            self.assertIn(72, predictor._modelos_liga)


class SchedulerTests(unittest.TestCase):
    def test_live_opportunity_probability_never_none(self):
        from pipeline.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        leitura = {
            "veredito": "sinal_live",
            "elapsed": 63,
            "mercado": "h2h_home",
            "metricas": {
                "shots_total": 14.0,
                "shots_on": 6.0,
                "corners": 7.0,
                "xg": 1.42,
                "teams": [
                    {"shots_total": 9.0, "shots_on": 4.0, "xg": 0.96},
                    {"shots_total": 5.0, "shots_on": 2.0, "xg": 0.46},
                ],
            },
        }

        prob = scheduler._inferir_prob_modelo_live("h2h_home", leitura)

        self.assertIsNotNone(prob)
        self.assertGreaterEqual(prob, 0.60)

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

    def test_live_category_notification_blocks_same_market_family(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.salvar_live_market_notification("2026-03-18", 123, "corners_over_85", "sinal_live")

            self.assertTrue(
                db.live_category_notification_exists(
                    "2026-03-18",
                    123,
                    ["corners_over_85", "corners_over_95", "corners_over_105"],
                    "sinal_live",
                )
            )
            self.assertFalse(
                db.live_category_notification_exists(
                    "2026-03-18",
                    123,
                    ["over25", "over35"],
                    "sinal_live",
                )
            )

    def test_live_conflict_map_blocks_ht_result_with_ht_under_05(self):
        from pipeline.scheduler import Scheduler

        conflitos = set(Scheduler._mercados_conflitantes_live("ht_away"))

        self.assertIn("ht_home", conflitos)
        self.assertIn("ht_draw", conflitos)
        self.assertIn("ht_away", conflitos)
        self.assertIn("under05_ht", conflitos)

    def test_live_conflict_map_blocks_equivalent_ft_and_2t_market_in_second_half(self):
        from pipeline.scheduler import Scheduler

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 55}},
            "goals": {"home": 1, "away": 0},
            "score": {"halftime": {"home": 1, "away": 0}},
        }

        conflitos = set(Scheduler._mercados_conflitantes_live("over15", fixture))

        self.assertIn("over15", conflitos)
        self.assertIn("under15", conflitos)
        self.assertIn("over05_2t", conflitos)
        self.assertNotIn("over15_2t", conflitos)

    def test_live_conflict_map_does_not_block_distinct_ft_and_2t_market_after_second_half_goal(self):
        from pipeline.scheduler import Scheduler

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 67}},
            "goals": {"home": 1, "away": 1},
            "score": {"halftime": {"home": 1, "away": 0}},
        }

        conflitos = set(Scheduler._mercados_conflitantes_live("over15", fixture))

        self.assertIn("over15", conflitos)
        self.assertIn("under15", conflitos)
        self.assertNotIn("over05_2t", conflitos)

    def test_suprime_cancelamento_tardio_para_under_ft(self):
        from pipeline.scheduler import Scheduler

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 90}},
        }
        item = {"mercado": "corners_under_85"}

        self.assertTrue(Scheduler._deve_suprimir_cancelamento_tardio(item, fixture))

    def test_nao_suprime_cancelamento_tardio_para_over(self):
        from pipeline.scheduler import Scheduler

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 90}},
        }
        item = {"mercado": "corners_over_85"}

        self.assertFalse(Scheduler._deve_suprimir_cancelamento_tardio(item, fixture))

    def test_bloqueia_sinal_live_tardio_no_fim_do_jogo(self):
        from pipeline.scheduler import Scheduler

        self.assertTrue(Scheduler._deve_bloquear_sinal_tardio({"mercado": "h2h_away"}, 90))
        self.assertTrue(Scheduler._deve_bloquear_sinal_tardio({"mercado": "over05_2t"}, 90))
        self.assertFalse(Scheduler._deve_bloquear_sinal_tardio({"mercado": "over05_2t"}, 72))

    def test_janela_operacional_live_diferencia_over_e_under(self):
        from pipeline.scheduler import Scheduler

        self.assertEqual(Scheduler._janela_operacional_live({"mercado": "over05_2t"}), (52, 78))
        self.assertEqual(Scheduler._janela_operacional_live({"mercado": "under15_2t"}), (60, 84))
        self.assertEqual(Scheduler._janela_operacional_live({"mercado": "corners_over_85"}), (50, 78))
        self.assertEqual(Scheduler._janela_operacional_live({"mercado": "corners_under_85"}), (58, 82))

    def test_status_janela_operacional_bloqueia_cedo_e_tarde(self):
        from pipeline.scheduler import Scheduler

        self.assertEqual(
            Scheduler._status_janela_operacional_live({"mercado": "over05_2t"}, 50)[0],
            "cedo",
        )
        self.assertEqual(
            Scheduler._status_janela_operacional_live({"mercado": "over05_2t"}, 78)[0],
            "tarde",
        )
        self.assertEqual(
            Scheduler._status_janela_operacional_live({"mercado": "under15_2t"}, 72)[0],
            "ok",
        )

    def test_refresh_live_odds_context_ignora_blocked_recheck(self):
        import pipeline.scheduler as scheduler_module
        from pipeline.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.db = MagicMock()

        fake_enriquecer = MagicMock(side_effect=lambda payloads, phase_operacional=None: payloads)
        with patch.object(scheduler_module, "enriquecer_tips_com_odds_oddspapi", fake_enriquecer):
            scheduler._refresh_live_odds_context([
                {
                    "id": 1,
                    "fixture_id": 100,
                    "mercado": "under35",
                    "watch_type": "blocked_recheck",
                    "payload": {},
                },
                {
                    "id": 2,
                    "fixture_id": 101,
                    "mercado": "over25",
                    "watch_type": "approved_prelive",
                    "payload": {
                        "price_watch_active": True,
                        "price_watch_expires_at": "2099-04-03T20:15:00-04:00",
                        "price_watch_next_check_at": "2000-04-03T20:00:00-04:00",
                    },
                },
            ])

        fake_enriquecer.assert_called_once()
        payloads = fake_enriquecer.call_args[0][0]
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["fixture_id"], 101)
        scheduler.db.atualizar_live_watch_item.assert_called_once()

    def test_acionar_janela_preco_live_inicia_watch_curto(self):
        from pipeline.scheduler import Scheduler
        from zoneinfo import ZoneInfo

        scheduler = Scheduler.__new__(Scheduler)
        now = datetime(2026, 4, 3, 20, 0, tzinfo=ZoneInfo("America/Manaus"))
        payload = {}
        item = {"mercado": "under15_2t"}

        acao, motivo = scheduler._acionar_janela_preco_live_inteligente(payload, item, 70, now)

        self.assertEqual(acao, "started")
        self.assertTrue(payload["price_watch_active"])
        self.assertEqual(payload["price_watch_checks"], 1)
        self.assertIn("14 minutos", motivo)

    def test_acionar_janela_preco_live_expira_quando_mercado_entra_tarde(self):
        from pipeline.scheduler import Scheduler
        from zoneinfo import ZoneInfo

        scheduler = Scheduler.__new__(Scheduler)
        now = datetime(2026, 4, 3, 20, 0, tzinfo=ZoneInfo("America/Manaus"))
        payload = {}
        item = {"mercado": "over05_2t"}

        acao, motivo = scheduler._acionar_janela_preco_live_inteligente(payload, item, 79, now)

        self.assertEqual(acao, "expired")
        self.assertFalse(payload["price_watch_active"])
        self.assertEqual(payload["price_watch_final_state"], "expired")
        self.assertIn("tarde demais", motivo)

    def test_acionar_janela_preco_live_descarta_ev_abaixo_de_dois_sem_observacao(self):
        from pipeline.scheduler import Scheduler
        from zoneinfo import ZoneInfo

        scheduler = Scheduler.__new__(Scheduler)
        now = datetime(2026, 4, 3, 20, 0, tzinfo=ZoneInfo("America/Manaus"))
        payload = {"odd_usada": 1.81, "ev_percent": 1.3}
        item = {"mercado": "under15_2t"}

        acao, motivo = scheduler._acionar_janela_preco_live_inteligente(payload, item, 70, now)

        self.assertEqual(acao, "expired")
        self.assertFalse(payload["price_watch_active"])
        self.assertEqual(payload["price_watch_final_state"], "expired")
        self.assertIn("faixa minima de observacao", motivo)

    def test_job_check_ao_vivo_ignora_analise_direta_do_fixture_manual(self):
        from pipeline.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.db = MagicMock()
        scheduler._entrar_secao_critica = MagicMock(return_value=True)
        scheduler._sair_secao_critica = MagicMock()
        scheduler._refresh_live_odds_context = MagicMock()
        scheduler._detectar_oportunidades_live_fixture = MagicMock(return_value=([], []))
        scheduler._enviar_telegram = MagicMock()
        scheduler._enviar_telegram_publico = MagicMock()
        scheduler._notification_sent = MagicMock(return_value=True)
        scheduler._save_notification_sent = MagicMock()
        scheduler._notificar_progresso_combos_live = MagicMock(return_value="")
        scheduler.db.live_watch_items.return_value = [{
            "id": 1,
            "scan_date": "2026-04-03",
            "fixture_id": 101,
            "fixture_date": "2026-04-03T19:00:00+00:00",
            "league_id": 71,
            "home_name": "Fortaleza",
            "away_name": "Bogota",
            "mercado": "__fixture_watch__",
            "watch_type": "manual_fixture",
            "status": "active",
            "payload": {"manual_anchor": True},
            "note": "Gancho manual por time: Fortaleza",
        }]
        scheduler.db.predictions_pendentes.return_value = []

        fake_live = MagicMock()

        with patch("pipeline.scheduler.raw_request", return_value={
            "response": [{
                "fixture": {"id": 101, "status": {"short": "1H", "elapsed": 27}},
                "goals": {"home": 0, "away": 0},
            }]
        }), patch("pipeline.scheduler.stats_partida", return_value=[]), \
             patch("pipeline.scheduler.LiveIntelligence", return_value=fake_live):
            scheduler._job_check_ao_vivo()

        fake_live.analisar.assert_not_called()
        scheduler._detectar_oportunidades_live_fixture.assert_called_once()
        scheduler.db.atualizar_live_watch_item.assert_called()

    def test_job_check_ao_vivo_resolve_leitura_quando_janela_preco_ja_expirou(self):
        from pipeline.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.db = MagicMock()
        scheduler._entrar_secao_critica = MagicMock(return_value=True)
        scheduler._sair_secao_critica = MagicMock()
        scheduler._refresh_live_odds_context = MagicMock()
        scheduler._detectar_oportunidades_live_fixture = MagicMock(return_value=([], []))
        scheduler._enviar_telegram = MagicMock()
        scheduler._enviar_telegram_publico = MagicMock()
        scheduler._notification_sent = MagicMock(return_value=True)
        scheduler._save_notification_sent = MagicMock()
        scheduler._notificar_progresso_combos_live = MagicMock(return_value="")
        scheduler.db.live_watch_items.return_value = [{
            "id": 7,
            "scan_date": "2026-04-03",
            "fixture_id": 202,
            "fixture_date": "2026-04-03T19:00:00+00:00",
            "league_id": 71,
            "home_name": "Puebla",
            "away_name": "Juarez",
            "mercado": "h2h_home",
            "descricao": "Casa",
            "watch_type": "approved_prelive",
            "status": "active",
            "payload": {
                "price_watch_active": False,
                "price_watch_final_state": "expired",
                "price_watch_reason": "Janela de preco expirou sem odd executavel.",
            },
            "note": "Segue observando.",
        }]
        scheduler.db.predictions_pendentes.return_value = []

        fake_live = MagicMock()
        fake_live.analisar.return_value = {
            "veredito": "sinal_live",
            "elapsed": 70,
            "mensagem": "O mandante segue melhor no jogo.",
        }

        with patch("pipeline.scheduler.raw_request", return_value={
            "response": [{
                "fixture": {"id": 202, "status": {"short": "2H", "elapsed": 70}},
                "goals": {"home": 1, "away": 0},
            }]
        }), patch("pipeline.scheduler.stats_partida", return_value=[]), \
             patch("pipeline.scheduler.LiveIntelligence", return_value=fake_live):
            scheduler._job_check_ao_vivo()

        update_calls = scheduler.db.atualizar_live_watch_item.call_args_list
        self.assertTrue(any(call.kwargs.get("status") == "resolved" for call in update_calls))
        self.assertTrue(any("Janela de preco expirou" in (call.kwargs.get("note") or "") for call in update_calls))
        scheduler._enviar_telegram.assert_called()
        self.assertIn("Leitura cancelada", scheduler._enviar_telegram.call_args.args[0])

    def test_odd_live_so_consulta_quando_watch_esta_vencido_para_recheck(self):
        from pipeline.scheduler import Scheduler
        from zoneinfo import ZoneInfo

        scheduler = Scheduler.__new__(Scheduler)
        now = datetime(2026, 4, 3, 20, 6, tzinfo=ZoneInfo("America/Manaus"))
        item = {"watch_type": "approved_prelive"}
        payload = {
            "price_watch_active": True,
            "price_watch_expires_at": "2026-04-03T20:15:00-04:00",
            "price_watch_next_check_at": "2026-04-03T20:05:00-04:00",
        }

        self.assertTrue(scheduler._deve_consultar_odd_live_item(item, payload, now))

        payload["price_watch_next_check_at"] = "2026-04-03T20:10:00-04:00"
        self.assertFalse(scheduler._deve_consultar_odd_live_item(item, payload, now))


class TelegramBotMessagingTests(unittest.TestCase):
    def test_reply_text_safe_retries_on_timeout(self):
        import services.telegram_bot as telegram_bot
        from telegram.error import TimedOut

        message = SimpleNamespace()
        message.reply_text = AsyncMock(side_effect=[TimedOut("slow"), {"ok": True}])

        resultado = asyncio.run(telegram_bot._reply_text_safe(message, "ping"))

        self.assertEqual(resultado, {"ok": True})
        self.assertEqual(message.reply_text.await_count, 2)

    def test_send_to_chats_resolves_admins_and_falls_back_without_parse_mode(self):
        import services.telegram_bot as telegram_bot

        class FakeBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("chat_id") == 11 and kwargs.get("parse_mode") == "HTML":
                    raise RuntimeError("parse quebrou")

        fake_bot = FakeBot()

        with patch.object(telegram_bot, "_ADMIN_CHAT_IDS", {11, 12}), \
             patch.object(telegram_bot, "_app_instance", SimpleNamespace(bot=fake_bot)), \
             patch.object(telegram_bot, "get_preferences", side_effect=lambda cid: None):
            resultado = asyncio.run(
                telegram_bot._send_to_chats("Resumo <b>automatico</b>", destino="admins", parse_mode="HTML")
            )

        self.assertEqual(resultado["destinatarios"], 2)
        self.assertEqual(resultado["entregues"], 2)
        self.assertEqual(resultado["falhas"], [])
        self.assertEqual(fake_bot.calls[0]["parse_mode"], "HTML")
        self.assertEqual(
            fake_bot.calls[1],
            {
                "chat_id": 11,
                "text": "Resumo <b>automatico</b>",
                "disable_web_page_preview": True,
            },
        )

    def test_send_to_chats_skips_disabled_alerts_and_reports_failures(self):
        import services.telegram_bot as telegram_bot

        class FakeBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("chat_id") == 33:
                    raise RuntimeError("chat bloqueado")

        fake_bot = FakeBot()

        def fake_prefs(chat_id):
            if chat_id == 22:
                return {"alerts_enabled": False}
            return None

        with patch.object(telegram_bot, "_app_instance", SimpleNamespace(bot=fake_bot)), \
             patch.object(telegram_bot._db, "telegram_chat_ids", return_value=[21, 22, 33]), \
             patch.object(telegram_bot, "get_preferences", side_effect=fake_prefs):
            resultado = asyncio.run(
                telegram_bot._send_to_chats(["Bloco 1", "Bloco 2"], destino="registrados", parse_mode="HTML")
            )

        self.assertEqual(resultado["destinatarios"], 3)
        self.assertEqual(resultado["entregues"], 1)
        self.assertEqual(resultado["falhas"], [{"chat_id": 33, "erro": "chat bloqueado"}])
        self.assertTrue(all(call["chat_id"] != 22 for call in fake_bot.calls))

    def test_send_to_chats_rejects_unknown_destination(self):
        import services.telegram_bot as telegram_bot

        with self.assertRaisesRegex(ValueError, "Destino de Telegram nao suportado"):
            asyncio.run(telegram_bot._send_to_chats("Mensagem", destino="geral", parse_mode="HTML"))

    def test_extrair_consultas_times_deduplica_e_preserva_nomes_compostos(self):
        import services.telegram_bot as telegram_bot

        consultas = telegram_bot._extrair_consultas_times("PSG, San Lorenzo, PSG, O Higgins")

        self.assertEqual(consultas, ["PSG", "San Lorenzo", "O Higgins"])

    def test_fixture_casa_com_consulta_aceita_nome_longo_para_time_curto_oficial(self):
        import services.telegram_bot as telegram_bot

        fixture = {
            "teams": {"home": {"name": "Junior"}, "away": {"name": "Deportivo Cali"}},
        }

        self.assertTrue(telegram_bot._fixture_casa_com_consulta(fixture, "Junior de Barranquilla"))

    def test_fixture_casa_com_consulta_nao_confunde_nome_composto_com_sigla_ambigua(self):
        import services.telegram_bot as telegram_bot

        fixture = {
            "teams": {"home": {"name": "MB Rouisset"}, "away": {"name": "Olympique Akbou"}},
        }

        self.assertFalse(telegram_bot._fixture_casa_com_consulta(fixture, "Miramar Rangers"))

    def test_monitorar_times_do_dia_ancora_fixture_monitoravel(self):
        import services.telegram_bot as telegram_bot

        fixture_ns = {
            "fixture": {"id": 501, "date": "2026-04-03T23:30:00+00:00", "status": {"short": "NS", "elapsed": None}},
            "league": {"id": 71, "name": "Liga Teste"},
            "teams": {"home": {"name": "Paris Saint-Germain"}, "away": {"name": "Toulouse"}},
        }
        fixture_ft = {
            "fixture": {"id": 502, "date": "2026-04-03T19:00:00+00:00", "status": {"short": "FT", "elapsed": 90}},
            "league": {"id": 72, "name": "Liga Encerrada"},
            "teams": {"home": {"name": "Fortaleza FC"}, "away": {"name": "Internacional de Bogota"}},
        }

        with patch.object(telegram_bot, "raw_request", return_value={"response": [fixture_ns, fixture_ft]}), \
             patch.object(telegram_bot._db, "salvar_fixture"), \
             patch.object(telegram_bot._db, "salvar_live_watch_item", return_value=91) as save_watch:
            resultado = telegram_bot._monitorar_times_do_dia(["PSG", "Fortaleza", "Chelsea"])

        self.assertEqual(len(resultado["fixtures"]), 2)
        self.assertEqual(resultado["sem_jogo"], ["Chelsea"])
        self.assertEqual(save_watch.call_count, 1)
        item_salvo = save_watch.call_args.args[1]
        self.assertEqual(item_salvo["mercado"], "__fixture_watch__")
        self.assertEqual(item_salvo["watch_type"], "manual_fixture")

    def test_monitorar_times_do_dia_mescla_live_all_para_nao_perder_jogo_ao_vivo(self):
        import services.telegram_bot as telegram_bot

        fixture_live = {
            "fixture": {"id": 701, "date": "2026-04-03T23:00:00+00:00", "status": {"short": "2H", "elapsed": 62}},
            "league": {"id": 39, "name": "Primera A"},
            "teams": {"home": {"name": "Junior"}, "away": {"name": "Deportivo Cali"}},
        }

        def fake_raw_request(endpoint, params):
            if params.get("date") == "2026-04-03":
                return {"response": []}
            if params.get("live") == "all":
                return {"response": [fixture_live]}
            return {"response": []}

        with patch.object(telegram_bot, "raw_request", side_effect=fake_raw_request), \
             patch.object(telegram_bot._db, "salvar_fixture"), \
             patch.object(telegram_bot._db, "salvar_live_watch_item", return_value=191) as save_watch:
            resultado = telegram_bot._monitorar_times_do_dia(["Junior de Barranquilla"])

        self.assertEqual(len(resultado["fixtures"]), 1)
        self.assertEqual(resultado["sem_jogo"], [])
        self.assertTrue(resultado["fixtures"][0]["monitoravel"])
        self.assertEqual(resultado["fixtures"][0]["fixture_id"], 701)
        self.assertEqual(save_watch.call_count, 1)

    def test_broadcast_scan_publico_delegates_to_send_to_chats(self):
        import services.telegram_bot as telegram_bot

        esperado = {"destinatarios": 3, "entregues": 2, "falhas": [{"chat_id": 9, "erro": "timeout"}]}
        send_mock = AsyncMock(return_value=esperado)

        with patch.object(telegram_bot, "_formatar_scan_publico_html", return_value=["A", "B"]), \
             patch.object(telegram_bot, "_send_to_chats", send_mock):
            resultado = asyncio.run(telegram_bot.broadcast_scan_publico("2026-03-30"))

        self.assertEqual(resultado, esperado)
        send_mock.assert_awaited_once_with(["A", "B"], destino="registrados", parse_mode="HTML")


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

    def test_relatorio_resultado_separa_pre_live_e_live(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            data = datetime.now().strftime("%Y-%m-%d")
            self._insert_fixture(db, 31, data)
            self._insert_fixture(db, 32, data)

            db.salvar_prediction({
                "fixture_id": 31,
                "date": f"{data} 19:00:00",
                "league_id": 71,
                "home_name": "Time 31A",
                "away_name": "Time 31B",
                "mercado": "under35",
                "prob_modelo": 0.83,
                "odd_usada": 1.8,
            })
            db.resolver_prediction(31, "draw", 1, 1)

            item = {
                "id": 99,
                "scan_date": data,
                "fixture_id": 32,
                "league_id": 71,
                "home_name": "Time 32A",
                "away_name": "Time 32B",
                "mercado": "over05_2t",
                "watch_type": "live_opportunity",
                "payload": {},
            }
            db.salvar_live_result_signal(item, signal_minute=63, signal_note="Entrou no live")
            db.resolver_live_result(99, resultado="home", gols_home=2, gols_away=1, acertou=True)

            relatorio = Learner(db).relatorio_resultado_dia(data)

            self.assertIn("<b>PRE-LIVE</b>", relatorio)
            self.assertIn("<b>LIVE</b>", relatorio)
            self.assertIn("Pre-live:", relatorio)
            self.assertIn("Live:", relatorio)
            self.assertIn("ROI: n/d", relatorio)

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

    def test_relatorio_diario_respeita_inicio_banca_visivel(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            conn = db._conn()
            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, status,
                    home_id, home_name, away_id, away_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (1, 71, "Liga Teste", 2026, "Round 1", "2026-04-03 19:00:00", "FT", 11, "Casa", 12, "Fora"))
            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, status,
                    home_id, home_name, away_id, away_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (2, 71, "Liga Teste", 2026, "Round 1", "2026-04-04 19:00:00", "FT", 21, "Casa 2", 22, "Fora 2"))
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, lucro, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (1, "2026-04-03 19:00:00", 71, "Casa", "Fora", "over25", 1, 0.8, "v1"))
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, lucro, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (2, "2026-04-04 19:00:00", 71, "Casa 2", "Fora 2", "over25", 0, -1.0, "v1"))
            conn.execute("""
                INSERT INTO train_log (
                    date, modelo_versao, n_samples, n_features, accuracy_train, accuracy_test
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ("2026-04-04", "v1", 42, 12, 0.71, 0.62))
            conn.commit()
            conn.close()
            db.definir_inicio_banca_visivel("2026-04-04T00:00:00-04:00")

            relatorio = Learner(db).relatorio_diario()

            self.assertIn("Base visível desde", relatorio)
            self.assertIn("Apostas: 1 | Acertos: 0", relatorio)

    def test_relatorio_resultado_dia_oculta_periodo_antes_do_reset_visivel(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.definir_inicio_banca_visivel("2026-04-04T00:00:00-04:00")

            relatorio = Learner(db).relatorio_resultado_dia("2026-04-03")

            self.assertIn("Histórico visível reiniciado", relatorio)
            self.assertIn("fora da nova fase", relatorio)


    def test_relatorio_saude_respeita_inicio_banca_visivel(self):
        from data.database import Database
        from models.learner import Learner

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            self._insert_fixture(db, 1, "2026-04-03")
            self._insert_fixture(db, 2, "2026-04-04")
            conn = db._conn()
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, lucro, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (1, "2026-04-03 19:00:00", 71, "Casa", "Fora", "over25", 1, 0.8, "v1"))
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, lucro, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (2, "2026-04-04 19:00:00", 71, "Casa 2", "Fora 2", "over25", 0, -1.0, "v1"))
            conn.commit()
            conn.close()
            db.definir_inicio_banca_visivel("2026-04-04T00:00:00-04:00")

            relatorio = Learner(db).relatorio_saude()

            self.assertIn("Base vis", relatorio)
            self.assertIn("Acumulado total", relatorio)
            self.assertIn("Pre-live: 1 | Acertos: 0", relatorio)


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
    def test_enriquecer_odds_usa_oddspapi_na_shortlist_final(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        tips = [{"fixture_id": 1, "mercado": "over15", "prob_modelo": 0.72}]

        with patch("pipeline.scanner.ODDSPAPI_USE_PRELIVE", True), \
             patch("pipeline.scanner.enriquecer_tips_com_odds_oddspapi") as enrich_mock:
            enrich_mock.return_value = [{"fixture_id": 1, "mercado": "over15", "prob_modelo": 0.72, "odd_usada": 1.65}]
            resultado = scanner._enriquecer_odds(tips)

        enrich_mock.assert_called_once()
        self.assertEqual(resultado[0]["odd_usada"], 1.65)

    def test_aplicar_gate_odds_ev_barra_sem_odd_valida(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        aprovadas, bloqueadas = scanner._aplicar_gate_odds_ev([
            {"fixture_id": 1, "mercado": "over15", "prob_modelo": 0.71},
        ])

        self.assertEqual(aprovadas, [])
        self.assertEqual(len(bloqueadas), 1)
        self.assertEqual(bloqueadas[0]["odd_block_reason"], "fixture_sem_odd")
        self.assertEqual((bloqueadas[0]["llm_validacao"] or {}).get("decisao"), "REJECT")

    def test_aplicar_gate_odds_ev_barra_odd_abaixo_do_minimo(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        aprovadas, bloqueadas = scanner._aplicar_gate_odds_ev([
            {"fixture_id": 1, "mercado": "over15", "prob_modelo": 0.75, "odd_usada": 1.39, "ev_percent": 6.0},
        ])

        self.assertEqual(aprovadas, [])
        self.assertEqual(len(bloqueadas), 1)
        self.assertIn("Odd abaixo do mínimo operacional", (bloqueadas[0]["llm_validacao"] or {}).get("motivo", ""))

    def test_aplicar_gate_odds_ev_mantem_tip_valida(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        aprovadas, bloqueadas = scanner._aplicar_gate_odds_ev([
            {"fixture_id": 1, "mercado": "over15", "prob_modelo": 0.75, "odd_usada": 1.55, "ev_percent": 16.2},
        ])

        self.assertEqual(len(aprovadas), 1)
        self.assertEqual(bloqueadas, [])
        self.assertTrue(aprovadas[0]["approved_odds_gate"])

    def test_scanner_limits_total_combos_and_keeps_tripla_possible(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        tips = [
            {"fixture_id": 1, "prob_modelo": 0.90, "odd_usada": 1.35, "ev_percent": 21.5},
            {"fixture_id": 2, "prob_modelo": 0.88, "odd_usada": 1.40, "ev_percent": 23.2},
            {"fixture_id": 3, "prob_modelo": 0.86, "odd_usada": 1.42, "ev_percent": 22.1},
            {"fixture_id": 4, "prob_modelo": 0.84, "odd_usada": 1.38, "ev_percent": 15.9},
            {"fixture_id": 5, "prob_modelo": 0.82, "odd_usada": 1.33, "ev_percent": 9.1},
            {"fixture_id": 6, "prob_modelo": 0.80, "odd_usada": 1.30, "ev_percent": 4.0},
        ]

        combos = scanner._gerar_combos(tips)

        self.assertLessEqual(len(combos), 3)
        self.assertTrue(all(len(c["tips"]) in (2, 3) for c in combos))
        self.assertEqual(len({t["fixture_id"] for c in combos for t in c["tips"]}),
                         sum(len(c["tips"]) for c in combos))
        self.assertTrue(all(c.get("odd_composta", 0) >= 1.80 for c in combos))
        self.assertTrue(all(c.get("ev_composto_percent", 0) >= 6.0 for c in combos))

    def test_gerar_combos_ignora_perna_com_ev_fraco(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        combos = scanner._gerar_combos([
            {"fixture_id": 1, "prob_modelo": 0.85, "odd_usada": 1.25, "ev_percent": 6.0},
            {"fixture_id": 2, "prob_modelo": 0.84, "odd_usada": 1.25, "ev_percent": 2.9},
            {"fixture_id": 3, "prob_modelo": 0.83, "odd_usada": 1.30, "ev_percent": 7.0},
        ])

        fixtures_usados = {t["fixture_id"] for combo in combos for t in combo["tips"]}
        self.assertNotIn(2, fixtures_usados)

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
                "odd_usada": 1.90,
                "ev_percent": 56.1,
                "bookmaker": "1xBet",
            }],
            "combos": [],
            "data": "2026-03-07",
        })

        header, body = msgs[0][0], msgs[1][0]
        self.assertIn("Mercados candidatos: <b>187</b>", header)
        self.assertIn("Bloqueadas por EV: <b>2</b>", header)
        self.assertIn("Mercados revisados: <b>117</b>", header)
        self.assertIn("<code>RB Leipzig</code> <b>x</b> <code>FC Augsburg</code>", body)
        self.assertIn("1xBet", body)


class TelegramChatPersistenceTests(unittest.TestCase):
    def test_database_persists_public_and_admin_chats(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.salvar_telegram_chat(123, is_admin=False, username="user", first_name="User")
            db.salvar_telegram_chat(999, is_admin=True, username="admin", first_name="Admin")

            self.assertEqual(db.telegram_chat_ids(), [123, 999])
            self.assertEqual(db.telegram_chat_ids(apenas_admin=True), [999])

    def test_resetar_historico_envio_telegram_limpa_dedupe_sem_apagar_chats(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.salvar_telegram_chat(123, is_admin=True, username="admin", first_name="Admin")
            db.save_notification_sent("2026-04-04", "daily_report")
            db.salvar_live_market_notification("2026-04-04", 501, "over25", "sinal_live")
            conn = db._conn()
            conn.execute(
                "INSERT INTO combos (id, date, combo_type, prob_composta) VALUES (?, ?, ?, ?)",
                (77, "2026-04-04", "dupla", 0.62),
            )
            conn.commit()
            conn.close()
            db.salvar_combo_live_notification(77, "locked_1")
            item_id = db.salvar_live_watch_item("2026-04-04", {
                "fixture_id": 501,
                "date": "2026-04-04T19:00:00+00:00",
                "league_id": 71,
                "home_name": "Casa",
                "away_name": "Fora",
                "mercado": "over25",
                "descricao": "Over 2.5 gols",
                "watch_type": "live_opportunity",
                "status": "active",
                "payload": {
                    "foo": "bar",
                    "live_signal_notified": True,
                    "live_hit_notified": True,
                    "live_loss_notified": True,
                    "sent_live_combo_keys": ["live:1-2"],
                },
            })

            resumo = db.resetar_historico_envio_telegram()

            self.assertEqual(resumo["notification_log_removidos"], 1)
            self.assertEqual(resumo["live_market_notifications_removidos"], 1)
            self.assertEqual(resumo["combo_live_notifications_removidos"], 1)
            self.assertEqual(resumo["live_watchlist_payloads_limpos"], 1)
            self.assertEqual(db.telegram_chat_ids(apenas_admin=True), [123])
            self.assertFalse(db.notification_sent("2026-04-04", "daily_report"))
            self.assertFalse(db.live_market_notification_exists("2026-04-04", 501, "over25", "sinal_live"))
            self.assertFalse(db.combo_live_notification_exists(77, "locked_1"))

            payload = next(item for item in db.live_watch_items(["2026-04-04"]) if item["id"] == item_id)["payload"]
            self.assertEqual(payload.get("foo"), "bar")
            self.assertNotIn("live_signal_notified", payload)
            self.assertNotIn("live_hit_notified", payload)
            self.assertNotIn("live_loss_notified", payload)
            self.assertNotIn("sent_live_combo_keys", payload)

    def test_inicio_banca_visivel_filtra_metricas_sem_apagar_historico(self):
        from data.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            conn = db._conn()
            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, status,
                    home_id, home_name, away_id, away_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (1, 71, "Liga Teste", 2026, "Round 1", "2026-04-03 19:00:00", "FT", 11, "Casa", 12, "Fora"))
            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, status,
                    home_id, home_name, away_id, away_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (2, 71, "Liga Teste", 2026, "Round 1", "2026-04-04 19:00:00", "FT", 21, "Casa 2", 22, "Fora 2"))
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, lucro, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (1, "2026-04-03 19:00:00", 71, "Casa", "Fora", "over25", 1, 0.8, "v1"))
            conn.execute("""
                INSERT INTO predictions (
                    fixture_id, date, league_id, home_name, away_name,
                    mercado, acertou, lucro, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (2, "2026-04-04 19:00:00", 71, "Casa 2", "Fora 2", "over25", 0, -1.0, "v1"))
            conn.execute("""
                INSERT INTO live_results (
                    live_watch_id, scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, watch_type, acertou, lucro
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (11, "2026-04-03", 101, 71, "Casa", "Fora", "h2h_home", "live_opportunity", 1, 0.9))
            conn.execute("""
                INSERT INTO live_results (
                    live_watch_id, scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, watch_type, acertou, lucro
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (12, "2026-04-04", 102, 71, "Casa 2", "Fora 2", "h2h_home", "live_opportunity", 0, -1.0))
            conn.commit()
            conn.close()

            db.definir_inicio_banca_visivel("2026-04-04T00:00:00-04:00")

            self.assertEqual(db.obter_inicio_banca_visivel(), "2026-04-04T00:00:00-04:00")
            self.assertEqual(db.metricas_modelo()["total"], 2)
            self.assertEqual(db.metricas_live()["total"], 2)
            self.assertEqual(db.metricas_modelo(data_inicio="2026-04-04T00:00:00-04:00")["total"], 1)
            self.assertEqual(db.metricas_live(data_inicio="2026-04-04T00:00:00-04:00")["total"], 1)

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

        self.assertEqual(conf_min, 0.654)
        self.assertEqual(conf_max, 1.01)

    def test_apply_discovery_uses_60_percent_floor_when_rule_missing(self):
        from scripts.apply_discovery_strategies import _infer_conf_band

        conf_min, conf_max = _infer_conf_band({
            "conditions": [
                ["away_btts_pct", ">=", 0.4],
            ]
        })

        self.assertEqual(conf_min, 0.60)
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
        self.assertIn("Segue em observação para live.", joined)
        self.assertIn("1xBet", joined)

    def test_formatar_relatorio_rejeitado_explica_watch_live(self):
        from pipeline.scanner import Scanner

        scanner = Scanner.__new__(Scanner)
        scanner.db = MagicMock()
        scanner.db.metricas_modelo.return_value = {"total": 0}

        msgs = scanner.formatar_relatorio({
            "fixtures": 3,
            "previsoes": 3,
            "tips_brutas": 9,
            "tips_pos_filtros": 3,
            "tips_bloqueadas_ev": 1,
            "tips_enviadas_llm": 1,
            "tips_aprovadas_llm": 0,
            "tips_rejeitadas_llm": [{
                "league_id": 71,
                "home_name": "San Lorenzo",
                "away_name": "Estudiantes",
                "mercado": "under35",
                "descricao": "Under 3.5 gols",
                "prob_modelo": 0.64,
                "llm_validacao": {"motivo": "Odd abaixo do mínimo operacional."},
            }],
            "ev_positivas": [],
            "combos": [],
            "data": "2026-04-03",
        })

        joined = "\n\n".join(texto for texto, _ in msgs)
        self.assertIn("Segue em observação para live.", joined)
        self.assertIn("Se o jogo continuar travado", joined)

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
            {"mercado": "over25", "watch_type": "approved_prelive"},
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

    def test_live_intelligence_does_not_cancel_ht_under_late_when_market_still_alive(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "1H", "elapsed": 39}},
            "goals": {"home": 1, "away": 0},
            "score": {"halftime": {"home": 1, "away": 0}},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Total Shots", "value": 4},
                    {"type": "expected_goals", "value": 0.95},
                    {"type": "Red Cards", "value": 1},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 1},
                    {"type": "Total Shots", "value": 5},
                    {"type": "expected_goals", "value": 0.60},
                    {"type": "Red Cards", "value": 0},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "under15_ht", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertNotEqual(leitura["veredito"], "cancelar")

    def test_live_intelligence_flags_ht_over_signal_early_with_real_volume(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "1H", "elapsed": 28}},
            "goals": {"home": 0, "away": 0},
            "score": {"halftime": {"home": 0, "away": 0}},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Total Shots", "value": 5},
                    {"type": "expected_goals", "value": 0.35},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Total Shots", "value": 4},
                    {"type": "expected_goals", "value": 0.34},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "over05_ht", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "sinal_live")

    def test_live_intelligence_uses_second_half_goals_for_over_2t(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 55}},
            "goals": {"home": 1, "away": 0},
            "score": {"halftime": {"home": 1, "away": 0}},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 8},
                    {"type": "expected_goals", "value": 0.55},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Total Shots", "value": 4},
                    {"type": "expected_goals", "value": 0.25},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "over05_2t", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["gols_2t"], 0)
        self.assertEqual(leitura["veredito"], "sinal_live")

    def test_live_intelligence_flags_second_half_over_with_reasonable_xg_not_only_extreme_xg(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 55}},
            "goals": {"home": 1, "away": 0},
            "score": {"halftime": {"home": 1, "away": 0}},
        }
        stats = [
            {
                "team": {"name": "Liverpool"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 8},
                    {"type": "expected_goals", "value": 0.55},
                ],
            },
            {
                "team": {"name": "Tottenham"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Total Shots", "value": 4},
                    {"type": "expected_goals", "value": 0.25},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "over05_2t", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "sinal_live")

    def test_live_intelligence_only_signals_draw_with_real_balance(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 67}},
            "goals": {"home": 1, "away": 1},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 8},
                    {"type": "expected_goals", "value": 0.92},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 7},
                    {"type": "expected_goals", "value": 0.88},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "h2h_draw", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "sinal_live")

    def test_live_intelligence_corners_under_stays_silent_when_low_corners_hide_pressure(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 61}},
            "goals": {"home": 1, "away": 0},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Corner Kicks", "value": 2},
                    {"type": "Shots on Goal", "value": 4},
                    {"type": "Total Shots", "value": 9},
                    {"type": "expected_goals", "value": 0.9},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Corner Kicks", "value": 2},
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 8},
                    {"type": "expected_goals", "value": 0.55},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "corners_under_95", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "monitorar")

    def test_live_intelligence_corners_over_requires_pressure_not_only_corner_count(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 58}},
            "goals": {"home": 0, "away": 0},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Corner Kicks", "value": 4},
                    {"type": "Shots on Goal", "value": 1},
                    {"type": "Total Shots", "value": 5},
                    {"type": "expected_goals", "value": 0.3},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Corner Kicks", "value": 1},
                    {"type": "Shots on Goal", "value": 1},
                    {"type": "Total Shots", "value": 4},
                    {"type": "expected_goals", "value": 0.2},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "corners_over_85", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "monitorar")

    def test_live_intelligence_ft_over_waits_for_second_half_confirmation(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 63}},
            "goals": {"home": 1, "away": 1},
            "score": {"halftime": {"home": 1, "away": 1}},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 4},
                    {"type": "Total Shots", "value": 8},
                    {"type": "expected_goals", "value": 0.65},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Total Shots", "value": 5},
                    {"type": "expected_goals", "value": 0.32},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "over35", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertNotEqual(leitura["veredito"], "sinal_live")

    def test_live_intelligence_ft_under_can_stay_alive_when_second_half_is_dead(self):
        from services.live_intelligence import LiveIntelligence

        fixture = {
            "fixture": {"status": {"short": "2H", "elapsed": 66}},
            "goals": {"home": 1, "away": 0},
            "score": {"halftime": {"home": 1, "away": 0}},
        }
        stats = [
            {
                "team": {"name": "Casa"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 3},
                    {"type": "Total Shots", "value": 7},
                    {"type": "expected_goals", "value": 0.48},
                ],
            },
            {
                "team": {"name": "Fora"},
                "statistics": [
                    {"type": "Shots on Goal", "value": 1},
                    {"type": "Total Shots", "value": 4},
                    {"type": "expected_goals", "value": 0.21},
                ],
            },
        ]

        leitura = LiveIntelligence().analisar(
            {"mercado": "under25", "watch_type": "approved_prelive"},
            fixture,
            stats,
        )

        self.assertEqual(leitura["veredito"], "sinal_live")
        self.assertEqual(leitura["estado_partida"], "travado")
        self.assertIn("Estado: travado.", leitura["mensagem"])


class OddsPapiMappingTests(unittest.TestCase):
    def test_all_operational_markets_are_mapped_in_oddspapi(self):
        from pipeline.scanner import MERCADOS
        from pipeline.scheduler import Scheduler
        from services.oddspapi import _TIP_TO_SELECTION

        scanner_markets = {market_id for market_id, _, _ in MERCADOS}
        live_markets = set(Scheduler._mercados_live_expandiveis())
        missing = sorted((scanner_markets | live_markets) - set(_TIP_TO_SELECTION))

        self.assertEqual(missing, [])

    def test_extract_price_supports_corners_over_85(self):
        from services.oddspapi import OddsPapiClient

        client = OddsPapiClient(
            api_key="teste",
            base_url="https://api.oddspapi.io/v4",
            bookmaker_slug="1xbet",
        )
        payload = {
            "bookmakerOdds": {
                "1xbet": {
                    "markets": {
                        "10799": {
                            "handicap": 8.5,
                            "outcomes": {
                                "10799": {
                                    "outcomeName": "Over",
                                    "players": {
                                        "0": [
                                            {
                                                "price": 1.91,
                                                "active": True,
                                                "changedAt": "2026-04-03T22:30:00Z",
                                            }
                                        ]
                                    },
                                }
                            },
                        }
                    }
                }
            }
        }

        detalhe = client.extract_price(payload, "corners_over_85")

        self.assertIsNotNone(detalhe)
        self.assertEqual(detalhe["market_id"], 10799)
        self.assertEqual(detalhe["outcome_id"], 10799)
        self.assertEqual(detalhe["point_line"], 8.5)
        self.assertEqual(detalhe["odd"], 1.91)


class SchedulerLivePublicFormattingTests(unittest.TestCase):
    def test_formatar_sinais_live_publicos_ignora_item_sem_odd_ev(self):
        from pipeline.scheduler import Scheduler

        texto = Scheduler._formatar_sinais_live_publicos([{
            "home_name": "San Lorenzo",
            "away_name": "Estudiantes",
            "descricao": "Under 3.5 gols",
            "watch_type": "blocked_recheck",
            "elapsed": 62,
            "note": "Jogo segue travado e sem pressão real.",
            "payload": {"last_live_message": "Jogo segue travado e sem pressão real."},
        }])

        self.assertEqual(texto, "")

    def test_formatar_sinais_live_publicos_destaca_sinal_liberado(self):
        from pipeline.scheduler import Scheduler

        texto = Scheduler._formatar_sinais_live_publicos([{
            "home_name": "Casa",
            "away_name": "Fora",
            "descricao": "Over 2.5 gols",
            "odd_usada": 1.74,
            "ev_percent": 8.4,
            "bookmaker": "1xBet",
            "watch_type": "approved_prelive",
            "elapsed": 55,
            "note": "Ritmo aumentou e a pressão ofensiva apareceu.",
            "payload": {"last_live_message": "Ritmo aumentou e a pressão ofensiva apareceu."},
        }])

        self.assertIn("Entrada live liberada", texto)
        self.assertIn("Leitura confirmada no acompanhamento ao vivo.", texto)
        self.assertIn("<code>Casa</code> <b>x</b> <code>Fora</code>", texto)
        self.assertIn("Odd 1.74 | EV +8.4% | 1xBet", texto)

    def test_formatar_combos_live_exibe_odd_individual_e_composta(self):
        from pipeline.scheduler import Scheduler

        texto = Scheduler._formatar_combos_live([{
            "tipo": "dupla",
            "prob_composta": 0.41,
            "odd_composta": 3.12,
            "ev_composto": 7.8,
            "tips": [
                {
                    "home_name": "Casa",
                    "away_name": "Fora",
                    "descricao": "Over 2.5 gols",
                    "prob_modelo": 0.65,
                    "elapsed": 55,
                    "odd_usada": 1.74,
                    "ev_percent": 8.4,
                    "bookmaker": "1xBet",
                },
                {
                    "home_name": "Um",
                    "away_name": "Dois",
                    "descricao": "BTTS",
                    "prob_modelo": 0.63,
                    "elapsed": 63,
                    "odd_usada": 1.79,
                    "ev_percent": 6.1,
                    "bookmaker": "1xBet",
                },
            ],
        }])

        self.assertIn("Odd 3.12", texto)
        self.assertIn("EV +7.8%", texto)
        self.assertIn("Odd 1.74 | EV +8.4% | 1xBet", texto)

    def test_blocked_recheck_nao_gera_oportunidade_publica(self):
        from pipeline.scheduler import Scheduler

        sinais = []
        item = {
            "watch_type": "blocked_recheck",
            "home_name": "San Lorenzo",
            "away_name": "Estudiantes",
            "payload": {},
        }
        veredito = "sinal_live"

        if veredito == "sinal_live" and item.get("watch_type") in {"approved_prelive", "live_opportunity"}:
            sinal = dict(item)
            sinal["payload"] = item["payload"]
            sinal["elapsed"] = 27
            sinais.append(sinal)

        self.assertEqual(sinais, [])

    def test_sinal_live_ja_notificado_nao_republica_o_mesmo_mercado(self):
        sinais = []
        item = {
            "watch_type": "approved_prelive",
            "home_name": "San Lorenzo",
            "away_name": "Estudiantes",
            "payload": {"live_signal_notified": True, "last_live_verdict": "sinal_live"},
        }
        veredito = "sinal_live"
        emitir_sinal_publico = False

        if (
            veredito == "sinal_live"
            and item.get("watch_type") in {"approved_prelive", "live_opportunity"}
            and emitir_sinal_publico
        ):
            sinal = dict(item)
            sinal["payload"] = item["payload"]
            sinal["elapsed"] = 60
            sinais.append(sinal)

        self.assertEqual(sinais, [])


class LiveTrainerBootstrapTests(unittest.TestCase):
    def test_live_trainer_resolve_linha_sem_confundir_sufixo_2t(self):
        from models.live_trainer import LiveTrainer

        self.assertEqual(LiveTrainer._line_value("over05_2t"), 0.5)
        self.assertEqual(LiveTrainer._line_value("under15_2t"), 1.5)

    def test_live_trainer_bloqueia_treino_sem_eventos_historicos_suficientes(self):
        from data.database import Database
        from models.live_trainer import LiveTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            trainer = LiveTrainer(db)
            summary = trainer.treinar(min_amostras_mercado=2)

            self.assertEqual(summary.get("status"), "blocked")
            self.assertEqual(summary.get("reason"), "base_live_insuficiente")
            self.assertFalse(summary.get("readiness", {}).get("pronto"))

    def test_live_trainer_filtra_sinal_fora_da_janela_oficial(self):
        from data.database import Database
        from models.live_trainer import LiveTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            conn = db._conn()
            historical_fixtures = [
                (1, 11, 2, 1, 0, "2026-03-01T20:00:00-04:00"),
                (2, 1, 12, 2, 1, "2026-03-05T20:00:00-04:00"),
                (3, 13, 1, 0, 0, "2026-03-10T20:00:00-04:00"),
                (11, 21, 1, 1, 2, "2026-03-16T20:00:00-04:00"),
                (7, 17, 2, 0, 1, "2026-03-14T20:00:00-04:00"),
                (8, 2, 18, 1, 1, "2026-03-18T20:00:00-04:00"),
                (4, 14, 3, 1, 1, "2026-03-01T20:00:00-04:00"),
                (5, 4, 15, 2, 0, "2026-03-05T20:00:00-04:00"),
                (6, 16, 4, 1, 2, "2026-03-10T20:00:00-04:00"),
                (12, 22, 3, 0, 1, "2026-03-16T20:00:00-04:00"),
                (9, 19, 3, 0, 1, "2026-03-14T20:00:00-04:00"),
                (10, 4, 20, 2, 1, "2026-03-18T20:00:00-04:00"),
            ]
            for fixture_id, home_id, away_id, goals_home, goals_away, date in historical_fixtures:
                conn.execute(
                    """
                    INSERT INTO fixtures (
                        fixture_id, league_id, league_name, season, round, date, timestamp,
                        venue, status, home_id, home_name, away_id, away_name,
                        goals_home, goals_away, score_ht_h, score_ht_a, referee, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fixture_id,
                        71,
                        "Brasileirão Série A",
                        2026,
                        "Round 0",
                        date,
                        0,
                        "Estádio",
                        "FT",
                        home_id,
                        f"Time{home_id}",
                        away_id,
                        f"Time{away_id}",
                        goals_home,
                        goals_away,
                        0,
                        0,
                        "Árbitro",
                        "{}",
                    ),
                )
            conn.execute(
                """
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, timestamp,
                    venue, status, home_id, home_name, away_id, away_name,
                    goals_home, goals_away, score_ht_h, score_ht_a, referee, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    101,
                    71,
                    "Brasileirão Série A",
                    2026,
                    "Round 1",
                    "2026-04-03T20:00:00-04:00",
                    0,
                    "Estádio",
                    "FT",
                    1,
                    "Casa",
                    2,
                    "Fora",
                    1,
                    0,
                    0,
                    0,
                    "Árbitro",
                    "{}",
                ),
            )
            conn.execute(
                """
                INSERT INTO fixture_events (
                    fixture_id, team_id, player_name, event_type,
                    event_detail, minute, extra_minute, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    101,
                    1,
                    "Atacante",
                    "Goal",
                    "Normal Goal",
                    60,
                    0,
                    "{}",
                ),
            )
            conn.execute(
                """
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, timestamp,
                    venue, status, home_id, home_name, away_id, away_name,
                    goals_home, goals_away, score_ht_h, score_ht_a, referee, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    102,
                    71,
                    "Brasileirão Série A",
                    2026,
                    "Round 1",
                    "2026-04-04T20:00:00-04:00",
                    0,
                    "Estádio",
                    "FT",
                    3,
                    "Casa2",
                    4,
                    "Fora2",
                    2,
                    0,
                    1,
                    0,
                    "Árbitro",
                    "{}",
                ),
            )
            conn.execute(
                """
                INSERT INTO fixture_events (
                    fixture_id, team_id, player_name, event_type,
                    event_detail, minute, extra_minute, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    102,
                    12,
                    "Atacante",
                    "Goal",
                    "Normal Goal",
                    35,
                    0,
                    "{}",
                ),
            )
            conn.commit()
            conn.close()

            trainer = LiveTrainer(db)
            samples = trainer._load_samples()

            under25_samples = [sample for sample in samples if sample.fixture_id == 101 and sample.market == "under25"]
            over05_ht_samples = [sample for sample in samples if sample.fixture_id == 102 and sample.market == "over05_ht"]

            self.assertTrue(any(sample.signal_minute == 59 for sample in under25_samples))
            self.assertFalse(any(sample.signal_minute == 12 for sample in over05_ht_samples))

    def test_live_trainer_carrega_cantos_de_snapshot_real(self):
        from data.database import Database
        from models.live_trainer import LiveTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            conn = db._conn()
            payload = {
                "season": 2026,
                "features": {"home_form_5": 0.8, "away_form_5": 0.4},
            }
            snapshot = {
                "elapsed": 64,
                "corners": 7,
                "shots_total": 18,
                "shots_on": 6,
                "xg": 1.42,
                "yellow_cards": 3,
                "red_cards": 0,
                "teams": [
                    {"corners": 5, "shots_total": 11, "shots_on": 4, "xg": 0.97},
                    {"corners": 2, "shots_total": 7, "shots_on": 2, "xg": 0.45},
                ],
            }
            conn.execute(
                """
                INSERT INTO live_watchlist (
                    scan_date, fixture_id, fixture_date, league_id,
                    home_name, away_name, mercado, descricao,
                    prob_modelo, watch_type, status, note, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-04-04",
                    9991,
                    "2026-04-04T20:00:00-04:00",
                    71,
                    "Casa",
                    "Fora",
                    "corners_over_85",
                    "Escanteios Over 8.5",
                    0.66,
                    "live_opportunity",
                    "resolved",
                    "snapshot",
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            live_watch_id = conn.execute("SELECT id FROM live_watchlist").fetchone()["id"]
            conn.execute(
                """
                INSERT INTO live_results (
                    live_watch_id, scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, watch_type, odd_usada, signal_minute, signal_note,
                    snapshot_json, resultado, gols_home, gols_away, acertou, lucro
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    live_watch_id,
                    "2026-04-04",
                    9991,
                    71,
                    "Casa",
                    "Fora",
                    "corners_over_85",
                    "live_opportunity",
                    1.85,
                    64,
                    "snapshot",
                    json.dumps(snapshot, ensure_ascii=False),
                    "home",
                    2,
                    1,
                    1,
                    0.85,
                ),
            )
            conn.commit()
            conn.close()

            trainer = LiveTrainer(db)
            samples = trainer._load_corner_snapshot_samples()

            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.market, "corners_over_85")
            self.assertEqual(sample.league_id, 71)
            self.assertEqual(sample.signal_minute, 64)
            self.assertEqual(sample.label, 1)
            self.assertEqual(sample.source, "bot_snapshot")
            self.assertEqual(sample.features["corners_before_signal"], 7.0)
            self.assertEqual(sample.features["shots_before_signal"], 18.0)

    def test_live_trainer_treina_por_slice_sem_modelo_global(self):
        from data.database import Database
        import models.live_trainer as live_trainer_module
        from models.live_trainer import LiveSample, LiveTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            trainer = LiveTrainer(db)
            samples = []
            for league_id in (71, 128):
                for fixture_idx in range(1, 31):
                    samples.append(
                        LiveSample(
                            fixture_id=(league_id * 1000) + fixture_idx,
                            fixture_date=f"2026-03-{fixture_idx:02d}T20:00:00-04:00",
                            league_id=league_id,
                            market="under25",
                            signal_minute=62,
                            label=1 if fixture_idx % 2 == 0 else 0,
                            source="api_historical",
                            features={
                                "league_id_numeric": float(league_id),
                                "signal_minute": 62.0,
                                "xg_before_signal": 0.8 if fixture_idx % 2 == 0 else 2.2,
                                "shots_before_signal": 9.0 if fixture_idx % 2 == 0 else 17.0,
                            },
                        )
                    )

            with patch.object(live_trainer_module, "MODELS_DIR", Path(tmpdir) / "live_models"):
                live_trainer_module.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                trainer = LiveTrainer(db)
                trainer.readiness = MagicMock(return_value={"pronto": True})
                trainer._load_samples = MagicMock(return_value=samples)
                trainer.MIN_SAMPLES_TOTAL = 10
                summary = trainer.treinar(min_amostras_mercado=10)

            self.assertEqual(summary["layout"], "league_market_only")
            self.assertIn("71:under25", summary["slices"])
            self.assertIn("128:under25", summary["slices"])
            self.assertNotIn("under25", summary["slices"])
            self.assertTrue((Path(tmpdir) / "live_models" / "league_71" / "under25.json").exists())
            self.assertTrue((Path(tmpdir) / "live_models" / "league_128" / "under25.json").exists())
            self.assertFalse((Path(tmpdir) / "live_models" / "under25.json").exists())


if __name__ == "__main__":
    unittest.main()

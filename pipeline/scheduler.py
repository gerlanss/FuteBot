"""
Scheduler — agendamento automático do pipeline.

Executa automaticamente:
  - 06:00  → Collector (resultados de ontem + resolver previsões)
  - 07:00  → Scanner (análise do dia + buscar odds + EV)
  - 22:00  → Relatório do dia (resultados das previsões de hoje)
  - 1º domingo do mês 14:00 → Treino per-league (~13h, termina antes do scanner de segunda)

Usa APScheduler para agendamento. Na VPS, roda como processo contínuo.
Localmente, pode rodar em background junto com o bot Telegram.

Uso:
  from pipeline.scheduler import Scheduler
  scheduler = Scheduler(telegram_callback=enviar_telegram)
  scheduler.iniciar()  # Bloqueia (roda infinitamente)

  # Ou em modo não-bloqueante:
  scheduler.iniciar(bloquear=False)
"""

from datetime import datetime, timedelta
import json
import threading
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from data.database import Database
from data.bulk_download import baixar_fixtures, baixar_stats, _check_limite
from pipeline.scanner import Scanner, _MERCADO_CATEGORIA, _CATEGORIA_MERCADOS
from pipeline.collector import Collector
from models.trainer import Trainer
from models.autotuner import AutoTuner
from models.market_discovery import MarketDiscoveryTrainer, MARKET_SPECS
from models.learner import Learner
from config import (
    SCAN_HORA, RESULTADOS_HORA, RETREINO_DIA, RETREINO_HORA, BULK_HORA,
    MIN_FIXTURES_TREINO, TIMEZONE, AUTO_RETREINO_HORA_INICIO, AUTO_RETREINO_HORA_FIM,
    AUTO_RETREINO_TRIALS, AUTO_RETREINO_MAX_LIGAS,
    DISCOVERY_SEMANAL_DIA, DISCOVERY_SEMANAL_HORA, DISCOVERY_TARGET_PRECISION,
    DISCOVERY_TARGET_PRECISION_COPA, DISCOVERY_MIN_TRAIN_SAMPLES, DISCOVERY_MIN_TRAIN_SAMPLES_COPA,
    DISCOVERY_MIN_TEST_SAMPLES, DISCOVERY_MIN_TEST_SAMPLES_COPA,
    DISCOVERY_OPTUNA_TRIALS, DISCOVERY_CUP_LEAGUE_IDS, LEAGUES, LIBERACAO_T30_INTERVALO_MIN,
    LIVE_CHECK_INTERVALO_MIN, LIVE_PRICE_WATCH_RECHECK_MIN, LIVE_PRICE_WATCH_WINDOW_MIN,
    MODEL_CONFIDENCE_MIN, ODDSPAPI_SIMPLE_MIN_EV_PERCENT, ODDSPAPI_SIMPLE_MIN_ODD,
    ODDSPAPI_SIMPLE_WATCH_MIN_EV_PERCENT,
    ODDSPAPI_USE_LIVE, SCAN_INTERVALO_HORAS, SCAN_LOOKAHEAD_HORAS,
)

from services.apifootball import raw_request, stats_partida
from services.live_intelligence import LiveIntelligence
from services.live_market_windows import (
    janela_operacional_live,
    status_janela_operacional_live,
)
from services.oddspapi import enriquecer_tips_com_odds_oddspapi

_LIVE_CONFLITOS_EXPLICITOS = {
    "under05_ht": {"ht_home", "ht_away"},
    "ht_home": {"under05_ht"},
    "ht_away": {"under05_ht"},
}
_MANUAL_LIVE_FIXTURE_MARKET = "__fixture_watch__"


class Scheduler:
    """Gerencia execução automática de todos os pipelines."""

    def __init__(self, db: Database = None, telegram_callback=None):
        """
        Parâmetros:
          db: instância do Database (compartilhada)
          telegram_callback: função async(texto) para enviar msgs no Telegram
        """
        self.db = db or Database()
        self.telegram_callback = telegram_callback
        self._pipeline_lock = threading.Lock()
        self._pipeline_job = None
        self.scheduler = BackgroundScheduler(
            timezone=TIMEZONE,
            job_defaults={
                "coalesce": True,
                "max_instances": 2,
                "misfire_grace_time": 120,
            },
        )
        self._configurar_jobs()

    def _configurar_jobs(self):
        """Configura todos os jobs agendados."""
        # Horários configuráveis via config.py
        h_resultados, m_resultados = RESULTADOS_HORA.split(":")
        h_scan, m_scan = SCAN_HORA.split(":")
        h_bulk, m_bulk = BULK_HORA.split(":")

        # 0. Bulk download incremental (madrugada — baixa stats pendentes)
        self.scheduler.add_job(
            self._job_bulk,
            CronTrigger(hour=int(h_bulk), minute=int(m_bulk)),
            id="bulk_incremental",
            name="Bulk download incremental",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._job_retreino_quarentena,
            CronTrigger(hour=f"{AUTO_RETREINO_HORA_INICIO}-{AUTO_RETREINO_HORA_FIM}", minute=0),
            id="retreino_quarentena",
            name="Retreino focal de ligas em quarentena",
            replace_existing=True,
        )

        h_discovery, m_discovery = DISCOVERY_SEMANAL_HORA.split(":")
        self.scheduler.add_job(
            self._job_discovery_semanal,
            CronTrigger(
                day_of_week=DISCOVERY_SEMANAL_DIA,
                hour=int(h_discovery),
                minute=int(m_discovery),
            ),
            id="discovery_semanal",
            name="Discovery semanal de strategies",
            replace_existing=True,
        )

        # 1. Coleta de resultados (diário)
        self.scheduler.add_job(
            self._job_coletar,
            CronTrigger(hour=int(h_resultados), minute=int(m_resultados)),
            id="coletar_resultados",
            name="Coleta de resultados",
            replace_existing=True,
        )

        # 2. Scanner de oportunidades (diário)
        self.scheduler.add_job(
            self._job_scanner,
            CronTrigger(hour=f"{int(h_scan)}-23/{max(1, SCAN_INTERVALO_HORAS)}", minute=int(m_scan)),
            id="scanner_diario",
            name="Scanner de oportunidades",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._job_liberacao_t30,
            CronTrigger(minute=f"*/{max(1, LIBERACAO_T30_INTERVALO_MIN)}"),
            id="liberacao_t30",
            name="Liberação final T-30",
            replace_existing=True,
        )

        # 3. Relatório diário (06:45 — após coleta das 06:00, com resultados do dia anterior)
        self.scheduler.add_job(
            self._job_relatorio,
            CronTrigger(hour=6, minute=45),
            id="relatorio_diario",
            name="Relatório diário (resultados de ontem)",
            replace_existing=True,
        )

        # 5. Acompanhamento ao vivo — verifica jogos previstos a cada 2h (10h-00h)
        self.scheduler.add_job(
            self._job_check_ao_vivo,
            CronTrigger(hour="10-23", minute=f"*/{max(1, LIVE_CHECK_INTERVALO_MIN)}"),
            id="check_ao_vivo",
            name="Check resultados ao vivo",
            replace_existing=True,
        )

        # 4. Retreino mensal per-league (1º domingo do mês às 14:00)
        #    ~13h de execução → termina ~03:00 de segunda, 4h antes do scanner
        h_retreino, m_retreino = RETREINO_HORA.split(":")
        # APScheduler: day='1st sun' não funciona diretamente.
        # Usamos day_of_week + week='1st' via campo 'day' com expressão.
        # Solução robusta: CronTrigger com dia 1-7 + day_of_week=sun = 1º domingo.
        self.scheduler.add_job(
            self._job_retreinar,
            CronTrigger(
                day="1-7",
                day_of_week=RETREINO_DIA,
                hour=int(h_retreino),
                minute=int(m_retreino),
            ),
            id="retreino_mensal",
            name="Retreino mensal per-league",
            replace_existing=True,
        )

    def iniciar(self, bloquear: bool = False):
        """
        Inicia o scheduler.

        Parâmetros:
          bloquear: se True, bloqueia a thread atual (para uso standalone)
        """
        print("⏰ Scheduler iniciado!")
        print(f"   📦 Bulk: {BULK_HORA} (diário, incremental)")
        print(
            f"   🧪 Retreino focal: {AUTO_RETREINO_HORA_INICIO:02d}:00-"
            f"{AUTO_RETREINO_HORA_FIM:02d}:00 (horário, ligas em quarentena)"
        )
        print(f"   🧬 Discovery semanal: {DISCOVERY_SEMANAL_DIA} {DISCOVERY_SEMANAL_HORA}")
        print(f"   📥 Coleta: {RESULTADOS_HORA} (diário)")
        print(
            f"   🔍 Scanner: a cada {SCAN_INTERVALO_HORAS}h desde {SCAN_HORA} "
            f"(janela pública de {SCAN_LOOKAHEAD_HORAS}h)"
        )
        print(f"   ⏳ Liberação T-30: a cada {LIBERACAO_T30_INTERVALO_MIN} min")
        print(f"   📋 Relatório: 06:45 (resultados de ontem)")
        print(f"   ⚽ Ao vivo: a cada {LIVE_CHECK_INTERVALO_MIN} min 10:00-23:59")
        print(f"   🤖 Retreino: 1º {RETREINO_DIA} do mês {RETREINO_HORA} per-league (mensal)")
        print()

        self.scheduler.start()
        try:
            self._garantir_radar_do_dia()
            # Recupera imediatamente qualquer janela T-30 perdida durante restart.
            self._job_liberacao_t30()
            self._garantir_relatorio_diario()
        except Exception as e:
            print(f"⚠️ Falha na recuperação inicial do T-30: {e}")

        if bloquear:
            try:
                import time
                while True:
                    time.sleep(60)
            except (KeyboardInterrupt, SystemExit):
                self.scheduler.shutdown()
                print("⏰ Scheduler encerrado.")

    def parar(self):
        """Para o scheduler."""
        self.scheduler.shutdown()

    def _entrar_secao_critica(self, job_name: str) -> bool:
        """Serializa jobs pesados para evitar travas por concorrencia no pipeline."""
        if self._pipeline_lock.acquire(blocking=False):
            self._pipeline_job = job_name
            return True

        atual = self._pipeline_job or "outro job crítico"
        print(f"[Scheduler] ⏭️ Pulando {job_name}: {atual} ainda está em execução")
        return False

    def _sair_secao_critica(self):
        """Libera a secao critica compartilhada do pipeline."""
        self._pipeline_job = None
        try:
            self._pipeline_lock.release()
        except RuntimeError:
            pass

    @staticmethod
    def _priorizar_ligas_quarentena(slices_ruins: list[dict]) -> list[int]:
        """
        Ordena ligas em quarentena pela necessidade de manutenção.

        Prioridade:
          1. Mais slices ruins na liga
          2. Pior ROI observado
          3. Maior número de previsões no slice pior
        """
        ranking = {}
        for item in slices_ruins:
            lid = item["league_id"]
            atual = ranking.setdefault(lid, {
                "count": 0,
                "worst_roi": 999.0,
                "worst_total": 0,
            })
            atual["count"] += 1
            atual["worst_roi"] = min(atual["worst_roi"], item.get("roi", 0))
            atual["worst_total"] = max(atual["worst_total"], item.get("total", 0))

        ordenadas = sorted(
            ranking.items(),
            key=lambda item: (item[1]["count"], -item[1]["worst_roi"], item[1]["worst_total"]),
            reverse=True,
        )
        return [lid for lid, _ in ordenadas]

    @staticmethod
    def _liga_eh_copa(league_id: int) -> bool:
        return int(league_id) in DISCOVERY_CUP_LEAGUE_IDS

    def _notification_sent(self, notification_date: str, kind: str) -> bool:
        fn = getattr(self.db, "notification_sent", None)
        if not callable(fn):
            return False
        try:
            value = fn(notification_date, kind)
        except Exception:
            return False
        return value is True

    def _save_notification_sent(self, notification_date: str, kind: str):
        fn = getattr(self.db, "save_notification_sent", None)
        if not callable(fn):
            return
        try:
            fn(notification_date, kind)
        except Exception:
            return

    def _refresh_live_odds_context(self, itens: list[dict]):
        """
        Atualiza contexto de odds apenas para fixtures já monitorados.
        Não abre varredura ampla nova no live.
        """
        if not ODDSPAPI_USE_LIVE or not itens:
            return
        now = datetime.now(ZoneInfo(TIMEZONE))
        payloads = []
        id_map: dict[tuple[int, str], tuple[int, dict, dict]] = {}
        for item in itens:
            if item.get("watch_type") not in {"approved_prelive", "live_opportunity"}:
                continue
            payload = dict(item.get("payload") or {})
            if not self._deve_consultar_odd_live_item(item, payload, now):
                continue
            payload.setdefault("fixture_id", item.get("fixture_id"))
            payload.setdefault("date", item.get("fixture_date") or item.get("date"))
            payload.setdefault("league_id", item.get("league_id"))
            payload.setdefault("home_name", item.get("home_name"))
            payload.setdefault("away_name", item.get("away_name"))
            payload.setdefault("mercado", item.get("mercado"))
            payloads.append(payload)
            id_map[(item.get("fixture_id"), item.get("mercado"))] = (item.get("id"), payload, item)

        try:
            enriquecer_tips_com_odds_oddspapi(payloads, phase_operacional="live_monitor")
        except Exception as exc:
            print(f"[Scheduler] ⚠️ Falha ao atualizar odds live OddsPapi: {exc}")
            return

        for payload in payloads:
            key = (payload.get("fixture_id"), payload.get("mercado"))
            item_ref = id_map.get(key)
            if not item_ref:
                continue
            item_id, original_payload, item = item_ref
            original_payload.update(payload)
            item["payload"] = original_payload
            try:
                self.db.atualizar_live_watch_item(item_id, payload=original_payload)
            except Exception:
                continue

    @staticmethod
    def _parse_payload_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo(TIMEZONE))
        return parsed

    def _deve_consultar_odd_live_item(self, item: dict, payload: dict, now: datetime) -> bool:
        if item.get("watch_type") not in {"approved_prelive", "live_opportunity"}:
            return False
        if payload.get("live_signal_notified"):
            return False
        if not payload.get("price_watch_active"):
            return False
        expires_at = self._parse_payload_datetime(payload.get("price_watch_expires_at"))
        if expires_at and now > expires_at:
            return False
        next_check_at = self._parse_payload_datetime(payload.get("price_watch_next_check_at"))
        if next_check_at and now < next_check_at:
            return False
        return True

    def _consultar_odd_live_agora(self, item: dict, payload: dict) -> dict:
        probe = dict(payload or {})
        probe.setdefault("fixture_id", item.get("fixture_id"))
        probe.setdefault("date", item.get("fixture_date") or item.get("date"))
        probe.setdefault("league_id", item.get("league_id"))
        probe.setdefault("home_name", item.get("home_name"))
        probe.setdefault("away_name", item.get("away_name"))
        probe.setdefault("mercado", item.get("mercado"))
        try:
            enriquecer_tips_com_odds_oddspapi([probe], phase_operacional="live_monitor")
        except Exception as exc:
            print(f"[Scheduler] ⚠️ Falha ao consultar odd live on-demand: {exc}")
        payload.update(probe)
        return payload

    @staticmethod
    def _clamp_live_probability(value: float) -> float:
        return max(MODEL_CONFIDENCE_MIN, min(0.89, round(float(value), 3)))

    @staticmethod
    def _snapshot_leitura_live(leitura: dict) -> dict:
        metricas = leitura.get("metricas") or {}
        teams = metricas.get("teams") or []
        return {
            "elapsed": leitura.get("elapsed"),
            "veredito": leitura.get("veredito"),
            "mensagem": leitura.get("mensagem"),
            "estado_partida": leitura.get("estado_partida"),
            "metricas": {
                "shots_total": float(metricas.get("shots_total") or 0.0),
                "shots_on": float(metricas.get("shots_on") or 0.0),
                "corners": float(metricas.get("corners") or 0.0),
                "xg": float(metricas.get("xg") or 0.0),
                "red_cards": float(metricas.get("red_cards") or 0.0),
                "yellow_cards": float(metricas.get("yellow_cards") or 0.0),
                "teams": [
                    {
                        "shots_total": float(team.get("shots_total") or 0.0),
                        "shots_on": float(team.get("shots_on") or 0.0),
                        "corners": float(team.get("corners") or 0.0),
                        "xg": float(team.get("xg") or 0.0),
                        "red_cards": float(team.get("red_cards") or 0.0),
                        "yellow_cards": float(team.get("yellow_cards") or 0.0),
                    }
                    for team in teams[:2]
                ],
            },
        }

    @staticmethod
    def _linha_live_mercado(mercado: str, default: float = 0.5) -> float:
        if "105" in (mercado or ""):
            return 10.5
        if "95" in (mercado or ""):
            return 9.5
        if "85" in (mercado or ""):
            return 8.5
        if "35" in (mercado or ""):
            return 3.5
        if "25" in (mercado or ""):
            return 2.5
        if "15" in (mercado or ""):
            return 1.5
        return default

    def _inferir_prob_modelo_live(self, mercado: str, leitura: dict) -> float:
        metricas = leitura.get("metricas") or {}
        corners = float(metricas.get("corners") or 0.0)
        shots_total = float(metricas.get("shots_total") or 0.0)
        shots_on = float(metricas.get("shots_on") or 0.0)
        xg = float(metricas.get("xg") or 0.0)
        teams = metricas.get("teams") or []
        home_team = teams[0] if len(teams) > 0 else {}
        away_team = teams[1] if len(teams) > 1 else {}
        home_on = float(home_team.get("shots_on") or 0.0)
        away_on = float(away_team.get("shots_on") or 0.0)
        home_xg = float(home_team.get("xg") or 0.0)
        away_xg = float(away_team.get("xg") or 0.0)
        home_shots = float(home_team.get("shots_total") or 0.0)
        away_shots = float(away_team.get("shots_total") or 0.0)
        elapsed = int(leitura.get("elapsed") or 0)
        estado = str(leitura.get("estado_partida") or "").lower()
        base = max(MODEL_CONFIDENCE_MIN, 0.60)

        if mercado.startswith("h2h") or mercado.startswith("ht_"):
            if mercado.endswith("home"):
                edge = max(0.0, (home_on - away_on) * 0.028) + max(0.0, (home_xg - away_xg) * 0.18)
                edge += max(0.0, (home_shots - away_shots) * 0.008)
                return self._clamp_live_probability(base + 0.05 + edge)
            if mercado.endswith("away"):
                edge = max(0.0, (away_on - home_on) * 0.028) + max(0.0, (away_xg - home_xg) * 0.18)
                edge += max(0.0, (away_shots - home_shots) * 0.008)
                return self._clamp_live_probability(base + 0.05 + edge)
            equilibrio = max(0.0, 1.0 - min(abs(home_on - away_on), 3.0) / 3.0)
            equilibrio += max(0.0, 1.0 - min(abs(home_xg - away_xg), 0.30) / 0.30)
            return self._clamp_live_probability(base + 0.03 + (equilibrio * 0.035))

        if "corners" in mercado:
            linha = self._linha_live_mercado(mercado, default=8.5)
            if "over" in mercado:
                progress = max(0.0, corners - max(4.0, linha - 5.0))
                pressure = max(0.0, shots_total - 11.0) * 0.004
                pressure += max(0.0, shots_on - 4.0) * 0.01
                pressure += max(0.0, xg - 0.95) * 0.045
                if elapsed >= 70:
                    pressure += 0.015
                return self._clamp_live_probability(base + 0.03 + (progress * 0.02) + pressure)
            cushion = max(0.0, max(4.0, linha - 4.0) - corners)
            control = max(0.0, 12.0 - shots_total) * 0.004
            control += max(0.0, 4.0 - shots_on) * 0.01
            control += max(0.0, 1.0 - xg) * 0.04
            if elapsed >= 65:
                control += 0.02
            return self._clamp_live_probability(base + 0.04 + (cushion * 0.02) + control)

        linha = self._linha_live_mercado(mercado, default=0.5)
        if "over" in mercado:
            pressure = max(0.0, shots_total - 11.0) * 0.004
            pressure += max(0.0, shots_on - 4.0) * 0.012
            pressure += max(0.0, xg - 0.95) * 0.05
            if estado == "quente":
                pressure += 0.04
            elif estado == "caotico":
                pressure += 0.03
            if elapsed >= 70 and linha <= 1.5:
                pressure += 0.015
            return self._clamp_live_probability(base + 0.03 + pressure)

        control = max(0.0, 13.0 - shots_total) * 0.004
        control += max(0.0, 5.0 - shots_on) * 0.012
        control += max(0.0, 1.0 - xg) * 0.05
        if estado == "frio":
            control += 0.04
        elif estado == "travado":
            control += 0.03
        if elapsed >= 70 and linha >= 1.5:
            control += 0.02
        return self._clamp_live_probability(base + 0.03 + control)

    @staticmethod
    def _odd_ev_live_aprovados(payload: dict) -> bool:
        odd = payload.get("odd_usada")
        ev = payload.get("ev_percent")
        try:
            odd_val = float(odd)
            ev_val = float(ev)
        except (TypeError, ValueError):
            return False
        return odd_val >= ODDSPAPI_SIMPLE_MIN_ODD and ev_val >= ODDSPAPI_SIMPLE_MIN_EV_PERCENT

    @staticmethod
    def _motivo_nao_executavel_live(payload: dict) -> str:
        odd = payload.get("odd_usada")
        ev = payload.get("ev_percent")
        try:
            odd_val = float(odd)
        except (TypeError, ValueError):
            odd_val = None
        try:
            ev_val = float(ev)
        except (TypeError, ValueError):
            ev_val = None

        if odd_val is None:
            return "a odd esta indisponivel na 1xBet para este mercado"
        if ev_val is None:
            return "o EV esta indisponivel para validar a entrada neste momento"
        if odd_val < ODDSPAPI_SIMPLE_MIN_ODD:
            return (
                "a odd ficou abaixo do minimo operacional "
                f"({odd_val:.2f} < {ODDSPAPI_SIMPLE_MIN_ODD:.2f})"
            )
        if ev_val < ODDSPAPI_SIMPLE_MIN_EV_PERCENT:
            return (
                "o EV ficou abaixo do minimo operacional "
                f"({ev_val:.1f}% < {ODDSPAPI_SIMPLE_MIN_EV_PERCENT:.1f}%)"
            )
        return "odd e EV ainda nao ficaram executaveis ao mesmo tempo"

    @staticmethod
    def _ev_live_permite_observacao(payload: dict) -> tuple[bool, str | None]:
        ev = payload.get("ev_percent")
        try:
            ev_val = float(ev)
        except (TypeError, ValueError):
            return True, None
        if ev_val < ODDSPAPI_SIMPLE_WATCH_MIN_EV_PERCENT:
            return (
                False,
                "o EV ficou abaixo da faixa minima de observacao "
                f"({ev_val:.1f}% < {ODDSPAPI_SIMPLE_WATCH_MIN_EV_PERCENT:.1f}%)",
            )
        return True, None

    def _normalizar_motivo_preco_live(self, payload: dict, motivo_preco: str | None) -> str:
        motivo = (motivo_preco or "").strip()
        motivo_real = self._motivo_nao_executavel_live(payload)
        motivo_lower = motivo.lower()

        if "vou rechecando" in motivo_lower:
            return (
                f"Cenario passou, mas {motivo_real}; "
                f"vou rechecando o preco por ate {LIVE_PRICE_WATCH_WINDOW_MIN} minutos."
            )
        if motivo_lower.startswith("cenario segue bom"):
            return f"Cenario segue bom, mas {motivo_real}."
        if "odd ainda" in motivo_lower or "odd e ev ainda" in motivo_lower:
            return f"Cenario passou, mas {motivo_real}."
        return motivo or motivo_real

    def _acionar_janela_preco_live(self, payload: dict, now: datetime) -> tuple[str, str]:
        if not payload.get("price_watch_active") and payload.get("price_watch_final_state") == "expired":
            return "expired", payload.get("price_watch_reason") or "Janela de preço já expirou para esta leitura."
        expires_at = self._parse_payload_datetime(payload.get("price_watch_expires_at"))
        if payload.get("price_watch_active") and expires_at and now > expires_at:
            payload["price_watch_active"] = False
            payload["price_watch_final_state"] = "expired"
            payload["price_watch_closed_at"] = now.isoformat()
            payload["price_watch_reason"] = "Janela de preço expirou sem odd executável."
            payload.pop("price_watch_next_check_at", None)
            return "expired", payload["price_watch_reason"]

        ev_observavel, motivo_ev = self._ev_live_permite_observacao(payload)
        if not ev_observavel:
            payload["price_watch_active"] = False
            payload["price_watch_final_state"] = "expired"
            payload["price_watch_closed_at"] = now.isoformat()
            payload["price_watch_reason"] = motivo_ev
            payload.pop("price_watch_next_check_at", None)
            return "expired", motivo_ev

        checks = int(payload.get("price_watch_checks") or 0)
        if not payload.get("price_watch_active"):
            payload["price_watch_active"] = True
            payload["price_watch_started_at"] = now.isoformat()
            payload["price_watch_expires_at"] = (now + timedelta(minutes=LIVE_PRICE_WATCH_WINDOW_MIN)).isoformat()
            payload["price_watch_checks"] = 1
            payload["price_watch_next_check_at"] = (now + timedelta(minutes=LIVE_PRICE_WATCH_RECHECK_MIN)).isoformat()
            payload["price_watch_final_state"] = "watching"
            return "started", (
                "Cenário passou, mas a odd ainda não ficou executável; "
                f"vou rechecando o preço por até {LIVE_PRICE_WATCH_WINDOW_MIN} minutos."
            )

        payload["price_watch_checks"] = checks + 1
        payload["price_watch_next_check_at"] = (now + timedelta(minutes=LIVE_PRICE_WATCH_RECHECK_MIN)).isoformat()
        payload["price_watch_final_state"] = "watching"
        return "watching", "Cenário segue bom, mas a odd ainda não ficou executável."

    @staticmethod
    def _encerrar_janela_preco_live(payload: dict, now: datetime):
        payload["price_watch_active"] = False
        payload["price_watch_closed_at"] = now.isoformat()
        payload["price_watch_final_state"] = "released"
        payload.pop("price_watch_next_check_at", None)

    def _acionar_janela_preco_live_inteligente(
        self,
        payload: dict,
        item: dict,
        elapsed: int | str | None,
        now: datetime,
    ) -> tuple[str, str]:
        if not payload.get("price_watch_active") and payload.get("price_watch_final_state") == "expired":
            return "expired", payload.get("price_watch_reason") or "Janela de preÃ§o jÃ¡ expirou para esta leitura."

        janela_status, janela_msg = self._status_janela_operacional_live(item, elapsed)
        if janela_status != "ok":
            payload["price_watch_active"] = False
            payload["price_watch_final_state"] = "expired"
            payload["price_watch_closed_at"] = now.isoformat()
            payload["price_watch_reason"] = janela_msg
            payload.pop("price_watch_next_check_at", None)
            return "expired", janela_msg

        expires_at = self._parse_payload_datetime(payload.get("price_watch_expires_at"))
        if payload.get("price_watch_active") and expires_at and now > expires_at:
            payload["price_watch_active"] = False
            payload["price_watch_final_state"] = "expired"
            payload["price_watch_closed_at"] = now.isoformat()
            payload["price_watch_reason"] = "Janela de preÃ§o expirou sem odd executÃ¡vel."
            payload.pop("price_watch_next_check_at", None)
            return "expired", payload["price_watch_reason"]

        try:
            minuto_atual = int(elapsed or 0)
        except Exception:
            minuto_atual = 0
        _, minuto_limite = self._janela_operacional_live(item)
        minutos_restantes = max(0, int(minuto_limite) - minuto_atual)
        if minutos_restantes < max(1, LIVE_PRICE_WATCH_RECHECK_MIN):
            payload["price_watch_active"] = False
            payload["price_watch_final_state"] = "expired"
            payload["price_watch_closed_at"] = now.isoformat()
            payload["price_watch_reason"] = "A leitura jÃ¡ entrou tarde demais para rechecagem Ãºtil de preÃ§o."
            payload.pop("price_watch_next_check_at", None)
            return "expired", payload["price_watch_reason"]

        ev_observavel, motivo_ev = self._ev_live_permite_observacao(payload)
        if not ev_observavel:
            payload["price_watch_active"] = False
            payload["price_watch_final_state"] = "expired"
            payload["price_watch_closed_at"] = now.isoformat()
            payload["price_watch_reason"] = motivo_ev
            payload.pop("price_watch_next_check_at", None)
            return "expired", motivo_ev

        janela_efetiva = min(LIVE_PRICE_WATCH_WINDOW_MIN, minutos_restantes)
        checks = int(payload.get("price_watch_checks") or 0)
        proximo_check = min(LIVE_PRICE_WATCH_RECHECK_MIN, janela_efetiva)
        if not payload.get("price_watch_active"):
            payload["price_watch_active"] = True
            payload["price_watch_started_at"] = now.isoformat()
            payload["price_watch_expires_at"] = (now + timedelta(minutes=janela_efetiva)).isoformat()
            payload["price_watch_checks"] = 1
            payload["price_watch_next_check_at"] = (now + timedelta(minutes=proximo_check)).isoformat()
            payload["price_watch_final_state"] = "watching"
            return "started", (
                f"CenÃ¡rio passou, mas {self._motivo_nao_executavel_live(payload)}; "
                f"vou rechecando o preÃ§o por atÃ© {janela_efetiva} minutos."
            )

        payload["price_watch_checks"] = checks + 1
        payload["price_watch_next_check_at"] = (now + timedelta(minutes=proximo_check)).isoformat()
        payload["price_watch_final_state"] = "watching"
        return "watching", f"CenÃ¡rio segue bom, mas {self._motivo_nao_executavel_live(payload)}."

    @classmethod
    def _formatar_alerta_preco_live(cls, item: dict, *, titulo: str, contexto: str) -> str:
        home_name = item.get("home_name", "?")
        away_name = item.get("away_name", "?")
        mercado_label = item.get("descricao") or item.get("mercado", "")
        payload = item.get("payload") or {}
        minuto = item.get("elapsed") or payload.get("last_live_minute") or "?"
        mensagem = payload.get("last_live_message") or item.get("note") or "Sem leitura adicional."
        contexto_bruto = (contexto or "").strip()
        contexto_lower = contexto_bruto.lower()
        if payload.get("price_watch_reason") and "odd ainda" in contexto_lower:
            contexto = payload.get("price_watch_reason")
        else:
            contexto = contexto_bruto
        meta = cls._meta_live_contexto(item)
        return "\n".join([
            titulo,
            f"<code>{home_name}</code> <b>x</b> <code>{away_name}</code>",
            f"• {mercado_label}" + (f" | {meta}" if meta else ""),
            f"• Minuto {minuto}",
            f"• {mensagem}",
            f"<i>{contexto}</i>",
        ]).rstrip()

    def _garantir_radar_do_dia(self, data: str = None) -> bool:
        """Reconstrói silenciosamente o radar do dia se o scheduler subiu depois do scan."""
        agora = datetime.now(ZoneInfo(TIMEZONE))
        data = data or agora.strftime("%Y-%m-%d")
        h_scan, m_scan = SCAN_HORA.split(":")
        horario_scan = agora.replace(hour=int(h_scan), minute=int(m_scan), second=0, microsecond=0)

        if agora < horario_scan:
            return False

        candidatos_existentes = self.db.candidatos_por_data(data)
        if candidatos_existentes:
            return False

        print(f"🔄 Recuperando radar do dia {data} (sem candidatos salvos após {SCAN_HORA})...")
        scanner = Scanner(self.db)
        resultado = scanner.executar(
            data=data,
            mode="preselect",
            reference_time=agora,
            lookahead_minutes=SCAN_LOOKAHEAD_HORAS * 60,
        )
        total = len(resultado.get("preselecionados") or [])
        print(f"   ✅ Radar reconstruído: {total} jogo(s) pré-selecionado(s)")
        return total > 0

    def _garantir_relatorio_diario(self) -> bool:
        """Recupera o relatório diário se o serviço subiu depois das 06:45."""
        agora = datetime.now(ZoneInfo(TIMEZONE))
        horario_relatorio = agora.replace(hour=6, minute=45, second=0, microsecond=0)
        if agora < horario_relatorio:
            return False

        hoje = agora.strftime("%Y-%m-%d")
        ontem = (agora - timedelta(days=1)).strftime("%Y-%m-%d")
        pendencias = (
            not self._notification_sent(ontem, "daily_results")
            or not self._notification_sent(hoje, "daily_report")
            or not self._notification_sent(hoje, "daily_health")
        )
        if not pendencias:
            return False

        print(f"🔄 Recuperando relatório diário perdido de {hoje} (janela 06:45)...")
        self._job_relatorio()
        return True

    @staticmethod
    def _infer_conf_band(best_rule: dict | None) -> tuple[float, float]:
        conf_min = MODEL_CONFIDENCE_MIN
        conf_max = 1.01
        if not best_rule:
            return conf_min, conf_max
        for feature, op, threshold in best_rule.get("conditions", []):
            if feature != "model_prob":
                continue
            if op == ">=":
                conf_min = max(conf_min, float(threshold))
            elif op == "<=":
                conf_max = min(conf_max, float(threshold))
        if conf_max <= conf_min:
            conf_max = min(1.01, conf_min + 0.05)
        return round(conf_min, 4), round(conf_max, 4)

    def _estrategias_faltantes_por_liga(self) -> dict[int, list[str]]:
        """
        Retorna mercados sem strategy ativa por liga.

        Considera apenas as ligas configuradas e o catálogo operacional atual.
        """
        ativos = self.db.strategies_ativas()
        cobertura = defaultdict(set)
        for item in ativos:
            cobertura[int(item["league_id"])].add(item["mercado"])

        todos_mercados = [item.market_id for item in MARKET_SPECS]
        faltantes = {}
        for info in LEAGUES.values():
            lid = int(info["id"])
            missing = sorted(set(todos_mercados) - cobertura.get(lid, set()))
            if missing:
                faltantes[lid] = missing
        return faltantes

    def _salvar_discovery_por_slice(self, run_summary: dict) -> int:
        """
        Promove apenas os slices aceitos do discovery, preservando o resto.
        """
        strategies = []
        for league in run_summary.get("leagues", []):
            if league.get("status") == "error":
                continue
            lid = int(league["league_id"])
            for market in league.get("markets", []):
                if market.get("status") != "accepted":
                    continue
                best = market.get("best_rule")
                conf_min, conf_max = self._infer_conf_band(best)
                test = best.get("test", {}) if best else {}
                strategies.append({
                    "mercado": market["market"],
                    "league_id": lid,
                    "conf_min": conf_min,
                    "conf_max": conf_max,
                    "accuracy": float(test.get("precision") or market.get("test_base_rate") or 0),
                    "n_samples": int(test.get("samples") or market.get("test_base_samples") or 0),
                    "ev_medio": 0,
                    "ativo": 1,
                    "params": {
                        "source": "market_discovery",
                        "rule": best.get("rule") if best else "",
                        "conditions": best.get("conditions", []) if best else [],
                        "train": best.get("train") if best else {},
                        "test": best.get("test") if best else {},
                        "run_id": run_summary.get("run_id"),
                    },
                    "modelo_versao": f"discovery_{run_summary.get('run_id')}",
                })

        if not strategies:
            return 0
        self.db.salvar_strategies_por_slice(strategies)
        return len(strategies)

    # ══════════════════════════════════════════════
    #  JOBS
    # ══════════════════════════════════════════════

    def _job_bulk(self):
        """
        Job: bulk download incremental (madrugada).

        Baixa stats de partidas finalizadas que ainda não têm stats.
        Se modelo não existe e dados suficientes, treina automaticamente.
        Tudo autônomo — sem intervenção humana.
        """
        print(f"\n{'='*60}")
        print(f"📦 JOB: Bulk download incremental — {datetime.now()}")
        print(f"{'='*60}")

        try:
            resumo = self.db.resumo()
            fixtures_total = resumo.get("fixtures", 0)
            com_stats = resumo.get("fixtures_com_stats", 0)
            fixtures_ft = resumo.get("fixtures_ft", 0)

            # Se banco de fixtures vazio, baixar primeiro
            if fixtures_total < 100:
                print("   📥 Banco vazio — baixando fixtures...")
                n = baixar_fixtures(self.db)
                self._enviar_telegram(f"📦 Bulk: {n} fixtures baixados (bootstrap)")
                resumo = self.db.resumo()
                fixtures_ft = resumo.get("fixtures_ft", 0)
                com_stats = resumo.get("fixtures_com_stats", 0)

            # Baixar stats pendentes
            pendentes = fixtures_ft - com_stats
            if pendentes > 0:
                usadas, pode = _check_limite()
                if pode:
                    print(f"   📊 {pendentes} stats pendentes — baixando...")
                    n = baixar_stats(self.db, resume=True)
                    self._enviar_telegram(
                        f"📦 Bulk incremental: {n} stats baixadas\n"
                        f"   Total com stats: {com_stats + n:,}/{fixtures_ft:,}"
                    )
                    com_stats += n
                else:
                    print(f"   ⏸️ Limite API atingido ({usadas}/7500)")
            else:
                print("   ✅ Todas as stats já baixadas")

            # Retreino é APENAS mensal (1º domingo do mês via _job_retreinar).
            # O bulk download NÃO dispara retreino automático — os modelos
            # per-league já existem em data/models/league_*/ e são atualizados
            # mensalmente pelo job dedicado.

        except Exception as e:
            print(f"❌ Erro no bulk: {e}")
            self._enviar_telegram(f"❌ Erro no bulk download:\n{e}")

    def _job_coletar(self):
        """Job: coleta resultados e resolve previsões.

        Coleta dados e resolve previsões silenciosamente.
        O relatório de resultados é enviado pelo _job_relatorio (22h)
        ou pelo _job_check_ao_vivo quando todos os jogos terminam.
        """
        print(f"\n{'='*60}")
        print(f"📥 JOB: Coleta de resultados — {datetime.now()}")
        print(f"{'='*60}")

        try:
            collector = Collector(self.db)
            resultado = collector.executar()
            # Relatório não é enviado aqui — sai às 22h ou no check ao vivo
            print(f"   ✅ Coleta concluída: {resultado.get('fixtures_atualizados', 0)} fixtures, "
                  f"{resultado.get('stats_baixadas', 0)} stats")

        except Exception as e:
            print(f"❌ Erro na coleta: {e}")
            self._enviar_telegram(f"❌ Erro na coleta de resultados:\n{e}")

    def _job_retreino_quarentena(self):
        """
        Job: retreino focal automático de ligas em quarentena.

        Mantém custo baixo:
          - roda no máximo AUTO_RETREINO_MAX_LIGAS por execução
          - usa poucos trials
          - retreina apenas ligas com slices degradados
        """
        print(f"\n{'='*60}")
        print(f"🧪 JOB: Retreino focal de quarentena — {datetime.now()}")
        print(f"{'='*60}")

        try:
            treino = self.db.ultimo_treino()
            versao = treino["modelo_versao"] if treino else None
            ruins = self.db.slices_degradados(modelo_versao=versao)
            if not ruins:
                print("   ✅ Nenhum slice em quarentena")
                return

            league_ids = self._priorizar_ligas_quarentena(ruins)
            league_ids = league_ids[:max(1, AUTO_RETREINO_MAX_LIGAS)]
            print(f"   🎯 Ligas alvo: {league_ids}")

            tuner = AutoTuner(self.db)
            resultado = tuner.executar(
                league_ids=league_ids,
                n_trials=AUTO_RETREINO_TRIALS,
            )

            msg = (
                "🧪 <b>Retreino focal automático concluído</b>\n\n"
                f"Ligas: {', '.join(str(l) for l in league_ids)}\n"
                f"Trials por modelo: {AUTO_RETREINO_TRIALS}\n"
                f"Strategies ativas geradas: {resultado.get('strategies_ativas', 0)}"
            )
            self._enviar_telegram(msg)

        except Exception as e:
            print(f"❌ Erro no retreino focal: {e}")
            self._enviar_telegram(f"❌ Erro no retreino focal automático:\n{e}")

    def _job_discovery_semanal(self):
        """
        Job: discovery semanal de strategies.

        Objetivos:
          - buscar cobertura para mercados sem strategy ativa
          - revisar slices degradados com dados mais novos
          - rodar de forma sequencial e leve para a VPS
        """
        print(f"\n{'='*60}")
        print(f"🧬 JOB: Discovery semanal — {datetime.now()}")
        print(f"{'='*60}")

        try:
            faltantes = self._estrategias_faltantes_por_liga()
            treino = self.db.ultimo_treino()
            versao = treino["modelo_versao"] if treino else None
            ruins = self.db.slices_degradados(modelo_versao=versao)
            ligas_ruins = self._priorizar_ligas_quarentena(ruins)

            candidate_leagues = []
            seen = set()
            for lid in ligas_ruins + sorted(faltantes.keys()):
                if lid not in seen:
                    candidate_leagues.append(lid)
                    seen.add(lid)

            if not candidate_leagues:
                print("   ✅ Nenhuma liga precisando discovery")
                return

            trainer = MarketDiscoveryTrainer(self.db)
            linhas_resumo = [
                "🧬 <b>Discovery semanal iniciado</b>",
                "",
                f"Ligas alvo: {', '.join(str(lid) for lid in candidate_leagues)}",
                f"Precisão mínima: {DISCOVERY_TARGET_PRECISION:.0%}",
                f"Train min: {DISCOVERY_MIN_TRAIN_SAMPLES}",
                f"Trials Optuna: {DISCOVERY_OPTUNA_TRIALS}",
            ]
            self._enviar_telegram("\n".join(linhas_resumo))

            total_saved = 0
            for lid in candidate_leagues:
                target_precision = DISCOVERY_TARGET_PRECISION_COPA if self._liga_eh_copa(lid) else DISCOVERY_TARGET_PRECISION
                min_train = DISCOVERY_MIN_TRAIN_SAMPLES_COPA if self._liga_eh_copa(lid) else DISCOVERY_MIN_TRAIN_SAMPLES
                min_test = DISCOVERY_MIN_TEST_SAMPLES_COPA if self._liga_eh_copa(lid) else DISCOVERY_MIN_TEST_SAMPLES
                markets = faltantes.get(lid) or None
                league_name = next(
                    (info["nome"] for info in LEAGUES.values() if int(info["id"]) == lid),
                    f"Liga {lid}",
                )
                print(
                    f"   🎯 Discovery liga {lid} ({league_name}) | mercados={markets or 'todos'} "
                    f"| target={target_precision:.0%} | min_train={min_train} | min_test={min_test}"
                )
                result = trainer.run(
                    league_ids=[lid],
                    markets=markets,
                    target_precision=target_precision,
                    min_train_samples=min_train,
                    min_test_samples=min_test,
                    optuna_trials=DISCOVERY_OPTUNA_TRIALS,
                )
                saved = self._salvar_discovery_por_slice(result)
                total_saved += saved

                league_summary = result["leagues"][0] if result.get("leagues") else {}
                msg = (
                    f"🧬 <b>Discovery liga concluída</b>\n\n"
                    f"Liga: {league_name} ({lid})\n"
                    f"Mercados testados: {len(league_summary.get('markets', []))}\n"
                    f"Mercados aceitos: {league_summary.get('accepted_markets', 0)}\n"
                    f"Strategies promovidas: {saved}\n"
                    f"Precisão alvo: {target_precision:.0%}\n"
                    f"Min train usado: {min_train}\n"
                    f"Min test usado: {min_test}\n"
                    f"Run: {result.get('run_id')}"
                )
                self._enviar_telegram(msg)

            self._enviar_telegram(
                "🧬 <b>Discovery semanal concluído</b>\n\n"
                f"Ligas processadas: {len(candidate_leagues)}\n"
                f"Strategies promovidas: {total_saved}"
            )
        except Exception as e:
            print(f"❌ Erro no discovery semanal: {e}")
            self._enviar_telegram(f"❌ Erro no discovery semanal:\n{e}")

    def _job_scanner(self):
        """Job: scanner de oportunidades."""
        if not self._entrar_secao_critica("scanner"):
            return
        print(f"\n{'='*60}")
        print(f"🔍 JOB: Scanner de oportunidades — {datetime.now()}")
        print(f"{'='*60}")

        try:
            agora = datetime.now(ZoneInfo(TIMEZONE))
            scanner = Scanner(self.db)
            resultado = scanner.executar(
                dias_adiante=0,
                mode="preselect",
                reference_time=agora,
                lookahead_minutes=SCAN_LOOKAHEAD_HORAS * 60,
            )

            # Enviar relatório via Telegram (tuplas: texto + botões ✏️ Odd)
            msgs = scanner.formatar_relatorio(resultado)
            for texto, botoes in msgs:
                reply_markup = None
                if botoes:
                    # Monta inline_keyboard para API HTTP do Telegram
                    reply_markup = {
                        "inline_keyboard": [
                            [{"text": label, "callback_data": cb}]
                            for label, cb in botoes
                        ]
                    }
                self._enviar_telegram_publico(texto, reply_markup=reply_markup)

        except Exception as e:
            print(f"❌ Erro no scanner: {e}")
            self._enviar_telegram(f"❌ Erro no scanner:\n{e}")
        finally:
            self._sair_secao_critica()

    def _job_liberacao_t30(self):
        """Job: libera os mercados na janela T-30 e publica o lote final."""
        if not self._entrar_secao_critica("liberacao_t30"):
            return
        print(f"\n{'='*60}")
        print(f"⏳ JOB: Liberação T-30 — {datetime.now()}")
        print(f"{'='*60}")

        try:
            self._garantir_radar_do_dia()
            scanner = Scanner(self.db)
            resultado = scanner.liberar_mercados()

            if not any([
                resultado.get("tips_enviadas_llm"),
                resultado.get("tips_aprovadas"),
                resultado.get("tips_rejeitadas_llm"),
                resultado.get("combos"),
            ]):
                print("   ✅ Nenhum candidato elegível nesta janela")
                return

            msgs = scanner.formatar_relatorio(resultado)
            for texto, botoes in msgs:
                reply_markup = None
                if botoes:
                    reply_markup = {
                        "inline_keyboard": [
                            [{"text": label, "callback_data": cb}]
                            for label, cb in botoes
                        ]
                    }
                self._enviar_telegram_publico(texto, reply_markup=reply_markup)

        except Exception as e:
            print(f"❌ Erro na liberação T-30: {e}")
            self._enviar_telegram(f"❌ Erro na liberação T-30:\n{e}")
        finally:
            self._sair_secao_critica()

    def _job_relatorio(self):
        """Job: relatório diário — resultados do dia ANTERIOR.

        Roda às 06:45 (após coleta das 06:00) e envia:
          1. Resultados dos jogos de ONTEM (todos já finalizados)
          2. Relatório de performance geral
          3. Saúde do modelo + alertas de degradação
        """
        print(f"\n{'='*60}")
        print(f"📋 JOB: Relatório diário — {datetime.now()}")
        print(f"{'='*60}")

        try:
            learner = Learner(self.db)
            hoje = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")

            # 1. Resultados de ONTEM (dia anterior completo)
            ontem = (datetime.now(ZoneInfo(TIMEZONE)) - timedelta(days=1)).strftime("%Y-%m-%d")
            resultado_dia = learner.relatorio_resultado_dia(ontem)
            if "Nenhum resultado" not in resultado_dia and not self._notification_sent(ontem, "daily_results"):
                self._enviar_telegram(resultado_dia)
                self._save_notification_sent(ontem, "daily_results")

            # 2. Relatório de performance geral
            if not self._notification_sent(hoje, "daily_report"):
                msg = learner.relatorio_diario()
                self._enviar_telegram(msg)
                self._save_notification_sent(hoje, "daily_report")

            # 3. Saúde do modelo (guard rails)
            if not self._notification_sent(hoje, "daily_health"):
                saude_msg = learner.relatorio_saude()
                self._enviar_telegram(saude_msg)
                self._save_notification_sent(hoje, "daily_health")

            # Se modelo degradado, alertar com urgência
            check = learner.verificar_degradacao()
            if (check["degradado"] or check["pausado"]) and not self._notification_sent(hoje, "daily_alert"):
                urgencia = (
                    "🚨 *ALERTA: Modelo precisa de atenção!*\n\n"
                    + "\n".join(check["alertas"])
                    + "\n\nUse /treinar para retreinar."
                )
                self._enviar_telegram(urgencia)
                self._save_notification_sent(hoje, "daily_alert")

        except Exception as e:
            print(f"❌ Erro no relatório: {e}")

    def _job_retreinar(self):
        """
        Job: retreinar modelos per-league (sem modelo global).

        Fluxo autônomo:
          1. Verifica se tem dados suficientes
          2. Treina 1 modelo por liga elegível
          3. Quality gate per-modelo decide aprovação/rejeição
          4. Modelos salvos em data/models/league_{id}/ (per-league)
        """
        print(f"\n{'='*60}")
        print(f"🤖 JOB: Treino per-league — {datetime.now()}")
        print(f"{'='*60}")

        try:
            # Verificar se tem dados suficientes
            resumo = self.db.resumo()
            com_stats = resumo.get("fixtures_com_stats", 0)

            if com_stats < MIN_FIXTURES_TREINO:
                msg = (
                    f"⏳ Treino adiado: {com_stats}/{MIN_FIXTURES_TREINO} jogos com stats.\n"
                    f"O bulk download continua coletando dados automaticamente."
                )
                self._enviar_telegram(msg)
                return

            trainer = Trainer(self.db)
            # Treino per-league: 1 modelo por liga com dados suficientes
            resultados = trainer.treinar_por_liga(
                train_seasons=[2020, 2021, 2022, 2023, 2024, 2025],
                test_season=2026
            )

            # Montar resumo para Telegram
            ligas_ok = 0
            ligas_total = 0
            for k, v in resultados.items():
                ligas_total += 1
                if v.get("modelos_aprovados"):
                    ligas_ok += 1

            msg = (
                f"🤖 *Treino per-league concluído!*\n\n"
                f"🏟️ *Ligas:* {ligas_ok}/{ligas_total} com modelos aprovados\n"
            )
            self._enviar_telegram(msg)

        except Exception as e:
            print(f"❌ Erro no treino per-league: {e}")
            import traceback
            traceback.print_exc()
            self._enviar_telegram(f"❌ Erro no treino per-league:\n{e}")

    def _job_check_ao_vivo(self):
        """
        Job: acompanha jogos monitorados pelo pre-live e sinaliza leitura live.

        Itens acompanhados:
          - entradas aprovadas no pre-live
          - bloqueios que merecem reavaliação live

        Em jogos ao vivo:
          - lê placar/status
          - puxa stats da partida
          - passa pela LiveIntelligence

        Em jogos finalizados:
          - resolve previsões aprovadas
          - fecha itens observados
        """
        if not self._entrar_secao_critica("check_ao_vivo"):
            return
        print(f"\n{'='*60}")
        print(f"⚽ JOB: Check ao vivo — {datetime.now()}")
        print(f"{'='*60}")

        try:
            hoje = datetime.now().strftime("%Y-%m-%d")
            ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            itens = self.db.live_watch_items(dates=[ontem, hoje], status="active")
            if not itens:
                print("   ✅ Nenhum jogo em observação live")
                return

            print(f"   🔍 Verificando {len(itens)} item(ns) monitorados...")
            self._refresh_live_odds_context(itens)
            itens_por_fixture = {}
            for item in itens:
                itens_por_fixture.setdefault(item["fixture_id"], []).append(item)
            fixture_cache = {}
            stats_cache = {}
            fixture_extras_processados = set()
            live = LiveIntelligence()
            resolvidos = 0
            acertos = 0
            erros = 0
            notificacoes_publicas = []
            alertas_admin = []
            alertas_preco_publicos = []
            sinais_live = []
            greens_antecipados = []
            reds_antecipados = []
            itens_tocados = []
            itens_resolvidos = []

            nomes_mercado = {
                "h2h_home": "Casa", "h2h_draw": "Empate", "h2h_away": "Fora",
                "over15": "Over 1.5", "under15": "Under 1.5",
                "over25": "Over 2.5", "under25": "Under 2.5",
                "over35": "Over 3.5", "under35": "Under 3.5",
                "ht_home": "1T Casa", "ht_draw": "1T Empate", "ht_away": "1T Fora",
                "over05_ht": "1T Over 0.5", "under05_ht": "1T Under 0.5",
                "over15_ht": "1T Over 1.5", "under15_ht": "1T Under 1.5",
                "over05_2t": "2T Over 0.5", "under05_2t": "2T Under 0.5",
                "over15_2t": "2T Over 1.5", "under15_2t": "2T Under 1.5",
                "corners_over_85": "Escanteios Over 8.5",
                "corners_under_85": "Escanteios Under 8.5",
                "corners_over_95": "Escanteios Over 9.5",
                "corners_under_95": "Escanteios Under 9.5",
                "corners_over_105": "Escanteios Over 10.5",
                "corners_under_105": "Escanteios Under 10.5",
            }

            for item in itens:
                fixture_id = item["fixture_id"]
                if fixture_id not in fixture_cache:
                    r = raw_request("fixtures", {"id": fixture_id})
                    fixture_cache[fixture_id] = (r.get("response", []) or [None])[0]
                game = fixture_cache.get(fixture_id)
                if not game:
                    continue

                status = game.get("fixture", {}).get("status", {}).get("short", "NS")
                item_id = item["id"]
                mercado = item.get("mercado", "")
                mercado_label = nomes_mercado.get(mercado, item.get("descricao") or mercado)
                home_name = item.get("home_name", "?")
                away_name = item.get("away_name", "?")
                payload = dict(item.get("payload") or {})
                scan_date_item = item.get("scan_date") or hoje

                if status in ("1H", "2H", "HT", "ET", "LIVE"):
                    stats = stats_partida(fixture_id)
                    stats_cache[fixture_id] = stats or []
                    if fixture_id not in fixture_extras_processados:
                        extras_alertas, extras_sinais = self._detectar_oportunidades_live_fixture(
                            scan_date=hoje,
                            fixture=game,
                            stats=stats or [],
                            itens_fixture=itens_por_fixture.get(fixture_id, []),
                            nomes_mercado=nomes_mercado,
                            live=live,
                        )
                        alertas_admin.extend(extras_alertas)
                        sinais_live.extend(extras_sinais)
                        fixture_extras_processados.add(fixture_id)
                    if item.get("watch_type") == "manual_fixture" or item.get("mercado") == _MANUAL_LIVE_FIXTURE_MARKET:
                        payload["last_live_verdict"] = "fixture_anchor"
                        payload["last_live_minute"] = ((game.get("fixture") or {}).get("status") or {}).get("elapsed") or "?"
                        self.db.atualizar_live_watch_item(
                            item_id,
                            note=item.get("note") or "Gancho manual do fixture mantido para exploracao live.",
                            payload=payload,
                        )
                        itens_tocados.append(item_id)
                        continue
                    if item.get("watch_type") in {"approved_prelive", "live_opportunity"} and (
                        payload.get("live_hit_notified") or payload.get("live_loss_notified")
                    ):
                        self.db.atualizar_live_watch_item(
                            item_id,
                            status="resolved",
                            note=item.get("note"),
                            payload=payload,
                        )
                        itens_tocados.append(item_id)
                        itens_resolvidos.append(item_id)
                        continue

                    if item.get("watch_type") == "approved_prelive" or payload.get("live_signal_notified"):
                        green_agora = self._mercado_green_antecipado(item, game, stats)
                        red_agora = self._mercado_red_antecipado(item, game, stats)
                        if green_agora and not payload.get("live_hit_notified"):
                            mercado_label_curto = nomes_mercado.get(mercado, item.get("descricao") or mercado)
                            greens_antecipados.append({
                                "fixture_id": fixture_id,
                                "mercado": mercado,
                                "home_name": home_name,
                                "away_name": away_name,
                                "mercado_label": mercado_label_curto,
                                "elapsed": ((game.get("fixture") or {}).get("status") or {}).get("elapsed") or "?",
                            })
                            payload["live_hit_notified"] = True
                            resultado_parcial = (
                                "home" if (int(game.get("goals", {}).get("home") or 0) > int(game.get("goals", {}).get("away") or 0))
                                else "draw" if (int(game.get("goals", {}).get("home") or 0) == int(game.get("goals", {}).get("away") or 0))
                                else "away"
                            )
                            self.db.resolver_live_result(
                                item_id,
                                resultado=resultado_parcial,
                                gols_home=int(game.get("goals", {}).get("home") or 0),
                                gols_away=int(game.get("goals", {}).get("away") or 0),
                                acertou=True,
                            )
                            self.db.atualizar_live_watch_item(
                                item_id,
                                status="resolved",
                                note=item.get("note"),
                                payload=payload,
                            )
                            itens_tocados.append(item_id)
                            itens_resolvidos.append(item_id)
                            continue
                        if red_agora and not payload.get("live_loss_notified"):
                            mercado_label_curto = nomes_mercado.get(mercado, item.get("descricao") or mercado)
                            reds_antecipados.append({
                                "fixture_id": fixture_id,
                                "mercado": mercado,
                                "home_name": home_name,
                                "away_name": away_name,
                                "mercado_label": mercado_label_curto,
                                "elapsed": ((game.get("fixture") or {}).get("status") or {}).get("elapsed") or "?",
                            })
                            payload["live_loss_notified"] = True
                            resultado_parcial = (
                                "home" if (int(game.get("goals", {}).get("home") or 0) > int(game.get("goals", {}).get("away") or 0))
                                else "draw" if (int(game.get("goals", {}).get("home") or 0) == int(game.get("goals", {}).get("away") or 0))
                                else "away"
                            )
                            self.db.resolver_live_result(
                                item_id,
                                resultado=resultado_parcial,
                                gols_home=int(game.get("goals", {}).get("home") or 0),
                                gols_away=int(game.get("goals", {}).get("away") or 0),
                                acertou=False,
                            )
                            self.db.atualizar_live_watch_item(
                                item_id,
                                status="resolved",
                                note=item.get("note"),
                                payload=payload,
                            )
                            itens_tocados.append(item_id)
                            itens_resolvidos.append(item_id)
                            continue

                    leitura = live.analisar(item, game, stats)
                    payload["live_reading"] = self._snapshot_leitura_live(leitura)
                    payload["metricas_live"] = payload["live_reading"].get("metricas")
                    payload["estado_partida"] = leitura.get("estado_partida")
                    veredito = leitura.get("veredito", "monitorar")
                    elapsed = leitura.get("elapsed") or "?"
                    last_verdict = payload.get("last_live_verdict")

                    bloqueio_operacional = False
                    if veredito == "sinal_live":
                        status_janela, mensagem_janela = self._status_janela_operacional_live(item, elapsed)
                        if status_janela == "cedo":
                            veredito = "monitorar"
                            bloqueio_operacional = True
                        elif status_janela == "tarde":
                            veredito = "cancelar"
                            leitura["mensagem"] = mensagem_janela or "A leitura apareceu tarde demais para virar tip operacional."
                            bloqueio_operacional = True

                    if veredito == "cancelar" and not bloqueio_operacional and self._deve_suprimir_cancelamento_tardio(item, game):
                        veredito = "monitorar"

                    if payload.get("live_signal_notified") and veredito == "cancelar":
                        veredito = "monitorar"

                    if veredito == "sinal_live" and item.get("watch_type") in {"approved_prelive", "live_opportunity"}:
                        now_local = datetime.now(ZoneInfo(TIMEZONE))
                        if payload.get("price_watch_final_state") == "expired" and not payload.get("price_watch_active"):
                            veredito = "cancelar"
                            leitura["mensagem"] = payload.get("price_watch_reason") or "Janela de preco encerrada sem odd executavel."
                        elif payload.get("price_watch_active") and not self._deve_consultar_odd_live_item(item, payload, now_local):
                            veredito = "monitorar"
                        else:
                            payload = self._consultar_odd_live_agora(item, payload)
                            item["payload"] = payload
                        if veredito == "sinal_live" and self._odd_ev_live_aprovados(payload):
                            self._encerrar_janela_preco_live(payload, now_local)
                        elif veredito == "sinal_live":
                            acao_preco, motivo_preco = self._acionar_janela_preco_live_inteligente(
                                payload,
                                item,
                                elapsed,
                                now_local,
                            )
                            motivo_preco = self._normalizar_motivo_preco_live(payload, motivo_preco)
                            payload["last_live_verdict"] = "price_watch"
                            payload["last_live_minute"] = elapsed
                            payload["last_live_message"] = leitura.get("mensagem", item.get("note"))
                            payload["price_watch_reason"] = motivo_preco
                            self.db.atualizar_live_watch_item(
                                item_id,
                                note=motivo_preco,
                                payload=payload,
                            )
                            item["note"] = motivo_preco
                            item["payload"] = payload
                            if acao_preco == "started":
                                bloco_preco = self._formatar_alerta_preco_live(
                                    {
                                        **item,
                                        "payload": payload,
                                        "elapsed": elapsed,
                                    },
                                    titulo="🟡 <b>Janela de preço em observação</b>",
                                    contexto=f"Cenário passou, mas a odd ainda não ficou boa; vou rechecando por até {LIVE_PRICE_WATCH_WINDOW_MIN} minutos.",
                                )
                                alertas_admin.append(
                                    bloco_preco
                                )
                                alertas_preco_publicos.append(bloco_preco)
                            if acao_preco == "expired":
                                veredito = "cancelar"
                                leitura["mensagem"] = motivo_preco
                            else:
                                veredito = "monitorar"

                    emitir_sinal_publico = False
                    if veredito in {"sinal_live", "cancelar"}:
                        repetir = last_verdict == veredito
                        if not repetir:
                            if veredito == "sinal_live":
                                if item.get("watch_type") in {"approved_prelive", "live_opportunity"}:
                                    emoji = "🟢"
                                    titulo = "Entrada live liberada"
                                else:
                                    emoji = "🟡"
                                    titulo = "Reavaliacao live"
                            else:
                                emoji = "🔴"
                                titulo = "Leitura cancelada"
                            alertas_admin.append(
                                f"{emoji} <b>{titulo}</b>\n"
                                f"<code>{home_name}</code> <b>x</b> <code>{away_name}</code>\n"
                                f"• {mercado_label}"
                                + (f" | {self._meta_live_contexto(item)}" if self._meta_live_contexto(item) else "")
                                + "\n"
                                f"• Minuto {elapsed}\n"
                                f"• {leitura.get('mensagem', 'Sem leitura adicional.')}"
                            )
                            payload["last_live_verdict"] = veredito
                            payload["last_live_minute"] = elapsed
                            payload["last_live_message"] = leitura.get("mensagem", item.get("note"))
                            note = leitura.get("mensagem", item.get("note"))
                            novo_status = None
                            if veredito == "cancelar":
                                novo_status = "resolved"
                                itens_resolvidos.append(item_id)
                            elif veredito == "sinal_live":
                                emitir_sinal_publico = True
                                payload["live_signal_notified"] = True
                                payload["training_snapshot"] = self._snapshot_leitura_live(leitura)
                                try:
                                    signal_minute = int(leitura.get("elapsed") or 0) or None
                                except Exception:
                                    signal_minute = None
                                self.db.salvar_live_result_signal(
                                    {
                                        **item,
                                        "id": item_id,
                                        "scan_date": scan_date_item,
                                        "payload": payload,
                                    },
                                    signal_minute=signal_minute,
                                    signal_note=note,
                                )
                            self.db.atualizar_live_watch_item(
                                item_id,
                                status=novo_status,
                                note=note,
                                payload=payload,
                            )
                    if (
                        veredito == "sinal_live"
                        and item.get("watch_type") in {"approved_prelive", "live_opportunity"}
                        and emitir_sinal_publico
                    ):
                        sinal = dict(item)
                        sinal["payload"] = payload
                        sinal["elapsed"] = elapsed
                        sinais_live.append(sinal)
                    elif veredito == "monitorar":
                        self.db.atualizar_live_watch_item(
                            item_id,
                            note=leitura.get("mensagem", item.get("note")),
                            payload=payload,
                        )
                    itens_tocados.append(item_id)
                    print(
                        f"   👀 {home_name} vs {away_name} | {mercado_label} | "
                        f"{status} {elapsed}' | {veredito}"
                    )
                    continue

                if status != "FT":
                    if item.get("watch_type") == "manual_fixture" or item.get("mercado") == _MANUAL_LIVE_FIXTURE_MARKET:
                        payload["last_live_verdict"] = "fixture_anchor_waiting"
                        self.db.atualizar_live_watch_item(
                            item_id,
                            note=item.get("note") or "Fixture manual aguardando kickoff/live.",
                            payload=payload,
                        )
                    itens_tocados.append(item_id)
                    continue

                self.db.salvar_fixture(game)
                gh = game.get("goals", {}).get("home")
                ga = game.get("goals", {}).get("away")
                if gh is None or ga is None:
                    continue

                if gh > ga:
                    resultado = "home"
                elif gh == ga:
                    resultado = "draw"
                else:
                    resultado = "away"

                itens_tocados.append(item_id)
                itens_resolvidos.append(item_id)

                if item.get("watch_type") not in {"approved_prelive", "live_opportunity"}:
                    print(f"   📌 {home_name} vs {away_name} finalizado | observação encerrada")
                    continue

                if item.get("watch_type") == "approved_prelive":
                    n = self.db.resolver_prediction(fixture_id, resultado, gh, ga)
                    if n == 0:
                        continue
                resolvidos += 1

                acertou = self._mercado_green_antecipado(item, game, stats_cache.get(fixture_id))
                self.db.resolver_live_result(
                    item_id,
                    resultado=resultado,
                    gols_home=int(gh),
                    gols_away=int(ga),
                    acertou=bool(acertou),
                )

                if acertou:
                    acertos += 1
                    emoji = "✅"
                    resultado_txt = "ACERTOU!"
                else:
                    erros += 1
                    emoji = "❌"
                    resultado_txt = "Errou"

                # Montar notificação individual
                notif = (
                    f"{emoji} <b>{home_name} {gh}-{ga} {away_name}</b>\n"
                    f"   Aposta: {mercado_label}\n"
                    f"   {resultado_txt}"
                )
                notificacoes_publicas.append(notif)
                print(f"   {emoji} {home_name} {gh}-{ga} {away_name} "
                      f"| {mercado_label} | {resultado_txt}")

            if itens_tocados:
                self.db.tocar_live_watchlist(itens_tocados)
            if itens_resolvidos:
                self.db.atualizar_status_live_watchlist(itens_resolvidos, "resolved")

            if alertas_admin:
                bloco = "⚽ <b>Leitura live do FuteBot</b>\n\n" + "\n\n".join(alertas_admin)
                self._enviar_telegram(bloco)

            if alertas_preco_publicos:
                self._enviar_telegram_publico("\n\n".join(alertas_preco_publicos))

            if sinais_live:
                bloco_sinais_live = self._formatar_sinais_live_publicos(sinais_live)
                if bloco_sinais_live:
                    self._enviar_telegram_publico(bloco_sinais_live)

            combos_live = self._gerar_combos_live(sinais_live)
            if combos_live:
                bloco_combo = self._formatar_combos_live(combos_live)
                self._enviar_telegram_publico(bloco_combo)

            if greens_antecipados:
                blocos_hit = []
                for item in greens_antecipados:
                    blocos_hit.append(
                        "✅ <b>Tip ja bateu no live</b>\n"
                        f"<b>{item['home_name']} x {item['away_name']}</b>\n"
                        f"• {item['mercado_label']}\n"
                        f"• Minuto {item['elapsed']}"
                    )
                self._enviar_telegram_publico("\n\n".join(blocos_hit))

            if reds_antecipados:
                blocos_loss = []
                for item in reds_antecipados:
                    blocos_loss.append(
                        "❌ <b>Tip ja perdeu no live</b>\n"
                        f"<b>{item['home_name']} x {item['away_name']}</b>\n"
                        f"• {item['mercado_label']}\n"
                        f"• Minuto {item['elapsed']}"
                    )
                self._enviar_telegram_publico("\n\n".join(blocos_loss))

            bloco_combo_status = self._notificar_progresso_combos_live(
                hoje,
                fixture_cache=fixture_cache,
                stats_cache=stats_cache,
            )
            if bloco_combo_status:
                self._enviar_telegram_publico(bloco_combo_status)

            if notificacoes_publicas:
                header = "⚽ <b>Resultados ao vivo</b>\n\n"
                corpo = "\n\n".join(notificacoes_publicas)
                total_check = acertos + erros
                if total_check > 0:
                    pct = acertos / total_check * 100
                    emoji_total = "🟢" if pct >= 55 else "🔴" if pct < 40 else "🟡"
                    resumo = (
                        f"\n\n{emoji_total} <b>Parcial:</b> "
                        f"{acertos}/{total_check} acertos ({pct:.0f}%)"
                    )
                else:
                    resumo = ""

                self._enviar_telegram_publico(header + corpo + resumo)

                pendentes_hoje = [
                    p for p in self.db.predictions_pendentes()
                    if p.get("date", "")[:10] == hoje
                ]
                if not pendentes_hoje:
                    print("   📋 Todas as previsões do dia resolvidas — enviando resumo")
                    learner = Learner(self.db)
                    resumo_dia = learner.relatorio_resultado_dia(hoje)
                    if "Nenhum resultado" not in resumo_dia and not self._notification_sent(hoje, "daily_results"):
                        self._enviar_telegram_publico(resumo_dia)
                        self._save_notification_sent(hoje, "daily_results")

            if not alertas_admin and not notificacoes_publicas:
                print("   📭 Nenhuma atualização relevante neste ciclo")

        except Exception as e:
            print(f"❌ Erro no check ao vivo: {e}")
            self._enviar_telegram(f"❌ Erro no check ao vivo:\n{e}")
        finally:
            self._sair_secao_critica()

    def _mercado_green_antecipado(self, item: dict, fixture: dict, stats: list[dict] | None = None) -> bool:
        """Detecta se um mercado já ficou matematicamente green antes do FT."""
        mercado = item.get("mercado", "")
        status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "NS")
        goals = fixture.get("goals") or {}
        score = fixture.get("score") or {}
        halftime = score.get("halftime") or {}
        gh = int(goals.get("home") or 0)
        ga = int(goals.get("away") or 0)
        total_gols = gh + ga
        ht_h = int(halftime.get("home") or 0)
        ht_a = int(halftime.get("away") or 0)
        total_ht = ht_h + ht_a
        total_2t = max(0, total_gols - total_ht)
        metricas = LiveIntelligence._stats_totais(stats or [])
        corners = metricas.get("corners", 0.0)

        if mercado == "over15":
            return total_gols > 1
        if mercado == "over25":
            return total_gols > 2
        if mercado == "over35":
            return total_gols > 3
        if mercado == "over05_ht":
            return total_ht > 0 or (status == "1H" and total_gols > 0)
        if mercado == "over15_ht":
            return total_ht > 1 or (status == "1H" and total_gols > 1)
        if mercado == "under05_ht" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return total_ht < 1
        if mercado == "under15_ht" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return total_ht < 2
        if mercado == "ht_home" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return ht_h > ht_a
        if mercado == "ht_draw" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return ht_h == ht_a
        if mercado == "ht_away" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return ht_h < ht_a
        if mercado == "over05_2t":
            return total_2t > 0
        if mercado == "over15_2t":
            return total_2t > 1
        if mercado == "corners_over_85":
            return corners > 8.5
        if mercado == "corners_over_95":
            return corners > 9.5
        if mercado == "corners_over_105":
            return corners > 10.5
        if mercado == "corners_under_85" and status in {"FT", "AET", "PEN"}:
            return corners < 8.5
        if mercado == "corners_under_95" and status in {"FT", "AET", "PEN"}:
            return corners < 9.5
        if mercado == "corners_under_105" and status in {"FT", "AET", "PEN"}:
            return corners < 10.5
        if mercado == "under15" and status in {"FT", "AET", "PEN"}:
            return total_gols < 2
        if mercado == "under25" and status in {"FT", "AET", "PEN"}:
            return total_gols < 3
        if mercado == "under35" and status in {"FT", "AET", "PEN"}:
            return total_gols < 4
        if mercado == "under05_2t" and status in {"FT", "AET", "PEN"}:
            return total_2t < 1
        if mercado == "under15_2t" and status in {"FT", "AET", "PEN"}:
            return total_2t < 2
        if mercado == "h2h_home" and status in {"FT", "AET", "PEN"}:
            return gh > ga
        if mercado == "h2h_draw" and status in {"FT", "AET", "PEN"}:
            return gh == ga
        if mercado == "h2h_away" and status in {"FT", "AET", "PEN"}:
            return gh < ga
        return False

    @staticmethod
    def _deve_suprimir_cancelamento_tardio(item: dict, fixture: dict) -> bool:
        """Evita cancelamento tardio em under quando o mercado ja esta na reta final."""
        mercado = (item.get("mercado") or "").lower()
        if "under" not in mercado:
            return False

        status_info = (fixture.get("fixture") or {}).get("status") or {}
        status = (status_info.get("short") or "").upper()
        elapsed = int(status_info.get("elapsed") or 0)

        if status not in {"1H", "2H", "HT", "LIVE", "ET"}:
            return False

        if mercado.endswith("_ht") or mercado.startswith("ht_"):
            return elapsed >= 43
        if mercado.endswith("_2t"):
            return elapsed >= 88
        return elapsed >= 88

    @staticmethod
    def _janela_operacional_live(item: dict) -> tuple[int, int]:
        return janela_operacional_live(item.get("mercado") or "")

    @classmethod
    def _status_janela_operacional_live(
        cls,
        item: dict,
        elapsed: int | str | None,
    ) -> tuple[str, str]:
        return status_janela_operacional_live(item.get("mercado") or "", elapsed)

    @classmethod
    def _deve_bloquear_sinal_tardio(cls, item: dict, elapsed: int | str | None) -> bool:
        status, _ = cls._status_janela_operacional_live(item, elapsed)
        return status == "tarde"

    def _mercado_red_antecipado(self, item: dict, fixture: dict, stats: list[dict] | None = None) -> bool:
        """Detecta se um mercado já ficou matematicamente red antes do FT."""
        mercado = item.get("mercado", "")
        status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "NS")
        goals = fixture.get("goals") or {}
        score = fixture.get("score") or {}
        halftime = score.get("halftime") or {}
        gh = int(goals.get("home") or 0)
        ga = int(goals.get("away") or 0)
        total_gols = gh + ga
        ht_h = int(halftime.get("home") or 0)
        ht_a = int(halftime.get("away") or 0)
        total_ht = ht_h + ht_a
        total_2t = max(0, total_gols - total_ht)
        metricas = LiveIntelligence._stats_totais(stats or [])
        corners = metricas.get("corners", 0.0)

        if mercado == "under15":
            return total_gols > 1
        if mercado == "under25":
            return total_gols > 2
        if mercado == "under35":
            return total_gols > 3
        if mercado == "under05_ht":
            return total_ht > 0 or (status == "1H" and total_gols > 0)
        if mercado == "under15_ht":
            return total_ht > 1 or (status == "1H" and total_gols > 1)
        if mercado == "over05_ht" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return total_ht < 1
        if mercado == "over15_ht" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return total_ht < 2
        if mercado == "under05_2t":
            return total_2t > 0
        if mercado == "under15_2t":
            return total_2t > 1
        if mercado == "over05_2t" and status in {"FT", "AET", "PEN"}:
            return total_2t < 1
        if mercado == "over15_2t" and status in {"FT", "AET", "PEN"}:
            return total_2t < 2
        if mercado == "corners_under_85":
            return corners > 8.5
        if mercado == "corners_under_95":
            return corners > 9.5
        if mercado == "corners_under_105":
            return corners > 10.5
        if mercado == "corners_over_85" and status in {"FT", "AET", "PEN"}:
            return corners < 8.5
        if mercado == "corners_over_95" and status in {"FT", "AET", "PEN"}:
            return corners < 9.5
        if mercado == "corners_over_105" and status in {"FT", "AET", "PEN"}:
            return corners < 10.5
        if mercado == "ht_home" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return ht_h <= ht_a
        if mercado == "ht_draw" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return ht_h != ht_a
        if mercado == "ht_away" and status in {"HT", "2H", "FT", "AET", "PEN"}:
            return ht_h >= ht_a
        if mercado == "over15" and status in {"FT", "AET", "PEN"}:
            return total_gols < 2
        if mercado == "over25" and status in {"FT", "AET", "PEN"}:
            return total_gols < 3
        if mercado == "over35" and status in {"FT", "AET", "PEN"}:
            return total_gols < 4
        if mercado == "h2h_home" and status in {"FT", "AET", "PEN"}:
            return gh <= ga
        if mercado == "h2h_draw" and status in {"FT", "AET", "PEN"}:
            return gh != ga
        if mercado == "h2h_away" and status in {"FT", "AET", "PEN"}:
            return gh >= ga
        return False

    @staticmethod
    def _mercados_live_expandiveis() -> list[str]:
        return [
            "over15", "under15", "over25", "under25", "over35", "under35",
            "ht_home", "ht_draw", "ht_away",
            "over05_ht", "under05_ht", "over15_ht", "under15_ht",
            "over05_2t", "under05_2t", "over15_2t", "under15_2t",
            "h2h_home", "h2h_draw", "h2h_away",
            "corners_over_85", "corners_under_85",
            "corners_over_95", "corners_under_95",
            "corners_over_105", "corners_under_105",
        ]

    @staticmethod
    def _categoria_live_mercado(mercado: str) -> str:
        return _MERCADO_CATEGORIA.get(mercado, mercado)

    @staticmethod
    def _mercados_da_categoria_live(categoria: str) -> tuple[str, ...]:
        return _CATEGORIA_MERCADOS.get(categoria, (categoria,))

    @staticmethod
    def _contexto_equivalencia_live(fixture: dict | None) -> dict[str, int | str] | None:
        if not fixture:
            return None
        status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "NS")
        goals = fixture.get("goals") or {}
        score = fixture.get("score") or {}
        halftime = score.get("halftime") or {}
        total_gols = int(goals.get("home") or 0) + int(goals.get("away") or 0)
        gols_ht = int(halftime.get("home") or 0) + int(halftime.get("away") or 0)
        gols_2t = max(0, total_gols - gols_ht)
        return {
            "status": status,
            "total_gols": total_gols,
            "gols_ht": gols_ht,
            "gols_2t": gols_2t,
        }

    @staticmethod
    def _mercados_equivalentes_live(mercado: str, fixture: dict | None = None) -> tuple[str, ...]:
        ctx = Scheduler._contexto_equivalencia_live(fixture)
        if not ctx:
            return ()
        status = str(ctx["status"])
        gols_ht = int(ctx["gols_ht"])
        gols_2t = int(ctx["gols_2t"])
        if status not in {"2H", "LIVE", "ET"} or gols_2t != 0:
            return ()

        equivalencias = {
            0: {
                "over15": {"over15_2t"},
                "under15": {"under15_2t"},
                "over15_2t": {"over15"},
                "under15_2t": {"under15"},
            },
            1: {
                "over15": {"over05_2t"},
                "under15": {"under05_2t"},
                "over25": {"over15_2t"},
                "under25": {"under15_2t"},
                "over05_2t": {"over15"},
                "under05_2t": {"under15"},
                "over15_2t": {"over25"},
                "under15_2t": {"under25"},
            },
            2: {
                "over25": {"over05_2t"},
                "under25": {"under05_2t"},
                "over35": {"over15_2t"},
                "under35": {"under15_2t"},
                "over05_2t": {"over25"},
                "under05_2t": {"under25"},
                "over15_2t": {"over35"},
                "under15_2t": {"under35"},
            },
            3: {
                "over35": {"over05_2t"},
                "under35": {"under05_2t"},
                "over05_2t": {"over35"},
                "under05_2t": {"under35"},
            },
        }
        return tuple(sorted(equivalencias.get(gols_ht, {}).get(mercado, set())))

    @staticmethod
    def _mercados_conflitantes_live(mercado: str, fixture: dict | None = None) -> tuple[str, ...]:
        categoria = _MERCADO_CATEGORIA.get(mercado, mercado)
        mercados = set(_CATEGORIA_MERCADOS.get(categoria, {mercado}))
        mercados.update(_LIVE_CONFLITOS_EXPLICITOS.get(mercado, set()))
        mercados.update(Scheduler._mercados_equivalentes_live(mercado, fixture))
        return tuple(sorted(mercados))

    def _detectar_oportunidades_live_fixture(
        self,
        *,
        scan_date: str,
        fixture: dict,
        stats: list[dict],
        itens_fixture: list[dict],
        nomes_mercado: dict[str, str],
        live: LiveIntelligence,
    ) -> tuple[list[str], list[dict]]:
        if not itens_fixture:
            return [], []

        fixture_id = itens_fixture[0]["fixture_id"]
        home_name = itens_fixture[0].get("home_name", "?")
        away_name = itens_fixture[0].get("away_name", "?")
        base_payload = dict((itens_fixture[0].get("payload") or {}))
        mercados_existentes = {item.get("mercado") for item in itens_fixture if item.get("mercado")}
        categorias_existentes = {
            self._categoria_live_mercado(mercado)
            for mercado in mercados_existentes
            if mercado
        }
        alertas = []
        sinais = []

        for mercado in self._mercados_live_expandiveis():
            if mercado in mercados_existentes:
                continue
            categoria = self._categoria_live_mercado(mercado)
            mercados_conflitantes = self._mercados_conflitantes_live(mercado, fixture)
            if categoria in categorias_existentes:
                continue
            if any(m in mercados_existentes for m in mercados_conflitantes):
                categorias_existentes.add(categoria)
                continue
            if self.db.live_category_notification_exists(
                scan_date,
                fixture_id,
                mercados_conflitantes,
                "sinal_live",
            ):
                categorias_existentes.add(categoria)
                continue

            item_virtual = {
                "scan_date": scan_date,
                "fixture_id": fixture_id,
                "date": itens_fixture[0].get("fixture_date") or itens_fixture[0].get("date"),
                "fixture_date": itens_fixture[0].get("fixture_date") or itens_fixture[0].get("date"),
                "league_id": itens_fixture[0].get("league_id"),
                "home_name": home_name,
                "away_name": away_name,
                "mercado": mercado,
                "descricao": nomes_mercado.get(mercado, mercado),
                "watch_type": "live_opportunity",
            }
            leitura = live.analisar(item_virtual, fixture, stats)
            if leitura.get("veredito") != "sinal_live":
                continue
            status_janela, _ = self._status_janela_operacional_live(item_virtual, leitura.get("elapsed"))
            if status_janela != "ok":
                categorias_existentes.add(categoria)
                continue
            if self.db.live_market_notification_exists(scan_date, fixture_id, mercado, "sinal_live"):
                categorias_existentes.add(categoria)
                continue
            if self.db.live_category_notification_exists(
                scan_date,
                fixture_id,
                mercados_conflitantes,
                "sinal_live",
            ):
                categorias_existentes.add(categoria)
                continue

            payload = dict(base_payload)
            payload["last_live_verdict"] = "sinal_live"
            payload["last_live_minute"] = leitura.get("elapsed")
            payload["last_live_message"] = leitura.get("mensagem")
            payload["origin"] = "fixture_live_scan"
            payload["live_reading"] = self._snapshot_leitura_live(leitura)
            payload["metricas_live"] = payload["live_reading"].get("metricas")
            payload["estado_partida"] = leitura.get("estado_partida")
            item_salvo = {
                **item_virtual,
                "prob_modelo": self._inferir_prob_modelo_live(mercado, leitura),
                "status": "active",
                "note": leitura.get("mensagem"),
                "payload": payload,
            }
            payload = self._consultar_odd_live_agora(item_salvo, payload)
            item_salvo["payload"] = payload
            item_id = self.db.salvar_live_watch_item(scan_date, item_salvo)
            now_local = datetime.now(ZoneInfo(TIMEZONE))

            if self._odd_ev_live_aprovados(payload):
                self._encerrar_janela_preco_live(payload, now_local)
                payload["training_snapshot"] = self._snapshot_leitura_live(leitura)
                self.db.atualizar_live_watch_item(item_id, payload=payload)
                try:
                    signal_minute = int(leitura.get("elapsed") or 0) or None
                except Exception:
                    signal_minute = None
                self.db.salvar_live_result_signal(
                    {
                        **item_salvo,
                        "id": item_id,
                        "scan_date": scan_date,
                        "payload": payload,
                    },
                    signal_minute=signal_minute,
                    signal_note=leitura.get("mensagem"),
                )
                self.db.salvar_live_market_notification(scan_date, fixture_id, mercado, "sinal_live")

                alertas.append(
                    f"🟢 <b>Entrada live liberada</b>\n"
                    f"<code>{home_name}</code> <b>x</b> <code>{away_name}</code>\n"
                    f"• {nomes_mercado.get(mercado, mercado)}"
                    + (f" | {self._meta_live_contexto({'payload': payload, **item_salvo})}" if self._meta_live_contexto({'payload': payload, **item_salvo}) else "")
                    + "\n"
                    f"• Minuto {leitura.get('elapsed') or '?'}\n"
                    f"• {leitura.get('mensagem', 'Sem leitura adicional.')}"
                )
                sinais.append({
                    **item_salvo,
                    "id": item_id,
                    "payload": payload,
                    "elapsed": leitura.get("elapsed") or "?",
                })
            else:
                acao_preco, motivo_preco = self._acionar_janela_preco_live_inteligente(
                    payload,
                    item_salvo,
                    leitura.get("elapsed"),
                    now_local,
                )
                motivo_preco = self._normalizar_motivo_preco_live(payload, motivo_preco)
                novo_status = "resolved" if acao_preco == "expired" else None
                self.db.atualizar_live_watch_item(item_id, status=novo_status, note=motivo_preco, payload=payload)
                if acao_preco == "started":
                    bloco_preco = self._formatar_alerta_preco_live(
                        {
                            **item_salvo,
                            "id": item_id,
                            "payload": payload,
                            "elapsed": leitura.get("elapsed") or "?",
                            "note": motivo_preco,
                        },
                        titulo="🟡 <b>Janela de preço em observação</b>",
                        contexto=f"Cenário passou, mas a odd ainda não ficou boa; vou rechecando por até {LIVE_PRICE_WATCH_WINDOW_MIN} minutos.",
                    )
                    alertas.append(bloco_preco)
                elif acao_preco == "expired":
                    alertas.append(
                        f"🔴 <b>Leitura cancelada</b>\n"
                        f"<code>{home_name}</code> <b>x</b> <code>{away_name}</code>\n"
                        f"• {nomes_mercado.get(mercado, mercado)}"
                        + (
                            f" | {self._meta_live_contexto({'payload': payload, **item_salvo})}"
                            if self._meta_live_contexto({'payload': payload, **item_salvo})
                            else ""
                        )
                        + "\n"
                        f"• Minuto {leitura.get('elapsed') or '?'}\n"
                        f"• {motivo_preco}"
                    )
            categorias_existentes.add(categoria)

        return alertas, sinais

    def _notificar_progresso_combos_live(
        self,
        data: str,
        *,
        fixture_cache: dict[int, dict],
        stats_cache: dict[int, list[dict]],
    ) -> str:
        """Gera bloco de progresso dos combos quando uma ou mais pernas já bateram."""
        combos = self.db.combos_por_data(data)
        if not combos:
            return ""

        linhas = []
        for idx, combo in enumerate(combos, start=1):
            itens = combo.get("items") or []
            if not itens:
                continue

            locked = []
            lost = []
            pendentes = []
            for item in itens:
                fixture_id = item.get("fixture_id")
                if item.get("acertou") == 1:
                    locked.append(item)
                    continue
                if item.get("acertou") == 0:
                    lost.append(item)
                    continue
                game = fixture_cache.get(fixture_id)
                stats = stats_cache.get(fixture_id)
                if game and self._mercado_green_antecipado(item, game, stats):
                    locked.append(item)
                elif game and self._mercado_red_antecipado(item, game, stats):
                    lost.append(item)
                else:
                    pendentes.append(item)

            if not locked and not lost:
                continue

            if lost:
                progress_key = f"lost_{len(lost)}_{len(locked)}"
            else:
                progress_key = f"locked_{len(locked)}"
            combo_id = combo.get("id")
            if combo_id and self.db.combo_live_notification_exists(combo_id, progress_key):
                continue
            if combo_id:
                self.db.salvar_combo_live_notification(combo_id, progress_key)

            tipo = "Dupla" if combo.get("combo_type") == "dupla" else "Tripla"
            if lost:
                linhas.append(f"❌ <b>{tipo} perdida #{idx}</b>")
                linhas.append(f"• {len(lost)} perna(s) ja deram red")
            else:
                linhas.append(f"🟣 <b>{tipo} em andamento #{idx}</b>")
                linhas.append(f"• {len(locked)}/{len(itens)} perna(s) ja bateram")
            for item in locked:
                meta_live = self._meta_live_contexto(item)
                linhas.append(
                    f"✅ {item.get('home_name', '?')} x {item.get('away_name', '?')} | {item.get('mercado', '?')}"
                    + (f" | {meta_live}" if meta_live else "")
                )
            for item in lost:
                meta_live = self._meta_live_contexto(item)
                linhas.append(
                    f"❌ {item.get('home_name', '?')} x {item.get('away_name', '?')} | {item.get('mercado', '?')}"
                    + (f" | {meta_live}" if meta_live else "")
                )
            for item in pendentes:
                meta_live = self._meta_live_contexto(item)
                linhas.append(
                    f"⏳ Falta: {item.get('home_name', '?')} x {item.get('away_name', '?')} | {item.get('mercado', '?')}"
                    + (f" | {meta_live}" if meta_live else "")
                )
            linhas.append("")

        if not linhas:
            return ""
        linhas.append("<i>Atualizacao live dos combos do dia.</i>")
        return "\n".join(linhas).strip()

    def _gerar_combos_live(self, sinais_live: list[dict]) -> list[dict]:
        """Gera duplas/triplas live sem repetir o mesmo combo a cada ciclo."""
        if len(sinais_live) < 2:
            return []

        scanner = Scanner(self.db)
        combos = scanner._gerar_combos(sinais_live)
        if not combos:
            return []

        novos = []
        for combo in combos:
            ids = sorted(int(item["id"]) for item in combo["tips"] if item.get("id") is not None)
            if len(ids) < 2:
                continue
            combo_key = "live:" + "-".join(str(i) for i in ids)
            ja_enviado = True
            for item in combo["tips"]:
                payload = dict(item.get("payload") or {})
                enviados = set(payload.get("sent_live_combo_keys") or [])
                if combo_key not in enviados:
                    ja_enviado = False
            if ja_enviado:
                continue

            for item in combo["tips"]:
                payload = dict(item.get("payload") or {})
                enviados = list(dict.fromkeys((payload.get("sent_live_combo_keys") or []) + [combo_key]))
                payload["sent_live_combo_keys"] = enviados
                self.db.atualizar_live_watch_item(item["id"], payload=payload)
                item["payload"] = payload
            combo["combo_key"] = combo_key
            novos.append(combo)
        return novos

    @staticmethod
    def _odd_live_contexto(item: dict) -> float | None:
        payload = item.get("payload") or {}
        for key in ("odd_usada", "odd_pinnacle", "odd"):
            value = item.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            value = payload.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _bookmaker_live_contexto(item: dict) -> str:
        payload = item.get("payload") or {}
        return (
            item.get("bookmaker")
            or item.get("odd_fonte")
            or payload.get("bookmaker")
            or payload.get("odd_fonte")
            or "1xBet"
        )

    @classmethod
    def _meta_live_contexto(cls, item: dict) -> str:
        partes = []
        odd = cls._odd_live_contexto(item)
        if odd and odd > 1:
            partes.append(f"Odd {odd:.2f}")
        payload = item.get("payload") or {}
        ev = item.get("ev_percent")
        if ev in (None, ""):
            ev = payload.get("ev_percent")
        if ev not in (None, ""):
            try:
                partes.append(f"EV {float(ev):+.1f}%")
            except (TypeError, ValueError):
                pass
        casa = cls._bookmaker_live_contexto(item)
        if casa and partes:
            partes.append(casa)
        return " | ".join(partes)

    @classmethod
    def _sinal_live_publicavel(cls, item: dict) -> bool:
        odd = cls._odd_live_contexto(item)
        if odd is None or odd <= 1:
            return False
        payload = item.get("payload") or {}
        ev = item.get("ev_percent")
        if ev in (None, ""):
            ev = payload.get("ev_percent")
        try:
            float(ev)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _formatar_combos_live(combos_live: list[dict]) -> str:
        """Formata sugestoes de combo live para o publico."""
        linhas = ["🟣 <b>Combos live em observacao</b>"]
        for idx, combo in enumerate(combos_live, 1):
            tipo = "Dupla" if combo.get("tipo") == "dupla" else "Tripla"
            detalhes_combo = [f"Conf composta {combo.get('prob_composta', 0):.0%}"]
            odd_composta = combo.get("odd_composta")
            if odd_composta:
                detalhes_combo.append(f"Odd {odd_composta:.2f}")
            ev_composto = combo.get("ev_composto")
            if ev_composto is not None:
                detalhes_combo.append(f"EV {ev_composto:+.1f}%")
            linhas.append(f"\n<b>{tipo} live #{idx}</b> | {' | '.join(detalhes_combo)}")
            for item in combo.get("tips", []):
                home = item.get("home_name", "?")
                away = item.get("away_name", "?")
                mercado = item.get("descricao") or item.get("mercado", "")
                minuto = item.get("elapsed") or "?"
                linhas.append(f"• <b>{home} x {away}</b>")
                detalhes = [mercado, f"Conf {item.get('prob_modelo', 0):.0%}", f"Min {minuto}"]
                meta_live = Scheduler._meta_live_contexto(item)
                if meta_live:
                    detalhes.append(meta_live)
                linhas.append(f"  <i>{' | '.join(detalhes)}</i>")
        linhas.append("\n<i>Combo live sugestivo. Nao entra nas metricas oficiais do dia.</i>")
        return "\n".join(linhas)

    @staticmethod
    def _formatar_sinais_live_publicos(sinais_live: list[dict]) -> str:
        """Formata sinais live unitarios para o publico, inclusive reavaliacoes."""
        sinais_validos = [item for item in sinais_live if Scheduler._sinal_live_publicavel(item)]
        if not sinais_validos:
            return ""

        linhas = ["⚽ <b>Oportunidades live do FuteBot</b>"]
        for item in sinais_validos:
            home = item.get("home_name", "?")
            away = item.get("away_name", "?")
            mercado = item.get("descricao") or item.get("mercado", "")
            minuto = item.get("elapsed") or "?"
            payload = item.get("payload") or {}
            mensagem = payload.get("last_live_message") or item.get("note") or "Sem leitura adicional."
            if item.get("watch_type") == "blocked_recheck":
                titulo = "🟡 <b>Reavaliacao live</b>"
                contexto = "Entrada barrada no pre-live, mas o jogo seguiu em observacao."
            else:
                titulo = "🟢 <b>Entrada live liberada</b>"
                contexto = "Leitura confirmada no acompanhamento ao vivo."
            meta_live = Scheduler._meta_live_contexto(item)

            linhas.append(
                "\n".join([
                    "",
                    titulo,
                    f"<code>{home}</code> <b>x</b> <code>{away}</code>",
                    f"• {mercado}" + (f" | {meta_live}" if meta_live else ""),
                    f"• Minuto {minuto}",
                    f"• {mensagem}",
                    f"<i>{contexto}</i>",
                ]).rstrip()
            )
        return "\n\n".join(linhas)

    def _post_telegram(self, chat_id: int, texto: str, reply_markup: dict | None = None) -> bool:
        """Envia uma mensagem para um chat específico via API HTTP do Telegram."""
        from config import TELEGRAM_TOKEN
        if not TELEGRAM_TOKEN:
            print("[Scheduler] ⚠️ TELEGRAM_TOKEN ausente, msg não enviada")
            return False

        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            msg = texto[:4000] + "\n\n... (truncado)" if len(texto) > 4000 else texto
            payload = {
                "chat_id": int(chat_id),
                "text": msg,
                "parse_mode": "HTML",
            }
            if reply_markup:
                import json as _json
                payload["reply_markup"] = _json.dumps(reply_markup)
            resp = requests.post(url, json=payload, timeout=(5, 10))
            if resp.status_code == 200:
                return True
            print(f"[Scheduler] ⚠️ Telegram API {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[Scheduler] ❌ Erro ao enviar Telegram: {e}")
        return False

    def _enviar_telegram(self, texto: str, reply_markup: dict | None = None):
        """Envia mensagem operacional ao chat admin principal.

        Não depende de _app_instance nem do event loop do bot.
        APScheduler roda jobs em ThreadPoolExecutor, então usamos
        requests.post direto — simples e confiável.

        Parâmetros:
          texto: mensagem HTML
          reply_markup: dict com inline_keyboard (opcional, para botões ✏️ Odd)
        """
        from config import TELEGRAM_CHAT_ID
        if TELEGRAM_CHAT_ID:
            ok = self._post_telegram(int(TELEGRAM_CHAT_ID), texto, reply_markup=reply_markup)
            if ok:
                print("[Scheduler] ✅ Msg enviada ao Telegram")
        else:
            print("[Scheduler] ⚠️ TELEGRAM_TOKEN ou CHAT_ID ausente, msg não enviada")
        print(texto)

    def _enviar_telegram_publico(self, texto: str, reply_markup: dict | None = None):
        """Envia radar/tips para todos os chats registrados."""
        chat_ids = self.db.telegram_chat_ids()
        if not chat_ids:
            print("[Scheduler] ⚠️ Nenhum chat registrado, fallback para admin")
            self._enviar_telegram(texto, reply_markup=reply_markup)
            return

        entregues = 0
        for chat_id in chat_ids:
            if self._post_telegram(chat_id, texto, reply_markup=reply_markup):
                entregues += 1
        print(f"[Scheduler] ✅ Msg pública enviada para {entregues}/{len(chat_ids)} chats")
        print(texto)

    # ══════════════════════════════════════════════
    #  EXECUÇÃO MANUAL (para testes)
    # ══════════════════════════════════════════════

    def executar_agora(self, job: str = "scanner"):
        """
        Executa um job imediatamente (para teste/debug).

        Parâmetros:
          job: 'scanner', 'coletar', 'relatorio', 'retreinar'
        """
        jobs = {
            "scanner": self._job_scanner,
            "liberacao_t30": self._job_liberacao_t30,
            "coletar": self._job_coletar,
            "relatorio": self._job_relatorio,
            "retreinar": self._job_retreinar,
            "bulk": self._job_bulk,
            "ao_vivo": self._job_check_ao_vivo,
        }

        if job in jobs:
            jobs[job]()
        else:
            print(f"Job desconhecido: {job}. Opções: {list(jobs.keys())}")

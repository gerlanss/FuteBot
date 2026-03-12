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
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from data.database import Database
from data.bulk_download import baixar_fixtures, baixar_stats, _check_limite
from pipeline.scanner import Scanner
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
    DISCOVERY_MIN_TRAIN_SAMPLES, DISCOVERY_MIN_TEST_SAMPLES, DISCOVERY_MIN_TEST_SAMPLES_COPA,
    DISCOVERY_OPTUNA_TRIALS, DISCOVERY_CUP_LEAGUE_IDS, LEAGUES, LIBERACAO_T30_INTERVALO_MIN,
)

from services.apifootball import raw_request


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
        self.scheduler = BackgroundScheduler(timezone=TIMEZONE)
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
            CronTrigger(hour=int(h_scan), minute=int(m_scan)),
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
            CronTrigger(hour="10-23/2", minute=30),
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
        print(f"   🔍 Scanner: {SCAN_HORA} (diário)")
        print(f"   ⏳ Liberação T-30: a cada {LIBERACAO_T30_INTERVALO_MIN} min")
        print(f"   📋 Relatório: 06:45 (resultados de ontem)")
        print(f"   ⚽ Ao vivo: a cada 2h 10:30-23:30")
        print(f"   🤖 Retreino: 1º {RETREINO_DIA} do mês {RETREINO_HORA} per-league (mensal)")
        print()

        self.scheduler.start()

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

    @staticmethod
    def _infer_conf_band(best_rule: dict | None) -> tuple[float, float]:
        conf_min = 0.65
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
                min_test = DISCOVERY_MIN_TEST_SAMPLES_COPA if self._liga_eh_copa(lid) else DISCOVERY_MIN_TEST_SAMPLES
                markets = faltantes.get(lid) or None
                league_name = next(
                    (info["nome"] for info in LEAGUES.values() if int(info["id"]) == lid),
                    f"Liga {lid}",
                )
                print(f"   🎯 Discovery liga {lid} ({league_name}) | mercados={markets or 'todos'} | min_test={min_test}")
                result = trainer.run(
                    league_ids=[lid],
                    markets=markets,
                    target_precision=DISCOVERY_TARGET_PRECISION,
                    min_train_samples=DISCOVERY_MIN_TRAIN_SAMPLES,
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
        print(f"\n{'='*60}")
        print(f"🔍 JOB: Scanner de oportunidades — {datetime.now()}")
        print(f"{'='*60}")

        try:
            scanner = Scanner(self.db)
            resultado = scanner.executar(dias_adiante=0)  # Apenas jogos do dia

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

    def _job_liberacao_t30(self):
        """Job: libera os mercados na janela T-30 e publica o lote final."""
        print(f"\n{'='*60}")
        print(f"⏳ JOB: Liberação T-30 — {datetime.now()}")
        print(f"{'='*60}")

        try:
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

            # 1. Resultados de ONTEM (dia anterior completo)
            ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            resultado_dia = learner.relatorio_resultado_dia(ontem)
            if "Nenhum resultado" not in resultado_dia:
                self._enviar_telegram(resultado_dia)

            # 2. Relatório de performance geral
            msg = learner.relatorio_diario()
            self._enviar_telegram(msg)

            # 3. Saúde do modelo (guard rails)
            saude_msg = learner.relatorio_saude()
            self._enviar_telegram(saude_msg)

            # Se modelo degradado, alertar com urgência
            check = learner.verificar_degradacao()
            if check["degradado"] or check["pausado"]:
                urgencia = (
                    "🚨 *ALERTA: Modelo precisa de atenção!*\n\n"
                    + "\n".join(check["alertas"])
                    + "\n\nUse /treinar para retreinar."
                )
                self._enviar_telegram(urgencia)

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
        Job: verifica jogos previstos que já finalizaram e notifica resultado.

        Roda a cada 2h (10:30-23:30). Para cada previsão pendente de hoje/ontem:
          1. Consulta status do jogo na API-Football (por fixture_id)
          2. Se FT → resolve previsão e envia notificação individual
          3. Se não FT → ignora (será verificado no próximo ciclo)

        Custo: ~1 request por previsão pendente (máximo ~10/ciclo).
        """
        print(f"\n{'='*60}")
        print(f"⚽ JOB: Check ao vivo — {datetime.now()}")
        print(f"{'='*60}")

        try:
            pendentes = self.db.predictions_pendentes()
            if not pendentes:
                print("   ✅ Nenhuma previsão pendente")
                return

            # Filtrar apenas previsões de hoje e ontem (não verificar antigas demais)
            hoje = datetime.now().strftime("%Y-%m-%d")
            ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            recentes = [
                p for p in pendentes
                if p.get("date", "")[:10] in (hoje, ontem)
            ]

            if not recentes:
                print("   ✅ Nenhuma previsão recente pendente")
                return

            print(f"   🔍 Verificando {len(recentes)} previsões pendentes...")

            resolvidos = 0
            acertos = 0
            erros = 0
            notificacoes = []

            for pred in recentes:
                fixture_id = pred["fixture_id"]

                # Consultar status atual do jogo na API
                r = raw_request("fixtures", {"id": fixture_id})
                resp = r.get("response", [])
                if not resp:
                    continue

                game = resp[0]
                status = game.get("fixture", {}).get("status", {}).get("short", "NS")

                if status != "FT":
                    # Jogo ainda não terminou — mostrar status atual
                    elapsed = game.get("fixture", {}).get("status", {}).get("elapsed", "")
                    if status in ("1H", "2H", "HT", "ET"):
                        elapsed = elapsed or "?"
                        print(f"   ⏳ {pred['home_name']} vs {pred['away_name']} — "
                              f"em andamento ({status} {elapsed}')")
                    continue

                # Jogo finalizado — resolver previsão
                self.db.salvar_fixture(game)

                gh = game.get("goals", {}).get("home")
                ga = game.get("goals", {}).get("away")
                if gh is None or ga is None:
                    continue

                # Determinar resultado
                if gh > ga:
                    resultado = "home"
                elif gh == ga:
                    resultado = "draw"
                else:
                    resultado = "away"

                # Resolver no banco
                n = self.db.resolver_prediction(fixture_id, resultado, gh, ga)
                if n == 0:
                    continue

                resolvidos += 1

                # Verificar se acertou
                mercado = pred.get("mercado", "")
                total_gols = gh + ga
                acertou = False

                if mercado == "h2h_home" and resultado == "home":
                    acertou = True
                elif mercado == "h2h_draw" and resultado == "draw":
                    acertou = True
                elif mercado == "h2h_away" and resultado == "away":
                    acertou = True
                elif mercado == "over25" and total_gols > 2:
                    acertou = True
                elif mercado == "under25" and total_gols < 3:
                    acertou = True
                elif mercado == "over15" and total_gols > 1:
                    acertou = True
                elif mercado == "under15" and total_gols < 2:
                    acertou = True
                elif mercado == "over35" and total_gols > 3:
                    acertou = True
                elif mercado == "under35" and total_gols < 4:
                    acertou = True
                elif mercado == "btts_yes" and gh > 0 and ga > 0:
                    acertou = True
                elif mercado == "btts_no" and (gh == 0 or ga == 0):
                    acertou = True

                # Nomes legíveis para os mercados
                nomes_mercado = {
                    "h2h_home": "Casa", "h2h_draw": "Empate", "h2h_away": "Fora",
                    "over15": "Over 1.5", "under15": "Under 1.5",
                    "over25": "Over 2.5", "under25": "Under 2.5",
                    "over35": "Over 3.5", "under35": "Under 3.5",
                    "btts_yes": "BTTS Sim", "btts_no": "BTTS Não",
                    "ht_home": "1T Casa", "ht_draw": "1T Empate", "ht_away": "1T Fora",
                }
                mercado_label = nomes_mercado.get(mercado, mercado)

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
                    f"{emoji} *{pred['home_name']} {gh}-{ga} {pred['away_name']}*\n"
                    f"   Aposta: {mercado_label}\n"
                    f"   {resultado_txt}"
                )
                notificacoes.append(notif)
                print(f"   {emoji} {pred['home_name']} {gh}-{ga} {pred['away_name']} "
                      f"| {mercado_label} | {resultado_txt}")

            # Enviar notificações agrupadas (se houver)
            if notificacoes:
                header = "⚽ <b>Resultados ao vivo</b>\n\n"
                corpo = "\n\n".join(notificacoes)

                # Resumo rápido
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

                self._enviar_telegram(header + corpo + resumo)

                # Se todas as previsões do dia foram resolvidas, enviar RESUMO DO DIA
                pendentes_hoje = [
                    p for p in self.db.predictions_pendentes()
                    if p.get("date", "")[:10] == hoje
                ]
                if not pendentes_hoje:
                    print("   📋 Todas as previsões do dia resolvidas — enviando resumo")
                    learner = Learner(self.db)
                    resumo_dia = learner.relatorio_resultado_dia(hoje)
                    if "Nenhum resultado" not in resumo_dia:
                        self._enviar_telegram(resumo_dia)
            else:
                print("   📭 Nenhum jogo finalizado neste ciclo")

        except Exception as e:
            print(f"❌ Erro no check ao vivo: {e}")
            self._enviar_telegram(f"❌ Erro no check ao vivo:\n{e}")

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
            resp = requests.post(url, json=payload, timeout=15)
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

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

from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from data.database import Database
from data.bulk_download import baixar_fixtures, baixar_stats, _check_limite
from pipeline.scanner import Scanner
from pipeline.collector import Collector
from models.trainer import Trainer
from models.learner import Learner
from config import SCAN_HORA, RESULTADOS_HORA, RETREINO_DIA, RETREINO_HORA, BULK_HORA, MIN_FIXTURES_TREINO, TIMEZONE

# Importar semana do retreino (1st, 2nd, etc.) — controla qual domingo do mês
try:
    from config import RETREINO_SEMANA
except ImportError:
    RETREINO_SEMANA = "1st"  # fallback: 1º domingo
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

        # 3. Relatório noturno (diário às 22h)
        self.scheduler.add_job(
            self._job_relatorio,
            CronTrigger(hour=22, minute=0),
            id="relatorio_noturno",
            name="Relatório noturno",
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
        print(f"   📥 Coleta: {RESULTADOS_HORA} (diário)")
        print(f"   🔍 Scanner: {SCAN_HORA} (diário)")
        print(f"   📋 Relatório: 22:00 (diário)")
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

    def _job_scanner(self):
        """Job: scanner de oportunidades."""
        print(f"\n{'='*60}")
        print(f"🔍 JOB: Scanner de oportunidades — {datetime.now()}")
        print(f"{'='*60}")

        try:
            scanner = Scanner(self.db)
            resultado = scanner.executar(dias_adiante=0)  # Apenas jogos do dia

            # Enviar relatório via Telegram (lista de mensagens: 1 por tip)
            msgs = scanner.formatar_relatorio(resultado)
            for msg in msgs:
                self._enviar_telegram(msg)

        except Exception as e:
            print(f"❌ Erro no scanner: {e}")
            self._enviar_telegram(f"❌ Erro no scanner:\n{e}")

    def _job_relatorio(self):
        """Job: relatório noturno — resultados do dia + saúde do modelo.

        Às 22h envia:
          1. Resultados dos jogos de HOJE (não de ontem)
          2. Relatório de performance geral
          3. Saúde do modelo + alertas de degradação
        """
        print(f"\n{'='*60}")
        print(f"📋 JOB: Relatório noturno — {datetime.now()}")
        print(f"{'='*60}")

        try:
            learner = Learner(self.db)

            # 1. Resultados do dia (jogos de HOJE)
            hoje = datetime.now().strftime("%Y-%m-%d")
            resultado_dia = learner.relatorio_resultado_dia(hoje)
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
            from datetime import timedelta
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

    def _enviar_telegram(self, texto: str):
        """Envia mensagem via API HTTP direta do Telegram (síncrono).

        Não depende de _app_instance nem do event loop do bot.
        APScheduler roda jobs em ThreadPoolExecutor, então usamos
        requests.post direto — simples e confiável.
        """
        from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            try:
                import requests
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                # Truncar se necessário (limite Telegram: 4096 chars)
                msg = texto[:4000] + "\n\n... (truncado)" if len(texto) > 4000 else texto
                payload = {
                    "chat_id": int(TELEGRAM_CHAT_ID),
                    "text": msg,
                    "parse_mode": "HTML",
                }
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 200:
                    print(f"[Scheduler] ✅ Msg enviada ao Telegram")
                else:
                    print(f"[Scheduler] ⚠️ Telegram API {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                print(f"[Scheduler] ❌ Erro ao enviar Telegram: {e}")
        else:
            print("[Scheduler] ⚠️ TELEGRAM_TOKEN ou CHAT_ID ausente, msg não enviada")
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

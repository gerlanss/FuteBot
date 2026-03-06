"""
bot.py — Entry-point principal do FuteBot.

Inicia simultaneamente:
  1. Bot Telegram (polling) — recebe comandos
  2. APScheduler — executa tarefas automáticas

Uso:
  python bot.py              → Modo normal (Telegram + Scheduler)
  python bot.py --no-sched   → Apenas Telegram (debug)
  python bot.py --test       → Testa conexão e sai

Ambiente:
  Todas as credenciais ficam em .env (veja .env.example)
"""

import asyncio
import sys
import os

# Garante que o diretório do projeto está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from config import TELEGRAM_TOKEN, DB_PATH, MIN_FIXTURES_TREINO
from data.database import Database
from data.bulk_download import baixar_fixtures, baixar_stats, _check_limite
from services.telegram_bot import criar_bot, enviar_mensagem
from pipeline.scheduler import Scheduler
from models.trainer import Trainer


def test_conexao():
    """Testa se tudo está configurado corretamente."""
    print("=" * 50)
    print("  FUTEBOT — Teste de Conexão")
    print("=" * 50)

    # 1. Banco de dados
    print("\n1️⃣  Banco de dados...")
    try:
        db = Database()
        r = db.resumo()
        print(f"   ✅ SQLite OK — {r['fixtures']:,} fixtures, {r['predictions']:,} previsões")
    except Exception as e:
        print(f"   ❌ Erro: {e}")

    # 2. API-Football
    print("\n2️⃣  API-Football...")
    try:
        from services.apifootball import raw_request
        status = raw_request("status")
        resp = status.get("response", {})
        sub = resp.get("subscription", {})
        req = resp.get("requests", {})
        print(f"   ✅ Plano: {sub.get('plan', '?')}")
        print(f"   ✅ Requests: {req.get('current', '?')}/{req.get('limit_day', '?')}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")

    # 3. The Odds API
    print("\n3️⃣  The Odds API...")
    try:
        from services.odds_api import creditos_restantes
        creditos = creditos_restantes()
        print(f"   ✅ Créditos restantes: {creditos}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")

    # 4. Telegram
    print("\n4️⃣  Telegram Bot...")
    try:
        import telegram
        async def _test_tg():
            bot = telegram.Bot(token=TELEGRAM_TOKEN)
            me = await bot.get_me()
            return me
        me = asyncio.run(_test_tg())
        print(f"   ✅ Bot: @{me.username} ({me.first_name})")
    except Exception as e:
        print(f"   ❌ Erro: {e}")

    # 5. Modelos
    print("\n5️⃣  Modelos XGBoost...")
    models_dir = os.path.join(os.path.dirname(__file__), "data", "models")
    if os.path.exists(models_dir):
        modelos = [f for f in os.listdir(models_dir) if f.endswith(".json")]
        if modelos:
            print(f"   ✅ Modelos encontrados: {', '.join(modelos)}")
        else:
            print("   ⚠️  Nenhum modelo treinado. Execute: python -m data.bulk_download && python -c \"from models.trainer import Trainer; from data.database import Database; Trainer(Database()).treinar()\"")
    else:
        print("   ⚠️  Diretório de modelos não existe ainda")

    print("\n" + "=" * 50)
    print("  Teste concluído!")
    print("=" * 50)


# Limite de stats baixadas durante o boot.
# Mantém a inicialização rápida (~2 min); o restante fica pro scheduler/bulk.
MAX_STATS_BOOTSTRAP = 200


async def main():
    """
    Loop principal totalmente autônomo.

    Na inicialização:
      1. Inicia Telegram polling PRIMEIRO (bot responde imediatamente)
      2. Inicia Scheduler
      3. Dispara auto-bootstrap em background (stats limitadas a MAX_STATS_BOOTSTRAP)

    O bot nunca depende de intervenção humana para começar a operar.
    """
    # Inicializar banco
    db = Database()
    resumo = db.resumo()
    print(f"📦 Banco: {resumo['fixtures']:,} fixtures, {resumo['fixtures_ft']:,} FT, "
          f"{resumo['fixtures_com_stats']:,} com stats")

    # ─── 1. Telegram polling PRIMEIRO — bot já responde a /start ───
    app = criar_bot()
    print("🚀 Iniciando bot Telegram...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    print("✅ Bot Telegram ouvindo comandos")

    # ─── 2. Scheduler ───
    usar_scheduler = "--no-sched" not in sys.argv
    if usar_scheduler:
        scheduler = Scheduler(db, telegram_callback=enviar_mensagem)
        scheduler.iniciar()
        print("📅 Scheduler iniciado com jobs automáticos")
    else:
        print("📅 Scheduler desativado (--no-sched)")

    # ─── 3. Mensagem de boot (usa app.bot direto, sem depender de _app_instance) ───
    ALL_MODELS = [
        "resultado_1x2", "over_under_15", "over_under_25", "over_under_35",
        "btts", "resultado_ht", "htft",
    ]
    modelos_ok = Trainer.ha_modelos_treinados()
    status_modelo = "✅ 7 modelos treinados" if modelos_ok else "⏳ Modelos pendentes (aguardando dados)"
    modelos_base = Trainer.contar_modelos_base()
    status_modelo = (
        f"Modelos disponiveis ({modelos_base}/{len(Trainer.CORE_MODEL_NAMES)} base)"
        if modelos_ok else
        "Modelos pendentes (aguardando dados)"
    )
    boot_msg = (
        f"🟢 *FuteBot iniciado!*\n\n"
        f"📦 {resumo['fixtures']:,} fixtures | {resumo['fixtures_com_stats']:,} com stats\n"
        f"🤖 {status_modelo}\n"
        f"📅 Scheduler ativo — tudo automático"
    )
    from services.telegram_bot import _chat_ids
    for cid in list(_chat_ids):
        try:
            await app.bot.send_message(chat_id=cid, text=boot_msg, parse_mode="Markdown")
        except Exception:
            pass

    # ─── 4. Bootstrap em background (não bloqueia o bot) ───
    asyncio.create_task(_auto_bootstrap(db, resumo, max_stats=MAX_STATS_BOOTSTRAP))

    # Manter rodando até Ctrl+C
    try:
        stop_event = asyncio.Event()
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n⏹️  Parando bot...")
        if usar_scheduler:
            scheduler.parar()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        print("👋 FuteBot finalizado.")


async def _auto_bootstrap(db: Database, resumo: dict, max_stats: int = 0):
    """
    Auto-provisionamento do banco e modelo — roda em BACKGROUND.

    Usa asyncio.to_thread() para cada operação bloqueante (HTTP/treino),
    garantindo que o event loop do Telegram não trave.

    Se max_stats > 0, limita a quantidade de stats baixadas neste boot.
    """
    try:
        fixtures_ft = resumo.get("fixtures_ft", 0)
        com_stats = resumo.get("fixtures_com_stats", 0)

        # ─── Passo 1: Fixtures ───
        if resumo.get("fixtures", 0) < 100:
            print("\n🔄 AUTO-BOOTSTRAP: Banco vazio — baixando fixtures...")
            try:
                n = await asyncio.to_thread(baixar_fixtures, db)
                print(f"   ✅ {n} fixtures baixados")
                resumo = db.resumo()
                fixtures_ft = resumo.get("fixtures_ft", 0)
            except Exception as e:
                print(f"   ⚠️ Erro no download de fixtures: {e}")

        # ─── Passo 2: Stats (limitado a max_stats no boot) ───
        if fixtures_ft > com_stats + 50:
            pendentes = fixtures_ft - com_stats
            limite_txt = f" (máx {max_stats})" if max_stats > 0 else ""
            print(f"\n🔄 AUTO-BOOTSTRAP: {pendentes} fixtures sem stats — baixando lote{limite_txt}...")
            try:
                usadas, pode = await asyncio.to_thread(_check_limite)
                if pode:
                    mf = max_stats if max_stats > 0 else 0
                    n = await asyncio.to_thread(baixar_stats, db, True, mf)
                    print(f"   ✅ {n} stats baixadas")
                    if pendentes > n:
                        print(f"   ℹ️ Restam {pendentes - n} stats — serão baixadas pelo scheduler/bulk.")
                    resumo = db.resumo()
                    com_stats = resumo.get("fixtures_com_stats", 0)
                else:
                    print(f"   ⏸️ Limite de API atingido ({usadas}/7500). Stats pendentes serão baixadas pelo scheduler.")
            except Exception as e:
                print(f"   ⚠️ Erro no download de stats: {e}")

        # ─── Passo 3: Treinar modelo se tiver dados suficientes ───
        ALL_MODELS = [
            "resultado_1x2", "over_under_15", "over_under_25", "over_under_35",
            "btts", "resultado_ht", "htft",
        ]
        modelos_existem = Trainer.ha_modelos_treinados()
        if not modelos_existem and com_stats >= MIN_FIXTURES_TREINO:
            print(f"\n🔄 AUTO-BOOTSTRAP: {com_stats} jogos com stats — treinando modelo...")
            try:
                trainer = Trainer(db)
                metricas = await asyncio.to_thread(trainer.treinar)
                if metricas.get("rejeitado"):
                    motivos = metricas.get("gate_motivos", [])
                    print(f"   ⚠️ Modelo rejeitado nos guard rails:")
                    for m in motivos:
                        print(f"      {m}")
                    print("   O scheduler vai tentar novamente com mais dados.")
                elif "erro" in metricas:
                    print(f"   ⚠️ {metricas['erro']}")
                else:
                    acc = metricas.get("resultado_1x2", {}).get("accuracy_test", 0)
                    print(f"   ✅ Modelo treinado! 1x2 accuracy: {acc:.1%}")
            except Exception as e:
                print(f"   ⚠️ Erro no treino: {e}")
        elif not modelos_existem:
            print(f"\n⏳ Dados insuficientes para treinar ({com_stats}/{MIN_FIXTURES_TREINO} com stats).")
            print("   O scheduler vai continuar baixando stats automaticamente.")

        print("\n✅ AUTO-BOOTSTRAP concluído.")
    except Exception as e:
        print(f"\n❌ AUTO-BOOTSTRAP erro inesperado: {e}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_conexao()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 FuteBot finalizado.")

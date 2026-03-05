"""
Bot Telegram do FuteBot.

Comandos disponíveis:
  /start       → Registra chat_id e mostra boas-vindas
  /status      → Status do bot, modelo e banco de dados
  /scan        → Executa scanner manualmente (oportunidades de hoje)
  /resultados  → Resultados de ontem (previsões vs realidade)
  /metricas    → Performance acumulada do modelo
  /treinar     → Força retreino do modelo XGBoost
  /bulk        → Status/início do bulk download
  /ajuda       → Lista de comandos

O bot também envia mensagens automáticas via scheduler:
  - 07:00 → Oportunidades do dia
  - 22:00 → Relatório noturno
  - Segunda → Resultado do retreino

Uso:
  python bot.py
"""

import asyncio
import os
import sys
from datetime import datetime

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, MenuButtonCommands
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DB_PATH, TIMEZONE
from data.database import Database
from pipeline.scanner import Scanner
from pipeline.collector import Collector
from models.trainer import Trainer
from models.learner import Learner
from models.autotuner import AutoTuner
from services.apifootball import raw_request
from datetime import timedelta


# ══════════════════════════════════════════════
#  INSTÂNCIAS COMPARTILHADAS
# ══════════════════════════════════════════════

_db = Database()
_chat_ids = set()

# Carregar chat_id salvo (se existir)
if TELEGRAM_CHAT_ID:
    _chat_ids.add(int(TELEGRAM_CHAT_ID))


# ══════════════════════════════════════════════
#  TECLADOS INLINE (botões interativos)
# ══════════════════════════════════════════════

def _teclado_menu() -> InlineKeyboardMarkup:
    """Monta o teclado inline do menu principal com todos os comandos."""
    botoes = [
        [InlineKeyboardButton("🔍 Scan — Oportunidades", callback_data="cmd_scan"),
         InlineKeyboardButton("⚽ Ao Vivo", callback_data="cmd_ao_vivo")],
        [InlineKeyboardButton("📊 Resultados", callback_data="cmd_resultados"),
         InlineKeyboardButton("📈 Métricas", callback_data="cmd_metricas")],
        [InlineKeyboardButton("🛡️ Saúde", callback_data="cmd_saude"),
         InlineKeyboardButton("⚙️ Status", callback_data="cmd_status")],
        [InlineKeyboardButton("🤖 Retreinar", callback_data="cmd_treinar"),
         InlineKeyboardButton("📦 Bulk Download", callback_data="cmd_bulk")],
        [InlineKeyboardButton("❓ Ajuda", callback_data="cmd_ajuda")],
    ]
    return InlineKeyboardMarkup(botoes)


def _botao_voltar() -> InlineKeyboardMarkup:
    """Botão para retornar ao menu principal."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("← Menu Principal", callback_data="cmd_menu")]
    ])


def _salvar_chat_id(chat_id: int):
    """Salva chat_id no .env para persistência."""
    _chat_ids.add(chat_id)
    # .env fica na raiz do projeto, não em services/
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "TELEGRAM_CHAT_ID=" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("TELEGRAM_CHAT_ID="):
                    lines[i] = f"TELEGRAM_CHAT_ID={chat_id}"
            content = "\n".join(lines)
        else:
            content += f"\nTELEGRAM_CHAT_ID={chat_id}\n"
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[Bot] Erro ao salvar chat_id: {e}")


# ══════════════════════════════════════════════
#  FUNÇÕES DE ENVIO (usadas pelo scheduler)
# ══════════════════════════════════════════════

_app_instance = None


async def enviar_mensagem(texto: str, chat_id: int = None):
    """
    Envia mensagem para o chat registrado.
    Usado pelo scheduler para enviar relatórios automáticos.
    """
    global _app_instance
    if _app_instance is None:
        print("[Bot] App não inicializado, mensagem não enviada")
        return

    ids = [chat_id] if chat_id else list(_chat_ids)
    if not ids:
        print("[Bot] Nenhum chat_id registrado")
        return

    for cid in ids:
        try:
            # Truncar mensagem se muito longa (Telegram limite: 4096 chars)
            if len(texto) > 4000:
                texto = texto[:4000] + "\n\n... (truncado)"

            await _app_instance.bot.send_message(
                chat_id=cid,
                text=texto,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            # Tentar sem Markdown se falhar
            try:
                await _app_instance.bot.send_message(
                    chat_id=cid,
                    text=texto,
                )
            except Exception as e2:
                print(f"[Bot] Erro ao enviar para {cid}: {e2}")


# ══════════════════════════════════════════════
#  HANDLERS DE COMANDOS
# ══════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Registra o chat e mostra boas-vindas com menu de botões."""
    chat_id = update.effective_chat.id
    _salvar_chat_id(chat_id)

    # Garantir botão de menu no canto inferior esquerdo para este chat
    try:
        await context.bot.set_chat_menu_button(
            chat_id=chat_id,
            menu_button=MenuButtonCommands()
        )
    except Exception:
        pass

    resumo = _db.resumo()
    treino = _db.ultimo_treino()

    msg = (
        "🤖 <b>FuteBot — Seu assistente de apostas com IA</b>\n\n"
        f"✅ Chat registrado: <code>{chat_id}</code>\n\n"
        f"📊 <b>Status do banco:</b>\n"
        f"  Fixtures: {resumo['fixtures']:,}\n"
        f"  Com stats: {resumo['fixtures_com_stats']:,}\n"
        f"  Previsões: {resumo['predictions']:,}\n\n"
    )

    if treino:
        versao = treino['modelo_versao'] or '?'
        acc = (treino['accuracy_test'] or 0) * 100
        msg += (
            f"🤖 <b>Modelo:</b> <code>{versao}</code>\n"
            f"  Accuracy: {acc:.1f}%\n\n"
        )
    else:
        # Verificar se modelos existem no disco mesmo sem train_log
        _ALL = ["resultado_1x2","over_under_15","over_under_25","over_under_35","btts","resultado_ht","htft"]
        n_modelos = sum(1 for m in _ALL if Trainer.modelo_existe(m))
        if n_modelos > 0:
            msg += f"🤖 <b>Modelo:</b> {n_modelos}/7 carregados (sem métricas — clique Retreinar)\n\n"
        else:
            msg += "⚠️ Modelo ainda não treinado\n\n"

    msg += "👇 <b>Escolha uma opção:</b>"

    try:
        await update.message.reply_text(
            msg, parse_mode=ParseMode.HTML, reply_markup=_teclado_menu()
        )
    except Exception:
        # Fallback sem formatação caso HTML falhe
        await update.message.reply_text(
            msg.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""),
            reply_markup=_teclado_menu()
        )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra status geral do bot."""
    resumo = _db.resumo()
    treino = _db.ultimo_treino()

    # Status da API-Football
    r = raw_request("status")
    api_status = r.get("response", {})
    req = api_status.get("requests", {})

    msg = (
        "📊 *Status do FuteBot*\n\n"
        f"*Banco de dados:*\n"
        f"  Fixtures: {resumo['fixtures']:,}\n"
        f"  Finalizados: {resumo['fixtures_ft']:,}\n"
        f"  Com stats: {resumo['fixtures_com_stats']:,}\n"
        f"  Previsões: {resumo['predictions']:,}\n"
        f"  Odds cache: {resumo['odds_cache']:,}\n\n"
        f"*API-Football:*\n"
        f"  Plano: {api_status.get('subscription', {}).get('plan', '?')}\n"
        f"  Requests hoje: {req.get('current', '?')}/{req.get('limit_day', '?')}\n\n"
    )

    if treino:
        # Data do treino em formato BR (dd/mm/yyyy)
        try:
            from datetime import datetime as _dt
            _d = _dt.strptime(treino['date'][:10], '%Y-%m-%d')
            data_treino_br = _d.strftime('%d/%m/%Y')
        except Exception:
            data_treino_br = treino['date'][:10]
        # Accuracy real = das previsões resolvidas (não do treino, que pode ser 0)
        metricas_modelo = _db.metricas_modelo()
        if metricas_modelo["total"] > 0:
            acc_display = f"{metricas_modelo['accuracy']}%"
        elif (treino['accuracy_test'] or 0) > 0:
            acc_display = f"{(treino['accuracy_test'] or 0)*100:.1f}%"
        else:
            acc_display = "sem dados ainda"
        msg += (
            f"*Modelo:*\n"
            f"  Versão: {treino['modelo_versao']}\n"
            f"  Último treino: {data_treino_br}\n"
            f"  Accuracy: {acc_display}\n"
            f"  Amostras: {treino['n_samples']:,}\n"
        )
    else:
        # Verificar se modelos existem no disco mesmo sem train_log
        _ALL = ["resultado_1x2","over_under_15","over_under_25","over_under_35","btts","resultado_ht","htft"]
        n_modelos = sum(1 for m in _ALL if Trainer.modelo_existe(m))
        if n_modelos > 0:
            msg += (
                f"*Modelo:*\n"
                f"  {n_modelos}/7 modelos carregados\n"
                f"  ⚠️ Sem métricas de treino — clique *Retreinar* para registrar\n"
            )
        else:
            msg += "⚠️ Modelo não treinado ainda\n"

    metricas = _db.metricas_modelo()
    if metricas["total"] > 0:
        msg += (
            f"\n*Performance:*\n"
            f"  Previsões: {metricas['total']}\n"
            f"  Acertos: {metricas['acertos']}\n"
            f"  Accuracy: {metricas['accuracy']}%\n"
            f"  ROI: {metricas['roi']:+.1f}%\n"
        )

    await update.message.reply_text(
        msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executa o scanner de oportunidades manualmente."""
    await update.message.reply_text("🔍 Executando scanner... aguarde.")

    try:
        scanner = Scanner(_db)
        resultado = scanner.executar()
        msgs = scanner.formatar_relatorio(resultado)
        # Envia cada bloco como mensagem separada (com botões ✏️ Odd)
        for i, (texto, botoes) in enumerate(msgs):
            kb = None
            if botoes:
                # Botões de odd manual (1 por tip)
                kb = InlineKeyboardMarkup(
                    [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in botoes]
                )
            elif i == len(msgs) - 1:
                kb = _botao_voltar()
            await update.message.reply_text(
                texto, parse_mode=ParseMode.HTML, reply_markup=kb
            )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro no scanner:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_resultados(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra resultados de ontem."""
    await update.message.reply_text("📥 Coletando resultados...")

    try:
        collector = Collector(_db)
        resultado = collector.executar()
        msg = collector.formatar_relatorio(resultado)
        await update.message.reply_text(
            msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
        )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_metricas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra métricas de performance do modelo (HTML para suportar nomes com _)."""
    learner = Learner(_db)
    msg = learner.relatorio_diario()
    await update.message.reply_text(
        msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
    )


async def cmd_treinar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Força retreino via AutoTuner (Optuna + strategy slicing)."""
    await update.message.reply_text(
        "🧠 Iniciando AutoTuner... Optuna (50 trials) + strategy slicing.\n"
        "Isso pode demorar ~30-60 min. Você receberá o resultado aqui."
    )

    try:
        tuner = AutoTuner(_db)
        resultado = tuner.executar()
        msg = AutoTuner.formatar_resultado(resultado)
        await update.message.reply_text(
            msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
        )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro no AutoTuner:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_bulk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Status do bulk download."""
    resumo = _db.resumo()

    msg = (
        "📦 *Bulk Download*\n\n"
        f"Fixtures no banco: {resumo['fixtures']:,}\n"
        f"Finalizados: {resumo['fixtures_ft']:,}\n"
        f"Com stats: {resumo['fixtures_com_stats']:,}\n\n"
        "Para iniciar/continuar o download:\n"
        "`python -m data.bulk_download`\n"
        "`python -m data.bulk_download --resume`\n"
    )
    await update.message.reply_text(
        msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
    )


async def cmd_ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista todos os comandos."""
    msg = (
        "🤖 *FuteBot — Comandos*\n\n"
        "🔍 /scan — Buscar oportunidades de EV+ agora\n"
        "⚽ /ao\\_vivo — Status dos jogos previstos\n"
        "📊 /resultados — Resultados de ontem\n"
        "📈 /metricas — Performance acumulada do modelo\n"
        "🛡️ /saude — Saúde do modelo (guard rails)\n"
        "⚙️ /status — Status geral (banco, API, modelo)\n"
        "🤖 /treinar — Forçar retreino do modelo\n"
        "📦 /bulk — Status do bulk download\n\n"
        "🔄 *Automático:*\n"
        "  07:00 → Oportunidades do dia\n"
        "  A cada 2h → Notificação de acerto/erro\n"
        "  22:00 → Relatório noturno + saúde do modelo\n"
        "  Segunda 04:00 → Retreino do modelo\n\n"
        "🛡️ *Guard Rails:*\n"
        "  Modelo só aceito se superar baselines\n"
        "  Scanner pausa se ROI < -15%\n"
        "  Alertas automáticos de degradação\n\n"
        "💡 Use os botões abaixo ou digite os comandos."
    )
    await update.message.reply_text(
        msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_teclado_menu()
    )


async def cmd_saude(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra saúde do modelo (guard rails, degradação, calibração)."""
    try:
        learner = Learner(_db)
        msg = learner.relatorio_saude()
        await update.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
        )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_ao_vivo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Verifica status dos jogos previstos hoje/ontem.
    Mostra quais já finalizaram (acertou/errou) e quais estão em andamento.
    """
    await update.message.reply_text("⚽ Verificando jogos previstos...")

    try:
        msg = await _logica_ao_vivo()
        await update.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
        )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro:\n{e}", reply_markup=_botao_voltar()
        )


async def _logica_ao_vivo() -> str:
    """
    Lógica compartilhada do /ao_vivo — usada tanto pelo comando
    quanto pelo callback de botão.

    Consulta previsões pendentes de hoje/ontem, verifica status
    de cada jogo na API-Football, e resolve as finalizadas.
    """
    pendentes = _db.predictions_pendentes()
    hoje = datetime.now().strftime("%Y-%m-%d")
    ontem = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    recentes = [
        p for p in pendentes
        if p.get("date", "")[:10] in (hoje, ontem)
    ]

    if not recentes:
        return "⚽ *Ao Vivo*\n\nNenhuma previsão pendente para hoje/ontem."

    lines = ["⚽ *Status dos jogos previstos*\n"]
    finalizados = 0
    em_andamento = 0
    aguardando = 0
    acertos = 0
    erros = 0

    # Nomes legíveis para os mercados
    nomes_mercado = {
        "h2h_home": "Casa", "h2h_draw": "Empate", "h2h_away": "Fora",
        "over25": "Over 2.5", "under25": "Under 2.5",
        "over15": "Over 1.5", "under15": "Under 1.5",
        "over35": "Over 3.5", "under35": "Under 3.5",
        "btts_yes": "BTTS Sim", "btts_no": "BTTS Não",
    }

    for pred in recentes:
        fixture_id = pred["fixture_id"]
        mercado = pred.get("mercado", "")
        mercado_label = nomes_mercado.get(mercado, mercado)
        odd_usada = pred.get("odd_usada", 0)

        # Consultar status na API
        r = raw_request("fixtures", {"id": fixture_id})
        resp = r.get("response", [])
        if not resp:
            lines.append(
                f"❓ *{pred['home_name']} vs {pred['away_name']}*\n"
                f"   {mercado_label} @ {odd_usada:.2f} | Sem dados"
            )
            aguardando += 1
            continue

        game = resp[0]
        status = game.get("fixture", {}).get("status", {}).get("short", "NS")
        elapsed = game.get("fixture", {}).get("status", {}).get("elapsed", "")
        gh = game.get("goals", {}).get("home", 0) or 0
        ga = game.get("goals", {}).get("away", 0) or 0

        if status == "NS":
            # Jogo ainda não começou — extrair horário
            date_str = game.get("fixture", {}).get("date", "")
            horario = ""
            if date_str and "T" in date_str:
                try:
                    from zoneinfo import ZoneInfo
                    dt_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    dt_local = dt_obj.astimezone(ZoneInfo(TIMEZONE))
                    horario = dt_local.strftime("%H:%M")
                except Exception:
                    horario = "?"
            lines.append(
                f"⏰ *{pred['home_name']} vs {pred['away_name']}* — {horario}\n"
                f"   {mercado_label} @ {odd_usada:.2f} | Aguardando início"
            )
            aguardando += 1

        elif status in ("1H", "2H", "HT", "ET", "P", "BT", "LIVE"):
            # Jogo em andamento
            status_labels = {
                "1H": "1º Tempo", "2H": "2º Tempo", "HT": "Intervalo",
                "ET": "Prorrogação", "P": "Pênaltis", "BT": "Intervalo",
                "LIVE": "Ao vivo"
            }
            status_label = status_labels.get(status, status)
            elapsed_txt = f" {elapsed}'" if elapsed else ""
            lines.append(
                f"🟢 *{pred['home_name']} {gh}-{ga} {pred['away_name']}*\n"
                f"   {status_label}{elapsed_txt} | {mercado_label} @ {odd_usada:.2f}"
            )
            em_andamento += 1

        elif status == "FT":
            # Jogo finalizado — resolver previsão
            _db.salvar_fixture(game)

            if gh > ga:
                resultado = "home"
            elif gh == ga:
                resultado = "draw"
            else:
                resultado = "away"

            _db.resolver_prediction(fixture_id, resultado, gh, ga)

            # Verificar acerto
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

            if acertou:
                acertos += 1
                lucro = round(odd_usada - 1, 2)
                lines.append(
                    f"✅ *{pred['home_name']} {gh}-{ga} {pred['away_name']}*\n"
                    f"   {mercado_label} @ {odd_usada:.2f} | *+{lucro:.2f}u*"
                )
            else:
                erros += 1
                lines.append(
                    f"❌ *{pred['home_name']} {gh}-{ga} {pred['away_name']}*\n"
                    f"   {mercado_label} @ {odd_usada:.2f} | *-1.00u*"
                )
            finalizados += 1

        else:
            # Outros status: PST (adiado), CANC (cancelado), etc.
            lines.append(
                f"⚠️ *{pred['home_name']} vs {pred['away_name']}*\n"
                f"   {mercado_label} | Status: {status}"
            )
            aguardando += 1

    # Resumo
    lines.append("")
    parts = []
    if aguardando > 0:
        parts.append(f"⏰ {aguardando} aguardando")
    if em_andamento > 0:
        parts.append(f"🟢 {em_andamento} ao vivo")
    if finalizados > 0:
        total_check = acertos + erros
        parts.append(f"🏁 {finalizados} finalizados ({acertos}/{total_check})")

    lines.append(" | ".join(parts))

    return "\n".join(lines)


# ══════════════════════════════════════════════
#  ODD MANUAL: Botão ✏️ Odd + input de texto
# ══════════════════════════════════════════════

async def _handle_odd_button(query, context, data: str):
    """
    Processa clique no botão ✏️ Odd.
    callback_data: 'odd:{fixture_id}:{mercado}:{prob_int}'
    Armazena estado em context.user_data e pede a odd ao usuário.
    """
    try:
        parts = data.split(":")
        fixture_id = int(parts[1])
        mercado = parts[2]
        prob_int = int(parts[3])
    except (IndexError, ValueError):
        await query.message.reply_text("❌ Botão inválido.")
        return

    prob = prob_int / 1000.0

    # Busca previsão no banco para contexto (nomes dos times)
    pred = _db.buscar_prediction(fixture_id, mercado)
    if not pred:
        await query.message.reply_text("❌ Previsão não encontrada no banco.")
        return

    # Armazena estado: aguardando digitação da odd
    context.user_data["odd_pendente"] = {
        "fixture_id": fixture_id,
        "mercado": mercado,
        "prob": prob,
        "home": pred["home_name"],
        "away": pred["away_name"],
    }

    await query.message.reply_text(
        f"✏️ <b>Definir odd manual</b>\n\n"
        f"⚽ {pred['home_name']} vs {pred['away_name']}\n"
        f"📊 {mercado} — Prob modelo: {prob:.0%}\n\n"
        f"Digite a odd (ex: <code>2.15</code>):",
        parse_mode=ParseMode.HTML,
    )


async def _handle_odd_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Recebe texto livre — se o bot está aguardando uma odd (odd_pendente),
    processa e calcula EV. Caso contrário, ignora silenciosamente.
    """
    pendente = context.user_data.get("odd_pendente")
    if not pendente:
        return  # Não está aguardando odd, ignora texto

    texto = update.message.text.strip().replace(",", ".")
    try:
        odd = float(texto)
    except ValueError:
        await update.message.reply_text(
            "❌ Odd inválida. Digite um número decimal (ex: <code>2.15</code>).",
            parse_mode=ParseMode.HTML,
        )
        return

    if odd <= 1.0:
        await update.message.reply_text("❌ Odd deve ser maior que 1.0")
        return

    fixture_id = pendente["fixture_id"]
    mercado = pendente["mercado"]
    prob = pendente["prob"]
    home = pendente["home"]
    away = pendente["away"]

    # Calcula EV: (prob × odd - 1) × 100
    ev = round((prob * odd - 1) * 100, 1)

    # Atualiza no banco
    _db.atualizar_odd_manual(fixture_id, mercado, round(odd, 2), ev)

    # Limpa estado
    context.user_data.pop("odd_pendente", None)

    emoji_ev = "✅" if ev > 0 else "⚠️"
    await update.message.reply_text(
        f"{emoji_ev} <b>Odd registrada!</b>\n\n"
        f"⚽ {home} vs {away}\n"
        f"📊 {mercado}\n"
        f"💰 Odd: <b>{odd:.2f}</b>\n"
        f"📈 EV: <b>{ev:+.1f}%</b>",
        parse_mode=ParseMode.HTML,
    )


# ══════════════════════════════════════════════
#  CALLBACK: BOTÕES INLINE
# ══════════════════════════════════════════════

# Mapeamento callback_data → função handler correspondente
_CALLBACK_MAP = {
    "cmd_scan": cmd_scan,
    "cmd_resultados": cmd_resultados,
    "cmd_metricas": cmd_metricas,
    "cmd_saude": cmd_saude,
    "cmd_status": cmd_status,
    "cmd_treinar": cmd_treinar,
    "cmd_bulk": cmd_bulk,
    "cmd_ajuda": cmd_ajuda,
    "cmd_ao_vivo": cmd_ao_vivo,
}


async def _callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler central para todos os botões inline.
    Roteia o callback_data para o handler correto.
    """
    query = update.callback_query
    await query.answer()  # Remove o "loading" do botão

    data = query.data

    # Botão "← Menu Principal" — reenvia o menu completo
    if data == "cmd_menu":
        msg = "🤖 *FuteBot — Menu Principal*\n\n👇 Escolha uma opção:"
        await query.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_teclado_menu()
        )
        return

    # ── Botão ✏️ Odd (odd manual por tip) ──
    if data.startswith("odd:"):
        await _handle_odd_button(query, context, data)
        return

    # Busca o handler mapeado
    handler_fn = _CALLBACK_MAP.get(data)
    if not handler_fn:
        await query.message.reply_text("❓ Comando não reconhecido.")
        return

    # Cria um Update fake com message para reusar os handlers existentes.
    # Os handlers esperam update.message, então usamos query.message.
    # Precisamos garantir que update.message exista.
    # Trick: setamos update._effective_message para query.message
    # e chamamos o handler com o update original — mas os handlers
    # usam update.message que é None em callbacks.
    # Solução: respondemos via query.message diretamente.
    await _executar_via_callback(query, handler_fn, context)


async def _executar_via_callback(query, handler_fn, context):
    """
    Executa um handler de comando a partir de um callback de botão.
    Como os handlers usam update.message.reply_text(), precisamos
    adaptar para funcionar a partir de callback_query.message.
    """
    # Mapeamento direto — executa a lógica inline ao invés de
    # tentar hackear o Update. Mais seguro e manutenível.
    chat = query.message.chat

    try:
        if handler_fn == cmd_scan:
            await query.message.reply_text("🔍 Executando scanner... aguarde.")
            scanner = Scanner(_db)
            resultado = scanner.executar()
            msgs = scanner.formatar_relatorio(resultado)
            for i, (texto, botoes) in enumerate(msgs):
                kb = None
                if botoes:
                    kb = InlineKeyboardMarkup(
                        [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in botoes]
                    )
                elif i == len(msgs) - 1:
                    kb = _botao_voltar()
                await query.message.reply_text(
                    texto, parse_mode=ParseMode.HTML, reply_markup=kb
                )

        elif handler_fn == cmd_resultados:
            await query.message.reply_text("📥 Coletando resultados...")
            collector = Collector(_db)
            resultado = collector.executar()
            msg = collector.formatar_relatorio(resultado)
            await query.message.reply_text(
                msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_metricas:
            learner = Learner(_db)
            msg = learner.relatorio_diario()
            await query.message.reply_text(
                msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_saude:
            learner = Learner(_db)
            msg = learner.relatorio_saude()
            await query.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_status:
            resumo = _db.resumo()
            treino = _db.ultimo_treino()
            r = raw_request("status")
            api_status = r.get("response", {})
            req = api_status.get("requests", {})

            msg = (
                "📊 *Status do FuteBot*\n\n"
                f"*Banco de dados:*\n"
                f"  Fixtures: {resumo['fixtures']:,}\n"
                f"  Finalizados: {resumo['fixtures_ft']:,}\n"
                f"  Com stats: {resumo['fixtures_com_stats']:,}\n"
                f"  Previsões: {resumo['predictions']:,}\n"
                f"  Odds cache: {resumo['odds_cache']:,}\n\n"
                f"*API-Football:*\n"
                f"  Plano: {api_status.get('subscription', {}).get('plan', '?')}\n"
                f"  Requests hoje: {req.get('current', '?')}/{req.get('limit_day', '?')}\n\n"
            )

            if treino:
                msg += (
                    f"*Modelo:*\n"
                    f"  Versão: {treino['modelo_versao']}\n"
                    f"  Último treino: {treino['date'][:10]}\n"
                    f"  Accuracy: {(treino['accuracy_test'] or 0)*100:.1f}%\n"
                    f"  Amostras: {treino['n_samples']:,}\n"
                )
            else:
                # Verificar se modelos existem no disco mesmo sem train_log
                _ALL = ["resultado_1x2","over_under_15","over_under_25","over_under_35","btts","resultado_ht","htft"]
                n_modelos = sum(1 for m in _ALL if Trainer.modelo_existe(m))
                if n_modelos > 0:
                    msg += (
                        f"*Modelo:*\n"
                        f"  {n_modelos}/7 modelos carregados\n"
                        f"  ⚠️ Sem métricas — clique *Retreinar* para registrar\n"
                    )
                else:
                    msg += "⚠️ Modelo não treinado ainda\n"

            metricas = _db.metricas_modelo()
            if metricas["total"] > 0:
                msg += (
                    f"\n*Performance:*\n"
                    f"  Previsões: {metricas['total']}\n"
                    f"  Acertos: {metricas['acertos']}\n"
                    f"  Accuracy: {metricas['accuracy']}%\n"
                    f"  ROI: {metricas['roi']:+.1f}%\n"
                )

            await query.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_treinar:
            await query.message.reply_text(
                "🧠 Iniciando AutoTuner... Optuna (50 trials) + strategy slicing.\n"
                "Isso pode demorar ~30-60 min."
            )
            tuner = AutoTuner(_db)
            resultado = tuner.executar()
            msg = AutoTuner.formatar_resultado(resultado)
            await query.message.reply_text(
                msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_bulk:
            resumo = _db.resumo()
            msg = (
                "📦 *Bulk Download*\n\n"
                f"Fixtures no banco: {resumo['fixtures']:,}\n"
                f"Finalizados: {resumo['fixtures_ft']:,}\n"
                f"Com stats: {resumo['fixtures_com_stats']:,}\n\n"
                "Para iniciar/continuar o download:\n"
                "`python -m data.bulk_download`\n"
                "`python -m data.bulk_download --resume`\n"
            )
            await query.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_ao_vivo:
            await query.message.reply_text("⚽ Verificando jogos previstos...")
            msg = await _logica_ao_vivo()
            await query.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_ajuda:
            msg = (
                "🤖 *FuteBot — Comandos*\n\n"
                "🔍 /scan — Buscar oportunidades de EV+ agora\n"
                "⚽ /ao\\_vivo — Status dos jogos previstos\n"
                "📊 /resultados — Resultados de ontem\n"
                "📈 /metricas — Performance acumulada do modelo\n"
                "🛡️ /saude — Saúde do modelo (guard rails)\n"
                "⚙️ /status — Status geral (banco, API, modelo)\n"
                "🤖 /treinar — Forçar retreino do modelo\n"
                "📦 /bulk — Status do bulk download\n\n"
                "🔄 *Automático:*\n"
                "  07:00 → Oportunidades do dia\n"
                "  A cada 2h → Notificação de acerto/erro\n"
                "  22:00 → Relatório noturno + saúde do modelo\n"
                "  Segunda 04:00 → Retreino do modelo\n\n"
                "💡 Use os botões abaixo ou digite os comandos."
            )
            await query.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_teclado_menu()
            )

    except Exception as e:
        await query.message.reply_text(
            f"❌ Erro: {e}", reply_markup=_botao_voltar()
        )


# ══════════════════════════════════════════════
#  CONFIGURAÇÃO DO BOT
# ══════════════════════════════════════════════

async def post_init(application: Application):
    """Callback pós-inicialização: configura comandos visíveis no Telegram."""
    global _app_instance
    _app_instance = application

    await application.bot.set_my_commands([
        BotCommand("start", "Registrar e ver boas-vindas"),
        BotCommand("scan", "Buscar oportunidades agora"),
        BotCommand("ao_vivo", "Status dos jogos previstos"),
        BotCommand("resultados", "Resultados de ontem"),
        BotCommand("metricas", "Performance do modelo"),
        BotCommand("saude", "Saúde do modelo (guard rails)"),
        BotCommand("status", "Status geral"),
        BotCommand("treinar", "Retreinar modelo"),
        BotCommand("ajuda", "Lista de comandos"),
    ])

    # Botão de menu persistente — aparece no canto inferior esquerdo do chat
    # Ao clicar, abre a lista de comandos do bot automaticamente
    # Define globalmente (padrão para todos os chats)
    await application.bot.set_chat_menu_button(
        menu_button=MenuButtonCommands()
    )
    # Define também para cada chat registrado individualmente
    for cid in _chat_ids:
        try:
            await application.bot.set_chat_menu_button(
                chat_id=cid,
                menu_button=MenuButtonCommands()
            )
        except Exception:
            pass

    print(f"🤖 Bot Telegram iniciado!")
    print(f"   Chat IDs: {_chat_ids or 'nenhum (envie /start)'}")


def criar_bot() -> Application:
    """Cria e configura a instância do bot Telegram."""
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN não configurado no .env!")
        sys.exit(1)

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Registrar handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("ao_vivo", cmd_ao_vivo))
    app.add_handler(CommandHandler("resultados", cmd_resultados))
    app.add_handler(CommandHandler("metricas", cmd_metricas))
    app.add_handler(CommandHandler("saude", cmd_saude))
    app.add_handler(CommandHandler("treinar", cmd_treinar))
    app.add_handler(CommandHandler("bulk", cmd_bulk))
    app.add_handler(CommandHandler("ajuda", cmd_ajuda))

    # Handler de callbacks dos botões inline (deve vir depois dos comandos)
    app.add_handler(CallbackQueryHandler(_callback_handler))

    # Handler de texto para receber odd manual (após clicar ✏️ Odd)
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, _handle_odd_input
    ))

    return app

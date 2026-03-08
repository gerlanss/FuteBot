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

import os
import sys
from datetime import datetime, timedelta

from telegram import (
    Update,
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
    MenuButtonWebApp,
    WebAppInfo,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE, MINI_APP_URL, ADMIN_CHAT_IDS
from data.database import Database
from pipeline.scanner import Scanner
from pipeline.collector import Collector
from models.trainer import Trainer
from models.learner import Learner
from models.autotuner import AutoTuner
from services.apifootball import raw_request
from services.user_prefs import get_preferences


# ══════════════════════════════════════════════
#  INSTÂNCIAS COMPARTILHADAS
# ══════════════════════════════════════════════

_db = Database()
_ADMIN_CHAT_IDS = set(ADMIN_CHAT_IDS or [])
_chat_ids = set(_ADMIN_CHAT_IDS)
_PUBLIC_COMMANDS = [
    BotCommand("start", "Abrir menu do bot"),
    BotCommand("resultados", "Resultados recentes"),
    BotCommand("app", "Abrir mini app"),
    BotCommand("ajuda", "Ver como o bot funciona"),
]
_ADMIN_COMMANDS = [
    BotCommand("start", "Abrir menu do bot"),
    BotCommand("scan", "Buscar oportunidades agora"),
    BotCommand("ao_vivo", "Status dos jogos previstos"),
    BotCommand("resultados", "Resultados recentes"),
    BotCommand("metricas", "Performance do modelo"),
    BotCommand("saude", "Saude do modelo"),
    BotCommand("status", "Status geral"),
    BotCommand("treinar", "Retreinar modelo"),
    BotCommand("bulk", "Status do bulk download"),
    BotCommand("app", "Abrir mini app"),
    BotCommand("ajuda", "Ver todos os comandos"),
]

# Carregar chat_id salvo (se existir)
if TELEGRAM_CHAT_ID:
    _chat_ids.add(int(TELEGRAM_CHAT_ID))
    _ADMIN_CHAT_IDS.add(int(TELEGRAM_CHAT_ID))


# ══════════════════════════════════════════════
#  TECLADOS INLINE (botões interativos)
# ══════════════════════════════════════════════

def _is_admin_chat(chat_id: int | None) -> bool:
    return chat_id is not None and int(chat_id) in _ADMIN_CHAT_IDS


def _teclado_menu(chat_id: int | None = None) -> InlineKeyboardMarkup:
    """Monta o teclado inline do menu principal conforme o nivel de acesso."""
    botoes = [
        [InlineKeyboardButton("📊 Resultados", callback_data="cmd_resultados")],
        [InlineKeyboardButton("❓ Ajuda", callback_data="cmd_ajuda")],
    ]
    if _is_admin_chat(chat_id):
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
    if MINI_APP_URL:
        botoes.insert(0, [InlineKeyboardButton("📲 Painel Mini App", web_app=WebAppInfo(url=MINI_APP_URL))])
    return InlineKeyboardMarkup(botoes)


def _botao_voltar() -> InlineKeyboardMarkup:
    """Botão para retornar ao menu principal."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("← Menu Principal", callback_data="cmd_menu")]
    ])



async def _configurar_menu(bot, chat_id: int | None = None):
    """Garante menu de comandos e botao lateral do Telegram."""
    await bot.set_my_commands(_PUBLIC_COMMANDS)
    await bot.set_my_commands(_PUBLIC_COMMANDS, scope=BotCommandScopeAllPrivateChats())
    for admin_chat_id in sorted(_ADMIN_CHAT_IDS):
        await bot.set_my_commands(_ADMIN_COMMANDS, scope=BotCommandScopeChat(admin_chat_id))

    if MINI_APP_URL:
        menu_button = MenuButtonWebApp("Painel", WebAppInfo(url=MINI_APP_URL))
    else:
        menu_button = MenuButtonCommands()

    if chat_id is None:
        await bot.set_chat_menu_button(menu_button=menu_button)
        return

    await bot.set_chat_menu_button(
        chat_id=chat_id,
        menu_button=menu_button,
    )


async def _negar_acesso(message):
    await _reply_html(
        message,
        "<b>Acesso restrito</b>\n\nEste comando fica disponivel apenas para o administrador do bot.",
        reply_markup=_teclado_menu(getattr(getattr(message, "chat", None), "id", None)),
    )


async def _garantir_admin_message(message) -> bool:
    chat_id = getattr(getattr(message, "chat", None), "id", None)
    if _is_admin_chat(chat_id):
        return True
    await _negar_acesso(message)
    return False


async def _reply_html(message, texto: str, reply_markup=None):
    """Envia resposta HTML com fallback para texto puro."""
    try:
        await message.reply_text(
            texto,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )
    except Exception:
        limpo = (
            texto.replace("<b>", "").replace("</b>", "")
            .replace("<code>", "").replace("</code>", "")
            .replace("<i>", "").replace("</i>", "")
        )
        await message.reply_text(
            limpo,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )


def _formatar_modelo_html(treino: dict | None) -> str:
    if treino:
        try:
            data_treino = datetime.strptime(treino["date"][:10], "%Y-%m-%d").strftime("%d/%m/%Y")
        except Exception:
            data_treino = treino["date"][:10]

        metricas_modelo = _db.metricas_modelo()
        if metricas_modelo["total"] > 0:
            acc_display = f"{metricas_modelo['accuracy']}%"
        elif (treino["accuracy_test"] or 0) > 0:
            acc_display = f"{(treino['accuracy_test'] or 0) * 100:.1f}%"
        else:
            acc_display = "sem dados ainda"

        return (
            "<b>Modelo</b>\n"
            f"- Versao: <code>{treino['modelo_versao']}</code>\n"
            f"- Ultimo treino: {data_treino}\n"
            f"- Accuracy: {acc_display}\n"
            f"- Amostras: {treino['n_samples']:,}"
        )

    n_modelos = Trainer.contar_modelos_base()
    if n_modelos > 0:
        return (
            "<b>Modelo</b>\n"
            f"- Base pronta: {n_modelos}/{len(Trainer.CORE_MODEL_NAMES)} modelos\n"
            "- Status: sem metricas salvas ainda\n"
            "- Acao: use <code>/treinar</code> para registrar desempenho"
        )

    return (
        "<b>Modelo</b>\n"
        "- Status: ainda nao treinado\n"
        "- Acao: o bootstrap vai baixar dados e tentar treinar automaticamente"
    )


def _formatar_start_html(chat_id: int) -> str:
    if not _is_admin_chat(chat_id):
        return (
            "<b>FuteBot</b>\n"
            "Bot privado de analise.\n\n"
            "Seu acesso atual eh limitado.\n"
            "Use o menu para abrir a Mini App ou ver a ajuda."
        )

    resumo = _db.resumo()
    treino = _db.ultimo_treino()
    return (
        "<b>FuteBot</b>\n"
        "Centro de controle do bot.\n\n"
        "<b>Chat</b>\n"
        f"- Registrado: <code>{chat_id}</code>\n\n"
        "<b>Banco</b>\n"
        f"- Fixtures: {resumo['fixtures']:,}\n"
        f"- Com stats: {resumo['fixtures_com_stats']:,}\n"
        f"- Previsoes: {resumo['predictions']:,}\n\n"
        f"{_formatar_modelo_html(treino)}\n\n"
        "Use o botao de menu do Telegram ou os atalhos abaixo."
    )


def _formatar_status_html() -> str:
    resumo = _db.resumo()
    treino = _db.ultimo_treino()
    api_status = raw_request("status").get("response", {})
    req = api_status.get("requests", {})
    metricas = _db.metricas_modelo()

    partes = [
        "<b>Status do FuteBot</b>",
        "",
        "<b>Banco</b>",
        f"- Fixtures: {resumo['fixtures']:,}",
        f"- Finalizados: {resumo['fixtures_ft']:,}",
        f"- Com stats: {resumo['fixtures_com_stats']:,}",
        f"- Previsoes: {resumo['predictions']:,}",
        f"- Odds cache: {resumo['odds_cache']:,}",
        "",
        "<b>API-Football</b>",
        f"- Plano: {api_status.get('subscription', {}).get('plan', '?')}",
        f"- Requests hoje: {req.get('current', '?')}/{req.get('limit_day', '?')}",
        "",
        _formatar_modelo_html(treino),
    ]

    if metricas["total"] > 0:
        partes.extend([
            "",
            "<b>Performance</b>",
            f"- Resolvidas: {metricas['total']}",
            f"- Acertos: {metricas['acertos']}",
            f"- Accuracy: {metricas['accuracy']}%",
            f"- ROI: {metricas['roi']:+.1f}%",
        ])

    return "\n".join(partes)


def _formatar_bulk_html() -> str:
    resumo = _db.resumo()
    return (
        "<b>Bulk Download</b>\n\n"
        f"- Fixtures: {resumo['fixtures']:,}\n"
        f"- Finalizados: {resumo['fixtures_ft']:,}\n"
        f"- Com stats: {resumo['fixtures_com_stats']:,}\n\n"
        "<b>Comandos uteis</b>\n"
        "- <code>python -m data.bulk_download</code>\n"
        "- <code>python -m data.bulk_download --resume</code>"
    )


def _formatar_ajuda_html() -> str:
    return (
        "<b>Comandos do FuteBot</b>\n\n"
        "<b>Operacao</b>\n"
        "- <code>/scan</code> - buscar oportunidades agora\n"
        "- <code>/ao_vivo</code> - acompanhar jogos previstos\n"
        "- <code>/resultados</code> - ver resultados recentes\n\n"
        "<b>Monitoramento</b>\n"
        "- <code>/status</code> - resumo geral do bot\n"
        "- <code>/metricas</code> - performance acumulada\n"
        "- <code>/saude</code> - guard rails e degradacao\n\n"
        "<b>Manutencao</b>\n"
        "- <code>/treinar</code> - forcar retreino\n"
        "- <code>/app</code> - abrir mini app\n"
        "- <code>/bulk</code> - status do bulk download\n\n"
        "<b>Rotina automatica</b>\n"
        "- 07:00 - scanner do dia\n"
        "- a cada 2h - acompanhamento ao vivo\n"
        "- 06:45 - relatorio de resultados\n"
        "- mensal - retreino per-league\n\n"
        "<b>Como o FuteBot escolhe as tips</b>\n"
        "Cada jogo passa por um funil antes de virar tip.\n\n"
        "1. <b>Modelo por liga</b>\n"
        "Cada liga tem modelos proprios para resultado, gols, BTTS, tempos e escanteios.\n\n"
        "2. <b>Confianca minima</b>\n"
        "Mercado com menos de 70% de confianca ja cai antes de tudo.\n\n"
        "3. <b>Strategy Gate</b>\n"
        "So passa se aquela combinacao liga + mercado + faixa de confianca tiver historico validado.\n\n"
        "4. <b>Anti-conflito</b>\n"
        "O bot nao manda mercados contraditorios no mesmo jogo. Ex.: Over 1.5 e Under 3.5 nao coexistem.\n\n"
        "5. <b>Odds e EV</b>\n"
        "Quando existe linha exata de odd, o bot calcula EV. Se a odd exata nao existir, ele nao inventa preco.\n\n"
        "6. <b>DeepSeek</b>\n"
        "A IA revisa contexto, forma, tabela, lesoes e coerencia da tip.\n\n"
        "7. <b>Selecao final</b>\n"
        "Depois de tudo, o bot ordena as oportunidades e envia so as 12 melhores do dia.\n\n"
        "8. <b>Combos</b>\n"
        "As combinacoes usam jogos diferentes, exigem no minimo 70% por tip e 50% de confianca composta, e o bot envia no maximo 3 combos priorizando os melhores."
    )

def _salvar_chat_id(chat_id: int):
    """Salva chat_id no .env para persistência."""
    if not _is_admin_chat(chat_id):
        return
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

    if chat_id is not None and not _is_admin_chat(chat_id):
        print(f"[Bot] Envio bloqueado para chat nao-admin: {chat_id}")
        return

    ids = [chat_id] if chat_id else sorted(_ADMIN_CHAT_IDS)
    if not ids:
        print("[Bot] Nenhum chat_id registrado")
        return

    for cid in ids:
        prefs = get_preferences(cid)
        if prefs and not prefs.get("alerts_enabled", True):
            continue
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
    """Registra o chat e mostra boas-vindas com menu de botoes."""
    chat_id = update.effective_chat.id
    _salvar_chat_id(chat_id)

    try:
        await _configurar_menu(context.bot, chat_id=chat_id)
    except Exception:
        pass

    await _reply_html(update.message, _formatar_start_html(chat_id), reply_markup=_teclado_menu(chat_id))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra status geral do bot."""
    if not await _garantir_admin_message(update.message):
        return
    await _reply_html(update.message, _formatar_status_html(), reply_markup=_botao_voltar())


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executa o scanner de oportunidades manualmente."""
    if not await _garantir_admin_message(update.message):
        return
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
    if not await _garantir_admin_message(update.message):
        return
    learner = Learner(_db)
    msg = learner.relatorio_diario()
    await update.message.reply_text(
        msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
    )


async def cmd_treinar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Força retreino via AutoTuner (Optuna + strategy slicing)."""
    if not await _garantir_admin_message(update.message):
        return
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
    if not await _garantir_admin_message(update.message):
        return
    await _reply_html(update.message, _formatar_bulk_html(), reply_markup=_botao_voltar())


async def cmd_app(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Abre a mini app quando configurada."""
    if not MINI_APP_URL:
        await update.message.reply_text(
            "Mini App preparada, mas a URL HTTPS publica ainda nao foi configurada.",
            reply_markup=_botao_voltar(),
        )
        return

    teclado = InlineKeyboardMarkup([
        [InlineKeyboardButton("Abrir Painel", web_app=WebAppInfo(url=MINI_APP_URL))],
        [InlineKeyboardButton("← Menu Principal", callback_data="cmd_menu")],
    ])
    await _reply_html(
        update.message,
        "<b>Mini App do FuteBot</b>\n\nAbra o painel para ver status, ultimas tips e preferencias.",
        reply_markup=teclado,
    )


async def cmd_ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista todos os comandos."""
    await _reply_html(update.message, _formatar_ajuda_html(), reply_markup=_teclado_menu(update.effective_chat.id))


async def cmd_saude(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra saúde do modelo (guard rails, degradação, calibração)."""
    if not await _garantir_admin_message(update.message):
        return
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
    if not await _garantir_admin_message(update.message):
        return
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
        if not _is_admin_chat(query.message.chat.id):
            await _reply_html(
                query.message,
                "<b>FuteBot</b>\nUse o painel e a ajuda para navegar.",
                reply_markup=_teclado_menu(query.message.chat.id),
            )
            return
        msg = "🤖 *FuteBot — Menu Principal*\n\n👇 Escolha uma opção:"
        await query.message.reply_text(
            msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_teclado_menu(query.message.chat.id)
        )
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
    if handler_fn not in (cmd_ajuda, cmd_resultados) and not _is_admin_chat(query.message.chat.id):
        await _negar_acesso(query.message)
        return

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
            await _reply_html(query.message, _formatar_status_html(), reply_markup=_botao_voltar())

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
            await _reply_html(query.message, _formatar_bulk_html(), reply_markup=_botao_voltar())

        elif handler_fn == cmd_ao_vivo:
            await query.message.reply_text("⚽ Verificando jogos previstos...")
            msg = await _logica_ao_vivo()
            await query.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=_botao_voltar()
            )

        elif handler_fn == cmd_ajuda:
            await _reply_html(
                query.message,
                _formatar_ajuda_html(),
                reply_markup=_teclado_menu(query.message.chat.id),
            )

    except Exception as e:
        await query.message.reply_text(
            f"❌ Erro: {e}", reply_markup=_botao_voltar()
        )


# ══════════════════════════════════════════════
#  CONFIGURAÇÃO DO BOT
# ══════════════════════════════════════════════

async def post_init(application: Application):
    """Callback pos-inicializacao: configura comandos visiveis no Telegram."""
    global _app_instance
    _app_instance = application

    await _configurar_menu(application.bot)
    for cid in _chat_ids:
        try:
            await _configurar_menu(application.bot, chat_id=cid)
        except Exception:
            pass

    print(f"Bot Telegram iniciado!")
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
    app.add_handler(CommandHandler("app", cmd_app))
    app.add_handler(CommandHandler("ajuda", cmd_ajuda))

    # Handler de callbacks dos botões inline (deve vir depois dos comandos)
    app.add_handler(CallbackQueryHandler(_callback_handler))

    return app

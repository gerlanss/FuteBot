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
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

from telegram import (
    Bot,
    Update,
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE, ADMIN_CHAT_IDS, BET365_URL
from data.database import Database
from pipeline.scanner import Scanner
from pipeline.collector import Collector
from models.trainer import Trainer
from models.learner import Learner
from models.autotuner import AutoTuner
from models.market_discovery import MARKET_SPECS
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
    BotCommand("ajuda", "Ver como o bot funciona"),
]
_ADMIN_COMMANDS = [
    BotCommand("start", "Abrir menu do bot"),
    BotCommand("scan", "Rodar radar do dia"),
    BotCommand("scan_final", "Liberar mercados T-30"),
    BotCommand("ao_vivo", "Status dos jogos previstos"),
    BotCommand("resultados", "Resultados recentes"),
    BotCommand("metricas", "Performance do modelo"),
    BotCommand("saude", "Saude do modelo"),
    BotCommand("status", "Status geral"),
    BotCommand("treinar", "Retreinar modelo"),
    BotCommand("bulk", "Status do bulk download"),
    BotCommand("ajuda", "Ver todos os comandos"),
]

_DISCOVERY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_market_discovery_training.py"
_DISCOVERY_RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "discovery_runs"

# Carregar chat_id salvo (se existir)
if TELEGRAM_CHAT_ID:
    _chat_ids.add(int(TELEGRAM_CHAT_ID))
    _ADMIN_CHAT_IDS.add(int(TELEGRAM_CHAT_ID))
for _cid in _db.telegram_chat_ids():
    _chat_ids.add(int(_cid))
for _cid in _ADMIN_CHAT_IDS:
    _db.salvar_telegram_chat(_cid, is_admin=True)


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
            [InlineKeyboardButton("🗓️ Radar do dia", callback_data="cmd_scan"),
             InlineKeyboardButton("🚨 Liberação T-30", callback_data="cmd_scan_final")],
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



def _link_bet365_markdown(home_name: str = "", away_name: str = "", mercado: str = "") -> str:
    query_parts = [
        "site:bet365.bet.br",
        str(home_name or "").strip(),
        str(away_name or "").strip(),
        str(mercado or "").strip(),
    ]
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        return f"[Bet365]({BET365_URL})"
    search_url = f"https://www.google.com/search?q={quote_plus(query)}"
    return f"[Buscar na Bet365]({search_url})"


async def _configurar_menu(bot, chat_id: int | None = None):
    """Garante menu de comandos e botao lateral do Telegram."""
    await bot.set_my_commands(_PUBLIC_COMMANDS)
    await bot.set_my_commands(_PUBLIC_COMMANDS, scope=BotCommandScopeAllPrivateChats())
    for admin_chat_id in sorted(_ADMIN_CHAT_IDS):
        await bot.set_my_commands(_ADMIN_COMMANDS, scope=BotCommandScopeChat(admin_chat_id))

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
            "Use o menu para ver resultados e a ajuda."
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
        "- <code>/scan</code> - radar do dia, sem revelar mercados\n"
        "- <code>/scan_final</code> - liberar mercados da janela T-30\n"
        "- <code>/ao_vivo</code> - acompanhar jogos previstos\n"
        "- <code>/resultados</code> - ver resultados recentes\n\n"
        "<b>Monitoramento</b>\n"
        "- <code>/status</code> - resumo geral do bot\n"
        "- <code>/metricas</code> - performance acumulada\n"
        "- <code>/saude</code> - guard rails e degradacao\n\n"
        "<b>Manutencao</b>\n"
        "- <code>/treinar</code> - retreino global\n"
        "- <code>/treinar 135 over35</code> - retreino focal por liga\n"
        "- <code>/treinar descoberta</code> - treino sequencial liga x mercado em background\n"
        "- <code>/treinar descoberta 253 over15</code> - descoberta focada\n"
        "- <code>/bulk</code> - status do bulk download\n\n"
        "<b>Rotina automatica</b>\n"
        "- 07:00 - scanner do dia\n"
        "- a cada 2h - acompanhamento ao vivo\n"
        "- 06:45 - relatorio de resultados\n"
        "- mensal - retreino per-league\n\n"
        "<b>Como o FuteBot escolhe as tips</b>\n"
        "Cada jogo passa por um funil antes de virar tip.\n\n"
        "1. <b>Modelo por liga</b>\n"
        "Cada liga tem modelos proprios para resultado, gols, tempos e escanteios.\n\n"
        "2. <b>Confianca minima</b>\n"
        "Mercado com menos de 65% de confianca ja cai antes de tudo.\n\n"
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
        "As combinacoes usam jogos diferentes, exigem no minimo 65% por tip e 50% de confianca composta, e o bot envia no maximo 3 combos priorizando os melhores."
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


def _registrar_chat(chat_id: int, username: str = None, first_name: str = None):
    """Persiste qualquer chat conhecido; admin segue salvo tambem no .env."""
    _chat_ids.add(int(chat_id))
    _db.salvar_telegram_chat(
        int(chat_id),
        is_admin=_is_admin_chat(chat_id),
        username=username,
        first_name=first_name,
    )
    if _is_admin_chat(chat_id):
        _salvar_chat_id(int(chat_id))


def _registrar_update(update: Update):
    """Registra chat e remetente sempre que houver interacao com o bot."""
    chat = getattr(update, "effective_chat", None)
    if not chat:
        return
    user = getattr(update, "effective_user", None)
    _registrar_chat(
        int(chat.id),
        username=getattr(user, "username", None),
        first_name=getattr(user, "first_name", None),
    )


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


def _quebrar_texto(texto: str, limite: int = 3800) -> list[str]:
    """Divide mensagem longa em blocos seguros para o Telegram."""
    if len(texto) <= limite:
        return [texto]

    partes = []
    atual = []
    tamanho = 0
    for linha in texto.splitlines():
        extra = len(linha) + 1
        if atual and tamanho + extra > limite:
            partes.append("\n".join(atual).strip())
            atual = [linha]
            tamanho = extra
        else:
            atual.append(linha)
            tamanho += extra
    if atual:
        partes.append("\n".join(atual).strip())
    return partes


def _formatar_scan_publico_html(data: str = None) -> list[str]:
    """Monta o scan vigente do dia a partir do lote salvo no banco."""
    from collections import defaultdict

    data = data or datetime.now().strftime("%Y-%m-%d")
    tips = _db.predictions_por_data(data)
    combos = _db.combos_por_data(data)
    data_br = Scanner._data_br(data)

    aviso = (
        "<b>Comunicado do FuteBot</b>\n"
        "Hoje o bot exibiu inconsistencias em parte das mensagens de scan.\n"
        "O problema foi corrigido, o lote invalido foi limpo e este eh o reenvio do scan valido de "
        f"{data_br}.\n\n"
        "Desculpa pelo ruido."
    )

    if not tips:
        return [aviso, f"<b>Scan do dia {data_br}</b>\nNenhuma tip aprovada no lote atual."]

    por_liga = defaultdict(lambda: defaultdict(list))
    for tip in tips:
        por_liga[tip.get("league_name") or f"Liga {tip.get('league_id', '?')}"][tip["fixture_id"]].append(tip)

    blocos = [aviso]
    header = [
        f"<b>Scan validado do dia {data_br}</b>",
        f"- Tips aprovadas: {len(tips)}",
    ]
    blocos.append("\n".join(header))

    for nome_liga in sorted(por_liga):
        linhas = [f"<b>{nome_liga}</b>"]
        fixtures_ordenados = sorted(
            por_liga[nome_liga].items(),
            key=lambda item: item[1][0].get("fixture_date") or item[1][0].get("date") or "",
        )
        for _, fix_tips in fixtures_ordenados:
            fix_tips.sort(key=lambda item: item.get("prob_modelo") or 0, reverse=True)
            primeira = fix_tips[0]
            horario = Scanner._horario_local(primeira.get("fixture_date") or primeira.get("date"))
            agenda = f" ({horario})" if horario else ""
            linhas.append(
                f"\n<code>{primeira.get('home_name', '?')}</code> <b>x</b> "
                f"<code>{primeira.get('away_name', '?')}</code>{agenda}"
            )
            for tip in fix_tips:
                detalhes = [f"Conf {(tip.get('prob_modelo') or 0):.0%}"]
                odd = tip.get("odd_usada")
                ev = tip.get("ev_percent")
                if odd and odd > 1:
                    detalhes.append(f"Odd {odd:.2f}")
                if ev is not None:
                    detalhes.append(f"EV {ev:+.1f}%")
                if tip.get("bookmaker"):
                    detalhes.append(tip["bookmaker"])
                linhas.append(f"• <b>{tip.get('mercado')}</b>")
                linhas.append(f"  <i>{' | '.join(detalhes)}</i>")
        blocos.extend(_quebrar_texto("\n".join(linhas).strip()))

    if combos:
        linhas_combo = [f"<b>Combos do dia {data_br}</b>"]
        for idx, combo in enumerate(combos, start=1):
            tipo = "Dupla" if combo.get("combo_type") == "dupla" else "Tripla"
            linhas_combo.append(f"\n<b>{tipo} #{idx}</b> | Conf composta {(combo.get('prob_composta') or 0):.0%}")
            for item in combo.get("items", []):
                linhas_combo.append(
                    f"• <code>{item.get('home_name', '?')}</code> <b>x</b> "
                    f"<code>{item.get('away_name', '?')}</code>"
                )
                linhas_combo.append(
                    f"  <i>{item.get('mercado')} | {(item.get('prob_modelo') or 0):.0%}</i>"
                )
        blocos.extend(_quebrar_texto("\n".join(linhas_combo).strip()))

    return blocos


async def broadcast_scan_publico(data: str = None) -> dict:
    """Envia o scan validado do dia para todos os chats persistidos."""
    destinos = sorted(set(_db.telegram_chat_ids()))
    if not destinos:
        return {"destinatarios": 0, "entregues": 0, "falhas": []}

    bot = Bot(token=TELEGRAM_TOKEN)
    blocos = _formatar_scan_publico_html(data=data)
    entregues = 0
    falhas = []

    for chat_id in destinos:
        try:
            for bloco in blocos:
                await bot.send_message(
                    chat_id=chat_id,
                    text=bloco,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            entregues += 1
        except Exception as exc:
            falhas.append({"chat_id": chat_id, "erro": str(exc)})

    return {
        "destinatarios": len(destinos),
        "entregues": entregues,
        "falhas": falhas,
    }


# ══════════════════════════════════════════════
#  HANDLERS DE COMANDOS
# ══════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Registra o chat e mostra boas-vindas com menu de botoes."""
    _registrar_update(update)
    chat_id = update.effective_chat.id

    try:
        await _configurar_menu(context.bot, chat_id=chat_id)
    except Exception:
        pass

    await _reply_html(update.message, _formatar_start_html(chat_id), reply_markup=_teclado_menu(chat_id))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra status geral do bot."""
    _registrar_update(update)
    if not await _garantir_admin_message(update.message):
        return
    await _reply_html(update.message, _formatar_status_html(), reply_markup=_botao_voltar())


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executa a pré-seleção do dia sem revelar mercados."""
    _registrar_update(update)
    if not await _garantir_admin_message(update.message):
        return
    await update.message.reply_text("🗓️ Executando radar do dia... aguarde.")

    try:
        scanner = Scanner(_db)
        resultado = scanner.executar(mode="preselect")
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

        # Se o /scan manual cair já dentro da janela T-30, dispara a revisão
        # imediatamente para não depender do próximo ciclo automático.
        release = scanner.liberar_mercados()
        if any([
            release.get("tips_enviadas_llm"),
            release.get("tips_aprovadas"),
            release.get("tips_rejeitadas_llm"),
            release.get("combos"),
        ]):
            await update.message.reply_text("🚨 Encontrei jogo(s) já na janela T-30. Rodando revisão final agora.")
            release_msgs = scanner.formatar_relatorio(release)
            for i, (texto, botoes) in enumerate(release_msgs):
                kb = None
                if botoes:
                    kb = InlineKeyboardMarkup(
                        [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in botoes]
                    )
                elif i == len(release_msgs) - 1:
                    kb = _botao_voltar()
                await update.message.reply_text(
                    texto, parse_mode=ParseMode.HTML, reply_markup=kb
                )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro no scanner:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_scan_final(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Força a liberação final dos mercados na janela T-30."""
    _registrar_update(update)
    if not await _garantir_admin_message(update.message):
        return
    await update.message.reply_text("🚨 Rodando liberação final T-30... aguarde.")

    try:
        scanner = Scanner(_db)
        resultado = scanner.liberar_mercados()
        msgs = scanner.formatar_relatorio(resultado)
        for i, (texto, botoes) in enumerate(msgs):
            kb = None
            if botoes:
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
            f"❌ Erro na liberação final:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_resultados(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _registrar_update(update)
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
    _registrar_update(update)
    """Mostra métricas de performance do modelo (HTML para suportar nomes com _)."""
    if not await _garantir_admin_message(update.message):
        return
    learner = Learner(_db)
    msg = learner.relatorio_diario()
    await update.message.reply_text(
        msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
    )


async def cmd_treinar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _registrar_update(update)
    """Força retreino via AutoTuner (global ou focal por liga)."""
    if not await _garantir_admin_message(update.message):
        return

    league_id = None
    mercado = None
    n_trials = None
    if context.args:
        if context.args[0].lower() == "descoberta":
            args = context.args[1:]
            league_ids = []
            markets = []

            if args:
                try:
                    league_ids.append(int(args[0]))
                    args = args[1:]
                except ValueError:
                    league_ids = []
            if args:
                mercados_validos = {item.market_id for item in MARKET_SPECS}
                markets = [item for item in args if item in mercados_validos]

            _DISCOVERY_RUNS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = _DISCOVERY_RUNS_DIR / f"run_{stamp}.log"
            command = [sys.executable, str(_DISCOVERY_SCRIPT)]
            if league_ids:
                command.extend(["--league-ids", ",".join(str(item) for item in league_ids)])
            if markets:
                command.extend(["--markets", ",".join(markets)])

            with open(log_path, "a", encoding="utf-8") as log_file:
                subprocess.Popen(
                    command,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            detalhe = []
            if league_ids:
                detalhe.append(f"liga(s): {', '.join(str(x) for x in league_ids)}")
            if markets:
                detalhe.append(f"mercado(s): {', '.join(markets)}")
            alvo = "\n".join(f"- {item}" for item in detalhe) if detalhe else "- todas as ligas\n- todos os mercados"

            await update.message.reply_text(
                "🧠 Treino por descoberta iniciado em background.\n"
                "Fluxo: liga por liga, mercado por mercado, com resumo por etapa no log.\n"
                f"{alvo}\n\n"
                f"Log: <code>{log_path}</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=_botao_voltar(),
            )
            return
        try:
            league_id = int(context.args[0])
            mercado = context.args[1] if len(context.args) > 1 else None
            n_trials = 8
            detalhe = f" da liga {league_id}"
            if mercado:
                detalhe += f" para revisar o slice {mercado}"
            await update.message.reply_text(
                "🧠 Iniciando retreino focal"
                f"{detalhe}.\n"
                "Modo leve para VPS: AutoTuner só dessa liga, com menos trials.\n"
                "As strategies da liga serão regeneradas sem mexer nas outras."
            )
        except ValueError:
            await update.message.reply_text(
                "Uso: /treinar ou /treinar <league_id> [mercado]"
            )
            return
    else:
        await update.message.reply_text(
            "🧠 Iniciando AutoTuner global... Optuna + strategy slicing.\n"
            "Isso pode demorar ~30-60 min. Você receberá o resultado aqui."
        )

    try:
        tuner = AutoTuner(_db)
        resultado = tuner.executar(
            league_ids=[league_id] if league_id else None,
            n_trials=n_trials,
        )
        msg = AutoTuner.formatar_resultado(resultado)
        await update.message.reply_text(
            msg, parse_mode=ParseMode.HTML, reply_markup=_botao_voltar()
        )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Erro no AutoTuner:\n{e}", reply_markup=_botao_voltar()
        )


async def cmd_bulk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _registrar_update(update)
    """Status do bulk download."""
    if not await _garantir_admin_message(update.message):
        return
    await _reply_html(update.message, _formatar_bulk_html(), reply_markup=_botao_voltar())


async def cmd_ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _registrar_update(update)
    """Lista todos os comandos."""
    await _reply_html(update.message, _formatar_ajuda_html(), reply_markup=_teclado_menu(update.effective_chat.id))


async def cmd_saude(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _registrar_update(update)
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
    _registrar_update(update)
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
                f"   {mercado_label} @ {odd_usada:.2f} | Sem dados\n"
                f"   🔗 {_link_bet365_markdown(pred['home_name'], pred['away_name'], mercado_label)}"
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
                f"   {mercado_label} @ {odd_usada:.2f} | Aguardando início\n"
                f"   🔗 {_link_bet365_markdown(pred['home_name'], pred['away_name'], mercado_label)}"
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
                f"   {status_label}{elapsed_txt} | {mercado_label} @ {odd_usada:.2f}\n"
                f"   🔗 {_link_bet365_markdown(pred['home_name'], pred['away_name'], mercado_label)}"
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
                    f"   {mercado_label} @ {odd_usada:.2f} | *+{lucro:.2f}u*\n"
                    f"   🔗 {_link_bet365_markdown(pred['home_name'], pred['away_name'], mercado_label)}"
                )
            else:
                erros += 1
                lines.append(
                    f"❌ *{pred['home_name']} {gh}-{ga} {pred['away_name']}*\n"
                    f"   {mercado_label} @ {odd_usada:.2f} | *-1.00u*\n"
                    f"   🔗 {_link_bet365_markdown(pred['home_name'], pred['away_name'], mercado_label)}"
                )
            finalizados += 1

        else:
            # Outros status: PST (adiado), CANC (cancelado), etc.
            lines.append(
                f"⚠️ *{pred['home_name']} vs {pred['away_name']}*\n"
                f"   {mercado_label} | Status: {status}\n"
                f"   🔗 {_link_bet365_markdown(pred['home_name'], pred['away_name'], mercado_label)}"
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
    "cmd_scan_final": cmd_scan_final,
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
    _registrar_update(update)
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
            await query.message.reply_text("🗓️ Executando radar do dia... aguarde.")
            scanner = Scanner(_db)
            resultado = scanner.executar(mode="preselect")
            msgs = scanner.formatar_relatorio(resultado)
            for i, (texto, botoes) in enumerate(msgs):
                kb = None
                if botoes:
                    kb = InlineKeyboardMarkup(
                        [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in botoes]
                    )
                elif i == len(msgs) - 1:
                    kb = _botao_voltar()
                await query.message.reply_text(texto, parse_mode=ParseMode.HTML, reply_markup=kb)
            release = scanner.liberar_mercados()
            if any([
                release.get("tips_enviadas_llm"),
                release.get("tips_aprovadas"),
                release.get("tips_rejeitadas_llm"),
                release.get("combos"),
            ]):
                await query.message.reply_text("🚨 Encontrei jogo(s) já na janela T-30. Rodando revisão final agora.")
                release_msgs = scanner.formatar_relatorio(release)
                for i, (texto, botoes) in enumerate(release_msgs):
                    kb = None
                    if botoes:
                        kb = InlineKeyboardMarkup(
                            [[InlineKeyboardButton(label, callback_data=cb)] for label, cb in botoes]
                        )
                    elif i == len(release_msgs) - 1:
                        kb = _botao_voltar()
                    await query.message.reply_text(texto, parse_mode=ParseMode.HTML, reply_markup=kb)
        elif handler_fn == cmd_scan_final:
            await query.message.reply_text("🚨 Rodando liberação final T-30... aguarde.")
            scanner = Scanner(_db)
            resultado = scanner.liberar_mercados()
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
    app.add_handler(CommandHandler("scan_final", cmd_scan_final))
    app.add_handler(CommandHandler("ao_vivo", cmd_ao_vivo))
    app.add_handler(CommandHandler("resultados", cmd_resultados))
    app.add_handler(CommandHandler("metricas", cmd_metricas))
    app.add_handler(CommandHandler("saude", cmd_saude))
    app.add_handler(CommandHandler("treinar", cmd_treinar))
    app.add_handler(CommandHandler("bulk", cmd_bulk))
    app.add_handler(CommandHandler("ajuda", cmd_ajuda))

    # Handler de callbacks dos botões inline (deve vir depois dos comandos)
    app.add_handler(CallbackQueryHandler(_callback_handler))

    return app

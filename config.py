"""
Configurações centrais do FuteBot.
Carrega variáveis de ambiente e define constantes de todas as APIs.

APIs utilizadas:
  - API-Football (api-sports.io) → stats, fixtures, predictions
  - The Odds API → odds reais de casas de aposta
  - Telegram Bot API → alertas e relatórios
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")

# ──────────────────────────────────────────────
# API-Football (api-sports.io)
# Autenticação: header x-apisports-key
# Plano Pro: 7.500 req/dia, todas as seasons
# ──────────────────────────────────────────────
API_FOOTBALL_KEY = _get_env_str("API_FOOTBALL_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# ──────────────────────────────────────────────
# Temporada padrão (Pro: todas disponíveis)
# ──────────────────────────────────────────────
DEFAULT_SEASON = _get_env_int("DEFAULT_SEASON", 2025)

# Seasons para bulk download e treino do modelo
TRAIN_SEASONS = [2024, 2025, 2026]

# ──────────────────────────────────────────────
# IDs de ligas populares (atalhos)
# ──────────────────────────────────────────────
LEAGUES = {
    # ── Brasil ──
    "brasileirao_a":   {"id": 71,  "nome": "Brasileirão Série A"},
    "brasileirao_b":   {"id": 72,  "nome": "Brasileirão Série B"},
    "copa_do_brasil":  {"id": 73,  "nome": "Copa do Brasil"},
    "paulistao":       {"id": 475, "nome": "Campeonato Paulista"},
    "carioca":         {"id": 479, "nome": "Campeonato Carioca"},
    # ── Europa — Top 5 ──
    "premier_league":  {"id": 39,  "nome": "Premier League"},
    "la_liga":         {"id": 140, "nome": "La Liga"},
    "serie_a_italia":  {"id": 135, "nome": "Serie A (Itália)"},
    "bundesliga":      {"id": 78,  "nome": "Bundesliga"},
    "ligue_1":         {"id": 61,  "nome": "Ligue 1"},
    # ── Europa — Copas ──
    "champions_league":  {"id": 2,   "nome": "Champions League"},
    "europa_league":     {"id": 3,   "nome": "Europa League"},

    # ── América do Sul ──
    "libertadores":       {"id": 13,  "nome": "Copa Libertadores"},
    "sulamericana":       {"id": 11,  "nome": "Copa Sul-Americana"},
    "argentina_primera":  {"id": 128, "nome": "Liga Profesional (Argentina)"},
    "colombiana_primera": {"id": 239, "nome": "Liga BetPlay (Colômbia)"},
    "chilena_primera":    {"id": 265, "nome": "Primera División (Chile)"},
    # ── América do Norte ──
    "mls":     {"id": 253, "nome": "MLS"},
    "liga_mx": {"id": 262, "nome": "Liga MX"},
    # ── Ásia / Oceania ──
    "j_league":         {"id": 98,  "nome": "J1 League (Japão)"},

    "saudi_pro_league": {"id": 307, "nome": "Saudi Pro League"},
    "a_league":         {"id": 188, "nome": "A-League (Austrália)"},
    # ── Seleções ──
    "copa_do_mundo": {"id": 1, "nome": "Copa do Mundo FIFA"},
}

# ──────────────────────────────────────────────
# The Odds API
# Plano Starter: 500 créditos/mês, odds em tempo real
# ──────────────────────────────────────────────
ODDS_API_KEY = _get_env_str("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Mapeamento liga API-Football → chave The Odds API
# None = liga sem cobertura na Odds API (tip sem odds, só confiança do modelo)
ODDS_SPORTS_MAP = {
    # Brasil
    71:  "soccer_brazil_campeonato",            # Brasileirão A
    72:  None,                                  # Brasileirão B (não tem na Odds API)
    73:  None,                                  # Copa do Brasil (não tem na Odds API)
    475: None,                                  # Paulistão (não tem)
    479: None,                                  # Carioca (não tem)

    # Europa — Top 5
    39:  "soccer_epl",                          # Premier League
    140: "soccer_spain_la_liga",                # La Liga
    135: "soccer_italy_serie_a",                # Serie A (Itália)
    78:  "soccer_germany_bundesliga",           # Bundesliga
    61:  "soccer_france_ligue_one",             # Ligue 1
    # Europa — Copas
    2:   "soccer_uefa_champs_league",           # Champions League
    3:   "soccer_uefa_europa_league",           # Europa League

    # América do Sul
    13:  "soccer_conmebol_copa_libertadores",   # Libertadores
    11:  "soccer_conmebol_copa_sudamericana",   # Sul-Americana
    128: "soccer_argentina_primera_division",   # Argentina Primera
    239: None,                                  # Colômbia Primera A (não tem na Odds API)
    265: "soccer_chile_campeonato",             # Chile Primera

    # América do Norte
    253: "soccer_usa_mls",                      # MLS
    262: "soccer_mexico_ligamx",                # Liga MX
    # Ásia / Oceania
    98:  "soccer_japan_j_league",               # J-League

    307: "soccer_saudi_arabia_pro_league",      # Saudi Pro League
    188: "soccer_australia_aleague",            # A-League
    # Seleções
    1:   "soccer_fifa_world_cup",               # Copa do Mundo
}

# ──────────────────────────────────────────────
# Telegram Bot
# ──────────────────────────────────────────────
TELEGRAM_TOKEN = _get_env_str("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _get_env_str("TELEGRAM_CHAT_ID")  # Preenchido no primeiro /start

# ──────────────────────────────────────────────
# Pipeline / Scanner
# ──────────────────────────────────────────────
MAX_JOGOS_ODDS = 10           # Máximo de jogos para buscar odds (economia de créditos)
EV_THRESHOLD = 0.03           # EV mínimo para considerar oportunidade (3%)

# Casa de apostas preferida (key do The Odds API)
# Pinnacle: odds sharp de referência, melhor cobertura na API, opera no Brasil
PREFERRED_BOOKMAKER = "pinnacle"
PREFERRED_BOOKMAKER_LABEL = "Pinnacle"
SCAN_HORA = "07:00"           # Horário do scan diário
RESULTADOS_HORA = "06:00"     # Horário da coleta de resultados
RETREINO_DIA = "sun"           # Dia do retreino mensal (domingo — termina antes do scanner de segunda)
RETREINO_HORA = "14:00"        # Horário do retreino mensal (14h dom → termina ~03h seg, 4h antes do scanner)
RETREINO_SEMANA = "1st"        # Qual semana do mês (1st, 2nd, 3rd, 4th, last) — 1º domingo

# Fuso horário local (IANA). Boa Vista/RR = America/Boa_Vista (UTC-4).
# Usado no scheduler, conversão de horários e exibição no Telegram.
TIMEZONE = _get_env_str("TIMEZONE", "America/Boa_Vista")

# ──────────────────────────────────────────────
# Auto-bootstrap — bot treina sozinho sem intervenção
# ──────────────────────────────────────────────
MIN_FIXTURES_TREINO = 200     # Mínimo de fixtures FT para tentar treinar
BULK_BATCH_DIARIO = 2000      # Máximo de stats para baixar por dia (deixa margem)
BULK_HORA = "03:00"           # Horário do bulk incremental (madrugada)

# ──────────────────────────────────────────────
# Guard Rails — proteção contra modelo "burro"
# ──────────────────────────────────────────────
# Accuracy mínima para aceitar modelo (baseline conservador: ~40% no 1x2 ≈ apostar num chute)
MODEL_ACC_MIN_1X2 = 0.38      # Se accuracy_test < 38% no 1x2 → modelo rejeitado
MODEL_ACC_MIN_BINARY = 0.50   # Se accuracy_test < 50% em O/U ou BTTS → pior que moeda
# Brier Score máximo aceitável para 1x2 multiclass (random ~0.667, bom <0.60)
# Para binário o threshold natural seria 0.25, mas usamos apenas o 1x2 no gate
MODEL_BRIER_MAX = 0.65
# Auto-pause: se ROI acumulado cair abaixo desse threshold, scanner para de operar
ROI_PAUSE_THRESHOLD = -15.0   # Scanner pausa se ROI < -15%
# Quantidade mínima de previsões resolvidas antes de ativar auto-pause
ROI_PAUSE_MIN_BETS = 20       # Precisa de pelo menos 20 previsões para julgar
# Degradação: alerta se accuracy dos últimos N dias cair abaixo de baseline
DEGRADATION_WINDOW_DAYS = 14  # Janela para medir degradação
DEGRADATION_ACC_MIN = 0.35    # Se accuracy janela < 35% → alerta de degradação

# ──────────────────────────────────────────────
# DeepSeek LLM — Validação inteligente de tips
# Atua como "segundo par de olhos" pós-XGBoost
# ──────────────────────────────────────────────
DEEPSEEK_API_KEY = _get_env_str("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"                # DeepSeek-V3 (barato e rápido)
USE_LLM_VALIDATION = _get_env_bool("USE_LLM_VALIDATION", False)
LLM_MIN_EV_FOR_REVIEW = 3.0   # Só manda para LLM tips com EV >= 3% (economiza tokens)

# ──────────────────────────────────────────────
# Banco de dados
# ──────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "futebot.db")

# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
SECRET_KEY = _get_env_str("FLASK_SECRET_KEY") or _get_env_str("SECRET_KEY")
FLASK_DEBUG = _get_env_bool("FLASK_DEBUG", True)
FLASK_PORT = _get_env_int("FLASK_PORT", 5000)

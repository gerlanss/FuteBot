"""
Client para The Odds API v4.

Busca odds reais de casas de aposta para jogos futuros.
Usado na etapa final do pipeline: após o modelo filtrar oportunidades,
buscamos odds apenas dos jogos selecionados.

Plano Starter (free): 500 créditos/mês.
Custo por chamada: 1 crédito por region × markets solicitados.

Referência: https://the-odds-api.com/liveapi/guides/v4/

Uso:
  from services.odds_api import buscar_odds_liga, creditos_restantes
  odds = buscar_odds_liga("soccer_brazil_campeonato")
"""

import requests
from config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORTS_MAP

_TIMEOUT = 15

# Armazena o último valor de créditos restantes
_ultimo_creditos = {"remaining": None, "used": None}


def _get(endpoint: str, params: dict = None) -> tuple[list | dict, dict]:
    """
    GET genérico na The Odds API.
    Retorna (data, headers_info) onde headers_info tem créditos restantes.
    """
    url = f"{ODDS_API_BASE}/{endpoint}"
    base_params = {"apiKey": ODDS_API_KEY}
    if params:
        base_params.update(params)

    try:
        resp = requests.get(url, params=base_params, timeout=_TIMEOUT)
        resp.raise_for_status()

        # Atualizar créditos restantes
        _ultimo_creditos["remaining"] = resp.headers.get("x-requests-remaining")
        _ultimo_creditos["used"] = resp.headers.get("x-requests-used")

        return resp.json(), _ultimo_creditos.copy()

    except requests.RequestException as e:
        print(f"[OddsAPI] ERRO ao acessar {url}: {e}")
        return [], _ultimo_creditos.copy()


def creditos_restantes() -> dict:
    """Retorna último status de créditos: {'remaining': '495', 'used': '5'}."""
    return _ultimo_creditos.copy()


def listar_esportes() -> list[dict]:
    """
    Lista todos os esportes disponíveis (não gasta crédito).
    Retorna lista de {key, group, title, description, active}.
    """
    data, _ = _get("sports")
    return data if isinstance(data, list) else []


def buscar_odds_liga(sport_key: str, markets: str = "h2h,totals,btts,h2h_h1",
                     regions: str = "eu", odds_format: str = "decimal") -> list[dict]:
    """
    Busca odds de todos os jogos futuros de uma liga.

    Parâmetros:
      sport_key: chave da liga (ex: 'soccer_brazil_campeonato')
      markets: mercados separados por vírgula ('h2h', 'totals', 'btts', 'h2h_h1', 'spreads')
      regions: regiões de casas ('eu', 'uk', 'us', 'au')
      odds_format: 'decimal' ou 'american'

    Retorna lista de jogos com odds de múltiplas casas.
    Custo: 1 crédito por market × region solicitado (4 markets = 4 créditos).
    """
    data, info = _get(f"sports/{sport_key}/odds", {
        "markets": markets,
        "regions": regions,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    })

    if info.get("remaining"):
        print(f"[OddsAPI] Créditos restantes: {info['remaining']}")

    return data if isinstance(data, list) else []


def buscar_odds_por_league_id(league_id: int, markets: str = "h2h,totals,btts,h2h_h1") -> list[dict]:
    """
    Busca odds usando o league_id da API-Football.
    Converte automaticamente para o sport_key do The Odds API.

    Retorna [] se a liga não está mapeada ou não tem jogos.
    """
    sport_key = ODDS_SPORTS_MAP.get(league_id)
    if not sport_key:
        print(f"[OddsAPI] Liga {league_id} não tem mapeamento no The Odds API")
        return []

    return buscar_odds_liga(sport_key, markets=markets)


def encontrar_odds_jogo(odds_lista: list[dict], home_name: str,
                        away_name: str) -> dict | None:
    """
    Encontra odds de um jogo específico na lista, por nome de time.
    Usa busca parcial (case-insensitive) porque nomes podem diferir entre APIs.

    Retorna o dict do jogo com todas as odds, ou None.
    """
    home_lower = home_name.lower()
    away_lower = away_name.lower()

    for game in odds_lista:
        gh = game.get("home_team", "").lower()
        ga = game.get("away_team", "").lower()

        # Busca parcial: qualquer palavra do nome
        home_match = any(w in gh for w in home_lower.split() if len(w) > 3)
        away_match = any(w in ga for w in away_lower.split() if len(w) > 3)

        if home_match and away_match:
            return game

    return None


def extrair_melhor_odd(game: dict, mercado: str = "h2h",
                       outcome: str = "Home",
                       point: float = None) -> tuple[float, str]:
    """
    Encontra a melhor odd (maior) para um outcome específico entre todas as casas.

    Parâmetros:
      game: dict de um jogo retornado por buscar_odds_liga
      mercado: 'h2h', 'totals', 'spreads'
      outcome: 'Home', 'Away', 'Draw', 'Over', 'Under'
      point: linha específica para totals (ex: 2.5, 1.5, 3.5).
             Obrigatório para mercado 'totals' — evita misturar linhas.

    Retorna (melhor_odd, nome_casa) ou (0.0, '') se não encontrar.
    """
    melhor_odd = 0.0
    melhor_casa = ""

    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != mercado:
                continue
            for oc in mkt.get("outcomes", []):
                # Filtrar pela linha (point) quando especificada
                # Essencial para totals: sem isso, Over 4.5 seria
                # confundido com Over 2.5
                if point is not None and oc.get("point") != point:
                    continue

                nome = oc.get("name", "")
                # Mapear nomes: The Odds API usa nome do time, não "Home"/"Away"
                if outcome == "Home" and nome == game.get("home_team"):
                    nome = "Home"
                elif outcome == "Away" and nome == game.get("away_team"):
                    nome = "Away"

                if nome == outcome or oc.get("name") == outcome:
                    price = oc.get("price", 0)
                    if price > melhor_odd:
                        melhor_odd = price
                        melhor_casa = bk.get("title", bk.get("key", ""))

    return melhor_odd, melhor_casa


def extrair_odd_preferida(game: dict, mercado: str = "h2h",
                          outcome: str = "Home",
                          bookmaker_key: str = None,
                          point: float = None) -> tuple[float, str, bool]:
    """
    Extrai odd da casa preferida (ex: Bet365) com fallback para melhor alternativa.

    Prioridade:
      1. Odd da casa preferida (bookmaker_key)
      2. Se não disponível, melhor odd entre todas as casas

    Parâmetros:
      game: dict de um jogo retornado por buscar_odds_liga
      mercado: 'h2h', 'totals', 'spreads'
      outcome: 'Home', 'Away', 'Draw', 'Over', 'Under'
      bookmaker_key: key da casa preferida (ex: 'bet365')
      point: linha específica para totals (ex: 2.5, 1.5, 3.5).
             Obrigatório para mercado 'totals' — evita misturar linhas.

    Retorna (odd, nome_casa, is_preferida).
      - is_preferida=True: odd é da casa preferida
      - is_preferida=False: fallback para melhor alternativa
    """
    from config import PREFERRED_BOOKMAKER, PREFERRED_BOOKMAKER_LABEL
    bk_key = bookmaker_key or PREFERRED_BOOKMAKER
    bk_label = PREFERRED_BOOKMAKER_LABEL

    # Tentar extrair da casa preferida
    odd_pref = 0.0
    for bk in game.get("bookmakers", []):
        if bk.get("key") != bk_key:
            continue
        for mkt in bk.get("markets", []):
            if mkt.get("key") != mercado:
                continue
            for oc in mkt.get("outcomes", []):
                # Filtrar pela linha (point) quando especificada
                if point is not None and oc.get("point") != point:
                    continue

                nome = oc.get("name", "")
                if outcome == "Home" and nome == game.get("home_team"):
                    nome = "Home"
                elif outcome == "Away" and nome == game.get("away_team"):
                    nome = "Away"
                if nome == outcome or oc.get("name") == outcome:
                    odd_pref = oc.get("price", 0)
                    break

    if odd_pref > 0:
        return odd_pref, bk_label, True

    # Fallback: melhor odd entre todas as casas
    melhor_odd, melhor_casa = extrair_melhor_odd(game, mercado, outcome, point=point)
    return melhor_odd, melhor_casa, False


def extrair_odds_pinnacle(game: dict, mercado: str = "h2h",
                          point: float = None) -> dict:
    """
    Extrai odds da Pinnacle (linha sharp de referência).
    Retorna dict com outcomes: {'Home': 1.85, 'Draw': 3.50, 'Away': 4.20}.
    """
    for bk in game.get("bookmakers", []):
        if bk.get("key") == "pinnacle":
            for mkt in bk.get("markets", []):
                if mkt.get("key") == mercado:
                    result = {}
                    for oc in mkt.get("outcomes", []):
                        # Filtrar pela linha quando especificada
                        if point is not None and oc.get("point") != point:
                            continue

                        nome = oc.get("name", "")
                        if nome == game.get("home_team"):
                            result["Home"] = oc["price"]
                        elif nome == game.get("away_team"):
                            result["Away"] = oc["price"]
                        elif nome == "Draw":
                            result["Draw"] = oc["price"]
                        elif nome == "Over":
                            result["Over"] = oc["price"]
                        elif nome == "Under":
                            result["Under"] = oc["price"]
                    return result
    return {}


def resumo_odds_jogo(game: dict) -> dict:
    """
    Gera um resumo estruturado das odds de um jogo.

    Usa a casa preferida (Bet365) como prioridade. Se a casa não tiver
    o jogo/mercado, faz fallback para a melhor odd disponível e sinaliza.

    Retorna dict com odd por mercado, Pinnacle, quantidade de casas
    e flag is_preferida por outcome.
    """
    n_casas = len(game.get("bookmakers", []))

    # H2H (1x2) — prioridade Bet365
    home_odd, home_casa, home_pref = extrair_odd_preferida(game, "h2h", game.get("home_team", ""))
    draw_odd, draw_casa, draw_pref = extrair_odd_preferida(game, "h2h", "Draw")
    away_odd, away_casa, away_pref = extrair_odd_preferida(game, "h2h", game.get("away_team", ""))

    # Over/Under 2.5 — prioridade Bet365, FILTRANDO por point=2.5
    over_odd, over_casa, over_pref = extrair_odd_preferida(
        game, "totals", "Over", point=2.5
    )
    under_odd, under_casa, under_pref = extrair_odd_preferida(
        game, "totals", "Under", point=2.5
    )

    # Over/Under 1.5 — prioridade Bet365, FILTRANDO por point=1.5
    over15_odd, over15_casa, over15_pref = extrair_odd_preferida(
        game, "totals", "Over", point=1.5
    )
    under15_odd, under15_casa, under15_pref = extrair_odd_preferida(
        game, "totals", "Under", point=1.5
    )

    # Over/Under 3.5 — prioridade Bet365, FILTRANDO por point=3.5
    over35_odd, over35_casa, over35_pref = extrair_odd_preferida(
        game, "totals", "Over", point=3.5
    )
    under35_odd, under35_casa, under35_pref = extrair_odd_preferida(
        game, "totals", "Under", point=3.5
    )

    # BTTS (Ambos Marcam) — prioridade Bet365
    # The Odds API usa market key "btts" com outcomes "Yes" / "No"
    btts_yes_odd, btts_yes_casa, btts_yes_pref = extrair_odd_preferida(
        game, "btts", "Yes"
    )
    btts_no_odd, btts_no_casa, btts_no_pref = extrair_odd_preferida(
        game, "btts", "No"
    )

    # H2H Primeiro Tempo (1T) — prioridade Bet365
    # The Odds API usa market key "h2h_h1" com outcomes = nomes dos times + "Draw"
    ht_home_odd, ht_home_casa, ht_home_pref = extrair_odd_preferida(
        game, "h2h_h1", game.get("home_team", "")
    )
    ht_draw_odd, ht_draw_casa, ht_draw_pref = extrair_odd_preferida(
        game, "h2h_h1", "Draw"
    )
    ht_away_odd, ht_away_casa, ht_away_pref = extrair_odd_preferida(
        game, "h2h_h1", game.get("away_team", "")
    )

    # Pinnacle (referência sharp, mantém separado)
    pinnacle_h2h = extrair_odds_pinnacle(game, "h2h")
    pinnacle_totals = extrair_odds_pinnacle(game, "totals", point=2.5)

    return {
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "commence_time": game.get("commence_time"),
        "n_casas": n_casas,
        "h2h": {
            "home": {"odd": home_odd, "casa": home_casa, "preferida": home_pref},
            "draw": {"odd": draw_odd, "casa": draw_casa, "preferida": draw_pref},
            "away": {"odd": away_odd, "casa": away_casa, "preferida": away_pref},
        },
        "totals": {
            "over": {"odd": over_odd, "casa": over_casa, "preferida": over_pref},
            "under": {"odd": under_odd, "casa": under_casa, "preferida": under_pref},
        },
        "totals_15": {
            "over": {"odd": over15_odd, "casa": over15_casa, "preferida": over15_pref},
            "under": {"odd": under15_odd, "casa": under15_casa, "preferida": under15_pref},
        },
        "totals_35": {
            "over": {"odd": over35_odd, "casa": over35_casa, "preferida": over35_pref},
            "under": {"odd": under35_odd, "casa": under35_casa, "preferida": under35_pref},
        },
        "btts": {
            "yes": {"odd": btts_yes_odd, "casa": btts_yes_casa, "preferida": btts_yes_pref},
            "no": {"odd": btts_no_odd, "casa": btts_no_casa, "preferida": btts_no_pref},
        },
        "h2h_h1": {
            "home": {"odd": ht_home_odd, "casa": ht_home_casa, "preferida": ht_home_pref},
            "draw": {"odd": ht_draw_odd, "casa": ht_draw_casa, "preferida": ht_draw_pref},
            "away": {"odd": ht_away_odd, "casa": ht_away_casa, "preferida": ht_away_pref},
        },
        "pinnacle": {
            "h2h": pinnacle_h2h,
            "totals": pinnacle_totals,
        },
    }

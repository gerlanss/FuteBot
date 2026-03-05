"""
Client para a API-Football (api-sports.io) v3.

Autenticação: header x-apisports-key
Base URL: https://v3.football.api-sports.io
Docs: https://www.api-football.com/documentation-v3

Plano Free:
  - 100 requisições/dia
  - Temporadas: 2022 a 2024
  - Sem acesso a parâmetros 'last', 'next', 'live'

Todas as funções retornam listas ou dicts Python, nunca None.
Em caso de erro, retornam lista/dict vazio e imprimem log no console.
"""

import requests
from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE, DEFAULT_SEASON

# Timeout padrão para todas as requisições (segundos)
_TIMEOUT = 15

# Headers de autenticação (reutilizados em todas as chamadas)
_HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}


# ══════════════════════════════════════════════
#  HELPER GENÉRICO
# ══════════════════════════════════════════════

def _get(endpoint: str, params: dict | None = None) -> list | dict:
    """
    Faz GET na API-Football.
    Retorna o conteúdo de 'response' do JSON, ou [] em caso de erro.
    Também retorna 'errors' e 'results' logando no console.
    """
    url = f"{API_FOOTBALL_BASE}/{endpoint}"
    try:
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # Log de erros da API (ex: plano free sem acesso)
        if data.get("errors") and len(data["errors"]) > 0:
            print(f"[API-Football] AVISO {endpoint}: {data['errors']}")

        return data.get("response", [])

    except requests.RequestException as e:
        print(f"[API-Football] ERRO ao acessar {url}: {e}")
        return []


def raw_request(endpoint: str, params: dict | None = None) -> dict:
    """
    Faz GET e retorna o JSON completo (com paging, errors, etc).
    Útil para o explorador de API.
    """
    url = f"{API_FOOTBALL_BASE}/{endpoint}"
    try:
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[API-Football] ERRO ao acessar {url}: {e}")
        return {"errors": {"request": str(e)}, "response": []}


# ══════════════════════════════════════════════
#  STATUS DA CONTA
# ══════════════════════════════════════════════

def status_conta() -> dict:
    """Retorna info da conta: plano, requests usadas, limite."""
    data = raw_request("status")
    return data.get("response", {})


# ══════════════════════════════════════════════
#  BUSCA DE TIMES
# ══════════════════════════════════════════════

def buscar_time(nome: str) -> list[dict]:
    """
    Busca times pelo nome.
    Retorna lista de {team: {...}, venue: {...}}.
    """
    return _get("teams", {"search": nome})


def detalhes_time(team_id: int) -> dict:
    """
    Retorna detalhes de um time pelo ID.
    Retorna {team: {...}, venue: {...}} ou {}.
    """
    result = _get("teams", {"id": team_id})
    return result[0] if result else {}


# ══════════════════════════════════════════════
#  ESTATÍSTICAS DO TIME (completas!)
# ══════════════════════════════════════════════

def stats_time(team_id: int, league_id: int, season: int = DEFAULT_SEASON) -> dict:
    """
    Retorna estatísticas detalhadas do time em uma liga/temporada.
    Inclui: form, fixtures (V/E/D), goals (por minuto, over/under),
    biggest wins/losses, clean sheets, penalties, lineups, cards.
    """
    result = _get("teams/statistics", {
        "team": team_id,
        "league": league_id,
        "season": season,
    })
    return result if isinstance(result, dict) else {}


# ══════════════════════════════════════════════
#  JOGOS (FIXTURES)
# ══════════════════════════════════════════════

def jogos_time(team_id: int, league_id: int, season: int = DEFAULT_SEASON) -> list[dict]:
    """
    Retorna todos os jogos de um time em uma liga/temporada.
    Cada item tem: fixture, league, teams, goals, score.
    """
    return _get("fixtures", {
        "team": team_id,
        "league": league_id,
        "season": season,
    })


def detalhe_jogo(fixture_id: int) -> dict:
    """
    Retorna detalhes de um jogo específico.
    Inclui: fixture, league, teams, goals, score, events, lineups, statistics.
    """
    result = _get("fixtures", {"id": fixture_id})
    return result[0] if result else {}


# ══════════════════════════════════════════════
#  ESTATÍSTICAS DE PARTIDA (chutes, posse, etc)
# ══════════════════════════════════════════════

def stats_partida(fixture_id: int) -> list[dict]:
    """
    Retorna stats detalhadas de uma partida: chutes, posse, cartões,
    escanteios, faltas, passes, xG, etc.
    Retorna lista com 2 dicts (um por time), cada um com:
      {team: {...}, statistics: [{type: '...', value: ...}, ...]}
    """
    return _get("fixtures/statistics", {"fixture": fixture_id})


def eventos_partida(fixture_id: int) -> list[dict]:
    """
    Retorna eventos (gols, cartões, substituições) de uma partida.
    Cada evento: {time, team, player, assist, type, detail, comments}.
    """
    return _get("fixtures/events", {"fixture": fixture_id})


def escalacao_partida(fixture_id: int) -> list[dict]:
    """
    Retorna escalações de uma partida.
    Cada item: {team, formation, startXI, substitutes, coach}.
    """
    return _get("fixtures/lineups", {"fixture": fixture_id})


# ══════════════════════════════════════════════
#  CLASSIFICAÇÃO (STANDINGS)
# ══════════════════════════════════════════════

def classificacao(league_id: int, season: int = DEFAULT_SEASON) -> list[dict]:
    """
    Retorna a tabela de classificação de uma liga.
    Cada item tem: rank, team, points, goalsDiff, form, all/home/away stats, etc.
    """
    result = _get("standings", {"league": league_id, "season": season})
    if result and isinstance(result, list) and len(result) > 0:
        # standings vem dentro de result[0]['league']['standings']
        liga = result[0].get("league", {})
        standings = liga.get("standings", [])
        # Pode ter múltiplos grupos (Champions League), flatten
        if standings and isinstance(standings[0], list):
            return standings[0]
        return standings
    return []


# ══════════════════════════════════════════════
#  JOGADORES — TOP SCORERS, TOP ASSISTS
# ══════════════════════════════════════════════

def artilheiros(league_id: int, season: int = DEFAULT_SEASON) -> list[dict]:
    """
    Retorna os artilheiros de uma liga/temporada.
    Cada item: {player: {...}, statistics: [{...}]}.
    """
    return _get("players/topscorers", {"league": league_id, "season": season})


def top_assistencias(league_id: int, season: int = DEFAULT_SEASON) -> list[dict]:
    """
    Retorna top assistentes de uma liga/temporada.
    """
    return _get("players/topassists", {"league": league_id, "season": season})


# ══════════════════════════════════════════════
#  ELENCO DO TIME
# ══════════════════════════════════════════════

def elenco_time(team_id: int, season: int = DEFAULT_SEASON) -> list[dict]:
    """
    Retorna o elenco completo de um time em uma temporada.
    Cada jogador: {player: {id, name, age, number, position, photo}, statistics: [...]}.
    """
    return _get("players", {"team": team_id, "season": season})


# ══════════════════════════════════════════════
#  LIGAS
# ══════════════════════════════════════════════

def listar_ligas(country: str = "Brazil") -> list[dict]:
    """
    Lista ligas de um país.
    Cada item: {league: {id, name, type, logo}, country: {...}, seasons: [...]}.
    """
    return _get("leagues", {"country": country})


# ══════════════════════════════════════════════
#  LESÕES / DESFALQUES
# ══════════════════════════════════════════════

def lesoes_fixture(fixture_id: int) -> list[dict]:
    """
    Retorna lesões/desfalques de uma partida.
    Cada item: {player: {id, name, photo, type, reason}, team: {...}}.
    type pode ser: 'Missing Fixture', 'Questionable', etc.
    """
    return _get("injuries", {"fixture": fixture_id})


def lesoes_time(team_id: int, season: int = DEFAULT_SEASON) -> list[dict]:
    """
    Retorna todas as lesões de um time na temporada.
    Útil para saber quem está fora antes do jogo ter fixture_id.
    """
    return _get("injuries", {"team": team_id, "season": season})


# ══════════════════════════════════════════════
#  PREVISÕES DA API-FOOTBALL
# ══════════════════════════════════════════════

def previsao_api(fixture_id: int) -> dict:
    """
    Retorna a previsão da própria API-Football para uma partida.
    Inclui: comparison (strength, form, h2h), predictions (winner, goals),
    teams (form, last_5), h2h, etc.
    Custo: 1 request.
    """
    result = _get("predictions", {"fixture": fixture_id})
    if result and isinstance(result, list) and len(result) > 0:
        return result[0]
    return {}

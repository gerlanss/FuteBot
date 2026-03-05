"""
FuteBot — App Web de estatísticas de futebol com API-Football.

Rotas:
    /                  → Dashboard com info da conta
    /buscar            → Busca de times por nome
    /time/<id>         → Stats detalhadas do time (V/E/D, gols, cartões, jogos)
    /partida/<id>      → Stats da partida (chutes, posse, xG, eventos, escalações)
    /classificacao     → Tabela de classificação de ligas
    /explorador        → Teste qualquer endpoint da API
"""

import json
from urllib.parse import parse_qs

from flask import Flask, render_template, request, flash

import config
from services import apifootball as api

# ──────────────────────────────────────────────
# Inicialização do Flask
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "futebot-dev-key-mude-em-producao"


# ══════════════════════════════════════════════
#  CONTEXT PROCESSOR — disponibiliza dados em todos os templates
# ══════════════════════════════════════════════

@app.context_processor
def inject_globals():
    """Injeta variáveis globais nos templates (req_info, leagues, config)."""
    return {
        "req_info": "",  # Preenchido sob demanda nas rotas que chamam status
        "leagues": config.LEAGUES,
        "config": config,
    }


# ══════════════════════════════════════════════
#  ROTAS
# ══════════════════════════════════════════════

@app.route("/")
def index():
    """Página inicial — dashboard com info da conta API-Football."""
    conta = api.status_conta()
    return render_template("index.html", conta=conta)


@app.route("/buscar")
def search_team():
    """Busca de times por nome."""
    query = request.args.get("q", "").strip()
    times = []
    if query:
        times = api.buscar_time(query)
    return render_template("search_team.html", query=query, times=times)


@app.route("/time/<int:team_id>")
def team_detail(team_id: int):
    """
    Detalhes e estatísticas de um time.
    Query params opcionais: league (ID), season (ano).
    Faz 3 chamadas: detalhes + stats do time + lista de jogos.
    """
    selected_league = request.args.get("league", config.LEAGUES["brasileirao_a"]["id"], type=int)
    selected_season = request.args.get("season", config.DEFAULT_SEASON, type=int)

    # Buscar info básica do time
    team_info = api.detalhes_time(team_id)
    if not team_info:
        flash("Time não encontrado.", "warning")
        return render_template("search_team.html", query="", times=[])

    # Stats do time na liga/temporada
    stats = api.stats_time(team_id, selected_league, selected_season)

    # Lista de jogos
    jogos = api.jogos_time(team_id, selected_league, selected_season)

    return render_template(
        "team_detail.html",
        team_info=team_info,
        stats=stats,
        jogos=jogos,
        selected_league=selected_league,
        selected_season=selected_season,
    )


@app.route("/partida/<int:fixture_id>")
def match_stats(fixture_id: int):
    """
    Estatísticas detalhadas de uma partida.
    Faz 4 chamadas: fixture + stats + eventos + escalações.
    """
    fixture = api.detalhe_jogo(fixture_id)
    stats = api.stats_partida(fixture_id)
    eventos = api.eventos_partida(fixture_id)
    escalacoes = api.escalacao_partida(fixture_id)

    return render_template(
        "match_stats.html",
        fixture=fixture,
        stats=stats,
        eventos=eventos,
        escalacoes=escalacoes,
    )


@app.route("/classificacao")
def standings_page():
    """Tabela de classificação de uma liga/temporada."""
    selected_league = request.args.get("league", config.LEAGUES["brasileirao_a"]["id"], type=int)
    selected_season = request.args.get("season", config.DEFAULT_SEASON, type=int)

    tabela = api.classificacao(selected_league, selected_season)

    return render_template(
        "standings.html",
        tabela=tabela,
        selected_league=selected_league,
        selected_season=selected_season,
    )


@app.route("/explorador", methods=["GET", "POST"])
def explorer():
    """Explorador genérico — testa qualquer endpoint da API-Football."""
    endpoint = ""
    params_str = ""
    resultado = None
    url_chamada = ""
    status_ok = True

    # Atalhos rápidos
    quick = request.args.get("quick", "")
    if quick == "status":
        endpoint = "status"
    elif quick == "teams":
        endpoint = "teams"
        params_str = "search=Corinthians"
    elif quick == "standings":
        endpoint = "standings"
        params_str = f"league=71&season={config.DEFAULT_SEASON}"
    elif quick == "fixtures":
        endpoint = "fixtures"
        params_str = f"team=131&league=71&season={config.DEFAULT_SEASON}"
    elif quick == "topscorers":
        endpoint = "players/topscorers"
        params_str = f"league=71&season={config.DEFAULT_SEASON}"

    # POST — executar
    if request.method == "POST":
        endpoint = request.form.get("endpoint", "").strip()
        params_str = request.form.get("params", "").strip()

        if not endpoint:
            flash("Informe o endpoint.", "warning")
        else:
            params = {}
            if params_str:
                parsed = parse_qs(params_str, keep_blank_values=True)
                params = {k: v[0] for k, v in parsed.items()}

            raw = api.raw_request(endpoint, params)
            url_chamada = f"/{endpoint}?{params_str}" if params_str else f"/{endpoint}"

            if raw and not raw.get("errors"):
                resultado = json.dumps(raw, indent=2, ensure_ascii=False)
                status_ok = True
            elif raw:
                resultado = json.dumps(raw, indent=2, ensure_ascii=False)
                status_ok = bool(raw.get("results", 0))
            else:
                resultado = "Erro na requisição."
                status_ok = False

    return render_template(
        "explorer.html",
        endpoint=endpoint,
        params_str=params_str,
        resultado=resultado,
        url_chamada=url_chamada,
        status_ok=status_ok,
    )


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  FuteBot — API-Football Explorer")
    print(f"  http://localhost:{config.FLASK_PORT}")
    print(f"  API Key: {config.API_FOOTBALL_KEY[:8]}****")
    print(f"  Temporada padrão: {config.DEFAULT_SEASON}")
    print(f"{'='*50}\n")

    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )

"""
Script de bulk download de dados históricos.

Baixa fixtures + stats de partida de todas as ligas/seasons configuradas
e salva no SQLite para treinamento do modelo.

Estratégia de consumo de API (Pro: 7.500 req/dia):
  - Fixtures por liga/season: 1 req (retorna ~380 jogos de uma vez)
  - Stats por partida: 1 req cada
  - Total estimado: ~32 req (fixtures) + ~8.390 req (stats) = ~8.422 req
  - Execução: pode precisar de 2 dias se > 7.500

Uso:
  python -m data.bulk_download           # Baixa tudo
  python -m data.bulk_download --stats   # Só stats (fixtures já baixados)
  python -m data.bulk_download --resume  # Continua de onde parou

Segurança:
  - Pausa automática se atingir 7.000 req/dia (margem de segurança)
  - Salva progresso a cada liga/season completada
  - Pode ser interrompido e retomado com --resume
"""

import sys
import time
import argparse
from datetime import datetime

from config import LEAGUES, TRAIN_SEASONS
from services.apifootball import (
    raw_request,
    status_conta,
    stats_partida,
    eventos_partida,
    escalacao_partida,
    jogadores_partida,
    lesoes_fixture,
    stats_time,
)
from data.database import Database

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Limite de segurança: para em 7.000 req para não estourar 7.500
LIMITE_SEGURANCA = 7000
# Delay entre requests para não sobrecarregar a API (milissegundos)
DELAY_MS = 200


def _check_limite() -> tuple[int, bool]:
    """Verifica requests usadas hoje. Retorna (usadas, pode_continuar)."""
    r = raw_request("status")
    req = r.get("response", {}).get("requests", {})
    usadas = req.get("current", 0)
    limite = req.get("limit_day", 7500)
    pode = usadas < LIMITE_SEGURANCA
    return usadas, pode


def _delay():
    """Pausa respeitosa entre requests."""
    time.sleep(DELAY_MS / 1000)


def baixar_fixtures(db: Database):
    """
    Etapa 1: Baixa todos os fixtures (agenda + resultados) de cada liga/season.
    Custo: ~32 requests (1 por liga/season).
    """
    print("=" * 60)
    print("ETAPA 1: BAIXANDO FIXTURES")
    print("=" * 60)

    total_salvos = 0
    for key, liga in LEAGUES.items():
        league_id = liga["id"]
        nome = liga["nome"]

        for season in TRAIN_SEASONS:
            # Verificar se já temos fixtures desta liga/season
            existentes = db.fixtures_por_liga(league_id, season)
            ft_existentes = sum(1 for f in existentes if f["status"] == "FT")

            print(f"\n[{nome}] Season {season}: ", end="")

            if ft_existentes > 100:
                print(f"já temos {ft_existentes} FT — pulando")
                continue

            r = raw_request("fixtures", {"league": league_id, "season": season})
            fixtures = r.get("response", [])

            if not fixtures:
                errs = r.get("errors", {})
                print(f"vazio (errors: {errs})")
                continue

            n = db.salvar_fixtures_batch(fixtures)
            ft = sum(1 for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "FT")
            total_salvos += n
            print(f"{n} fixtures ({ft} FT)")
            _delay()

    print(f"\n✅ Total fixtures salvos: {total_salvos}")
    return total_salvos


def baixar_stats(db: Database, resume: bool = False, max_fixtures: int = 0):
    """
    Etapa 2: Baixa stats de cada partida finalizada.
    Filtra apenas seasons em TRAIN_SEASONS para não desperdiçar requests.
    Custo: 1 req por fixture.
    Para automaticamente se atingir limite de segurança.

    Args:
        max_fixtures: Se > 0, limita a quantidade de fixtures processadas.
                      Útil no bootstrap para não travar a inicialização.
    """
    print("\n" + "=" * 60)
    print("ETAPA 2: BAIXANDO STATS DE PARTIDAS")
    print("=" * 60)

    # Filtrar apenas seasons relevantes
    placeholders = ",".join("?" for _ in TRAIN_SEASONS)
    print(f"Seasons: {TRAIN_SEASONS}")

    # Buscar fixtures finalizados das seasons relevantes que ainda não têm stats
    conn = db._conn()
    if resume:
        rows = conn.execute(f"""
            SELECT f.fixture_id, f.home_name, f.away_name, f.league_name, f.season
            FROM fixtures f
            LEFT JOIN fixture_stats fs ON f.fixture_id = fs.fixture_id
            WHERE f.status = 'FT' AND fs.fixture_id IS NULL
            AND f.season IN ({placeholders})
            ORDER BY f.date
        """, TRAIN_SEASONS).fetchall()
    else:
        rows = conn.execute(f"""
            SELECT f.fixture_id, f.home_name, f.away_name, f.league_name, f.season
            FROM fixtures f
            WHERE f.status = 'FT'
            AND f.season IN ({placeholders})
            ORDER BY f.date
        """, TRAIN_SEASONS).fetchall()
    conn.close()

    total = len(rows)
    # Limitar se max_fixtures definido (usado no bootstrap para boot rápido)
    if max_fixtures > 0 and total > max_fixtures:
        rows = rows[:max_fixtures]
        total_real = total
        total = max_fixtures
        print(f"Fixtures para baixar stats: {total} (de {total_real} pendentes, limitado a {max_fixtures})")
    else:
        print(f"Fixtures para baixar stats: {total}")

    if total == 0:
        print("Nenhum fixture pendente!")
        return 0

    baixados = 0
    erros = 0
    ultimo_check = 0

    for i, row in enumerate(rows):
        fix_id = row["fixture_id"]

        # Verificar limite a cada 500 requests
        if (i - ultimo_check) >= 500:
            usadas, pode = _check_limite()
            print(f"\n📊 Progresso: {i}/{total} | Requests hoje: {usadas} | ", end="")
            if not pode:
                print(f"⚠️ LIMITE ATINGIDO ({usadas}/{LIMITE_SEGURANCA}). Retome amanhã com --resume")
                break
            print("OK")
            ultimo_check = i

        # Baixar stats
        stats = stats_partida(fix_id)
        if stats:
            db.salvar_fixture_stats(fix_id, stats)
            baixados += 1
        else:
            erros += 1

        # Log a cada 100
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] {row['league_name']} {row['season']}: "
                  f"{row['home_name']} vs {row['away_name']} "
                  f"({'✅' if stats else '❌'})")

        _delay()

    print(f"\n✅ Stats baixadas: {baixados} | Erros: {erros} | Pendentes: {total - baixados - erros}")
    return baixados


def baixar_eventos(db: Database, resume: bool = False):
    """
    Etapa 3 (opcional): Baixa eventos (gols, cartões) de partidas.
    Filtra apenas seasons em TRAIN_SEASONS para eficiência.
    Menos prioritário que stats, mas útil para features avançadas.
    """
    print("\n" + "=" * 60)
    print("ETAPA 3: BAIXANDO EVENTOS DE PARTIDAS")
    print("=" * 60)

    placeholders = ",".join("?" for _ in TRAIN_SEASONS)

    conn = db._conn()
    if resume:
        rows = conn.execute(f"""
            SELECT f.fixture_id, f.home_name, f.away_name
            FROM fixtures f
            LEFT JOIN fixture_events fe ON f.fixture_id = fe.fixture_id
            WHERE f.status = 'FT' AND fe.fixture_id IS NULL
            AND f.season IN ({placeholders})
            ORDER BY f.date
        """, TRAIN_SEASONS).fetchall()
    else:
        rows = conn.execute(f"""
            SELECT fixture_id, home_name, away_name
            FROM fixtures WHERE status = 'FT'
            AND season IN ({placeholders})
            ORDER BY date
        """, TRAIN_SEASONS).fetchall()
    conn.close()

    total = len(rows)
    print(f"Fixtures para baixar eventos: {total}")

    baixados = 0
    for i, row in enumerate(rows):
        # Verificar limite a cada 500
        if i > 0 and i % 500 == 0:
            usadas, pode = _check_limite()
            if not pode:
                print(f"\n⚠️ LIMITE ATINGIDO. Retome com --resume")
                break

        events = eventos_partida(row["fixture_id"])
        if events:
            db.salvar_eventos(row["fixture_id"], events)
            baixados += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] eventos baixados: {baixados}")

        _delay()

    print(f"\n✅ Eventos baixados: {baixados}")
    return baixados


def baixar_team_stats_contexto(db: Database, max_items: int = 0):
    """
    Baixa team_stats agregadas por team x league x season.
    Alto valor para pre-live com custo bem menor que lineups historicos.
    """
    print("\n" + "=" * 60)
    print("ETAPA 4: BAIXANDO TEAM STATS AGREGADAS")
    print("=" * 60)

    conn = db._conn()
    rows = conn.execute("""
        SELECT x.team_id, x.team_name, x.league_id, x.season
        FROM (
            SELECT DISTINCT home_id AS team_id, home_name AS team_name, league_id, season
            FROM fixtures
            WHERE season IN ({})
            UNION
            SELECT DISTINCT away_id AS team_id, away_name AS team_name, league_id, season
            FROM fixtures
            WHERE season IN ({})
        ) x
        LEFT JOIN team_stats ts
          ON ts.team_id = x.team_id
         AND ts.league_id = x.league_id
         AND ts.season = x.season
        WHERE ts.team_id IS NULL
        ORDER BY x.season, x.league_id, x.team_id
    """.format(",".join("?" for _ in TRAIN_SEASONS), ",".join("?" for _ in TRAIN_SEASONS)), TRAIN_SEASONS + TRAIN_SEASONS).fetchall()
    conn.close()

    if max_items > 0:
        rows = rows[:max_items]

    total = len(rows)
    print(f"Combinações team/league/season para baixar: {total}")
    salvos = 0
    erros = 0

    for i, row in enumerate(rows):
        if i > 0 and i % 100 == 0:
            usadas, pode = _check_limite()
            print(f"  [checkpoint {i}/{total}] requests hoje: {usadas}")
            if not pode:
                print("⚠️ Limite de segurança atingido durante team_stats. Encerrando fase.")
                break

        payload = stats_time(int(row["team_id"]), int(row["league_id"]), int(row["season"]))
        if payload:
            db.salvar_team_stats(
                int(row["team_id"]),
                int(row["league_id"]),
                int(row["season"]),
                str(row["team_name"] or ""),
                payload,
            )
            salvos += 1
        else:
            erros += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] team_stats salvas: {salvos} | erros: {erros}")
        _delay()

    print(f"\n✅ Team stats salvas: {salvos} | Erros: {erros}")
    return salvos


def baixar_contexto_operacional_fixtures(db: Database, *, lookahead_days: int = 3, include_players: bool = False):
    """
    Baixa contexto barato para fixtures recentes/proximos:
      - lineups
      - injuries
      - opcionalmente player stats da partida

    Nao tenta baixar lineup historico de todo o universo porque isso seria
    uma ideia de bosta em custo/beneficio.
    """
    print("\n" + "=" * 60)
    print("ETAPA 5: BAIXANDO CONTEXTO OPERACIONAL DE FIXTURES")
    print("=" * 60)

    conn = db._conn()
    rows = conn.execute(
        """
        SELECT fixture_id, home_name, away_name, date, status
        FROM fixtures
        WHERE datetime(date) >= datetime('now', '-1 day')
          AND datetime(date) <= datetime('now', ?)
        ORDER BY date
        """,
        (f"+{int(lookahead_days)} day",),
    ).fetchall()
    conn.close()

    total = len(rows)
    print(f"Fixtures no recorte operacional: {total}")
    lineups_salvas = 0
    injuries_salvas = 0
    players_salvos = 0

    conn = db._conn()
    fixtures_com_lineups = {int(r["fixture_id"]) for r in conn.execute("SELECT DISTINCT fixture_id FROM fixture_lineups").fetchall()}
    fixtures_com_injuries = {int(r["fixture_id"]) for r in conn.execute("SELECT DISTINCT fixture_id FROM fixture_injuries").fetchall()}
    fixtures_com_players = {int(r["fixture_id"]) for r in conn.execute("SELECT DISTINCT fixture_id FROM fixture_player_stats").fetchall()}
    conn.close()

    for i, row in enumerate(rows):
        if i > 0 and i % 25 == 0:
            usadas, pode = _check_limite()
            print(f"  [checkpoint {i}/{total}] requests hoje: {usadas}")
            if not pode:
                print("⚠️ Limite de segurança atingido durante contexto operacional. Encerrando fase.")
                break

        fixture_id = int(row["fixture_id"])

        if fixture_id not in fixtures_com_lineups:
            lineups = escalacao_partida(fixture_id)
            if lineups:
                db.salvar_lineups(fixture_id, lineups)
                lineups_salvas += len(lineups)
            _delay()

        if fixture_id not in fixtures_com_injuries:
            injuries = lesoes_fixture(fixture_id)
            db.salvar_injuries(fixture_id, injuries or [])
            injuries_salvas += len(injuries or [])
            _delay()

        if include_players and fixture_id not in fixtures_com_players:
            players = jogadores_partida(fixture_id)
            if players:
                db.salvar_fixture_player_stats(fixture_id, players)
                players_salvos += sum(len(team.get("players") or []) for team in players)
            _delay()

        if (i + 1) % 50 == 0:
            print(
                f"  [{i+1}/{total}] fixture contextos | lineups={lineups_salvas} "
                f"| injuries={injuries_salvas} | players={players_salvos}"
            )

    print(
        f"\n✅ Contexto operacional salvo: lineups={lineups_salvas} "
        f"| injuries={injuries_salvas} | players={players_salvos}"
    )
    return {
        "lineups_salvas": lineups_salvas,
        "injuries_salvas": injuries_salvas,
        "players_salvos": players_salvos,
    }


def baixar_contexto_treino(db: Database, *, lookahead_days: int = 3, include_players: bool = False):
    """
    Fase completa de contexto de treino:
      - team stats agregadas (baratas e fortes para pre-live)
      - lineups/injuries operacionais (baratos e úteis para o jogo do dia)
    """
    team_stats_salvas = baixar_team_stats_contexto(db)
    contexto = baixar_contexto_operacional_fixtures(
        db,
        lookahead_days=lookahead_days,
        include_players=include_players,
    )
    return {
        "team_stats_salvas": team_stats_salvas,
        **contexto,
    }


def baixar_eventos_live_treino(db: Database, max_items: int = 0):
    """
    Baixa eventos só dos fixtures que realmente entraram no fluxo live.
    Muito mais inteligente do que tentar baixar 19 mil fixtures na tora.
    """
    print("\n" + "=" * 60)
    print("ETAPA LIVE: BAIXANDO EVENTOS DOS FIXTURES DE TREINO")
    print("=" * 60)

    conn = db._conn()
    rows = conn.execute(
        """
        SELECT DISTINCT lw.fixture_id, lw.home_name, lw.away_name
        FROM live_watchlist lw
        LEFT JOIN fixture_events fe ON fe.fixture_id = lw.fixture_id
        WHERE lw.watch_type IN ('approved_prelive', 'live_opportunity')
          AND fe.fixture_id IS NULL
        ORDER BY lw.fixture_id
        """
    ).fetchall()
    conn.close()

    if max_items > 0:
        rows = rows[:max_items]

    total = len(rows)
    print(f"Fixtures live sem eventos: {total}")
    baixados = 0
    erros = 0

    for i, row in enumerate(rows):
        if i > 0 and i % 25 == 0:
            usadas, pode = _check_limite()
            print(f"  [checkpoint {i}/{total}] requests hoje: {usadas}")
            if not pode:
                print("⚠️ Limite de segurança atingido durante eventos live. Encerrando fase.")
                break

        events = eventos_partida(int(row["fixture_id"]))
        if events:
            db.salvar_eventos(int(row["fixture_id"]), events)
            baixados += len(events)
        else:
            erros += 1
        _delay()

    print(f"\n✅ Eventos live salvos: {baixados} | Fixtures sem retorno: {erros}")
    return baixados


def baixar_eventos_historicos_live(
    db: Database,
    max_items: int = 0,
    league_ids: list[int] | None = None,
    seasons: list[int] | None = None,
    reserve: int = 150,
):
    """
    Baixa eventos históricos para o treino live real.

    Fonte:
      - fixtures FT das seasons de treino
      - ligas configuradas
      - prioridade por fixtures mais recentes

    Isso aqui é o que interessa para modelo live de verdade.
    Watchlist do bot não manda porra nenhuma como fonte primária de treino.
    """
    print("\n" + "=" * 60)
    print("ETAPA LIVE HISTÓRICA: BAIXANDO EVENTOS PARA TREINO REAL")
    print("=" * 60)

    seasons = list(seasons or TRAIN_SEASONS)
    if not league_ids:
        league_ids = [info["id"] for info in LEAGUES.values()]

    season_placeholders = ",".join("?" for _ in seasons)
    league_placeholders = ",".join("?" for _ in league_ids)

    conn = db._conn()
    rows = conn.execute(
        f"""
        SELECT f.fixture_id, f.home_name, f.away_name, f.league_id, f.season, f.date
        FROM fixtures f
        LEFT JOIN fixture_events fe ON fe.fixture_id = f.fixture_id
        WHERE f.status = 'FT'
          AND fe.fixture_id IS NULL
          AND f.season IN ({season_placeholders})
          AND f.league_id IN ({league_placeholders})
        GROUP BY f.fixture_id
        ORDER BY f.season DESC, f.date DESC
        """,
        tuple(seasons) + tuple(league_ids),
    ).fetchall()
    conn.close()

    if max_items > 0:
        rows = rows[:max_items]

    total = len(rows)
    print(f"Fixtures históricas sem eventos: {total}")
    fixtures_salvos = 0
    eventos_salvos = 0
    erros = 0

    for i, row in enumerate(rows):
        if i > 0 and i % 25 == 0:
            status = status_conta() or {}
            req = status.get("requests", {})
            usadas = int(req.get("current") or 0)
            limite = int(req.get("limit_day") or 7500)
            pode = usadas < max(limite - reserve, 0)
            print(f"  [checkpoint {i}/{total}] requests hoje: {usadas}/{limite} | reserva: {reserve}")
            if not pode:
                print("⚠️ Orçamento real da API atingido durante eventos live históricos. Encerrando fase.")
                break

        events = eventos_partida(int(row["fixture_id"]))
        if events:
            db.salvar_eventos(int(row["fixture_id"]), events)
            fixtures_salvos += 1
            eventos_salvos += len(events)
        else:
            erros += 1
        _delay()

    print(
        f"\n✅ Eventos históricos live salvos: fixtures={fixtures_salvos} "
        f"| eventos={eventos_salvos} | fixtures sem retorno={erros}"
    )
    return {
        "fixtures_salvos": fixtures_salvos,
        "eventos_salvos": eventos_salvos,
        "fixtures_sem_retorno": erros,
    }


def main():
    parser = argparse.ArgumentParser(description="Bulk download de dados históricos")
    parser.add_argument("--stats", action="store_true", help="Só baixar stats (fixtures já baixados)")
    parser.add_argument("--eventos", action="store_true", help="Só baixar eventos")
    parser.add_argument("--eventos-live", action="store_true", help="Baixar eventos apenas dos fixtures usados no treino live")
    parser.add_argument("--eventos-historicos-live", action="store_true", help="Baixar eventos históricos para o treino live real")
    parser.add_argument("--team-stats", action="store_true", help="Só baixar team stats agregadas")
    parser.add_argument("--contexto", action="store_true", help="Baixar lineups/injuries para fixtures operacionais")
    parser.add_argument("--contexto-operacional-only", action="store_true", help="Baixar só lineups/injuries para fixtures operacionais")
    parser.add_argument("--players", action="store_true", help="Incluir player stats no modo --contexto")
    parser.add_argument("--lookahead-days", type=int, default=3, help="Janela de dias para o modo --contexto")
    parser.add_argument("--max-items", type=int, default=0, help="Limita itens processados em fases de backfill")
    parser.add_argument("--resume", action="store_true", help="Continuar de onde parou")
    args = parser.parse_args()

    print("FuteBot - Bulk Download")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Verificar estado da conta
    usadas, pode = _check_limite()
    print(f"Requests hoje: {usadas}/7500 (limite seguranca: {LIMITE_SEGURANCA})")

    if not pode:
        print("Limite de seguranca ja atingido. Tente novamente amanha.")
        return

    db = Database()

    if args.stats:
        baixar_stats(db, resume=args.resume)
    elif args.eventos:
        baixar_eventos(db, resume=args.resume)
    elif args.eventos_live:
        baixar_eventos_live_treino(db, max_items=args.max_items)
    elif args.eventos_historicos_live:
        baixar_eventos_historicos_live(db, max_items=args.max_items)
    elif args.team_stats:
        baixar_team_stats_contexto(db, max_items=args.max_items)
    elif args.contexto_operacional_only:
        baixar_contexto_operacional_fixtures(
            db,
            lookahead_days=args.lookahead_days,
            include_players=args.players,
        )
    elif args.contexto:
        baixar_contexto_treino(
            db,
            lookahead_days=args.lookahead_days,
            include_players=args.players,
        )
    else:
        # Pipeline completo
        baixar_fixtures(db)
        baixar_stats(db, resume=args.resume)

    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DO BANCO")
    print("=" * 60)
    resumo = db.resumo()
    for k, v in resumo.items():
        print(f"  {k}: {v:,}")

    usadas, _ = _check_limite()
    print(f"\nRequests usadas hoje: {usadas}/7500")
    print("Download concluido!")


if __name__ == "__main__":
    main()

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

import time
import argparse
from datetime import datetime

from config import LEAGUES, TRAIN_SEASONS
from services.apifootball import raw_request, stats_partida, eventos_partida
from data.database import Database

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


def main():
    parser = argparse.ArgumentParser(description="Bulk download de dados históricos")
    parser.add_argument("--stats", action="store_true", help="Só baixar stats (fixtures já baixados)")
    parser.add_argument("--eventos", action="store_true", help="Só baixar eventos")
    parser.add_argument("--resume", action="store_true", help="Continuar de onde parou")
    args = parser.parse_args()

    print("🤖 FuteBot — Bulk Download")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Verificar estado da conta
    usadas, pode = _check_limite()
    print(f"📊 Requests hoje: {usadas}/7500 (limite segurança: {LIMITE_SEGURANCA})")

    if not pode:
        print("⚠️ Limite de segurança já atingido. Tente novamente amanhã.")
        return

    db = Database()

    if args.stats:
        baixar_stats(db, resume=args.resume)
    elif args.eventos:
        baixar_eventos(db, resume=args.resume)
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
    print(f"\n📊 Requests usadas hoje: {usadas}/7500")
    print("🏁 Download concluído!")


if __name__ == "__main__":
    main()

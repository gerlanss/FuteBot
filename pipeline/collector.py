"""
Collector — coleta resultados de jogos e alimenta o ciclo de aprendizado.

Executado diariamente (antes do scanner) para:
  1. Atualizar fixtures de ontem com resultados finais
  2. Baixar stats de partidas finalizadas (para enriquecer o dataset)
  3. Resolver previsões pendentes (preencher acertou/lucro)
  4. Gerar relatório de resultados

Uso:
  from pipeline.collector import Collector
  collector = Collector()
  resumo = collector.executar()
"""

from datetime import datetime, timedelta
from data.database import Database
from services.apifootball import raw_request, stats_partida
from models.learner import Learner
from config import LEAGUES


class Collector:
    """Coleta resultados de jogos e fecha o loop de aprendizado."""

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.learner = Learner(self.db)

    def executar(self, data: str = None) -> dict:
        """
        Executa coleta completa de resultados.

        Parâmetros:
          data: data de referência YYYY-MM-DD (default: ontem)

        Retorna resumo da coleta.
        """
        if data is None:
            data = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"📥 Collector iniciando para {data}")

        # ─── 1. Atualizar fixtures do dia ───
        print("\n1️⃣ Atualizando fixtures...")
        fixtures_atualizados = self._atualizar_fixtures(data)
        print(f"   Atualizados: {fixtures_atualizados}")

        # ─── 2. Baixar stats das partidas finalizadas ───
        print("\n2️⃣ Baixando stats de partidas...")
        stats_baixadas = self._baixar_stats_dia(data)
        print(f"   Stats baixadas: {stats_baixadas}")

        # ─── 3. Resolver previsões pendentes ───
        print("\n3️⃣ Resolvendo previsões...")
        resolucao = self.learner.resolver_pendentes()
        print(f"   Resolvidas: {resolucao['resolvidos']} | "
              f"Acertos: {resolucao.get('acertos', 0)} | "
              f"Erros: {resolucao.get('erros', 0)}")

        # ─── 4. Relatório ───
        relatorio = self.learner.relatorio_resultado_dia(data)

        return {
            "data": data,
            "fixtures_atualizados": fixtures_atualizados,
            "stats_baixadas": stats_baixadas,
            "resolucao": resolucao,
            "relatorio": relatorio,
        }

    def _atualizar_fixtures(self, data: str) -> int:
        """Busca resultados de fixtures de uma data específica."""
        league_ids = {l["id"] for l in LEAGUES.values()}
        r = raw_request("fixtures", {"date": data})
        fixtures = r.get("response", [])

        # Filtrar nossas ligas e finalizados
        nossas = [
            f for f in fixtures
            if f.get("league", {}).get("id") in league_ids
            and f.get("fixture", {}).get("status", {}).get("short") == "FT"
        ]

        for f in nossas:
            self.db.salvar_fixture(f)

        return len(nossas)

    def _baixar_stats_dia(self, data: str) -> int:
        """Baixa stats de partidas finalizadas que ainda não têm stats no banco."""
        conn = self.db._conn()
        rows = conn.execute("""
            SELECT f.fixture_id
            FROM fixtures f
            LEFT JOIN fixture_stats fs ON f.fixture_id = fs.fixture_id
            WHERE f.date LIKE ? AND f.status = 'FT' AND fs.fixture_id IS NULL
        """, (f"{data}%",)).fetchall()
        conn.close()

        baixadas = 0
        for row in rows:
            stats = stats_partida(row["fixture_id"])
            if stats:
                self.db.salvar_fixture_stats(row["fixture_id"], stats)
                baixadas += 1

        return baixadas

    def formatar_relatorio(self, resultado: dict) -> str:
        """Formata resultado do collector para Telegram."""
        return resultado.get("relatorio", "Nenhum resultado disponível.")


if __name__ == "__main__":
    collector = Collector()
    resultado = collector.executar()
    print("\n" + collector.formatar_relatorio(resultado))

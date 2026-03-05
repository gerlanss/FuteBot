"""Executa o AutoTuner completo e mostra resumo."""
import sys, os
sys.path.insert(0, ".")
os.environ.setdefault("API_FOOTBALL_KEY", "7d40d96b3852438ee6fd1d4896bc54b9")

from models.autotuner import AutoTuner

tuner = AutoTuner()
resultado = tuner.executar()

print(f"\n{'='*60}")
print(f"RESUMO FINAL")
print(f"{'='*60}")

strategies = resultado.get("strategies", [])
ativas = [s for s in strategies if s.get("ativo")]

print(f"Strategies geradas: {len(strategies)}")
print(f"Strategies ativas: {len(ativas)}")

mercados_set = set(s["mercado"] for s in ativas)
print(f"\nMercados com strategy ativa: {len(mercados_set)}")
for m in sorted(mercados_set):
    n = sum(1 for s in ativas if s["mercado"] == m)
    print(f"  {m}: {n} slices")

ligas_set = set(s["league_id"] for s in ativas)
print(f"\nLigas com strategy ativa: {len(ligas_set)}")
for lid in sorted(ligas_set):
    n = sum(1 for s in ativas if s["league_id"] == lid)
    mercs = set(s["mercado"] for s in ativas if s["league_id"] == lid)
    print(f"  Liga {lid}: {n} strategies ({len(mercs)} mercados)")

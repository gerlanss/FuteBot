"""Re-run AutoTuner apenas nas 9 ligas que tiveram modelos falhados pelo bug scale_pos_weight."""
import sys
import os
import gc

sys.path.insert(0, "/opt/futebot")
os.chdir("/opt/futebot")

from data.database import Database
from models.autotuner import AutoTuner
from models.features import FeatureExtractor

db = Database()
tuner = AutoTuner(db)

# Ligas que tiveram modelos falhados pelo bug scale_pos_weight
LIGAS_RERUN = [253, 188, 98, 39, 265, 307, 78, 61, 2]
NOMES = {
    253: "MLS",
    188: "A-League",
    98: "J1 League",
    39: "Premier League",
    265: "Primera Div Chile",
    307: "Saudi Pro",
    78: "Bundesliga",
    61: "Ligue 1",
    2: "Champions League",
}

feat_names = FeatureExtractor.feature_names()

print("=" * 60)
print("Re-run: 9 ligas com modelos falhados (fix scale_pos_weight)")
print("=" * 60)

for i, lid in enumerate(LIGAS_RERUN, 1):
    nome = NOMES.get(lid, f"Liga {lid}")
    print(f"\n[{i}/9] Liga {lid}: {nome}")

    resultado = tuner._processar_liga(
        league_id=lid,
        nome_liga=nome,
        train_seasons=[2024],
        test_season=2025,
        trials_per_model=25,
        feat_names=feat_names,
    )

    if "erro" in resultado:
        print(f"  ERRO: {resultado['erro']}")
    else:
        n_mod = resultado.get("modelos_salvos", 0)
        n_strat = len([s for s in resultado.get("strategies", []) if s.get("ativo")])
        print(f"  OK: {n_mod} modelos, {n_strat} strategies ativas")

    gc.collect()

print("\n" + "=" * 60)
print("Re-run concluido!")
print("=" * 60)

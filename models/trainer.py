"""
Treinamento do modelo XGBoost para previsão de resultados.

Treina 8 modelos independentes:
  1. resultado_1x2: classificação (home/draw/away) — FT
  2. over_under_15: binário (over/under 1.5 gols)
  3. over_under_25: binário (over/under 2.5 gols)
  4. over_under_35: binário (over/under 3.5 gols)
  5. btts: binário (ambos marcam sim/não)
  6. resultado_ht: classificação (home/draw/away) — 1º tempo
  7. htft: classificação (9 classes) — combinado HT/FT
  8. total_corners: regressão (prever total de escanteios — futuro)

Split temporal: treina com seasons anteriores, testa na última.
Salva modelos em data/models/ como arquivos .json do XGBoost.

Uso:
  from models.trainer import Trainer
  trainer = Trainer(db)
  metricas = trainer.treinar()

Retreino automático: chamado semanalmente pelo scheduler.
"""

import os
import json
import numpy as np
from datetime import datetime
from collections import Counter

from data.database import Database
from models.features import FeatureExtractor
from config import MODEL_ACC_MIN_1X2, MODEL_ACC_MIN_BINARY, MODEL_BRIER_MAX, LEAGUES

# Tentativa de importar XGBoost (será instalado via requirements.txt)
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, log_loss
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("[Trainer] ⚠️ XGBoost/sklearn não instalado. Instale com: pip install xgboost scikit-learn")

# Diretório para salvar modelos treinados
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")


class Trainer:
    """Treina e gerencia os modelos de previsão XGBoost."""

    def __init__(self, db: Database):
        self.db = db
        self.fe = FeatureExtractor(db)
        os.makedirs(MODELS_DIR, exist_ok=True)

    # ══════════════════════════════════════════════
    #  TREINO PER-LEAGUE (principal)
    # ══════════════════════════════════════════════

    # Mínimo de jogos FT por liga para treinar modelo próprio.
    # Ligas com menos dados simplesmente não treinam (sem fallback global).
    MIN_JOGOS_LIGA = 300

    def treinar_por_liga(self, train_seasons: list[int] = None,
                         test_season: int = None) -> dict:
        """
        Treina 1 modelo por liga (100% per-league, sem modelo global).

        Para cada liga em config.LEAGUES com >= MIN_JOGOS_LIGA fixtures FT:
          - Constrói dataset filtrado por league_id
          - Treina os 14 modelos e salva em data/models/league_{id}/

        Ligas com poucos dados não geram modelos nem tips.

        Retorna dict com métricas por liga.
        """
        train_seasons = train_seasons or [2020, 2021, 2022, 2023, 2024, 2025]
        test_season = test_season or 2026

        resultados = {}

        # 1. Identificar ligas elegíveis (com dados suficientes)
        ligas_elegiveis = []
        ligas_insuficientes = []

        for slug, info in LEAGUES.items():
            lid = info["id"]
            nome = info["nome"]
            fixtures = self.db.fixtures_finalizados(league_id=lid)
            n_ft = len(fixtures)
            if n_ft >= self.MIN_JOGOS_LIGA:
                ligas_elegiveis.append((lid, nome, n_ft))
            else:
                ligas_insuficientes.append((lid, nome, n_ft))

        print("=" * 60)
        print("🤖 TREINO PER-LEAGUE (sem modelo global)")
        print(f"   Seasons treino: {train_seasons}")
        print(f"   Season teste:   {test_season}")
        print(f"   Ligas elegíveis: {len(ligas_elegiveis)} (>= {self.MIN_JOGOS_LIGA} jogos)")
        print(f"   Ligas sem dados: {len(ligas_insuficientes)} (puladas)")
        print("=" * 60)

        # 2. Treinar modelo por liga (sem modelo global)
        for lid, nome, n_ft in sorted(ligas_elegiveis, key=lambda x: -x[2]):
            print(f"\n{'─' * 60}")
            print(f"🏟️  Liga {lid}: {nome} ({n_ft} jogos FT)")
            print(f"{'─' * 60}")
            try:
                resultado_liga = self.treinar(
                    train_seasons=train_seasons,
                    test_season=test_season,
                    league_id=lid
                )
                resultados[f"league_{lid}"] = resultado_liga
            except Exception as e:
                print(f"   ❌ Erro na liga {lid}: {e}")
                resultados[f"league_{lid}"] = {"erro": str(e)}

        # 3. Resumo final
        print("\n" + "=" * 60)
        print("📊 RESUMO PER-LEAGUE")
        print("=" * 60)
        for key, res in resultados.items():
            if "erro" in res:
                print(f"   {key}: ❌ {res['erro']}")
            else:
                aprov = res.get('modelos_aprovados', [])
                reprov = res.get('modelos_reprovados', [])
                print(f"   {key}: ✅ {len(aprov)} aprovados, ❌ {len(reprov)} reprovados")

        if ligas_insuficientes:
            print(f"\n   Ligas sem dados suficientes (< {self.MIN_JOGOS_LIGA} jogos):")
            for lid, nome, n_ft in ligas_insuficientes:
                print(f"   ↳ {lid} {nome}: {n_ft} jogos")

        return resultados

    # ══════════════════════════════════════════════
    #  TREINO DE UM CONJUNTO DE MODELOS (per-league)
    # ══════════════════════════════════════════════

    def treinar(self, train_seasons: list[int] = None,
                test_season: int = None,
                league_id: int = None) -> dict:
        """
        Treina os 14 modelos com split temporal para uma liga específica.

        Parâmetros:
          train_seasons: seasons para treino
          test_season: season para teste (default: 2026)
          league_id: ID da liga (obrigatório — não existe mais treino global)

        Modelos salvos em data/models/league_{id}/{nome}.json

        Retorna dict com métricas de cada modelo.
        """
        if not HAS_ML:
            return {"erro": "XGBoost/sklearn não instalado"}

        if not league_id:
            return {"erro": "league_id é obrigatório (treino global foi removido)"}

        train_seasons = train_seasons or [2020, 2021, 2022, 2023, 2024, 2025]
        test_season = test_season or 2026

        # Diretório de saída: sempre per-league
        models_dir = os.path.join(MODELS_DIR, f"league_{league_id}")
        label = f"Liga {league_id}"
        os.makedirs(models_dir, exist_ok=True)

        print(f"🤖 Treinando [{label}]...")
        print(f"   Treino: seasons {train_seasons}")
        print(f"   Teste: season {test_season}")
        print(f"   Saída: {models_dir}")

        # Construir dataset (filtrado por liga se league_id informado)
        all_features, all_labels = self.fe.build_dataset(
            league_id=league_id,
            seasons=train_seasons + [test_season]
        )

        if len(all_features) < 50:
            print(f"⚠️ Poucos dados ({len(all_features)} jogos). Mínimo recomendado: 50.")
            return {"erro": f"Dados insuficientes: {len(all_features)} jogos"}

        # Split temporal: treina com seasons anteriores, testa na última
        feat_names = FeatureExtractor.feature_names()
        X_train, X_test, y_train, y_test = [], [], [], []

        for f, l in zip(all_features, all_labels):
            row = [f.get(name, 0) or 0 for name in feat_names]
            if f.get("_season") in train_seasons:
                X_train.append(row)
                y_train.append(l)
            elif f.get("_season") == test_season:
                X_test.append(row)
                y_test.append(l)

        # Fallback: se test_season não tem dados FT, usar random split 80/20
        if len(X_test) == 0:
            print(f"   ⚠️ Season {test_season} sem jogos finalizados — usando random split 80/20")
            all_rows = X_train  # Tudo que temos
            all_labels_list = y_train
            from sklearn.model_selection import train_test_split as tts
            idx = list(range(len(all_rows)))
            idx_tr, idx_te = tts(idx, test_size=0.2, random_state=42)
            X_train = [all_rows[i] for i in idx_tr]
            X_test = [all_rows[i] for i in idx_te]
            y_train = [all_labels_list[i] for i in idx_tr]
            y_test = [all_labels_list[i] for i in idx_te]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)

        print(f"   Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

        metricas = {}

        # ─── Lista de modelos a treinar ───
        # Cada entrada: (nome, chave_label, objetivo, num_class)
        MODELOS = [
            ("resultado_1x2",  "resultado",    "multi:softprob",  3),
            ("over_under_15",  "over15",        "binary:logistic", None),
            ("over_under_25",  "over25",        "binary:logistic", None),
            ("over_under_35",  "over35",        "binary:logistic", None),
            ("btts",           "btts",          "binary:logistic", None),
            ("resultado_ht",   "resultado_ht",  "multi:softprob",  3),
            ("htft",           "htft",          "multi:softprob",  9),
            # ── Gols por tempo (1T e 2T) ──
            ("over_under_05_ht", "over05_ht",   "binary:logistic", None),
            ("over_under_15_ht", "over15_ht",   "binary:logistic", None),
            ("over_under_05_2t", "over05_2t",   "binary:logistic", None),
            ("over_under_15_2t", "over15_2t",   "binary:logistic", None),
            # ── Escanteios ──
            ("corners_over_85",  "corners_over_85",  "binary:logistic", None),
            ("corners_over_95",  "corners_over_95",  "binary:logistic", None),
            ("corners_over_105", "corners_over_105", "binary:logistic", None),
        ]

        for nome, chave_label, objetivo, num_class in MODELOS:
            print(f"\n📊 Treinando modelo: {nome}...")
            y_tr = np.array([l[chave_label] for l in y_train])
            y_te = np.array([l[chave_label] for l in y_test])

            m = self._treinar_modelo(
                X_train, y_tr, X_test, y_te,
                objetivo=objetivo,
                nome=nome,
                num_class=num_class,
                models_dir=models_dir,
            )
            metricas[nome] = m

        # ─── BASELINES (comparar modelo vs chutes) ───
        print("\n📏 Calculando baselines...")
        baselines = self._calcular_baselines(y_train, y_test)
        metricas["baselines"] = baselines

        # ─── BRIER SCORE (calibração) ───
        print("📐 Calculando Brier Score (calibração)...")
        brier = self._brier_scores(metricas)
        metricas["brier"] = brier

        # ─── GATE DE QUALIDADE (por modelo) ───
        # Cada modelo é avaliado e revertido individualmente.
        # Isso evita que a falha do 1x2 destrua modelos novos com boa accuracy.
        print("\n🚦 Verificando gate de qualidade (por modelo)...")
        aprovados, reprovados, motivos = self._gate_qualidade_por_modelo(
            metricas, baselines, MODELOS, models_dir=models_dir
        )
        metricas["aprovado"] = len(reprovados) == 0
        metricas["gate_motivos"] = motivos
        metricas["modelos_aprovados"] = aprovados
        metricas["modelos_reprovados"] = reprovados

        if reprovados:
            print(f"\n⚠️ {len(reprovados)} modelo(s) reprovado(s) (revertidos):")
            for nome_r in reprovados:
                print(f"   ↩️ {nome_r}")
        if aprovados:
            print(f"\n✅ {len(aprovados)} modelo(s) aprovado(s):")
            for nome_a in aprovados:
                acc = metricas.get(nome_a, {}).get('accuracy_test', 0)
                print(f"   ✅ {nome_a}: {acc:.1%}")
        for m in motivos:
            print(f"   📋 {m}")

        # Salvar metadados do treino
        # Limpar arrays numpy antes de serializar (não são JSON serializáveis)
        metricas_json = {}
        for k, v in metricas.items():
            if isinstance(v, dict):
                metricas_json[k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                    for kk, vv in v.items()
                    if not isinstance(vv, np.ndarray)
                }
            elif isinstance(v, (np.floating, np.integer)):
                metricas_json[k] = float(v)
            elif not isinstance(v, np.ndarray):
                metricas_json[k] = v

        versao = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.db.salvar_treino(
            versao=versao,
            n_samples=len(X_train),
            n_features=len(feat_names),
            acc_train=metricas["resultado_1x2"].get("accuracy_train", 0),
            acc_test=metricas["resultado_1x2"].get("accuracy_test", 0),
            metrics=metricas_json,
            params=self._default_params(),
        )

        # Salvar feature importance
        self._salvar_feature_importance(feat_names, models_dir=models_dir)

        print(f"\n✅ Treinamento [{label}] concluído! Versão: {versao}")
        for nome_m, _, _, _ in MODELOS:
            acc = metricas.get(nome_m, {}).get("accuracy_test", 0)
            print(f"   {nome_m}: {acc:.1%}")

        return metricas

    def _treinar_modelo(self, X_train, y_train, X_test, y_test,
                        objetivo: str, nome: str,
                        num_class: int = None,
                        models_dir: str = None) -> dict:
        """
        Treina um modelo XGBoost individual.
        Salva no models_dir informado (per-league).
        """
        models_dir = models_dir or MODELS_DIR
        params = self._default_params()
        params["objective"] = objetivo
        if num_class:
            params["num_class"] = num_class
            params["eval_metric"] = "mlogloss"   # Multiclass
        else:
            params["eval_metric"] = "logloss"    # Binário

        dtrain = xgb.DMatrix(X_train, label=y_train,
                             feature_names=FeatureExtractor.feature_names())
        dtest = xgb.DMatrix(X_test, label=y_test,
                            feature_names=FeatureExtractor.feature_names())

        # Treinar com early stopping
        evals = [(dtrain, "train"), (dtest, "test")]
        model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Previsões (probabilidades brutas para Brier Score)
        pred_train = model.predict(dtrain)
        pred_test = model.predict(dtest)

        if objetivo == "multi:softprob":
            pred_train_cls = np.argmax(pred_train, axis=1) if pred_train.ndim > 1 else pred_train
            pred_test_cls = np.argmax(pred_test, axis=1) if pred_test.ndim > 1 else pred_test
            acc_train = accuracy_score(y_train, pred_train_cls)
            acc_test = accuracy_score(y_test, pred_test_cls)
        else:
            pred_train_cls = (pred_train > 0.5).astype(int)
            pred_test_cls = (pred_test > 0.5).astype(int)
            acc_train = accuracy_score(y_train, pred_train_cls)
            acc_test = accuracy_score(y_test, pred_test_cls)

        # Salvar modelo temporariamente (pode ser revertido se reprovar no gate)
        model_path = os.path.join(models_dir, f"{nome}.json")
        backup_path = os.path.join(models_dir, f"{nome}_backup.json")

        # Backup do modelo anterior (se existir)
        if os.path.exists(model_path):
            import shutil
            shutil.copy2(model_path, backup_path)

        model.save_model(model_path)
        print(f"   📊 {nome}: train={acc_train:.3f} test={acc_test:.3f}")

        return {
            "accuracy_train": round(acc_train, 4),
            "accuracy_test": round(acc_test, 4),
            "best_iteration": model.best_iteration,
            "model_path": model_path,
            "pred_test_probs": pred_test,   # Probabilidades para Brier Score
            "y_test": y_test,               # Labels reais para Brier Score
        }

    def _default_params(self) -> dict:
        """Hiperparâmetros padrão do XGBoost (conservadores para evitar overfit)."""
        return {
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "tree_method": "hist",       # Rápido, funciona sem GPU
            "seed": 42,
            "verbosity": 0,
        }

    def _salvar_feature_importance(self, feat_names: list[str], models_dir: str = None):
        """Salva importância das features do modelo resultado_1x2."""
        models_dir = models_dir or MODELS_DIR
        try:
            model_path = os.path.join(models_dir, "resultado_1x2.json")
            if not os.path.exists(model_path):
                return

            model = xgb.Booster()
            model.load_model(model_path)
            importance = model.get_score(importance_type="gain")

            # Ordenar por importância
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

            imp_path = os.path.join(models_dir, "feature_importance.json")
            with open(imp_path, "w", encoding="utf-8") as f:
                json.dump(sorted_imp, f, indent=2, ensure_ascii=False)

            print(f"\n📊 Top 10 features mais importantes:")
            for name, score in sorted_imp[:10]:
                print(f"   {name}: {score:.1f}")

        except Exception as e:
            print(f"[Trainer] Erro ao salvar feature importance: {e}")

    # ══════════════════════════════════════════════
    #  GUARD RAILS — Proteção contra modelo "burro"
    # ══════════════════════════════════════════════

    def _calcular_baselines(self, y_train: list, y_test: list) -> dict:
        """
        Calcula baselines para todos os modelos.
        Recebe listas de dicts de labels (como vindos de build_dataset).
        """
        # Extrair arrays por mercado
        y1_train = np.array([l["resultado"] for l in y_train])
        y1_test = np.array([l["resultado"] for l in y_test])
        y2_test = np.array([l["over25"] for l in y_test])
        y3_test = np.array([l["btts"] for l in y_test])

        # ─── 1x2 ───
        contagem_1x2 = Counter(y1_train)
        classe_majority = contagem_1x2.most_common(1)[0][0]
        majority_acc_1x2 = np.mean(y1_test == classe_majority)
        random_acc_1x2 = 1.0 / 3

        n_train = len(y1_train)
        dist_1x2 = {c: n / n_train for c, n in contagem_1x2.items()}
        prop_acc_1x2 = sum((np.sum(y1_test == c) / len(y1_test)) * p
                           for c, p in dist_1x2.items())

        print(f"   1x2 baselines → Random: {random_acc_1x2:.1%} | "
              f"Majority({classe_majority}): {majority_acc_1x2:.1%} | "
              f"Proporcional: {prop_acc_1x2:.1%}")

        # ─── Over/Under 2.5 ───
        majority_ou = Counter(y2_test).most_common(1)[0][0]
        majority_acc_ou = np.mean(y2_test == majority_ou)

        # ─── BTTS ───
        majority_btts = Counter(y3_test).most_common(1)[0][0]
        majority_acc_btts = np.mean(y3_test == majority_btts)

        print(f"   O/U baselines → Majority({majority_ou}): {majority_acc_ou:.1%}")
        print(f"   BTTS baselines → Majority({majority_btts}): {majority_acc_btts:.1%}")

        return {
            "1x2": {
                "random": round(float(random_acc_1x2), 4),
                "majority": round(float(majority_acc_1x2), 4),
                "majority_class": int(classe_majority),
                "proporcional": round(float(prop_acc_1x2), 4),
            },
            "over_under_25": {
                "random": 0.5,
                "majority": round(float(majority_acc_ou), 4),
                "majority_class": int(majority_ou),
            },
            "btts": {
                "random": 0.5,
                "majority": round(float(majority_acc_btts), 4),
                "majority_class": int(majority_btts),
            },
        }

    def _brier_scores(self, metricas: dict) -> dict:
        """
        Calcula Brier Score para medir calibração das probabilidades.

        Brier Score = média de (probabilidade prevista - outcome real)²
        - 0.0 = calibração perfeita
        - 0.25 = chute 50/50 em binário
        - >0.30 = calibração ruim, probabilidades "mentem"

        Se o modelo diz "70% home" mas homes ganham só 40%, Brier vai ser alto.
        """
        result = {}

        # ─── 1x2 (Brier multiclasse) ───
        m1x2 = metricas.get("resultado_1x2", {})
        probs = m1x2.get("pred_test_probs")
        y_true = m1x2.get("y_test")
        if probs is not None and y_true is not None:
            # One-hot encode verdade
            n = len(y_true)
            if probs.ndim > 1 and probs.shape[1] == 3:
                one_hot = np.zeros((n, 3))
                for i, cls in enumerate(y_true.astype(int)):
                    if 0 <= cls <= 2:
                        one_hot[i, cls] = 1.0
                brier_1x2 = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
            else:
                brier_1x2 = 999.0  # Formato inesperado
            result["1x2"] = round(brier_1x2, 4)
            print(f"   Brier 1x2: {brier_1x2:.4f} (ref: random=0.667, bom<0.30)")

        # ─── Over/Under e BTTS (Brier binário para todos) ───
        for key in ["over_under_15", "over_under_25", "over_under_35", "btts",
                    "over_under_05_ht", "over_under_15_ht",
                    "over_under_05_2t", "over_under_15_2t",
                    "corners_over_85", "corners_over_95", "corners_over_105"]:
            m_bin = metricas.get(key, {})
            probs_bin = m_bin.get("pred_test_probs")
            y_bin = m_bin.get("y_test")
            if probs_bin is not None and y_bin is not None:
                brier_bin = float(np.mean((probs_bin - y_bin) ** 2))
                result[key] = round(brier_bin, 4)
                print(f"   Brier {key}: {brier_bin:.4f}")

        # ─── Resultado HT (Brier multiclasse) ───
        m_ht = metricas.get("resultado_ht", {})
        probs_ht = m_ht.get("pred_test_probs")
        y_ht = m_ht.get("y_test")
        if probs_ht is not None and y_ht is not None and probs_ht.ndim > 1:
            n = len(y_ht)
            one_hot = np.zeros((n, probs_ht.shape[1]))
            for i, cls in enumerate(y_ht.astype(int)):
                if 0 <= cls < probs_ht.shape[1]:
                    one_hot[i, cls] = 1.0
            brier_ht = float(np.mean(np.sum((probs_ht - one_hot) ** 2, axis=1)))
            result["resultado_ht"] = round(brier_ht, 4)
            print(f"   Brier HT: {brier_ht:.4f}")

        return result

    def _gate_qualidade_por_modelo(
        self, metricas: dict, baselines: dict, modelos_lista: list,
        models_dir: str = None
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Gate de qualidade POR MODELO: cada modelo é avaliado individualmente.

        Critérios por tipo:
          - Multiclasse 1x2: accuracy > majority baseline E > MODEL_ACC_MIN_1X2
          - Multiclasse HT/HTFT: accuracy > 33% (melhor que random 3 classes)
          - Binário (O/U, BTTS, corners, gols/tempo): accuracy > 55%
          - Calibração 1x2: Brier < MODEL_BRIER_MAX

        Modelo reprovado → revertido para backup (ou removido se sem backup).
        Modelo aprovado → mantido no disco.

        Retorna (aprovados: list[str], reprovados: list[str], motivos: list[str]).
        """
        import shutil
        models_dir = models_dir or MODELS_DIR
        aprovados = []
        reprovados = []
        motivos = []

        majority_1x2 = baselines.get("1x2", {}).get("majority", 0.40)
        brier_1x2 = metricas.get("brier", {}).get("1x2", 999)

        # Limiar mínimo por tipo de modelo
        # Binários: 55% (acima de moeda com margem)
        # Multiclasse 3 classes (1x2, HT): precisa superar majority
        # HTFT (9 classes): 15% já é razoável (random = 11%)
        LIMIAR_BINARIO = 0.55
        LIMIAR_HTFT = 0.15

        for nome, chave_label, objetivo, num_class in modelos_lista:
            acc = metricas.get(nome, {}).get("accuracy_test", 0)
            model_path = os.path.join(models_dir, f"{nome}.json")
            backup_path = os.path.join(models_dir, f"{nome}_backup.json")

            passou = True
            motivo = ""

            if nome == "resultado_1x2":
                # Gate especial: precisa superar majority + mínimo absoluto + Brier
                if acc <= majority_1x2:
                    passou = False
                    motivo = f"{nome}: {acc:.1%} ≤ majority ({majority_1x2:.1%})"
                elif acc < MODEL_ACC_MIN_1X2:
                    passou = False
                    motivo = f"{nome}: {acc:.1%} < mínimo ({MODEL_ACC_MIN_1X2:.0%})"
                elif brier_1x2 > MODEL_BRIER_MAX:
                    passou = False
                    motivo = f"{nome}: Brier {brier_1x2:.4f} > {MODEL_BRIER_MAX}"
                else:
                    motivo = f"{nome}: {acc:.1%} > majority ({majority_1x2:.1%}), Brier OK"

            elif nome == "htft":
                # HTFT tem 9 classes — limiar mais baixo
                if acc < LIMIAR_HTFT:
                    passou = False
                    motivo = f"{nome}: {acc:.1%} < {LIMIAR_HTFT:.0%} (mín 9 classes)"
                else:
                    motivo = f"{nome}: {acc:.1%} ≥ {LIMIAR_HTFT:.0%}"

            elif objetivo == "multi:softprob":
                # Multiclasse 3 classes (resultado_ht): > 33% random
                if acc < 0.34:
                    passou = False
                    motivo = f"{nome}: {acc:.1%} < 34% (pior que random)"
                else:
                    motivo = f"{nome}: {acc:.1%} ≥ 34%"

            else:
                # Binário (O/U, BTTS, corners, gols/tempo): > 55%
                if acc < LIMIAR_BINARIO:
                    passou = False
                    motivo = f"{nome}: {acc:.1%} < {LIMIAR_BINARIO:.0%}"
                else:
                    motivo = f"{nome}: {acc:.1%} ≥ {LIMIAR_BINARIO:.0%}"

            if passou:
                aprovados.append(nome)
                motivos.append(f"✅ {motivo}")
            else:
                reprovados.append(nome)
                motivos.append(f"❌ {motivo}")
                # Reverter apenas ESTE modelo
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, model_path)
                elif os.path.exists(model_path):
                    os.remove(model_path)

        return aprovados, reprovados, motivos

    # _reverter_backups() removido — agora o revert é feito por modelo
    # dentro de _gate_qualidade_por_modelo()

    @staticmethod
    def modelo_existe(nome: str) -> bool:
        """Verifica se um modelo treinado existe no disco."""
        return os.path.exists(os.path.join(MODELS_DIR, f"{nome}.json"))


if __name__ == "__main__":
    # Execução direta: treino per-league (principal)
    db = Database()
    trainer = Trainer(db)
    resultados = trainer.treinar_por_liga()
    # Exibir resumo
    print("\n" + json.dumps(
        {k: v.get("modelos_aprovados", []) for k, v in resultados.items() if isinstance(v, dict)},
        indent=2, default=str
    ))

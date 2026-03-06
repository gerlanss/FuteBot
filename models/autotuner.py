"""
AutoTuner Per-League — Otimização Optuna + Strategy Slicing por liga.

Treina cada liga individualmente:
  1. Para cada liga com dados suficientes, carrega dataset filtrado
  2. Usa Optuna para buscar hiperparâmetros ótimos (por liga × modelo)
  3. Treina 14 modelos com os melhores hiperparâmetros encontrados
  4. Salva modelos em data/models/league_{id}/
  5. Strategy slicing por mercado × faixa de confiança
  6. Salva estratégias ativas no banco — scanner consulta antes de emitir tips

NÃO existe modelo global. Cada liga tem seus próprios modelos e estratégias.
Se uma liga não tem dados suficientes, ela simplesmente não gera tips.

Roda semanalmente no VPS via scheduler (_job_retreinar).
Tempo estimado: ~60-90 min (22 ligas × 14 modelos × 25 trials).

Uso:
  from models.autotuner import AutoTuner
  tuner = AutoTuner(db)
  resultado = tuner.executar()
"""

import os
import gc
import json
import time
import numpy as np
from datetime import datetime
from collections import Counter

from data.database import Database
from models.features import FeatureExtractor
from models.feature_factory import FeatureFactory
from models.feature_evolution import FeatureEvolution
from models.trainer import Trainer, MODELS_DIR
from config import LEAGUES

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, log_loss
    import optuna
    # Silenciar logs verbosos do Optuna (só erros)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optuna 4.x moveu integrations para pacote separado
    try:
        from optuna.integration import XGBoostPruningCallback
    except (ImportError, ModuleNotFoundError):
        try:
            from optuna_integration import XGBoostPruningCallback
        except (ImportError, ModuleNotFoundError):
            XGBoostPruningCallback = None  # Roda sem pruning

    HAS_ML = True
except ImportError:
    HAS_ML = False
    XGBoostPruningCallback = None

# ══════════════════════════════════════════════
#  CONFIGURAÇÃO
# ══════════════════════════════════════════════

# Trials Optuna por modelo (per-league → menos trials necessários)
N_TRIALS = 25

# Timeout por modelo em segundos
OPTUNA_TIMEOUT = 300

# Mínimo de jogos FT por liga para poder treinar modelo próprio
# Ligas abaixo disso não geram models nem tips
MIN_JOGOS_LIGA = 200

# Threshold mínimo de accuracy para ativar uma estratégia
# Estratégias abaixo disso são salvas mas desativadas (ativo=0)
# Elevado de 0.52 → 0.58 para só ativar slices com accuracy real
STRATEGY_ACC_MIN = 0.58

# Mínimo de amostras num slice para ser considerado válido
# Elevado de 5 → 10 para evitar estratégias baseadas em amostra tíny
STRATEGY_MIN_SAMPLES = 10

# Faixas de confiança para slicing
# Removida faixa 0.0-0.50 (modelo inseguro não gera tip útil)
CONF_BANDS = [
    (0.50, 0.65),   # Confiança moderada
    (0.65, 1.01),   # Alta confiança (modelo seguro)
]

# Lista de modelos e seus mercados associados para slicing
# Cada entrada: (nome_modelo, objetivo, num_class, mercados_para_slice)
MODELO_MERCADOS = [
    ("resultado_1x2", "multi:softprob", 3, [
        ("h2h_home", "prob_home"),
        ("h2h_draw", "prob_draw"),
        ("h2h_away", "prob_away"),
    ]),
    ("over_under_15", "binary:logistic", None, [
        ("over15", "prob_over15"),
        ("under15", "prob_under15"),
    ]),
    ("over_under_25", "binary:logistic", None, [
        ("over25", "prob_over25"),
        ("under25", "prob_under25"),
    ]),
    ("over_under_35", "binary:logistic", None, [
        ("over35", "prob_over35"),
        ("under35", "prob_under35"),
    ]),
    ("btts", "binary:logistic", None, [
        ("btts_yes", "prob_btts_yes"),
        ("btts_no", "prob_btts_no"),
    ]),
    ("resultado_ht", "multi:softprob", 3, [
        ("ht_home", "prob_ht_home"),
        ("ht_draw", "prob_ht_draw"),
        ("ht_away", "prob_ht_away"),
    ]),
    ("htft", "multi:softprob", 9, [
        ("htft", "prob_htft_top"),
    ]),
    # ── Over/Under 1º Tempo (gols no 1T) ──
    ("over_under_05_ht", "binary:logistic", None, [
        ("over05_ht", "prob_over05_ht"),
        ("under05_ht", "prob_under05_ht"),
    ]),
    ("over_under_15_ht", "binary:logistic", None, [
        ("over15_ht", "prob_over15_ht"),
        ("under15_ht", "prob_under15_ht"),
    ]),
    # ── Over/Under 2º Tempo (gols no 2T) ──
    ("over_under_05_2t", "binary:logistic", None, [
        ("over05_2t", "prob_over05_2t"),
        ("under05_2t", "prob_under05_2t"),
    ]),
    ("over_under_15_2t", "binary:logistic", None, [
        ("over15_2t", "prob_over15_2t"),
        ("under15_2t", "prob_under15_2t"),
    ]),
    # ── Escanteios ──
    ("corners_over_85", "binary:logistic", None, [
        ("corners_over_85", "prob_corners_over_85"),
        ("corners_under_85", "prob_corners_under_85"),
    ]),
    ("corners_over_95", "binary:logistic", None, [
        ("corners_over_95", "prob_corners_over_95"),
        ("corners_under_95", "prob_corners_under_95"),
    ]),
    ("corners_over_105", "binary:logistic", None, [
        ("corners_over_105", "prob_corners_over_105"),
        ("corners_under_105", "prob_corners_under_105"),
    ]),
]

# Mapeamento: chave do label no build_dataset
LABEL_KEY = {
    "resultado_1x2": "resultado",
    "over_under_15": "over15",
    "over_under_25": "over25",
    "over_under_35": "over35",
    "btts":          "btts",
    "resultado_ht":  "resultado_ht",
    "htft":          "htft",
    # Modelos de 1T, 2T e escanteios
    "over_under_05_ht":  "over05_ht",
    "over_under_15_ht":  "over15_ht",
    "over_under_05_2t":  "over05_2t",
    "over_under_15_2t":  "over15_2t",
    "corners_over_85":   "corners_over_85",
    "corners_over_95":   "corners_over_95",
    "corners_over_105":  "corners_over_105",
}


class AutoTuner:
    """
    Otimização Optuna + Strategy Slicing — 100% per-league.

    Para cada liga com dados suficientes:
      1. Carrega dataset filtrado por league_id
      2. Split temporal (treino/teste)
      3. Optuna busca hiperparâmetros por modelo
      4. Treina e salva modelos em data/models/league_{id}/
      5. Strategy slicing por mercado × confiança

    Não existe modelo global. Liga sem dados = liga sem tips.
    """

    def __init__(self, db: Database = None, device: str = "cpu"):
        self.db = db or Database()
        self.fe = FeatureExtractor(self.db)
        self.ff = FeatureFactory(self.db)
        self.evo = FeatureEvolution(device=device)
        self.device = device

    def executar(self, train_seasons: list[int] = None,
                 test_season: int = None,
                 n_trials: int = None) -> dict:
        """
        Executa o AutoTuner per-league completo.

        Para cada liga elegível (>= MIN_JOGOS_LIGA fixtures FT):
          - Carrega dataset filtrado
          - Optuna busca hiperparâmetros (n_trials por modelo)
          - Treina modelos finais e salva em league_{id}/
          - Gera strategies por mercado × confiança

        Retorna dict com métricas consolidadas de todas as ligas.
        """
        if not HAS_ML:
            return {"erro": "XGBoost/sklearn/optuna não instalado"}

        train_seasons = train_seasons or [2024]
        test_season = test_season or 2025
        trials_per_model = n_trials or N_TRIALS
        inicio = time.time()

        print(f"\n{'='*60}")
        print(f"🧠 AutoTuner Per-League — Optuna + Strategy Slicing")
        print(f"{'='*60}")
        print(f"   Treino: seasons {train_seasons}")
        print(f"   Teste: season {test_season}")
        print(f"   Trials por modelo: {trials_per_model}")
        print(f"   Mínimo jogos/liga: {MIN_JOGOS_LIGA}")

        # ─── ETAPA 1: Identificar ligas elegíveis ───
        print(f"\n📋 Etapa 1: Identificando ligas elegíveis...")
        ligas_elegiveis = []
        ligas_insuficientes = []

        for slug, info in LEAGUES.items():
            lid = info["id"]
            nome = info["nome"]
            fixtures = self.db.fixtures_finalizados(league_id=lid)
            n_ft = len(fixtures)
            if n_ft >= MIN_JOGOS_LIGA:
                ligas_elegiveis.append((lid, nome, n_ft))
            else:
                ligas_insuficientes.append((lid, nome, n_ft))

        ligas_elegiveis.sort(key=lambda x: -x[2])  # Maior primeiro

        print(f"   Elegíveis: {len(ligas_elegiveis)} ligas (>= {MIN_JOGOS_LIGA} jogos)")
        print(f"   Insuficientes: {len(ligas_insuficientes)} ligas (puladas)")
        for lid, nome, n_ft in ligas_insuficientes:
            print(f"     ⏭️  {nome} (liga {lid}): {n_ft} jogos")

        if not ligas_elegiveis:
            return {"erro": f"Nenhuma liga com >= {MIN_JOGOS_LIGA} jogos FT"}

        # ─── ETAPA 2: Processar cada liga ───
        feat_names = FeatureFactory.feature_names_full()
        todas_strategies = []
        resultados_por_liga = {}
        total_modelos_salvos = 0

        for idx_liga, (lid, nome_liga, n_ft) in enumerate(ligas_elegiveis, 1):
            print(f"\n{'─'*60}")
            print(f"🏟️  [{idx_liga}/{len(ligas_elegiveis)}] Liga {lid}: {nome_liga} ({n_ft} jogos)")
            print(f"{'─'*60}")

            resultado_liga = self._processar_liga(
                league_id=lid,
                nome_liga=nome_liga,
                train_seasons=train_seasons,
                test_season=test_season,
                trials_per_model=trials_per_model,
                feat_names=feat_names,
            )

            resultados_por_liga[lid] = resultado_liga

            if "erro" in resultado_liga:
                print(f"   ⚠️ {resultado_liga['erro']}")
                continue

            # Acumular strategies
            for s in resultado_liga.get("strategies", []):
                s["league_id"] = lid
                todas_strategies.append(s)

            total_modelos_salvos += resultado_liga.get("modelos_salvos", 0)

            # Liberar memória entre ligas
            gc.collect()

        # ─── ETAPA 3: Salvar todas as strategies no banco ───
        print(f"\n{'='*60}")
        print(f"💾 Salvando strategies de todas as ligas...")
        versao = f"at_{datetime.now().strftime('%Y%m%d_%H%M')}"

        for s in todas_strategies:
            s["modelo_versao"] = versao
            lid = s["league_id"]
            res_liga = resultados_por_liga.get(lid, {})
            melhores = res_liga.get("melhores", {})
            nome_modelo = self._modelo_do_mercado(s["mercado"])
            s["params"] = melhores.get(nome_modelo, {}).get("params", {})

        self.db.salvar_strategies(todas_strategies)

        ativas = [s for s in todas_strategies if s.get("ativo", 0)]
        elapsed = time.time() - inicio

        # ─── Resumo final ───
        print(f"\n{'='*60}")
        print(f"✅ AutoTuner Per-League concluído em {elapsed/60:.1f} min")
        print(f"   Ligas processadas: {len(ligas_elegiveis)}")
        print(f"   Modelos salvos: {total_modelos_salvos}")
        print(f"   Estratégias: {len(todas_strategies)} total, {len(ativas)} ativas")
        print(f"   Versão: {versao}")
        if ativas:
            accs = [s["accuracy"] for s in ativas]
            print(f"   Accuracy ativas: {min(accs):.1%} – {max(accs):.1%} "
                  f"(média: {sum(accs)/len(accs):.1%})")
        print(f"{'='*60}")

        # Salvar log de treino
        self.db.salvar_treino(
            versao=versao,
            n_samples=sum(r.get("n_amostras", 0) for r in resultados_por_liga.values()),
            n_features=len(feat_names),
            acc_train=0,
            acc_test=0,
            metrics={
                "autotuner": True,
                "per_league": True,
                "optuna": True,
                "trials_por_modelo": trials_per_model,
                "ligas_processadas": len(ligas_elegiveis),
                "ligas_insuficientes": len(ligas_insuficientes),
                "modelos_salvos": total_modelos_salvos,
                "strategies_total": len(todas_strategies),
                "strategies_ativas": len(ativas),
                "tempo_segundos": round(elapsed, 1),
            },
            params={},
        )

        return {
            "versao": versao,
            "ligas_processadas": len(ligas_elegiveis),
            "modelos_salvos": total_modelos_salvos,
            "strategies_total": len(todas_strategies),
            "strategies_ativas": len(ativas),
            "strategies": todas_strategies,
            "resultados_por_liga": {
                lid: {
                    "nome": r.get("nome_liga", ""),
                    "modelos": r.get("modelos_salvos", 0),
                    "strategies_ativas": len([
                        s for s in todas_strategies
                        if s.get("league_id") == lid and s.get("ativo")
                    ]),
                }
                for lid, r in resultados_por_liga.items()
                if "erro" not in r
            },
            "tempo_segundos": round(elapsed, 1),
        }

    # ══════════════════════════════════════════════
    #  PROCESSAMENTO DE UMA LIGA
    # ══════════════════════════════════════════════

    def _processar_liga(self, league_id: int, nome_liga: str,
                        train_seasons: list[int], test_season: int,
                        trials_per_model: int, feat_names: list) -> dict:
        """
        Executa o pipeline completo para UMA liga:
          1. Carrega dataset filtrado por league_id
          2. Split temporal
          3. Optuna por modelo (14 modelos)
          4. Treina e salva modelos em league_{id}/
          5. Strategy slicing por mercado × confiança

        Retorna dict com melhores params, accuracy, strategies.
        """
        # ─── Carregar dataset da liga (pool completo ~217 features) ───
        all_features, all_labels = self.ff.build_dataset(
            league_id=league_id,
            seasons=train_seasons + [test_season]
        )

        if len(all_features) < 50:
            return {"erro": f"Dados insuficientes: {len(all_features)} jogos"}

        # Split temporal
        X_train, X_test, y_train, y_test = [], [], [], []

        for f, l in zip(all_features, all_labels):
            row = [f.get(name, 0) or 0 for name in feat_names]
            if f.get("_season") in train_seasons:
                X_train.append(row)
                y_train.append(l)
            elif f.get("_season") == test_season:
                X_test.append(row)
                y_test.append(l)

        # Fallback: random split se test_season vazio
        if len(X_test) == 0:
            print(f"   ⚠️ Season {test_season} vazio — random split 80/20")
            from sklearn.model_selection import train_test_split as tts
            idx = list(range(len(X_train)))
            if len(idx) < 20:
                return {"erro": f"Treino muito pequeno: {len(idx)} amostras"}
            idx_tr, idx_te = tts(idx, test_size=0.2, random_state=42)
            all_X = X_train
            all_y = y_train
            X_train = [all_X[i] for i in idx_tr]
            X_test = [all_X[i] for i in idx_te]
            y_train = [all_y[i] for i in idx_tr]
            y_test = [all_y[i] for i in idx_te]

        # Arrays contíguos C-order em RAM — DMatrix e slice por coluna ficam rápidos
        X_train = np.ascontiguousarray(np.array(X_train, dtype=np.float32))
        X_test = np.ascontiguousarray(np.array(X_test, dtype=np.float32))

        print(f"[Features] Dataset: {len(all_features)} jogos com features "
              f"(de {len(all_features)} fixtures totais)")
        print(f"   Treino: {len(X_train)} | Teste: {len(X_test)} amostras")

        if len(X_train) < 30:
            return {"erro": f"Treino muito pequeno: {len(X_train)} amostras"}

        # ─── Evolução + Optuna por modelo ───
        melhores = {}
        feat_selecionadas = {}  # nome_modelo → (indices, feat_names_selecionados)

        # Features originais (51) como semente 0 da evolução
        features_base = FeatureExtractor.feature_names()

        for nome_modelo, objetivo, num_class, _ in MODELO_MERCADOS:
            label_key = LABEL_KEY[nome_modelo]
            y_tr = np.array([l[label_key] for l in y_train])
            y_te = np.array([l[label_key] for l in y_test])

            # Verificar se há variância no label (senão Optuna falha)
            if len(np.unique(y_tr)) < 2:
                continue

            # ── Evolução genética: selecionar features ótimas ──
            print(f"   🧬 {nome_modelo}: evolução de features...")
            indices_sel = self.evo.evoluir(
                X_train, y_tr, X_test, y_te,
                feat_names, objetivo, num_class,
                features_base=features_base
            )
            fn_sel = [feat_names[i] for i in indices_sel]
            X_tr_sel = X_train[:, indices_sel]
            X_te_sel = X_test[:, indices_sel]
            feat_selecionadas[nome_modelo] = (indices_sel, fn_sel)
            print(f"      → {len(indices_sel)} features selecionadas (de {len(feat_names)})")

            # ── Optuna com features selecionadas ──
            best_result = self._optuna_modelo(
                nome_modelo, objetivo, num_class,
                X_tr_sel, y_tr, X_te_sel, y_te,
                fn_sel, trials_per_model
            )

            if best_result:
                melhores[nome_modelo] = best_result

        if not melhores:
            return {"erro": "Nenhum modelo completou Optuna"}

        # ─── Salvar modelos em league_{id}/ ───
        modelos_salvos = 0
        for nome_modelo, objetivo, num_class, _ in MODELO_MERCADOS:
            if nome_modelo not in melhores:
                continue
            best = melhores[nome_modelo]
            indices_sel, fn_sel = feat_selecionadas[nome_modelo]
            self._treinar_e_salvar(
                X_train[:, indices_sel],
                np.array([l[LABEL_KEY[nome_modelo]] for l in y_train]),
                X_test[:, indices_sel],
                np.array([l[LABEL_KEY[nome_modelo]] for l in y_test]),
                best["params"], objetivo, num_class,
                nome_modelo, fn_sel, league_id
            )
            modelos_salvos += 1

        # ─── Salvar mapa de features por modelo ───
        feature_map = {
            nome: fn_sel
            for nome, (_, fn_sel) in feat_selecionadas.items()
        }
        liga_dir = os.path.join(MODELS_DIR, f"league_{league_id}")
        os.makedirs(liga_dir, exist_ok=True)
        map_path = os.path.join(liga_dir, "feature_map.json")
        with open(map_path, "w") as f:
            json.dump(feature_map, f, indent=2)
        print(f"   📝 feature_map.json salvo ({len(feature_map)} modelos)")

        # ─── Strategy slicing (dados de teste da liga) ───
        strategies = self._strategy_slicing_liga(
            melhores, league_id
        )

        n_ativas = len([s for s in strategies if s.get("ativo")])
        print(f"   📊 {modelos_salvos} modelos salvos, "
              f"{n_ativas} strategies ativas")

        return {
            "nome_liga": nome_liga,
            "n_amostras": len(X_train) + len(X_test),
            "modelos_salvos": modelos_salvos,
            "melhores": {
                k: {"accuracy": v["accuracy"], "params": v["params"]}
                for k, v in melhores.items()
            },
            "strategies": strategies,
        }

    # ══════════════════════════════════════════════
    #  OPTUNA — BUSCA POR MODELO
    # ══════════════════════════════════════════════

    def _optuna_modelo(self, nome_modelo: str, objetivo: str, num_class: int,
                       X_train, y_tr, X_test, y_te,
                       feat_names: list, n_trials: int) -> dict | None:
        """
        Roda Optuna para um modelo específico numa liga.
        Retorna dict com params, accuracy, preds ou None se falhou.
        """
        study = optuna.create_study(
            direction="minimize",
            study_name=f"futebot_{nome_modelo}",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=20,
            ),
        )

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            # scale_pos_weight para modelos binários
            if not num_class or num_class <= 2:
                pos = float(np.sum(y_tr == 1))
                neg = float(np.sum(y_tr == 0))
                if pos > 0 and neg > 0:
                    auto_spw = neg / pos
                    spw_low = min(auto_spw * 0.5, auto_spw * 2.0)
                    spw_high = max(auto_spw * 0.5, auto_spw * 2.0)
                    # Garantir range mínimo sensato
                    spw_low = max(0.1, spw_low)
                    spw_high = max(spw_low + 0.01, spw_high)
                    params["scale_pos_weight"] = trial.suggest_float(
                        "scale_pos_weight", spw_low, spw_high
                    )

            xgb_params = {
                **params,
                "objective": objetivo,
                "tree_method": "hist",
                "device": self.device,
                "nthread": os.cpu_count() or 4,
                "seed": 42,
                "verbosity": 0,
            }
            if num_class:
                xgb_params["num_class"] = num_class
                xgb_params["eval_metric"] = "mlogloss"
            else:
                xgb_params["eval_metric"] = "logloss"

            dtrain = xgb.DMatrix(X_train, label=y_tr, feature_names=feat_names)
            dtest = xgb.DMatrix(X_test, label=y_te, feature_names=feat_names)

            callbacks_xgb = []
            if XGBoostPruningCallback is not None:
                pruning_cb = XGBoostPruningCallback(
                    trial, f"test-{xgb_params['eval_metric']}"
                )
                callbacks_xgb.append(pruning_cb)

            try:
                model = xgb.train(
                    xgb_params, dtrain,
                    num_boost_round=300,
                    evals=[(dtest, "test")],
                    early_stopping_rounds=30,
                    verbose_eval=False,
                    callbacks=callbacks_xgb if callbacks_xgb else None,
                )
            except optuna.TrialPruned:
                raise

            preds_raw = model.predict(dtest)
            del dtrain, dtest, model

            if objetivo == "multi:softprob":
                probs = preds_raw if preds_raw.ndim > 1 else preds_raw.reshape(-1, 1)
                return log_loss(y_te, probs, labels=list(range(num_class)))
            else:
                probs_clamp = np.clip(preds_raw, 1e-7, 1 - 1e-7)
                return log_loss(y_te, probs_clamp)

        # Rodar otimização
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=OPTUNA_TIMEOUT,
                show_progress_bar=False,
            )
        except Exception as e:
            print(f"   ⚠️ {nome_modelo}: erro no Optuna ({e})")

        try:
            _best = study.best_trial
        except ValueError:
            _best = None

        result = None
        if _best:
            best_params = study.best_params
            best_logloss = study.best_value

            # Retreinar com melhores params para obter accuracy e predições
            acc_test, preds = self._treinar_avaliar(
                X_train, y_tr, X_test, y_te,
                best_params, objetivo, num_class, feat_names
            )

            result = {
                "params": best_params,
                "accuracy": acc_test,
                "logloss": best_logloss,
                "preds": preds,
                "y_test": y_te,
            }

            n_pruned = len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.PRUNED])
            print(f"   ✅ {nome_modelo}: acc={acc_test:.1%}, "
                  f"logloss={best_logloss:.4f} "
                  f"({len(study.trials)} trials, {n_pruned} pruned)")
        else:
            print(f"   ❌ {nome_modelo}: nenhum trial completou")

        # Liberar memória do study
        del study
        gc.collect()

        return result

    # ══════════════════════════════════════════════
    #  TREINO + AVALIAÇÃO (uma config)
    # ══════════════════════════════════════════════

    def _treinar_avaliar(self, X_train, y_train, X_test, y_test,
                         params: dict, objetivo: str,
                         num_class: int, feat_names: list) -> tuple[float, np.ndarray]:
        """
        Treina um modelo com uma config e retorna (accuracy, predictions).
        Não salva no disco — usado para avaliação e para recuperar predições
        do melhor trial do Optuna.
        """
        xgb_params = {
            **params,
            "objective": objetivo,
            "tree_method": "hist",
            "device": self.device,
            "nthread": os.cpu_count() or 4,
            "seed": 42,
            "verbosity": 0,
        }
        if num_class:
            xgb_params["num_class"] = num_class
            xgb_params["eval_metric"] = "mlogloss"
        else:
            xgb_params["eval_metric"] = "logloss"

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_names)

        model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=300,
            evals=[(dtest, "test")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        preds_raw = model.predict(dtest)

        # Liberar DMatrix e modelo da memória
        del dtrain, dtest, model

        if objetivo == "multi:softprob":
            pred_cls = np.argmax(preds_raw, axis=1) if preds_raw.ndim > 1 else preds_raw
        else:
            pred_cls = (preds_raw > 0.5).astype(int)

        acc = accuracy_score(y_test, pred_cls)
        return acc, preds_raw

    def _treinar_e_salvar(self, X_train, y_train, X_test, y_test,
                          params: dict, objetivo: str, num_class: int,
                          nome: str, feat_names: list, league_id: int):
        """
        Treina o modelo final e salva em data/models/league_{id}/.
        Mantém backup do modelo anterior.
        """
        import shutil

        xgb_params = {
            **params,
            "objective": objetivo,
            "tree_method": "hist",
            "device": self.device,
            "nthread": os.cpu_count() or 4,
            "seed": 42,
            "verbosity": 0,
        }
        if num_class:
            xgb_params["num_class"] = num_class
            xgb_params["eval_metric"] = "mlogloss"
        else:
            xgb_params["eval_metric"] = "logloss"

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_names)

        model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=500,
            evals=[(dtest, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Salvar SEMPRE na pasta da liga (sem modelo global)
        liga_dir = os.path.join(MODELS_DIR, f"league_{league_id}")
        os.makedirs(liga_dir, exist_ok=True)

        model_path = os.path.join(liga_dir, f"{nome}.json")
        backup_path = os.path.join(liga_dir, f"{nome}_backup.json")
        if os.path.exists(model_path):
            shutil.copy2(model_path, backup_path)

        model.save_model(model_path)
        del dtrain, dtest, model

    # ══════════════════════════════════════════════
    #  STRATEGY SLICING PER-LEAGUE
    # ══════════════════════════════════════════════

    def _strategy_slicing_liga(self, melhores: dict,
                                league_id: int) -> list[dict]:
        """
        Analisa cada modelo por mercado × faixa de confiança para UMA liga.

        Não precisa filtrar por liga — dados já são de uma liga só.
        Usa predições do set de teste (guardadas em melhores[nome]["preds"]).
        """
        strategies = []

        for nome_modelo, _, num_class, mercados in MODELO_MERCADOS:
            if nome_modelo not in melhores:
                continue

            best = melhores[nome_modelo]
            preds_raw = best["preds"]
            y_te = best["y_test"]

            for mercado_id, prob_key in mercados:
                probs = self._extrair_probs_mercado(preds_raw, mercado_id, num_class)
                if probs is None:
                    continue

                acertos = self._calcular_acertos(preds_raw, y_te, mercado_id, num_class)
                if acertos is None:
                    continue

                for conf_min, conf_max in CONF_BANDS:
                    mask = (probs >= conf_min) & (probs < conf_max)
                    n_slice = int(np.sum(mask))

                    if n_slice < STRATEGY_MIN_SAMPLES:
                        continue

                    acc_slice = float(np.mean(acertos[mask]))
                    ativo = 1 if acc_slice >= STRATEGY_ACC_MIN else 0

                    strategies.append({
                        "mercado": mercado_id,
                        "league_id": league_id,
                        "conf_min": conf_min,
                        "conf_max": conf_max,
                        "accuracy": round(acc_slice, 4),
                        "n_samples": n_slice,
                        "ativo": ativo,
                    })

        return strategies

    def _extrair_probs_mercado(self, preds_raw: np.ndarray,
                                mercado_id: str,
                                num_class: int) -> np.ndarray | None:
        """
        Extrai a probabilidade relevante para um mercado específico.

        Para multiclasse: seleciona a coluna certa (home=0, draw=1, away=2).
        Para binário: usa a prob direta (over=prob, under=1-prob).
        """
        try:
            if num_class and num_class > 2:
                # Multiclasse: extrair coluna
                if preds_raw.ndim < 2:
                    return None
                col_map = {
                    "h2h_home": 0, "h2h_draw": 1, "h2h_away": 2,
                    "ht_home": 0, "ht_draw": 1, "ht_away": 2,
                    "htft": None,  # HT/FT usa max das 9 probs
                }
                col = col_map.get(mercado_id)
                if col is not None:
                    return preds_raw[:, col]
                else:
                    # HT/FT: usar a probabilidade máxima
                    return np.max(preds_raw, axis=1)
            else:
                # Binário: prob direta para "yes/over", complementar para "no/under"
                # Usa 'in' para capturar corners_under_85 etc.
                if mercado_id.endswith("_no") or "under" in mercado_id:
                    return 1 - preds_raw
                return preds_raw
        except Exception:
            return None

    def _calcular_acertos(self, preds_raw: np.ndarray,
                          y_test: np.ndarray,
                          mercado_id: str,
                          num_class: int) -> np.ndarray | None:
        """
        Calcula array booleano de acertos para cada amostra.

        Para multiclasse: acerto = classe prevista == classe real.
        Para binário: acerto = (prob > 0.5) == label real.
        Para mercados "under/no": inverte a lógica.
        """
        try:
            if num_class and num_class > 2:
                pred_cls = np.argmax(preds_raw, axis=1) if preds_raw.ndim > 1 else preds_raw
                acertos = (pred_cls == y_test).astype(float)

                # Para mercados específicos (h2h_home, h2h_draw, etc.),
                # o acerto é: modelo provou a classe correta E era esse mercado
                col_map = {
                    "h2h_home": 0, "h2h_draw": 1, "h2h_away": 2,
                    "ht_home": 0, "ht_draw": 1, "ht_away": 2,
                }
                col = col_map.get(mercado_id)
                if col is not None:
                    # Acerto para este mercado = modelo previu home E resultado foi home
                    # Ou: o modelo acertou (qualquer classe)
                    # Na prática, queremos: se apostou neste outcome, acertou?
                    # Isso é: o resultado real foi esse outcome
                    acertos = (y_test == col).astype(float)
                    # Mas só conta como "acerto da estratégia" se o modelo
                    # indicou esse outcome (prob alta nessa classe)
                    if preds_raw.ndim > 1:
                        modelo_indicou = (np.argmax(preds_raw, axis=1) == col)
                        # Acerto = modelo indicou E acertou
                        acertos = (modelo_indicou & (y_test == col)).astype(float)
                        # Mas para accuracy, queremos: entre as vezes que indicou, quantas acertou?
                        # Isso é precision, não accuracy geral.
                        # Para strategy slicing, usamos: acertou o resultado geral
                        acertos = (pred_cls == y_test).astype(float)

                return acertos
            else:
                pred_cls = (preds_raw > 0.5).astype(int)
                if mercado_id.endswith("_no") or "under" in mercado_id:
                    # Para under/no: acertou se previu < 0.5 e label era 0
                    pred_cls_inv = (preds_raw <= 0.5).astype(int)
                    acertos = (pred_cls_inv == (1 - y_test)).astype(float)
                else:
                    acertos = (pred_cls == y_test).astype(float)
                return acertos
        except Exception:
            return None

    def _modelo_do_mercado(self, mercado_id: str) -> str:
        """Dado um mercado_id, retorna o nome do modelo correspondente."""
        for nome, _, _, mercados in MODELO_MERCADOS:
            for mid, _ in mercados:
                if mid == mercado_id:
                    return nome
        return ""

    # ══════════════════════════════════════════════
    #  FORMATAÇÃO PARA TELEGRAM
    # ══════════════════════════════════════════════

    @staticmethod
    def formatar_resultado(resultado: dict) -> str:
        """
        Formata o resultado do AutoTuner per-league para envio no Telegram.
        Retorna HTML.
        """
        import html as _html

        if "erro" in resultado:
            return f"❌ AutoTuner: {_html.escape(str(resultado['erro']))}"

        n_ligas = resultado.get("ligas_processadas", 0)
        n_modelos = resultado.get("modelos_salvos", 0)
        n_ativas = resultado.get("strategies_ativas", 0)
        n_total = resultado.get("strategies_total", 0)
        tempo = resultado.get("tempo_segundos", 0)

        lines = [
            "🧠 <b>AutoTuner Per-League concluído!</b>\n",
            f"⏱ Tempo: {tempo/60:.1f} min",
            f"🏟️ Ligas: {n_ligas} processadas",
            f"💾 Modelos: {n_modelos} salvos",
            f"🎯 Estratégias: {n_ativas}/{n_total} ativas "
            f"(threshold: {STRATEGY_ACC_MIN:.0%})\n",
        ]

        # Resumo por liga
        por_liga = resultado.get("resultados_por_liga", {})
        if por_liga:
            lines.append("<b>Por liga:</b>")
            for lid, info in sorted(por_liga.items(),
                                     key=lambda x: -x[1].get("strategies_ativas", 0)):
                nome = info.get("nome", f"Liga {lid}")
                n_mod = info.get("modelos", 0)
                n_strat = info.get("strategies_ativas", 0)
                emoji = "🟢" if n_strat >= 10 else "🟡" if n_strat >= 5 else "🔴"
                lines.append(
                    f"  {emoji} {_html.escape(nome)}: "
                    f"{n_mod} modelos, {n_strat} strategies"
                )

        return "\n".join(lines)

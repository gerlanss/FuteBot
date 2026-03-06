"""
Feature Evolution — Algoritmo genético para seleção de features por mercado × liga.

Cada cromossomo é uma máscara binária sobre o pool de ~217 features.
A população evolui por gerações, selecionando os subconjuntos que
maximizam a accuracy do XGBoost para cada mercado específico.

Parâmetros genéticos (padrão):
  - População: 24 indivíduos
  - Gerações: 15
  - Seleção: torneio (k=3)
  - Crossover: uniforme (50% dos genes)
  - Mutação: flip de ~2 bits aleatórios
  - Elitismo: top 2 sobrevivem intactos entre gerações
  - Semente 0: 51 features originais do FeatureExtractor (baseline)
  - Semente 1: pool completo de 217 features (todas ligadas)

Fitness:
  XGBoost rápido (50 rounds, GPU) → accuracy no conjunto de teste.
  Resultado cacheado por cromossomo para evitar reavaliações.

Uso:
  from models.feature_evolution import FeatureEvolution
  evo = FeatureEvolution()
  indices = evo.evoluir(X_train, y_tr, X_test, y_te, feat_names, ...)
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Workers paralelos para avaliação de fitness (6 workers × 2 threads XGB = plena utilização CPU)
# Leve oversubscription é OK — XGBoost tem idle time entre rounds
N_WORKERS = min(6, os.cpu_count() or 6)
NTHREAD_PER_WORKER = 2


# ══════════════════════════════════════════════
#  PARÂMETROS DO ALGORITMO GENÉTICO
# ══════════════════════════════════════════════

POP_SIZE = 24           # Tamanho da população
N_GERACOES = 15         # Número de gerações
TORNEIO_K = 3           # Indivíduos por torneio de seleção
PROB_CROSSOVER = 0.7    # Probabilidade de crossover entre 2 pais
N_MUTACAO = 2           # Bits a flipar por mutação
ELITISMO = 2            # Indivíduos preservados por elitismo
MIN_FEATURES = 10       # Mínimo de features ativas num cromossomo
XGB_ROUNDS = 50         # Rounds rápidos de XGBoost para avaliação


class FeatureEvolution:
    """
    Algoritmo genético para seleção de features ótimas por mercado.

    Cada indivíduo da população é um vetor binário (0/1) com tamanho
    igual ao número total de features no pool. A evolução favorece
    subconjuntos que maximizam accuracy, descartando ruído.

    É executado PER-MERCADO × PER-LIGA: um resultado de 1x2 pode precisar
    de features diferentes de um mercado de escanteios ou BTTS.
    """

    def __init__(self, seed: int = 42, device: str = "cuda"):
        """
        Parâmetros:
          seed: semente para reproducibilidade do genético
          device: dispositivo para Optuna (GPU). Evolução usa CPU para micro-treinos.
        """
        self.rng = np.random.RandomState(seed)
        self.device = device
        # Evolução SEMPRE usa CPU — micro-treinos de 50 rounds em <500 amostras
        # são mais rápidos em CPU do que com overhead de transferência GPU
        self._evo_device = "cpu"

    def evoluir(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                feat_names: list[str], objetivo: str,
                num_class: int,
                features_base: list[str] = None) -> list[int]:
        """
        Executa evolução genética e retorna índices das features selecionadas.

        Parâmetros:
          X_train, X_test: arrays com TODAS as features (~217 colunas)
          y_train, y_test: labels do mercado (array 1D)
          feat_names: nomes de todas as features (217 strings)
          objetivo: 'multi:softprob' ou 'binary:logistic'
          num_class: nº de classes (None para binário)
          features_base: nomes das 51 features originais (semente 0)

        Retorna:
          Lista de índices (colunas em X) das features vencedoras.
        """
        n_features = len(feat_names)

        # ── Criar população inicial ──
        pop = self._criar_populacao(n_features, feat_names, features_base)

        # Cache de fitness: evita reavaliar cromossomos idênticos
        cache = {}

        # Pré-calcular parâmetros invariantes por mercado (evita recalcular a cada fitness)
        spw = None
        if not num_class or (num_class and num_class <= 2):
            pos = float(np.sum(y_train == 1))
            neg = float(np.sum(y_train == 0))
            if pos > 0 and neg > 0:
                spw = neg / pos

        melhor_fitness = -1.0
        melhor_individuo = None

        for gen in range(N_GERACOES):
            # Avaliar fitness em paralelo (ThreadPool — XGB libera GIL)
            pendentes = []
            for i, ind in enumerate(pop):
                key = tuple(np.where(ind == 1)[0])
                if key in cache:
                    pendentes.append((i, None, key))  # já cacheado
                else:
                    pendentes.append((i, ind, key))

            # Avaliar não-cacheados em paralelo via ThreadPool
            def _eval_ind(args):
                _, ind, key = args
                if ind is None:
                    return cache[key]
                return self._fitness(
                    ind, X_train, y_train, X_test, y_test,
                    feat_names, objetivo, num_class, cache, spw
                )

            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                resultados = list(pool.map(_eval_ind, pendentes))

            fitness = np.array(resultados)

            # Registrar melhor da geração
            idx_best = int(np.argmax(fitness))
            if fitness[idx_best] > melhor_fitness:
                melhor_fitness = fitness[idx_best]
                melhor_individuo = pop[idx_best].copy()

            n_ativas = int(np.sum(pop[idx_best]))
            print(
                f"      Gen {gen + 1:2d}/{N_GERACOES}: "
                f"best={fitness[idx_best]:.1%} "
                f"({n_ativas} feats) "
                f"avg={np.mean(fitness):.1%}"
            )

            # ── Elitismo: copiar top-N para a nova geração ──
            elite_idx = np.argsort(fitness)[-ELITISMO:]
            nova_pop = [pop[i].copy() for i in elite_idx]

            # ── Seleção + Crossover + Mutação até completar população ──
            while len(nova_pop) < POP_SIZE:
                pai1 = self._selecao_torneio(pop, fitness)
                pai2 = self._selecao_torneio(pop, fitness)

                if self.rng.random() < PROB_CROSSOVER:
                    filho1, filho2 = self._crossover_uniforme(pai1, pai2)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()

                self._mutar(filho1, n_features)
                self._mutar(filho2, n_features)

                # Garantir mínimo de features ativas
                self._garantir_minimo(filho1)
                self._garantir_minimo(filho2)

                nova_pop.append(filho1)
                if len(nova_pop) < POP_SIZE:
                    nova_pop.append(filho2)

            pop = nova_pop[:POP_SIZE]

        # ── Retornar índices das features do melhor indivíduo ──
        indices = [i for i, v in enumerate(melhor_individuo) if v == 1]
        return indices

    # ══════════════════════════════════════════════
    #  CRIAÇÃO DA POPULAÇÃO INICIAL
    # ══════════════════════════════════════════════

    def _criar_populacao(self, n_features: int, feat_names: list[str],
                         features_base: list[str] = None) -> list[np.ndarray]:
        """
        Cria população inicial com sementes estratégicas.

        Sementes:
          0 — features originais do FeatureExtractor (baseline de 51)
          1 — pool completo (todas as 217 ligadas)
          2..N — aleatórias (~40-60% das features ativas)
        """
        pop = []

        # Semente 0: features originais (baseline comprovado)
        if features_base:
            seed0 = np.zeros(n_features, dtype=np.int8)
            base_set = set(features_base)
            for i, nome in enumerate(feat_names):
                if nome in base_set:
                    seed0[i] = 1
            # Se nenhuma feature base foi encontrada, ativa todas
            if np.sum(seed0) == 0:
                seed0 = np.ones(n_features, dtype=np.int8)
            pop.append(seed0)

        # Semente 1: pool completo (benchmark — sem seleção)
        pop.append(np.ones(n_features, dtype=np.int8))

        # Resto: aleatório com ~50% de features ativas
        while len(pop) < POP_SIZE:
            ind = self.rng.binomial(1, 0.5, n_features).astype(np.int8)
            if np.sum(ind) >= MIN_FEATURES:
                pop.append(ind)

        return pop

    # ══════════════════════════════════════════════
    #  AVALIAÇÃO DE FITNESS
    # ══════════════════════════════════════════════

    def _fitness(self, individuo: np.ndarray,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 feat_names: list[str], objetivo: str,
                 num_class: int, cache: dict,
                 spw: float = None) -> float:
        """
        Treina XGBoost rápido com o subconjunto de features e retorna accuracy.

        Usa CPU (self._evo_device) para micro-treinos — mais rápido que GPU
        em datasets <1000 amostras com 50 rounds.
        Cache evita reavaliar cromossomos idênticos.
        spw: scale_pos_weight pré-calculado (evita recalcular por indivíduo).
        """
        key = tuple(np.where(individuo == 1)[0])
        if key in cache:
            return cache[key]

        idx = list(key)
        if len(idx) < MIN_FEATURES:
            cache[key] = 0.0
            return 0.0

        # Slice de colunas — arrays já em RAM, sem cópia profunda
        X_tr = X_train[:, idx]
        X_te = X_test[:, idx]
        fn = [feat_names[i] for i in idx]

        # CPU + nthread limitado (cada worker usa NTHREAD_PER_WORKER)
        xgb_params = {
            "objective": objetivo,
            "tree_method": "hist",
            "device": self._evo_device,
            "nthread": NTHREAD_PER_WORKER,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "verbosity": 0,
        }
        if num_class:
            xgb_params["num_class"] = num_class
            xgb_params["eval_metric"] = "mlogloss"
        else:
            xgb_params["eval_metric"] = "logloss"
            if spw is not None:
                xgb_params["scale_pos_weight"] = spw

        try:
            dtrain = xgb.DMatrix(X_tr, label=y_train, feature_names=fn)
            dtest = xgb.DMatrix(X_te, label=y_test, feature_names=fn)

            model = xgb.train(
                xgb_params, dtrain,
                num_boost_round=XGB_ROUNDS,
                evals=[(dtest, "test")],
                verbose_eval=False,
            )

            preds = model.predict(dtest)

            if objetivo == "multi:softprob":
                pred_cls = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
            else:
                pred_cls = (preds > 0.5).astype(int)

            acc = float(np.mean(pred_cls == y_test))

            del dtrain, dtest, model
        except Exception:
            acc = 0.0

        cache[key] = acc
        return acc

    # ══════════════════════════════════════════════
    #  OPERADORES GENÉTICOS
    # ══════════════════════════════════════════════

    def _selecao_torneio(self, pop: list[np.ndarray],
                         fitness: np.ndarray) -> np.ndarray:
        """Seleciona um indivíduo via torneio de tamanho k."""
        idx = self.rng.choice(len(pop), size=TORNEIO_K, replace=False)
        vencedor = idx[int(np.argmax(fitness[idx]))]
        return pop[vencedor].copy()

    def _crossover_uniforme(self, pai1: np.ndarray,
                            pai2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crossover uniforme: cada gene vem de um pai aleatório."""
        mask = self.rng.randint(0, 2, len(pai1)).astype(np.int8)
        filho1 = np.where(mask, pai1, pai2)
        filho2 = np.where(mask, pai2, pai1)
        return filho1, filho2

    def _mutar(self, individuo: np.ndarray, n_features: int):
        """Flip de N_MUTACAO bits aleatórios (liga↔desliga)."""
        bits = self.rng.choice(n_features, size=N_MUTACAO, replace=False)
        for b in bits:
            individuo[b] = 1 - individuo[b]

    def _garantir_minimo(self, individuo: np.ndarray):
        """Se o cromossomo tem menos de MIN_FEATURES ativas, liga bits aleatórios."""
        ativas = int(np.sum(individuo))
        if ativas < MIN_FEATURES:
            desligadas = np.where(individuo == 0)[0]
            faltam = MIN_FEATURES - ativas
            ligar = self.rng.choice(
                desligadas, size=min(faltam, len(desligadas)), replace=False
            )
            for i in ligar:
                individuo[i] = 1

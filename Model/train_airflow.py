"""
train_airflow.py -- Treinamento do modelo DEDICADO (5 features de negocio)
com K-Fold Cross Validation.

Baseado no train.py (versao real e atual do projeto). A UNICA diferenca
funcional em relacao ao train.py original esta na selecao de features:

    train.py (original):
        feats = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
        # usa TODAS as colunas da ABT (bureau, previous_application,
        # POS_CASH, installments, credit_card, etc. -- centenas de colunas)

    train_airflow.py (este arquivo):
        feats = [c for c in API_FEATURES if c in train_df.columns]
        # usa SOMENTE as 5 features de negocio definidas em API_FEATURES
        # (common/config_base.py) -- as mesmas que a API (FastAPI) vai
        # perguntar ao usuario no /predict.

Por que um modelo dedicado?
    Se a API so pergunta 5 features e o modelo foi treinado com ~200,
    e preciso "inventar" valores (ex: mediana do treino) para as demais
    colunas na hora de servir -- isso distorce a predicao real do cliente
    (o modelo aprendeu a usar sinais de bureau/POS/credit_card que a API
    nunca vai fornecer de verdade). Treinando um modelo SOMENTE com as 5
    features que a API realmente coleta, o AUC-OOF reportado aqui passa a
    ser EXATAMENTE o AUC que a API entrega em producao -- sem "mentira"
    estatistica.

Por que K-Fold? (igual ao train.py original)
    Usa TODOS os dados de treino para treinar e validar (em folds diferentes),
    dando uma estimativa mais robusta da performance real.
    A media das predicoes dos N folds tambem reduz variancia.

AirFlow via PythonOperator:
    task = PythonOperator(task_id="train_airflow", python_callable=run)
"""

import gc
import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import json
import joblib
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ABT_DATA_PATH, DATA_DIR,
    NUM_FOLDS, STRATIFIED, RANDOM_STATE,
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS, LOG_PERIOD,
    API_FEATURES,                    # NOVO: 5 features de negocio da API
    MODEL_PKL_PATH_5FEATS,           # NOVO: artefato do modelo dedicado
    SUBMISSION_PATH_5FEATS,          # NOVO: submissao do modelo dedicado
    FEATURE_IMPORTANCE_PATH_5FEATS,  # NOVO: importancia de features dedicada
)

# Caminho do JSON gerado pelo tune.py (mesmo arquivo do train.py original --
# reaproveita os hiperparametros ja tunados, ja que a arquitetura do LightGBM
# nao muda, so muda a QUANTIDADE de features usadas)
BEST_PARAMS_JSON = os.path.join(DATA_DIR, "best_params.json")


# ============================================================
# CARREGAMENTO DE PARAMETROS (JSON -> config.py)
# ============================================================
# Identico ao train.py original -- mantido aqui para o modulo ser
# autossuficiente (nao depender de importar train.py).

def load_lgbm_params() -> dict:
    """
    Carrega os parametros do LightGBM com a seguinte prioridade:

    1. best_params.json (gerado pelo tune.py) -- se existir e for valido
    2. LGBM_PARAMS do config.py              -- fallback padrao
    """
    if os.path.exists(BEST_PARAMS_JSON):
        try:
            with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            params = data.get("params", {})
            meta   = data.get("meta", {})

            if not params:
                raise ValueError("Chave 'params' ausente ou vazia no JSON.")

            print(f"[train_airflow] Parametros carregados de: {BEST_PARAMS_JSON}")
            print(f"                 AUC estimado (1 fold): {meta.get('best_auc_1fold', '?')}")
            print(f"                 Gerado em:             {meta.get('tuned_at', '?')}")
            return params

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[train_airflow] Aviso: falha ao ler {BEST_PARAMS_JSON} ({e}).")
            print(f"                 Usando LGBM_PARAMS do config.py como fallback.")

    else:
        print(f"[train_airflow] {BEST_PARAMS_JSON} nao encontrado.")
        print(f"                 Usando LGBM_PARAMS do config.py.")

    return LGBM_PARAMS


# ============================================================
# TREINAMENTO COM K-FOLD CROSS VALIDATION (MODELO DEDICADO - 5 FEATURES)
# ============================================================

def kfold_lightgbm_5feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treina o LightGBM com K-Fold CV usando SOMENTE as features definidas
    em API_FEATURES (common/config_base.py) e retorna a importancia de
    features.

    Estrutura identica a kfold_lightgbm() do train.py original -- unica
    mudanca e a linha de selecao de `feats` (comentada abaixo).

    Retorna DataFrame com colunas: feature, importance, fold.
    """
    params = load_lgbm_params()

    # Separa treino (TARGET preenchido) de teste (TARGET = NaN)
    train_df = df[df["TARGET"].notnull()].copy()
    test_df  = df[df["TARGET"].isnull()].copy()

    # Sanitiza nomes de colunas: LightGBM nao aceita [ ] { } nos nomes
    # (gerados pelo pd.get_dummies -- ex: 'NAME_CONTRACT_STATUS_[Approved]')
    def clean_cols(frame):
        frame.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in frame.columns]
        return frame

    train_df = clean_cols(train_df)
    test_df  = clean_cols(test_df)

    # ---------------------------------------------------------------
    # UNICA DIFERENCA em relacao ao train.py original:
    #   original -> feats = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    #   aqui     -> feats = somente as 5 features de API_FEATURES
    # ---------------------------------------------------------------
    feats = [c for c in API_FEATURES if c in train_df.columns]
    faltando = [c for c in API_FEATURES if c not in train_df.columns]
    if faltando:
        raise ValueError(
            f"As seguintes API_FEATURES nao existem na ABT: {faltando}. "
            f"Confira o abt_transform.py ou ajuste API_FEATURES em "
            f"common/config_base.py."
        )
    
    # ---- MONOTONICIDADE (features DERIVADAS de API_FEATURES) ----
    # +1 -> feature sobe => prob. de default só pode subir ou ficar igual
    # -1 -> feature sobe => prob. de default só pode cair ou ficar igual
    MONO_MAP = {
        "PAYMENT_RATE":         +1,  # AMT_ANNUITY/AMT_CREDIT: parcela pesada rel. ao crédito -> mais risco
        "ANNUITY_INCOME_PERC":  +1,  # AMT_ANNUITY/AMT_INCOME_TOTAL: peso da parcela na renda -> mais risco (comentário explícito na ABT)
        "INCOME_CREDIT_PERC":   -1,  # AMT_INCOME_TOTAL/AMT_CREDIT: cliente mais "folgado" -> menos risco (comentário explícito na ABT)
        "DEBT_INCOME_RATIO":    +1,  # dívida bureau/renda: mais dívida relativa -> mais risco
        "DAYS_EMPLOYED_PERC":   -1,  # fração da vida empregado: mais estabilidade -> menos risco
    }
    monotone_constraints = [MONO_MAP.get(f, 0) for f in feats]

    n_restritas = sum(1 for v in monotone_constraints if v != 0)
    if n_restritas == 0:
        raise ValueError(
            "Nenhuma feature de MONO_MAP casou com API_FEATURES "
            f"({feats}). A monotonicidade seria ignorada silenciosamente. "
            "Ajuste as chaves de MONO_MAP para os nomes REAIS de API_FEATURES."
        )
    params = {**params, "monotone_constraints": monotone_constraints}
    print(f"[train_airflow] Monotonicidade: {n_restritas}/{len(feats)} features "
          f"restritas -> {dict(zip(feats, monotone_constraints))}")

    print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")
    print(f"Features usadas neste modelo dedicado ({len(feats)}): {feats}")
    del df; gc.collect()

    # Escolha da estrategia de fold (identico ao original)
    if STRATIFIED:
        folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        folds = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    # Acumula os modelos treinados em cada fold para poder serializa-los
    # ao final (necessario para servir o modelo depois via API/batch scoring).
    fold_models = []

    # ---- Loop de validacao cruzada (identico ao original) ----
    for fold_n, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x = train_df[feats].iloc[train_idx]
        train_y = train_df["TARGET"].iloc[train_idx]
        valid_x = train_df[feats].iloc[valid_idx]
        valid_y = train_df["TARGET"].iloc[valid_idx]

        clf = LGBMClassifier(**params)
        clf.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(period=LOG_PERIOD),
            ],
        )

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]

        sub_preds += clf.predict_proba(
            test_df[feats], num_iteration=clf.best_iteration_
        )[:, 1] / folds.n_splits

        fold_imp = pd.DataFrame({
            "feature":    feats,
            "importance": clf.feature_importances_,
            "fold":       fold_n + 1,
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)

        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        print(f"Fold {fold_n + 1:2d} | AUC: {fold_auc:.6f} | best iter: {clf.best_iteration_}")

        fold_models.append(clf)

        del train_x, train_y, valid_x, valid_y
        gc.collect()

    # AUC final: concatena todas as predicoes OOF -- estimativa mais honesta
    # e, neste modelo dedicado, e EXATAMENTE o AUC real que a API entrega.
    full_auc = roc_auc_score(train_df["TARGET"], oof_preds)
    print(f"\nAUC total (OOF) - modelo dedicado 5 features: {full_auc:.6f}")

    # Salva submissao (arquivo separado do modelo "completo" -- nao sobrescreve)
    test_df["TARGET"] = sub_preds
    test_df[["SK_ID_CURR", "TARGET"]].to_csv(SUBMISSION_PATH_5FEATS, index=False)
    print(f"Submissao salva em: {SUBMISSION_PATH_5FEATS}")

    # ------------------------------------------------------------------
    # Serializa o ENSEMBLE completo (todos os N modelos do K-Fold) em um
    # unico arquivo model_5feats.pkl via joblib -- e este arquivo que a
    # API (FastAPI) carrega para servir o modelo dedicado em producao.
    #
    # Por que nao precisamos de "feature_defaults" (mediana) aqui?
    #   Porque este modelo foi TREINADO so com as 5 features -- ele nunca
    #   espera nem depende de nenhuma outra coluna. Isso elimina o risco
    #   de mascarar o risco real do cliente completando ~200 colunas com
    #   valores "medios" ficticios (ver discussao sobre confiabilidade
    #   do modelo servido via FastAPI).
    # ------------------------------------------------------------------
    model_bundle = {
        "models": fold_models,          # lista com os N modelos (um por fold)
        "feats": feats,                 # ordem exata das 5 features esperadas
        "n_folds": folds.n_splits,
        "full_auc_oof": full_auc,       # metrica de referencia -- E a metrica real da API
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(model_bundle, MODEL_PKL_PATH_5FEATS)
    print(f"Modelo dedicado (ensemble de {folds.n_splits} folds, {len(feats)} features) "
          f"salvo em: {MODEL_PKL_PATH_5FEATS}")

    return feature_importance_df


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run():
    """
    Le a ABT, treina o modelo DEDICADO (5 features) e salva:
      - o ensemble completo (model_5feats.pkl) -- consumido pela API
      - a submissao (submission_5feats.csv)
      - a importancia de features (feature_importance_5feats.csv)

    Chamavel diretamente (python train_airflow.py) ou via
    AirFlow PythonOperator (task_id="train_airflow_5feats").
    """
    print("=== Iniciando treinamento (modelo dedicado - 5 features) ===")
    df = pd.read_csv(ABT_DATA_PATH)

    # Converte colunas object remanescentes (seguança antes do LightGBM)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # >>> Reduz memória: float64->float32 e int64->int32 (corta ~50% da RAM)
    # Mantém TARGET intacto (precisa de NaN p/ separar treino/teste).
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        if col != "TARGET":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    gc.collect()

    feat_importance = kfold_lightgbm_5feats(df)

    feat_importance.to_csv(FEATURE_IMPORTANCE_PATH_5FEATS, index=False)
    print(f"Importancia de features salva em {FEATURE_IMPORTANCE_PATH_5FEATS}")


if __name__ == "__main__":
    run()

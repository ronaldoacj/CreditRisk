"""
model/predict.py — Lógica PURA de predição do modelo LightGBM (Home Credit).

  - carregamento/cache do bundle de modelos (ensemble K-Fold)
  - engenharia de features derivadas (fórmulas idênticas ao abt_transform.py)
  - predict_default(): recebe features cruas e retorna (probabilidade, features usadas)
"""

import os

import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model_artifacts/model_5feats.pkl")
DAYS_PER_YEAR = 365.25

_models: Optional[List] = None
_feats: Optional[List[str]] = None


def get_model():
    """Carrega o bundle (ensemble K-Fold) uma vez e mantém em cache."""
    global _models, _feats
    if _models is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict):
            _models = bundle["models"]   # <- LISTA, não "model"
            _feats  = bundle["feats"]    # <- ['PAYMENT_RATE', ...]
        else:
            _models = [bundle]
            _feats  = None
    return _models, _feats


def div(a, b, eps=0.0):
    """Divisão robusta que reproduz o abt: inf/erro -> NaN."""
    try:
        if a is None or b is None:
            return np.nan
        denom = b + eps
        if denom == 0:
            return np.nan
        return a / denom
    except Exception:
        return np.nan


def _to_days(years):
    """
    Converte anos positivos (idade / tempo de emprego) para DAYS_* negativo,
    na MESMA convenção do dataset de treino (DAYS_BIRTH, DAYS_EMPLOYED).
    """
    if years is None:
        return None
    try:
        return int(round(-float(years) * DAYS_PER_YEAR))
    except Exception:
        return None


def engineer_features(raw: dict, feats: list) -> dict:
    """
    Converte os campos CRUS enviados pelo Streamlit nas features derivadas
    que o modelo espera. Fórmulas idênticas ao abt_transform.py.
    """
    # Payload já traz as features prontas? usa como está.
    if feats and all(f in raw for f in feats):
        return raw

    inc    = raw.get("AMT_INCOME_TOTAL")
    cred   = raw.get("AMT_CREDIT")
    ann    = raw.get("AMT_ANNUITY")

    # Fonte PRIMÁRIA (Streamlit): anos positivos, amigáveis ao usuário.
    dbirth = _to_days(raw.get("IDADE_ANOS"))
    demp   = _to_days(raw.get("ANOS_EMPREGO"))

    # Fallback: aceita DAYS_BIRTH/DAYS_EMPLOYED direto (ex.: outro consumidor
    # da API que já mande no formato original da ABT).
    if dbirth is None:
        dbirth = raw.get("DAYS_BIRTH")
    if demp is None:
        demp = raw.get("DAYS_EMPLOYED")

    # dívida do bureau: a UI pode ou não enviar -> DEBT_INCOME_RATIO fica NaN
    debt  = raw.get("BURO_AMT_CREDIT_SUM_DEBT_SUM")

    return {
        "PAYMENT_RATE":        div(ann, cred),     # AMT_ANNUITY / AMT_CREDIT
        "ANNUITY_INCOME_PERC": div(ann, inc),      # AMT_ANNUITY / AMT_INCOME_TOTAL
        "INCOME_CREDIT_PERC":  div(inc, cred),     # AMT_INCOME_TOTAL / AMT_CREDIT
        "DAYS_EMPLOYED_PERC":  div(demp, dbirth),  # DAYS_EMPLOYED / DAYS_BIRTH
        "DEBT_INCOME_RATIO":   div(debt, inc, eps=1e-5),
    }


def predict_default(raw_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Executa a predição completa a partir das features CRUAS do cliente.

    Retorna:
      (probabilidade_de_default, features_derivadas_usadas)

    Levanta exceção em caso de erro — quem chama (main.py) decide como
    traduzir isso em resposta HTTP.
    """
    models, feats = get_model()

    # 1) calcula (ou reaproveita) as features derivadas
    eng = engineer_features(raw_features, feats)

    # 2) DataFrame na ORDEM EXATA do treino (colunas faltantes viram NaN)
    if feats is not None:
        X = pd.DataFrame([{f: eng.get(f, np.nan) for f in feats}])[feats]
    else:
        X = pd.DataFrame([eng])

    # 3) ENSEMBLE: média do predict_proba dos N folds (igual ao OOF do treino)
    proba = float(np.mean([m.predict_proba(X)[:, 1][0] for m in models]))

    return proba, eng
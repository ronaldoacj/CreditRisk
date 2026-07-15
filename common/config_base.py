"""
common/config_base.py -- Fonte unica de verdade para todas as configuracoes
do pipeline (bruta -> clear -> abt -> model).

"""

import os

# ----------------------------------------------------------------------------
# CAMINHOS BASE
# ----------------------------------------------------------------------------
# DATA_ROOT e injetado via variavel de ambiente pelo docker-compose
# (ver docker-compose.yml -> environment: DATA_ROOT). Isso permite rodar o
# MESMO codigo local (sem Docker) e em container, so trocando a env var.
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_ROOT = os.path.abspath(DATA_ROOT)

# Camadas do pipeline (bruta -> clear -> abt -> model), espelhando o
# diagrama do projeto. Cada camada tem sua propria subpasta em /data.
RAW_DIR   = os.path.join(DATA_ROOT, "raw")     # bruta
CLEAN_DIR = os.path.join(DATA_ROOT, "clean")   # clear
ABT_DIR   = os.path.join(DATA_ROOT, "abt")     # abt
MODEL_DIR = os.path.join(DATA_ROOT, "model")   # model.pkl, submission.csv, etc.

# Alias usado pelo train.py/tune.py original (DATA_DIR = pasta de saida dos
# artefatos do modelo: best_params.json, model.pkl, submission.csv)
DATA_DIR = MODEL_DIR

for _dir in (RAW_DIR, CLEAN_DIR, ABT_DIR, MODEL_DIR):
    os.makedirs(_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# ARQUIVOS BRUTOS (camada "bruta") -- dataset Home Credit Default Risk
# ----------------------------------------------------------------------------
RAW_DATA = {
    "application_train":    os.path.join(RAW_DIR, "application_train.csv"),
    "application_test":     os.path.join(RAW_DIR, "application_test.csv"),
    "bureau":                os.path.join(RAW_DIR, "bureau.csv"),
    "bureau_balance":        os.path.join(RAW_DIR, "bureau_balance.csv"),
    "previous_application": os.path.join(RAW_DIR, "previous_application.csv"),
    "pos_cash":              os.path.join(RAW_DIR, "POS_CASH_balance.csv"),
    "installments":          os.path.join(RAW_DIR, "installments_payments.csv"),
    "credit_card":           os.path.join(RAW_DIR, "credit_card_balance.csv"),
}

# Limita quantidade de linhas lidas (util para testes rapidos locais).
# None = le tudo. Sobrescrever via env var permite rodar "modo debug" no
# Airflow sem alterar codigo (Variable/Param do Airflow no futuro).
NUM_ROWS = os.environ.get("NUM_ROWS")
NUM_ROWS = int(NUM_ROWS) if NUM_ROWS else None

# Valor sentinela conhecido do Home Credit para DAYS_EMPLOYED (365243 =
# "aposentado/nao empregado" codificado como um numero absurdo de dias).
SENTINEL_VALUE = 365243

# Colunas de data (em dias, valores negativos) da tabela previous_application
# que tambem podem conter o sentinela acima e precisam de tratamento.
PREV_APP_DATE_COLS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]

# ----------------------------------------------------------------------------
# ARQUIVOS INTERMEDIARIOS (camada "clear" e "abt")
# ----------------------------------------------------------------------------
CLEAN_DATA_PATH = os.path.join(CLEAN_DIR, "clean_data.csv")
ABT_DATA_PATH   = os.path.join(ABT_DIR, "abt.csv")

# ----------------------------------------------------------------------------
# IDENTIFICADORES E TARGET
# ----------------------------------------------------------------------------
ID_COLUMN     = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# Colunas que NUNCA devem entrar como feature do modelo
NON_FEATURE_COLS = [ID_COLUMN, TARGET_COLUMN]

# ----------------------------------------------------------------------------
# AGREGACOES (usadas pelo abt_transform.py)
# ----------------------------------------------------------------------------
BUREAU_NUM_AGG = {
    "DAYS_CREDIT":         ["mean", "var"],
    "DAYS_CREDIT_ENDDATE": ["mean"],
    "AMT_CREDIT_SUM":      ["mean", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
}

PREV_APP_NUM_AGG = {
    "AMT_ANNUITY":     ["mean", "max"],
    "AMT_CREDIT":      ["mean", "sum"],
    "APP_CREDIT_PERC": ["mean", "var"],
}

POS_CASH_AGG_BASE = {
    "MONTHS_BALANCE": ["max", "mean", "size"],
    "SK_DPD":         ["max", "mean"],
}

INSTALLMENTS_AGG_BASE = {
    "NUM_INSTALMENT_VERSION": ["nunique"],
    "DPD":          ["max", "mean", "sum"],
    "DBD":          ["max", "mean", "sum"],
    "PAYMENT_PERC": ["max", "mean", "var"],
    "PAYMENT_DIFF": ["max", "mean", "var"],
}

# Colunas ja cobertas explicitamente acima -- evita duplicar na agregacao
# generica "mean" de todas as colunas numericas restantes.
INSTALLMENTS_EXPLICIT_COLS = [
    "NUM_INSTALMENT_VERSION", "DPD", "DBD", "PAYMENT_PERC", "PAYMENT_DIFF",
]

# ----------------------------------------------------------------------------
# TREINAMENTO / CROSS VALIDATION
# ----------------------------------------------------------------------------
NUM_FOLDS      = int(os.environ.get("NUM_FOLDS", 5))
STRATIFIED     = True
RANDOM_STATE   = 42
EARLY_STOPPING_ROUNDS = 100
LOG_PERIOD     = 200

# Parametros fallback do LightGBM (usados se best_params.json nao existir,
# ou seja, se o tune.py ainda nao rodou)
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "n_estimators":     2000,
    "num_leaves":       34,
    "max_depth":        8,
    "min_child_samples": 70,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "reg_alpha":        0.05,
    "reg_lambda":       0.05,
    "learning_rate":    0.02,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ----------------------------------------------------------------------------
# ARTEFATOS DE SAIDA (camada "model")
# ----------------------------------------------------------------------------
SUBMISSION_PATH = os.path.join(MODEL_DIR, "submission.csv")

# NOVO (necessario para o deploy -- ver Model/PATCH_train_py.md):
# caminho onde o ensemble de modelos treinados sera serializado via joblib.
MODEL_PKL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")

# ----------------------------------------------------------------------------
# FEATURES EXPOSTAS PELA API (modelo dedicado -- ver Model/train_airflow.py)
# ----------------------------------------------------------------------------
# Sao as 5 features de negocio que a API (FastAPI) pede ao usuario no
# /predict. O train_airflow.py treina um modelo DEDICADO usando SOMENTE
# estas colunas, para que o AUC-OOF reportado no treino seja exatamente
# o AUC real entregue em producao.

API_FEATURES = [
    "PAYMENT_RATE",
    "ANNUITY_INCOME_PERC",
    "INCOME_CREDIT_PERC",
    "DEBT_INCOME_RATIO",
    "DAYS_EMPLOYED_PERC",
]

# Artefato de saida do modelo dedicado (nao sobrescreve o model.pkl completo)
MODEL_PKL_PATH_5FEATS = os.path.join(MODEL_DIR, "model_5feats.pkl")
SUBMISSION_PATH_5FEATS = os.path.join(MODEL_DIR, "submission_5feats.csv")
FEATURE_IMPORTANCE_PATH_5FEATS = os.path.join(MODEL_DIR, "feature_importance_5feats.csv")

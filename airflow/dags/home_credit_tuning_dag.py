"""
home_credit_tuning_dag.py -- DAG separada para a busca de hiperparametros (Optuna).

POR QUE UMA DAG SEPARADA DA DAG DE TREINO?
    O tuning (tune.py) e uma operacao PONTUAL e cara (30-90 min, dezenas de
    trials), enquanto o pipeline de treino e recorrente (ex: diario). Rodar
    o tuning automaticamente dentro do pipeline diario seria caro e
    desnecessario -- o resultado (best_params.json) muda pouco entre
    execucoes recorrentes.

    Por isso: schedule_interval=None (nao roda sozinha) -- o time aciona
    manualmente pela UI do Airflow (Trigger DAG) quando quiser re-otimizar
    hiperparametros (ex: apos mudanca grande na ABT ou degradacao de
    performance detectada no monitoramento).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from Model.tune import run as run_tune

default_args = {
    "owner": "data-science-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="home_credit_tuning",
    description="Busca de hiperparametros do LightGBM via Optuna (execucao manual)",
    default_args=default_args,
    schedule_interval=None,   # so roda via trigger manual na UI do Airflow
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["home-credit", "ml", "tuning"],
) as dag:

    task_tune = PythonOperator(
        task_id="tune_hyperparameters",
        python_callable=run_tune,
        # op_kwargs vira parametro de run(n_trials, timeout) do tune.py.
        # Para rodar com outros valores, use "Trigger DAG w/ config" na UI
        # do Airflow (melhoria futura: sobrescrever via params).
        op_kwargs={"n_trials": 50, "timeout": None},
        doc_md="Roda o Optuna e salva Model/data/model/best_params.json (lido automaticamente pelo train.py).",
    )

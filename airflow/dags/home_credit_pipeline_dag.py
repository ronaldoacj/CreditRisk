"""
home_credit_pipeline_dag.py -- Orquestra o pipeline completo de dados + treino.

Camadas (ver diagrama do projeto): bruta -> clear -> abt -> model

    sanitize_data  (data_sanitization.run)   bruta -> clear
        |
        v
    build_abt      (abt_transform.run)       clear -> abt
        |
        v
    train_model    (train.run)               abt -> model.pkl + submission.csv

"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Resolvidos via PYTHONPATH=/opt/airflow/project (docker-compose.yml), que
# aponta para a raiz do projeto onde ficam DataPipeline/ e Model/ (volumes).
from DataPipeline.data_sanitization import run as run_sanitization
from DataPipeline.abt_transform import run as run_abt_transform
#from Model.train import run as run_train
from Model.train_airflow import run as run_train_5feats

default_args = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False
}

with DAG(
    dag_id="home_credit_pipeline",
    description="Pipeline bruta -> clear -> abt -> treino do modelo LightGBM",
    default_args=default_args,
    schedule_interval="@daily",  # TODO: ajuste conforme frequencia real de chegada de dados
    start_date=datetime(2024, 1, 1),
    catchup=False,           # nao roda execucoes retroativas ao subir a DAG
    max_active_runs=1,       # evita 2 execucoes do pipeline em paralelo
    tags=["home-credit", "ml", "training"],
) as dag:

    task_sanitize = PythonOperator(
        task_id="sanitize_data",
        python_callable=run_sanitization,
        doc_md="Le os CSVs brutos (data/raw), limpa e salva clean_data.csv (data/clean).",
    )

    task_build_abt = PythonOperator(
        task_id="build_abt",
        python_callable=run_abt_transform,
        doc_md="Constroi a ABT (features agregadas de todas as tabelas) e salva abt.csv (data/abt).",
    )

    # task_train = PythonOperator(
    #     task_id="train_model",
    #     python_callable=run_train,
    #     doc_md=(
    #         "Treina o LightGBM com K-Fold CV, salva submission.csv, "
    #         "feature_importance.csv e model.pkl (data/model)."
    #     ),
    # )

    task_train_5feats = PythonOperator(
        task_id="train_model_5feats",
        python_callable=run_train_5feats,
        doc_md=(
            "Treina o modelo DEDICADO (somente as 5 features de negocio da API) "
            "com K-Fold CV. Salva model_5feats.pkl, submission_5feats.csv e "
            "feature_importance_5feats.csv (data/model). Roda em paralelo a "
            "train_model, pois ambos partem da mesma ABT."
        ),
    )

    task_sanitize >> task_build_abt >> task_train_5feats

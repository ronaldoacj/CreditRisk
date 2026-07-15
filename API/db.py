"""db.py — Camada de persistência (Postgres) da API Home Credit."""

import os
import time
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float,
    DateTime, insert,
)
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("uvicorn.error")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://appuser:apppass@app-postgres:5432/simulacoes",
)
DAYS_PER_YEAR = 365.25

_engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
_metadata = MetaData()

simulacoes = Table(
    "simulacoes", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("nome", String, nullable=False, index=True),
    Column("criado_em", DateTime(timezone=True), nullable=False),
    Column("renda_anual", Float),
    Column("valor_credito", Float),
    Column("anuidade", Float),
    Column("divida_outros_bancos", Float),
    Column("idade", Float),
    Column("anos_emprego", Float),
    Column("membros_familia", Float),
    Column("probabilidade_default", Float),
    Column("classe_risco", String),
    Column("limiar_usado", Float),
)

_table_ready = False


def init_db(retries: int = 10, delay: float = 3.0):
    """Cria a tabela 'simulacoes' com RETRY aguardando o Postgres subir."""
    global _table_ready
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            _metadata.create_all(_engine)
            _table_ready = True
            logger.info("Tabela 'simulacoes' pronta (tentativa %d).", attempt)
            return
        except SQLAlchemyError as e:
            last_err = e
            logger.warning(
                "Postgres ainda não disponível (tentativa %d/%d): %s",
                attempt, retries, e,
            )
            time.sleep(delay)
    logger.error("Não foi possível inicializar o banco após %d tentativas: %s",
                 retries, last_err)


def ensure_table():
    """Garante que a tabela exista ANTES de qualquer INSERT/SELECT."""
    global _table_ready
    if _table_ready:
        return
    _metadata.create_all(_engine)  # idempotente
    _table_ready = True


def _to_years(days):
    """Converte DAYS_* (negativo) do modelo de volta para anos positivos."""
    if days is None:
        return None
    try:
        return round(abs(float(days)) / DAYS_PER_YEAR, 1)
    except Exception:
        return None


def save_simulation(nome: str, raw: dict, proba: float, risk_class: str,
                    threshold: float):
    """Persiste a simulação no Postgres. Falha aqui NÃO derruba a predição."""
    try:
        ensure_table()

        idade = raw.get("IDADE_ANOS")
        if idade is None:
            idade = _to_years(raw.get("DAYS_BIRTH"))

        anos_emprego = raw.get("ANOS_EMPREGO")
        if anos_emprego is None:
            anos_emprego = _to_years(raw.get("DAYS_EMPLOYED"))

        row = {
            "nome": nome,
            "criado_em": datetime.now(timezone.utc),
            "renda_anual": raw.get("AMT_INCOME_TOTAL"),
            "valor_credito": raw.get("AMT_CREDIT"),
            "anuidade": raw.get("AMT_ANNUITY"),
            "divida_outros_bancos": raw.get("BURO_AMT_CREDIT_SUM_DEBT_SUM"),
            "idade": idade,
            "anos_emprego": anos_emprego,
            "membros_familia": raw.get("CNT_FAM_MEMBERS"),
            "probabilidade_default": proba,
            "classe_risco": risk_class,
            "limiar_usado": threshold,
        }
        with _engine.begin() as conn:
            conn.execute(insert(simulacoes).values(**row))
    except SQLAlchemyError as e:
        logger.warning("Falha ao salvar simulação no banco: %s", e)

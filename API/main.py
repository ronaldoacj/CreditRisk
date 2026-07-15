"""
main.py — API FastAPI que serve o modelo LightGBM treinado (Home Credit).

Endpoints:
  GET  /health              -> healthcheck (usado pelo docker-compose)
  POST /predict             -> recebe features do cliente, retorna a
                               probabilidade de default E persiste a
                               simulação no Postgres
  GET  /simulations?nome=.. -> lista as simulações salvas para um nome
                               (busca parcial, case-insensitive)

Camadas:
  - model/predict.py -> lógica pura de predição (modelo + features)
  - db.py            -> persistência no Postgres

Como rodar localmente:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db import _engine, simulacoes, init_db, ensure_table, save_simulation
from Model.predict import get_model, predict_default

app = FastAPI(
    title="Home Credit Default Risk - Prediction API",
    description="Serve o modelo LightGBM treinado via train.py",
    version="1.3.0",
)


@app.on_event("startup")
def _startup():
    init_db()


class PredictRequest(BaseModel):
    """
    Payload de entrada:
      - nome: identificador da pessoa analisada (obrigatório)
      - features: dicionário de features do cliente (schema da ABT)
    """
    nome: str = Field(..., min_length=1, description="Nome da pessoa analisada")
    features: Dict[str, Any] = Field(
        ..., description="Dicionário de features do cliente (mesmo schema da ABT)"
    )


class PredictResponse(BaseModel):
    probability_default: float
    risk_class: str
    threshold_used: float
    features_used: Dict[str, Any]


class SimulationOut(BaseModel):
    id: int
    nome: str
    criado_em: datetime
    renda_anual: Optional[float] = None
    valor_credito: Optional[float] = None
    anuidade: Optional[float] = None
    divida_outros_bancos: Optional[float] = None
    idade: Optional[float] = None
    anos_emprego: Optional[float] = None
    membros_familia: Optional[float] = None
    probabilidade_default: Optional[float] = None
    classe_risco: Optional[str] = None
    limiar_usado: Optional[float] = None


@app.get("/health")
def health():
    """Healthcheck simples usado no docker-compose."""
    try:
        models, feats = get_model()
        return {"status": "ok", "n_models": len(models), "feats": feats}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Recebe as features de um cliente, executa predict_proba do LightGBM,
    persiste a simulação no Postgres e retorna a probabilidade de default.
    """
    threshold = float(os.getenv("RISK_THRESHOLD", "0.15"))

    try:
        # Toda a manipulação da predição fica em model/predict.py
        proba, eng = predict_default(payload.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {e}")

    risk_class = "ALTO_RISCO" if proba >= threshold else "BAIXO_RISCO"

    # Persiste a simulação (usa os dados CRUS enviados pelo Streamlit)
    save_simulation(payload.nome, payload.features, proba, risk_class, threshold)

    return PredictResponse(
        probability_default=proba,
        risk_class=risk_class,
        threshold_used=threshold,
        features_used=eng,
    )


@app.get("/simulations", response_model=List[SimulationOut])
def list_simulations(nome: str):
    """
    Lista todas as simulações salvas para um nome (busca parcial,
    case-insensitive), da mais recente para a mais antiga.
    """
    if not nome or not nome.strip():
        raise HTTPException(status_code=400, detail="Parâmetro 'nome' é obrigatório.")
    try:
        ensure_table()  # garante que a tabela exista antes de consultar
        stmt = (
            select(simulacoes)
            .where(simulacoes.c.nome.ilike(f"%{nome.strip()}%"))
            .order_by(simulacoes.c.criado_em.desc())
        )
        with _engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"Erro ao consultar o banco: {e}")


@app.get("/")
def root():
    return {"message": "Home Credit Default Risk API - ver /docs para Swagger UI"}

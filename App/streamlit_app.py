"""
streamlit_app.py — Interface visual que consome a API FastAPI de predição.

Versão de produção: labels de negócio (sem nomes técnicos), campos de idade/
tempo de emprego em ANOS (sem números negativos) e campo de dívida atual para
alimentar o DEBT_INCOME_RATIO do modelo.

Novidades desta versão:
  - Campo "Nome da pessoa" (obrigatório) para identificar a análise.
  - Botão "Consultar" que lista todas as simulações já feitas para aquele nome.
"""

import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
DAYS_PER_YEAR = 365.25

st.set_page_config(page_title="Home Credit - Risco de Default", layout="centered")
st.title("🏦 Home Credit — Predição de Risco de Default")
st.caption("Preencha os dados do cliente para estimar a probabilidade de inadimplência.")

st.markdown("### Identificação")
nome = st.text_input(
    "Nome da pessoa analisada *",
    value="",
    help="Obrigatório. Usado para salvar e consultar as simulações desta pessoa.",
)

st.markdown("### Dados do cliente")

col1, col2 = st.columns(2)

with col1:
    amt_income = st.number_input(
        "Renda anual total (R$)",
        min_value=0.0, value=150_000.0, step=1_000.0, format="%.2f",
        help="Renda bruta anual declarada pelo cliente.",
    )
    amt_credit = st.number_input(
        "Valor do crédito solicitado (R$)",
        min_value=0.0, value=500_000.0, step=1_000.0, format="%.2f",
        help="Valor total do empréstimo/financiamento que o cliente está pedindo.",
    )
    amt_annuity = st.number_input(
        "Valor da parcela anual — anuidade (R$)",
        min_value=0.0, value=25_000.0, step=500.0, format="%.2f",
        help="Quanto o cliente pagará por ano. Parcelas altas frente à renda aumentam o risco.",
    )
    debt_other_banks = st.number_input(
        "Dívida atual em outros bancos (R$)",
        min_value=0.0, value=0.0, step=1_000.0, format="%.2f",
        help="Total já devido pelo cliente em outras instituições (histórico de crédito). "
             "Usado para calcular o comprometimento da renda com dívidas.",
    )

with col2:
    idade = st.number_input(
        "Idade do cliente (anos)",
        min_value=18, max_value=100, value=41, step=1,
        help="Idade atual em anos.",
    )
    anos_emprego = st.number_input(
        "Tempo no emprego atual (anos)",
        min_value=0.0, max_value=60.0, value=5.0, step=0.5, format="%.1f",
        help="Há quantos anos o cliente está no emprego atual.",
    )
    cnt_fam = st.number_input(
        "Membros na família",
        min_value=1, value=2, step=1,
        help="Número de pessoas na família, incluindo o cliente.",
    )

if st.button("🔍 Calcular risco", use_container_width=True):
    if not nome.strip():
        st.warning("⚠️ Informe o **nome da pessoa** antes de calcular o risco.")
    else:
        features = {
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "IDADE_ANOS": idade,
            "ANOS_EMPREGO": anos_emprego,
            "CNT_FAM_MEMBERS": cnt_fam,
            # Alimenta o DEBT_INCOME_RATIO (BURO_AMT_CREDIT_SUM_DEBT_SUM / AMT_INCOME_TOTAL).
            # Só é enviado quando > 0; caso contrário a API trata como sem histórico.
            "BURO_AMT_CREDIT_SUM_DEBT_SUM": debt_other_banks if debt_other_banks > 0 else None,
        }

        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"nome": nome.strip(), "features": features},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()

            proba = result["probability_default"]
            risk = result["risk_class"]

            st.markdown("### Resultado")
            st.metric("Probabilidade de inadimplência (default)", f"{proba:.2%}")

            if risk == "ALTO_RISCO":
                st.error("⚠️ Classificação: **ALTO RISCO**")
            else:
                st.success("✅ Classificação: **BAIXO RISCO**")

            st.caption(f"Limiar de decisão usado: {result.get('threshold_used', 0.5):.0%}")
            st.info(f"Simulação salva para **{nome.strip()}**.")

            # Transparência: mostra as features derivadas que o modelo realmente usou
            if "features_used" in result:
                with st.expander("Ver features usadas pelo modelo"):
                    st.json(result["features_used"])

        except requests.exceptions.HTTPError:
            st.error(f"Erro da API ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Não foi possível conectar à API: {e}")

# -------------------------------------------------------------------------
# Consulta de simulações anteriores por nome
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 Consultar simulações anteriores")
st.caption("Lista todas as simulações já feitas para o nome informado acima.")

if st.button("🔎 Consultar", use_container_width=True):
    if not nome.strip():
        st.warning("⚠️ Informe o **nome da pessoa** para consultar as simulações.")
    else:
        try:
            resp = requests.get(
                f"{API_URL}/simulations",
                params={"nome": nome.strip()},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                st.info(f"Nenhuma simulação encontrada para **{nome.strip()}**.")
            else:
                df = pd.DataFrame(data)

                # Renomeia colunas para exibição amigável
                colmap = {
                    "criado_em": "Data/Hora",
                    "nome": "Nome",
                    "renda_anual": "Renda anual (R$)",
                    "valor_credito": "Crédito solicitado (R$)",
                    "anuidade": "Anuidade (R$)",
                    "divida_outros_bancos": "Dívida outros bancos (R$)",
                    "idade": "Idade (anos)",
                    "anos_emprego": "Emprego (anos)",
                    "membros_familia": "Membros família",
                    "probabilidade_default": "Prob. default",
                    "classe_risco": "Classe de risco",
                    "limiar_usado": "Limiar",
                }
                cols_order = [c for c in colmap if c in df.columns]
                df = df[cols_order].rename(columns=colmap)

                if "Prob. default" in df.columns:
                    df["Prob. default"] = (df["Prob. default"] * 100).round(2).astype(str) + "%"

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"{len(data)} simulação(ões) encontrada(s).")

        except requests.exceptions.HTTPError:
            st.error(f"Erro da API ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Não foi possível conectar à API: {e}")

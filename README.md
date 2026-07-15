# 🏦 Home Credit Default Risk — Pipeline de ML End-to-End

Solução completa de Machine Learning para **predição de risco de inadimplência**
em concessão de crédito, cobrindo desde a ingestão dos dados brutos até o deploy
do modelo como serviço de predição — com orquestração, API, interface visual e
persistência de simulações.

---

## 📌 Índice

1. [Descrição do Projeto](#-descrição-do-projeto)
2. [Objetivo de Negócio](#-objetivo-de-negócio)
3. [Arquitetura da Solução](#-arquitetura-da-solução)
4. [Estrutura do Repositório](#-estrutura-do-repositório)
5. [Metodologia](#-metodologia)
6. [Como Treinar o Modelo](#-como-treinar-o-modelo)
7. [Como Executar o Serviço de Predição](#-como-executar-o-serviço-de-predição)
8. [Orquestração com Airflow](#-orquestração-com-airflow)
9. [Monitoramento de Dados e Modelo em Produção](#-monitoramento-de-dados-e-modelo-em-produção)
10. [Ações Automatizadas a partir das Previsões](#-ações-automatizadas-a-partir-das-previsões)
11. [Próximos Passos](#-próximos-passos)

---

## 📖 Descrição do Projeto

Este projeto implementa um pipeline de ML para o problema **Home Credit Default
Risk**: estimar a probabilidade de um cliente não honrar um empréstimo, a partir
de dados cadastrais, histórico de crédito (bureau), aplicações anteriores e
comportamento de pagamento.

O fluxo completo é:

**dados brutos → sanitização → ABT (Analytical Base Table) → tuning de
hiperparâmetros (Optuna) → treinamento (LightGBM + K-Fold CV) → serialização do
modelo → serviço de predição (FastAPI) → interface de simulação (Streamlit) →
persistência das simulações (PostgreSQL)** — tudo orquestrado pelo **Apache
Airflow** e containerizado com **Docker Compose**.

---

## 🎯 Objetivo de Negócio

Instituições de crédito perdem dinheiro de duas formas opostas:

| Risco | Consequência |
|---|---|
| **Aprovar clientes maus pagadores** | Perda direta por inadimplência |
| **Negar clientes bons pagadores** | Perda de receita e de market share |

O objetivo do modelo é **ordenar os clientes pelo risco de default** com a maior
capacidade discriminativa possível (AUC), permitindo que a área de crédito:

- **Aprove automaticamente** clientes de baixo risco (probabilidade < 15%);
- **Encaminhe para análise / negue** clientes de alto risco (probabilidade ≥ 15%);
- Reduza a perda esperada da carteira sem sacrificar volume de concessão.

> **Limiar de decisão (threshold): 15%** — definido na API (`API/main.py`).
> Clientes com probabilidade de default ≥ 0.15 são classificados como
> `ALTO_RISCO`; abaixo disso, `BAIXO_RISCO`.

---

## 🏗 Arquitetura da Solução

```
                        ┌─────────────────────  Apache Airflow (agendamento diário)  ─────────────────────┐
                        │                                                                                  │
 ┌──────────┐   ┌───────▼────────┐   ┌────────────────┐   ┌──────────────┐   ┌───────────────────────┐    │
 │ raw_data │──▶│ data_          │──▶│ abt_transform  │──▶│  tune.py     │──▶│  train.py /           │    │
 │  (CSVs)  │   │ sanitization.py│   │ .py  (ABT)     │   │  (Optuna,    │   │  train_airflow.py     │    │
 └──────────┘   └────────────────┘   └────────────────┘   │  sob demanda)│   │  (LightGBM K-Fold)    │    │
                                                          └──────────────┘   └───────────┬───────────┘    │
                                                                                          │ model .pkl     │
                        ┌─────────────────────────────────────────────────────────────────┘               │
                        ▼                                                                                  │
                ┌───────────────┐    POST /predict     ┌──────────────┐                                    │
                │   FastAPI     │◀────────────────────│  Streamlit    │  ◀── usuário final                 │
                │  (porta 8000) │────────────────────▶│  (porta 8501) │                                    │
                └───────┬───────┘    score + risco     └──────────────┘                                    │
                        │ persiste simulação                                                               │
                        ▼                                                                                  │
                ┌───────────────┐                                                                          │
                │  PostgreSQL   │  banco `simulacoes` (porta 5433 no host)                                 │
                └───────────────┘                                                                          │
```

### Componentes

| Componente | Tecnologia | Responsabilidade |
|---|---|---|
| **DataPipeline** | Python / Pandas | Limpeza (`data_sanitization.py`) e construção da ABT (`abt_transform.py`) |
| **Tuning** | Optuna (TPESampler + MedianPruner) | Busca de hiperparâmetros do LightGBM, salva `best_params.json` |
| **Treino** | LightGBM + StratifiedKFold (5 folds) | Treinamento com CV, early stopping e serialização do ensemble |
| **Orquestração** | Apache Airflow | DAG diária de retreino + DAG de tuning sob demanda |
| **API de predição** | FastAPI | Endpoints `/health`, `/predict`, `/simulations` |
| **Interface** | Streamlit | Formulário de simulação e consulta de histórico |
| **Persistência** | PostgreSQL | Armazena todas as simulações realizadas |
| **Infraestrutura** | Docker Compose | Sobe todos os serviços integrados |

---

## 📂 Estrutura do Repositório

```
HOME_CREDIT/
├── airflow/                     # DAGs do Airflow
│   ├── home_credit_pipeline_dag.py   # DAG diária: sanitização → ABT → treino
│   └── home_credit_tuning_dag.py     # DAG manual: tuning Optuna
├── API/                         # API de predição (FastAPI)
│   └── main.py                       # Endpoints + threshold de decisão (15%)
│   └── db.py                         # Camada de persistência (Postgres)
├── App/                         # Interface visual (Streamlit)
│   └── streamlit_app.py
├── common/                      # Código compartilhado (config_base.py)
├── data/                        # Dados: raw_data, clean_data, abt
├── DataPipeline/                # Scripts de dados
│   ├── data_sanitization.py          # Limpeza e padronização
│   └── abt_transform.py              # Construção da ABT
├── Model/                       # Treinamento
│   ├── config.py                     # Variáveis, parâmetros e metadados
│   ├── train.py                      # Treino completo (todas as features)
│   ├── train_airflow.py              # Treino do modelo dedicado à API (features reduzidas)
│   ├── predict.py                    # Serviço de Predição
│   ├── tune.py                       # Busca de hiperparâmetros (Optuna)
│   ├── Dockerfile
│   └── requirements.txt
├── model_artifacts/             # Modelos serializados (.pkl), best_params.json
├── docker-compose.yml           # Orquestra todos os serviços
├── Dockerfile.airflow
├── requirements-airflow.txt
├── payload.json                 # Exemplo de payload para o /predict
├── .env.example                 # Modelo de variáveis de ambiente
└── readme.md
```

---

## 🔬 Metodologia

### 1. Sanitização (`DataPipeline/data_sanitization.py`)
Limpeza e padronização dos dados brutos: tratamento de valores inválidos e
inconsistências, gerando os dados limpos (`clean_data`).

### 2. Construção da ABT (`DataPipeline/abt_transform.py`)
Consolida as múltiplas fontes (aplicação principal, bureau, aplicações
anteriores, POS/cash, installments) em uma única tabela analítica, com
agregações numéricas por cliente (médias, somas, variâncias — ex.:
`BURO_AMT_CREDIT_SUM_DEBT_SUM`) e encoding de variáveis categóricas.

### 3. Tuning de Hiperparâmetros (`Model/tune.py`)
- **Optuna** com `TPESampler` (aprende com trials anteriores) e `MedianPruner`
  (interrompe trials ruins cedo).
- Estratégia de avaliação rápida: **1 fold fixo** com early stopping por trial,
  permitindo explorar ~50 combinações em 30–90 min.
- Espaço de busca: `num_leaves`, `max_depth`, `min_child_samples`, `subsample`,
  `colsample_bytree`, `reg_alpha`, `reg_lambda`, `learning_rate`.
- Resultado salvo em `best_params.json`, lido automaticamente pelo `train.py`.

### 4. Treinamento (`Model/train.py`)
- **LightGBM** (gradient boosting) com **StratifiedKFold de 5 folds** — mantém a
  proporção do target em cada fold (dataset desbalanceado).
- **Early stopping** (100 rodadas) para evitar overfitting.
- **Predições out-of-fold (OOF)**: cada amostra é prevista pelo fold em que
  estava na validação → estimativa honesta do AUC, sem data leakage.
- O ensemble dos 5 folds é serializado (`.pkl`) junto com a importância de
  features (`feature_importance.csv`).

### 5. Modelo dedicado à API (`Model/train_airflow.py`)
Treina um modelo com **conjunto reduzido de features** (as mesmas que o usuário
informa no formulário do Streamlit), garantindo que o AUC OOF reportado seja
**exatamente** a performance que a API entrega. Gera `model_5feats.pkl`.

---

## 🏋️ Como Treinar o Modelo

### Via Airflow 

Com os containers de pé (ver seção seguinte), acesse a UI do Airflow e:

- A DAG **`home_credit_pipeline`** roda **diariamente** de forma automática
  (sanitização → ABT → treino);
- A DAG **`home_credit_tuning`** é acionada **manualmente** (Trigger DAG)
  quando desejar re-otimizar hiperparâmetros.

---

## 🚀 Como Executar o Serviço de Predição

### 1. Configuração inicial

```bash
# Copie o modelo de variáveis de ambiente e ajuste se necessário
cp .env.example .env
```

### 2. Suba a infraestrutura completa

```bash
docker-compose up -d --build
```

### 3. Serviços disponíveis

| Serviço | URL / Acesso |
|---|---|
| **FastAPI (docs interativas)** | http://localhost:8000/docs |
| **Streamlit (interface)** | http://localhost:8501 |
| **Postgres da aplicação** | `localhost:5433` — banco `simulacoes`, usuário `appuser`, senha `apppass` |

### 4. Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Healthcheck (usado pelo docker-compose para liberar dependências) |
| `POST` | `/predict` | Recebe as features do cliente, retorna probabilidade de default + classificação (`ALTO_RISCO` / `BAIXO_RISCO`, threshold de **15%**) e persiste a simulação no Postgres |
| `GET` | `/simulations?nome=...` | Lista as simulações salvas para um nome (busca parcial, case-insensitive) |

### 5. Exemplo de chamada ao `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
```

```json
{
  "nome": "Ronaldo",
  "features": {
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 500000.0,
    "AMT_ANNUITY": 25000.0,
    "IDADE_ANOS": 41,
    "ANOS_EMPREGO": 5.0,
    "CNT_FAM_MEMBERS": 2,
    "BURO_AMT_CREDIT_SUM_DEBT_SUM": null
  }
}
```

### 6. Consultar simulações persistidas

```bash
docker exec -it hc-app-postgres psql -U appuser -d simulacoes

\dt                          -- lista as tabelas
SELECT * FROM simulacoes;    -- ver todos os registros
```

---

## ⚙️ Orquestração com Airflow

| DAG | Agendamento | O que faz |
|---|---|---|
| `home_credit_pipeline` | **Diário** | Executa a esteira completa: `data_sanitization` → `abt_transform` → treino do modelo. Garante que o modelo servido reflita sempre os dados mais recentes. |
| `home_credit_tuning` | **Manual** (`schedule_interval=None`) | Roda a busca Optuna (50 trials) e salva `best_params.json`. Separada da DAG diária por ser uma operação pontual e cara (30–90 min) — é acionada quando há mudança grande na ABT ou degradação de performance detectada no monitoramento. |

---

## 📊 Monitoramento de Dados e Modelo em Produção

Para acompanhar a saúde do modelo e da solução em produção, foram definidas
**5 métricas de confiança & governança**. Para cada uma: o que é, onde é
aplicada, por que monitorar e com qual frequência.


### 1. AUC OOF 

| | |
|---|---|
| **O que é** | Área sob a curva ROC calculada sobre as predições *out-of-fold* do K-Fold CV — mede o poder discriminativo do modelo (capacidade de ordenar maus pagadores acima de bons pagadores). |
| **Onde é aplicada** | Calculada automaticamente em `Model/train.py` e `Model/train_airflow.py` ao final do loop de K-Fold, concatenando as predições OOF dos 5 folds. |
| **Por quê** | É a métrica-alvo do problema: como o retreino é **diário**, uma queda no AUC OOF sinaliza imediatamente que os dados novos degradaram o modelo (mudança de comportamento, problema na ABT, quebra de feature). |
| **Frequência** | **Diária** — a cada execução da DAG `home_credit_pipeline`. |
| **Alerta sugerido** | Queda > 0.02 em relação ao AUC de referência (baseline) → bloquear a promoção do novo modelo e investigar. |

### 2. Estabilidade (CV) 

| | |
|---|---|
| **O que é** | Consistência do AUC entre os 5 folds do K-Fold (desvio-padrão / amplitude dos AUCs por fold). |
| **Onde é aplicada** | Os AUCs por fold já são impressos em `Model/train.py` durante o treino (`Fold N | AUC: ...`). |
| **Por quê** | Variância alta entre folds indica modelo instável — sensível demais à amostra de treino. Um AUC médio bom com folds muito discrepantes não é confiável para produção. |
| **Frequência** | **Diária** — junto com cada retreino. |
| **Alerta sugerido** | Desvio-padrão do AUC entre folds > 0.01 → revisar features e regularização (ou acionar a DAG de tuning). |

### 3. Calibração 

| | |
|---|---|
| **O que é** | O quão próximas as probabilidades previstas estão da frequência real de inadimplência observada (reliability diagram / Brier score). Ex.: entre os clientes com score 15%, cerca de 15% deveriam de fato entrar em default. |
| **Onde será aplicada** | Sobre as predições OOF do treino diário e, em produção, comparando os scores salvos na tabela `simulacoes` (Postgres) com os desfechos reais quando disponíveis. |
| **Por quê** | O threshold de decisão da API é **15% de probabilidade** — se o modelo estiver mal calibrado, esse corte perde o significado de negócio e a régua de aprovação fica errada, mesmo com AUC alto. |
| **Frequência** | **Semanal** sobre as predições OOF; **mensal** contra desfechos reais de produção (que demoram a maturar). |
| **Alerta sugerido** | Aumento relevante do Brier score vs. baseline ou desvio sistemático no reliability diagram → recalibrar (ex.: Platt scaling / isotonic regression). |

### 4. PSI / Drift 

| | |
|---|---|
| **O que é** | *Population Stability Index* — mede o quanto a distribuição das features de entrada em produção se afastou da distribuição vista no treino. |
| **Onde será aplicada** | Comparando as features das requisições ao `/predict` (persistidas na tabela `simulacoes`) contra a distribuição de referência da ABT de treino. Também sobre a distribuição dos **scores** de saída. |
| **Por quê** | O modelo foi treinado numa "fotografia" da população. Se o perfil dos clientes muda (ex.: renda média das solicitações cai, crédito médio sobe), as predições perdem validade **antes** de o AUC cair — o PSI é o alarme antecipado de drift. |
| **Frequência** | **Semanal** por feature de entrada; consolidado **mensal** para relatório de governança. |
| **Alerta sugerido** | PSI < 0.10 → estável; 0.10–0.25 → atenção (investigar); > 0.25 → drift significativo → acionar retreino/tuning e revisar a ABT. |

### 5. Fairness por Subgrupo 

| | |
|---|---|
| **O que é** | Avaliação de desempenho (AUC, taxas de erro) e de taxa de classificação `ALTO_RISCO` segmentada por subgrupos de clientes — ex.: gênero (`CODE_GENDER`), faixas de idade (`IDADE_ANOS`), faixas de renda (`AMT_INCOME_TOTAL`). |
| **Onde será aplicada** | Sobre as predições OOF do treino diário (por subgrupo da ABT) e sobre as simulações reais registradas no Postgres. |
| **Por quê** | Crédito é um domínio regulado e sensível: o modelo não pode penalizar sistematicamente um subgrupo por atributos protegidos como **gênero** (viés discriminatório), o que gera risco reputacional e legal. Disparidades por subgrupo também costumam revelar problemas de representatividade nos dados de treino. |
| **Frequência** | **Mensal**, com revisão obrigatória a cada mudança relevante de modelo (novo tuning, mudança de features). |
| **Alerta sugerido** | Diferença de AUC > 0.05 entre gêneros, ou taxa de `ALTO_RISCO` desproporcional entre homens e mulheres sem justificativa nas variáveis de risco → análise de viés em manter o modelo em produção. |


### Métrica operacional complementar: Taxa de Acionamento

| | |
|---|---|
| **O que é** | Percentual de clientes classificados como `ALTO_RISCO` (probabilidade ≥ 15%) sobre o total de simulações. |
| **Onde será aplicada** | Consulta direta à tabela `simulacoes` no Postgres. |
| **Por quê** | Mudanças bruscas na taxa de acionamento (sem mudança de threshold) indicam drift de população ou problema no modelo — e impactam diretamente a operação de crédito (fila de análise manual, volume aprovado). |
| **Frequência** | **Diária**, junto com o dashboard de operação. |

---

## 🤖 Ações Automatizadas a partir das Previsões

Propostas de automação conectando **Machine Learning + automação + agentes de
IA** ao contexto de negócio:

| Gatilho | Ação automatizada |
|---|---|
| Score < 15% (`BAIXO_RISCO`) | **Aprovação automática** do crédito, sem intervenção humana — reduz tempo de resposta e custo operacional. |
| Score ≥ 15% (`ALTO_RISCO`) | Encaminhamento automático para **fila de análise manual**, com as features usadas e a explicação do score anexadas ao caso. |
| Score em zona limítrofe (ex.: 12–18%) | **Agente de IA** solicita automaticamente documentação complementar ao cliente (comprovante de renda, extratos) antes da decisão final. |
| AUC OOF diário cai abaixo do limite | Pipeline **bloqueia a promoção** do novo modelo e mantém o artefato anterior (`model_artifacts/`); alerta enviado ao time. |
| PSI > 0.25 em qualquer feature | Acionamento automático da DAG `home_credit_tuning` (re-otimização) seguida de retreino, + notificação ao time de dados. |
| Taxa de acionamento foge do padrão histórico | Alerta ao time de crédito para revisar a régua (threshold) e a origem das solicitações. |
| Mesmo cliente simula repetidamente e o score piora a cada tentativa (ex.: aumentando o valor solicitado) | **Agente de IA** analisa o histórico de simulações (`/simulations`), identifica qual variável está levando o score para `ALTO_RISCO` (ex.: valor do crédito muito alto para a renda) e sugere proativamente uma combinação de valor/prazo/anuidade que enquadraria a solicitação em `BAIXO_RISCO`, aumentando a conversão sem intervenção manual. |


---

*Projeto desenvolvido como entrega final do curso — LABDATA FIA.*

*Disponível em: https://github.com/ronaldoacj/CreditRisk/tree/individual*

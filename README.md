# 🎯 MLOps na Prática - Predição de Inadimplência

## O que é este projeto?

Este é um projeto de **MLOps completo** que demonstra como colocar um modelo de Machine Learning em produção usando Google Cloud Platform.

**Problema de negócio:** Uma empresa financeira precisa prever se um cliente vai se tornar inadimplente (deixar de pagar) para tomar decisões de crédito.

**Solução:** Um modelo de classificação binária que recebe dados do cliente e retorna a probabilidade de inadimplência.

---

## 🔄 Ciclo de Vida MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   📊 DADOS        →    🔧 PRÉ-PROC     →    🤖 TREINO      →    📈 AVALIA   │
│   (CSV/GCS)            (Limpeza)            (XGBoost)           (Métricas)  │
│                                                                             │
│       ↑                                                              ↓      │
│                                                                             │
│   📡 MONITORA     ←    🎯 SCORING      ←    🚀 DEPLOY      ←    📦 REGISTRO │
│   (Performance)        (Predições)         (Endpoint)          (MLflow)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cada etapa explicada:

| # | Etapa | O que faz | Arquivo |
|---|-------|-----------|---------|
| 1 | **Dados** | Carrega dados brutos de clientes | `data/*.csv` |
| 2 | **Pré-processamento** | Limpa dados, cria features, normaliza | `src/pre_processamento.py` |
| 3 | **Treinamento** | Treina modelos (XGBoost, RandomForest) | `src/model_registry.py` |
| 4 | **Avaliação** | Calcula accuracy, F1-score, AUC | `src/model_registry.py` |
| 5 | **Registro** | Versiona modelo no MLflow | `src/model_registry.py` |
| 6 | **Deploy** | Cria API REST no Vertex AI | `deploy/02_deploy_endpoint.py` |
| 7 | **Scoring** | Faz predições em novos dados | `src/pipeline_scoring.py` |

---

## 📁 Estrutura do Projeto

```
MLFlow-GCP/
│
├── data/                          # Dados
│   ├── base_clientes_inadimplencia.csv    # Treino (10k clientes)
│   └── base_clientes_inadimplencia_2.csv  # Scoring (novos clientes)
│
├── src/                           # Código principal
│   ├── pre_processamento.py       # Limpeza e feature engineering
│   ├── model_registry.py          # Treina e registra no MLflow
│   ├── pipeline_scoring.py        # Faz predições
│   └── scoring_model_final.py     # Scoring alternativo
│
├── deploy/                        # Deploy em Produção
│   ├── 01_upload_model_to_vertex.py   # Sobe modelo para Vertex AI
│   ├── 02_deploy_endpoint.py          # Cria API REST
│   ├── 03_test_endpoint.py            # Testa a API
│   └── 05_cleanup.py                  # Remove recursos (evita custos)
│
├── jobs/                          # Pipeline automatizado
│   └── vertex_pipeline.py         # Pipeline Kubeflow/Vertex AI
│
├── mlruns/                        # MLflow (tracking local)
│
└── requirements.txt               # Dependências Python
```

---

## 🚀 Como Executar

### Pré-requisitos

1. Python 3.10+
2. Conta no Google Cloud Platform
3. Projeto GCP com billing ativo

### Passo 1: Configurar ambiente

```bash
# Clonar repositório
git clone https://github.com/ViniciusCantanhede/MLFlow-GCP.git
cd MLFlow-GCP

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Passo 2: Configurar GCP

```bash
# Autenticar no GCP
gcloud auth login
gcloud auth application-default login

# Definir projeto
gcloud config set project mlops-484912
```

### Passo 3: Executar Pipeline de ML

```bash
# 1. Pré-processamento (limpa e transforma dados)
python src/pre_processamento.py

# 2. Treinamento (treina modelo e registra no MLflow)
python src/model_registry.py

# 3. Scoring (faz predições em novos dados)
python src/pipeline_scoring.py
```

### Passo 4: Deploy em Produção (opcional)

```bash
# 1. Upload do modelo para Vertex AI
python deploy/01_upload_model_to_vertex.py

# 2. Criar endpoint REST (demora ~10 min)
python deploy/02_deploy_endpoint.py

# 3. Testar endpoint
python deploy/03_test_endpoint.py

# 4. IMPORTANTE: Limpar recursos para não gastar dinheiro!
python deploy/05_cleanup.py
```

---

## 🛠️ Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
|-----------|------------|-----|
| **Cloud** | Google Cloud Platform | Infraestrutura |
| **Storage** | Google Cloud Storage | Armazenar dados/modelos |
| **ML Platform** | Vertex AI | Deploy e endpoints |
| **Experiment Tracking** | MLflow | Versionar experimentos |
| **Pipeline** | Kubeflow Pipelines | Orquestração |
| **ML** | XGBoost, scikit-learn | Algoritmos |
| **Python** | pandas, numpy | Manipulação de dados |

---

## 📊 Métricas do Modelo

Os modelos treinados alcançam aproximadamente:

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| XGBoost | ~85% | ~0.84 | ~0.90 |
| RandomForest | ~83% | ~0.82 | ~0.88 |

---

## 🔑 Conceitos MLOps para Entrevistas

### O que é MLOps?
MLOps (Machine Learning Operations) combina práticas de DevOps com Machine Learning para automatizar o ciclo de vida de modelos em produção.

### Diferença Dev vs Prod

| Aspecto | Desenvolvimento | Produção |
|---------|-----------------|----------|
| Dados | Estáticos (CSV) | Streaming/Batch |
| Modelo | Notebook | API REST |
| Tracking | Local | Servidor MLflow |
| Infra | Laptop | Cloud (auto-scaling) |
| Monitoramento | Nenhum | Alertas, dashboards |

### Por que MLflow?
- **Tracking**: Registra métricas, parâmetros, artefatos
- **Registry**: Versiona modelos (v1, v2, staging, prod)
- **Reprodutibilidade**: Qualquer pessoa pode recriar o experimento

### Por que Vertex AI?
- **Integrado com GCP**: IAM, logging, monitoring
- **Endpoints**: API REST com auto-scaling
- **Pipelines**: Orquestração serverless
- **Feature Store**: Features consistentes

---

## 💰 Custos GCP

| Recurso | Custo | Nota |
|---------|-------|------|
| Cloud Storage | ~$0.02/GB/mês | Dados |
| Vertex AI Endpoint | ~$0.10/hora | Por réplica |
| Batch Prediction | ~$0.0001/predição | Scoring |

⚠️ **Importante**: Execute `deploy/05_cleanup.py` ao terminar para evitar cobranças!

---

## 📚 Referências

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)

---

## 👤 Autor

**Vinicius Cantanhede**

- GitHub: [@ViniciusCantanhede](https://github.com/ViniciusCantanhede)


# 🎯 Projeto SPIN - Sistema de Predição de Inadimplência

> **Projeto de MLOps completo** usando Google Cloud Platform (Vertex AI, GCS) e MLflow.

---

## 📋 Índice

1. [Visão Geral](#-visão-geral)
2. [Arquitetura MLOps](#-arquitetura-mlops)
3. [Estrutura do Projeto](#-estrutura-do-projeto)
4. [Passo a Passo](#-passo-a-passo)
5. [Como Funciona Cada Etapa](#-como-funciona-cada-etapa)
6. [Executando o Projeto](#-executando-o-projeto)
7. [Conceitos para Entrevista](#-conceitos-para-entrevista)

---

## 🎯 Visão Geral

Este projeto implementa um **pipeline completo de Machine Learning** para prever inadimplência de clientes. 

**Objetivo:** Dado um cliente com suas características, prever se ele será inadimplente ou não.

### Stack Tecnológica

| Categoria | Ferramenta | Para que serve |
|-----------|------------|----------------|
| **Cloud** | Google Cloud Platform | Infraestrutura |
| **Storage** | Google Cloud Storage (GCS) | Armazenar dados e modelos |
| **ML Platform** | Vertex AI | Executar pipelines de ML |
| **Experiment Tracking** | MLflow | Rastrear experimentos e versionar modelos |
| **Pipeline** | Kubeflow Pipelines (KFP) | Orquestrar etapas do ML |
| **Linguagem** | Python 3.10+ | Desenvolvimento |
| **ML** | scikit-learn, XGBoost | Algoritmos de ML |

---

## 🏗️ Arquitetura MLOps

### O que é MLOps?

MLOps = **Machine Learning + DevOps**. É o conjunto de práticas para automatizar e monitorar o ciclo de vida de modelos de ML.

### Fluxo Completo

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CICLO DE VIDA MLOps                                │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐      ┌─────────────┐      ┌───────────┐      ┌──────────┐
    │  DADOS  │ ───▶ │ PRÉ-PROCESS │ ───▶ │ TREINO    │ ───▶ │ AVALIAÇÃO│
    │  (GCS)  │      │             │      │ (MLflow)  │      │          │
    └─────────┘      └─────────────┘      └───────────┘      └────┬─────┘
         ▲                                                        │
         │                                                        ▼
    ┌─────────┐      ┌─────────────┐      ┌───────────┐      ┌──────────┐
    │MONITORA │ ◀─── │ PREDIÇÕES   │ ◀─── │  DEPLOY   │ ◀─── │ REGISTRO │
    │ MENTO   │      │ (Scoring)   │      │(Vertex AI)│      │(Registry)│
    └─────────┘      └─────────────┘      └───────────┘      └──────────┘
         │                                                        
         └──────────────── RETREINO (se necessário) ─────────────┘
```

### Por que cada etapa é importante?

| Etapa | O que faz | Por que é importante |
|-------|-----------|---------------------|
| **Dados** | Armazena dados brutos | Fonte única de verdade |
| **Pré-processamento** | Limpa e transforma dados | Dados ruins = modelo ruim |
| **Treinamento** | Treina o modelo | Aprende padrões dos dados |
| **Avaliação** | Calcula métricas | Sabe se o modelo é bom |
| **Registro** | Versiona o modelo | Rastreabilidade e rollback |
| **Deploy** | Coloca em produção | Gera valor para o negócio |
| **Monitoramento** | Acompanha performance | Detecta degradação |

---

## 📁 Estrutura do Projeto

```
Projeto-SPIN/
│
├── 📂 data/                              # Dados do projeto
│   ├── base_clientes_inadimplencia.csv   # Dados para treino
│   └── base_clientes_inadimplencia_2.csv # Dados para scoring (produção)
│
├── 📂 src/                               # Código fonte principal
│   ├── pre_processamento.py              # ETL e Feature Engineering
│   ├── model_registry.py                 # Treino + Registro no MLflow
│   └── scoring_model_final.py            # Predições em batch
│
├── 📂 jobs/                              # Pipelines e automação
│   └── vertex_pipeline.py                # Pipeline Vertex AI (KFP)
│
├── 📂 notebooks/                         # Notebooks interativos
│   └── fluxo_completo_mlops.ipynb        # Tutorial completo
│
├── 📂 scripts/                           # Scripts auxiliares
│   └── upload_to_gcs.py                  # Upload para GCS
│
├── 📂 tests/                             # Testes unitários
│   ├── test_model.py
│   └── test_pre_processamento.py
│
├── requirements.txt                       # Dependências Python
└── README.md                              # Este arquivo
```

---

## 📚 Passo a Passo

### ⚙️ Configuração do Ambiente GCP

**Projeto GCP:** `mlops-484912`  
**Bucket GCS:** `meu-bucket-29061999`  
**Região:** `us-central1`

Os dados já estão no bucket: `gs://meu-bucket-29061999/data/`

---

### Passo 1️⃣: Instalar Dependências

```bash
# Entre na pasta do projeto
cd Projeto-SPIN

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

**O que está sendo instalado:**
- `pandas`, `numpy` - Manipulação de dados
- `scikit-learn`, `xgboost` - Algoritmos de ML
- `mlflow` - Tracking de experimentos
- `google-cloud-storage` - Acesso ao GCS
- `google-cloud-aiplatform` - Vertex AI
- `kfp` - Kubeflow Pipelines

---

### Passo 2️⃣: Entender os Dados

Os dados estão em dois lugares:
- **Local:** `data/base_clientes_inadimplencia.csv`
- **GCS:** `gs://meu-bucket-29061999/data/base_clientes_inadimplencia.csv`

| Arquivo | Descrição | Uso |
|---------|-----------|-----|
| `base_clientes_inadimplencia.csv` | Dados históricos **com label** | Treino do modelo |
| `base_clientes_inadimplencia_2.csv` | Novos dados **sem label** | Scoring em produção |

**Principais colunas:**

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `ID_Cliente` | int | Identificador único |
| `Status_Pagamento` | str | **TARGET** - Adimplente ou Inadimplente |
| `Valor_Contrato` | float | Valor do contrato |
| `Tempo_Assinatura` | int | Meses como cliente |
| `Valor_em_Aberto` | float | Valor pendente |

---

### Passo 3️⃣: Pré-processamento

```bash
cd src
python pre_processamento.py
```

**O que esse script faz:**

```
DADOS BRUTOS → TRATAMENTO → FEATURE ENGINEERING → ENCODING → NORMALIZAÇÃO → DADOS PRONTOS
```

1. **Carrega dados** do CSV (local ou GCS)
2. **Trata valores nulos:**
   - Numéricos: preenche com mediana
   - Categóricos: preenche com "desconhecido"
3. **Cria features (Feature Engineering):**
   - Calcula idade a partir da data de nascimento
   - Calcula tempo de assinatura em meses
   - Calcula dias em atraso
4. **Codifica categóricas:**
   - One-hot encoding para variáveis com poucas categorias
   - Frequency encoding para alta cardinalidade (ex: cidade)
5. **Normaliza** valores numéricos (StandardScaler)
6. **Salva** `df_transformado.csv`

**Output:** Arquivo `df_transformado.csv` pronto para treino

---

### Passo 4️⃣: Treinamento com MLflow

```bash
python model_registry.py
```

**O que esse script faz:**

```
DADOS PROCESSADOS → SPLIT → TREINO → MÉTRICAS → REGISTRO MLFLOW
```

1. **Carrega** dados processados
2. **Divide** em treino (80%) e teste (20%)
3. **Treina 2 modelos:**
   - XGBoost (gradient boosting)
   - Random Forest (ensemble de árvores)
4. **Calcula métricas:**
   - Accuracy, Precision, Recall, F1-Score
5. **Registra no MLflow:**
   - Parâmetros do modelo
   - Métricas de avaliação
   - Modelo serializado
6. **Versiona** no Model Registry

**O que é MLflow?**

MLflow é a ferramenta padrão de mercado para rastrear experimentos de ML:

```python
# Exemplo simplificado
with mlflow.start_run():
    # Log parâmetros
    mlflow.log_param("model_type", "XGBoost")
    
    # Treina
    model.fit(X_train, y_train)
    
    # Log métricas
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.82)
    
    # Salva modelo
    mlflow.sklearn.log_model(model, "model")
```

**Por que usar MLflow?**
- ✅ Compara diferentes experimentos
- ✅ Reproduz resultados
- ✅ Versiona modelos
- ✅ Deploy fácil

---

### Passo 5️⃣: Scoring em Batch (Produção)

```bash
python scoring_model_final.py \
    --model-name ModelRFC-GCP \
    --model-version 1 \
    --input-csv ../data/base_clientes_inadimplencia_2.csv \
    --upload-output true
```

**O que esse script faz:**

```
MODELO (Registry) + NOVOS DADOS → PREDIÇÕES → SALVA RESULTADO
```

1. **Carrega modelo** do MLflow Registry
2. **Lê novos dados** (local ou GCS)
3. **Aplica modelo** - gera predições
4. **Salva resultados** (local e/ou GCS)

**Parâmetros:**

| Parâmetro | Descrição | Exemplo |
|-----------|-----------|---------|
| `--model-name` | Nome do modelo no Registry | `ModelRFC-GCP` |
| `--model-version` | Versão do modelo | `1` |
| `--input-csv` | Dados para scoring | `gs://bucket/data/novos.csv` |
| `--upload-output` | Fazer upload para GCS? | `true` |

**Output:** CSV com predições (cliente + probabilidade de inadimplência)

---

### Passo 6️⃣: Pipeline Automatizado (Vertex AI)

> ⚠️ **Este passo precisa rodar no GCP** (Cloud Shell ou Vertex AI Workbench)

```bash
cd jobs
python vertex_pipeline.py
```

**O que esse script faz:**

```
DEFINE COMPONENTES → COMPILA PIPELINE → SUBMETE PARA VERTEX AI
```

1. **Define componentes** (cada etapa é um container):
   - `preprocessamento`: limpa dados
   - `treinamento`: treina modelo
   - `scoring`: faz predições
2. **Conecta componentes** (output de um → input do próximo)
3. **Compila** para formato Vertex AI
4. **Submete** o job

**Acompanhe a execução:**
- Console: https://console.cloud.google.com/vertex-ai/pipelines

**Por que usar Pipeline?**
- ✅ **Reprodutibilidade:** Mesmo código = mesmo resultado
- ✅ **Automação:** Pode agendar (ex: todo dia às 6h)
- ✅ **Escalabilidade:** Roda em máquinas potentes
- ✅ **Rastreabilidade:** Log de tudo que rodou

---

## 🔍 Como Funciona Cada Etapa

### Diagrama de Fluxo de Dados

```
                    ┌────────────────────────────────────────┐
                    │           Google Cloud Storage          │
                    │  gs://meu-bucket-29061999/              │
                    └───────────────┬────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │   data/   │   │  mlflow/  │   │predictions│
            │  (CSVs)   │   │ (modelos) │   │ (outputs) │
            └─────┬─────┘   └─────┬─────┘   └─────▲─────┘
                  │               │               │
                  │               │               │
    ┌─────────────┼───────────────┼───────────────┼─────────────┐
    │             │     PIPELINE  │               │             │
    │             ▼               ▼               │             │
    │    ┌────────────────┐  ┌─────────┐  ┌──────┴──────┐      │
    │    │ Pré-processamento│─▶│ Treino  │─▶│  Scoring    │      │
    │    └────────────────┘  └────┬────┘  └─────────────┘      │
    │                             │                             │
    │                             ▼                             │
    │                      ┌─────────────┐                      │
    │                      │   MLflow    │                      │
    │                      │  Registry   │                      │
    │                      └─────────────┘                      │
    └───────────────────────────────────────────────────────────┘
```

### Fluxo Detalhado

```
1. DADOS BRUTOS (GCS)
   └─▶ base_clientes_inadimplencia.csv
       • 10.000 registros
       • 20 colunas
       • Contém valores nulos
       • Variáveis categóricas como strings

2. PRÉ-PROCESSAMENTO
   └─▶ df_transformado.csv
       • Nulos tratados
       • Features criadas (idade, tempo_assinatura)
       • Categóricas codificadas
       • Valores normalizados

3. TREINAMENTO
   └─▶ MLflow Experiment
       • XGBoost: accuracy=0.85, f1=0.82
       • RandomForest: accuracy=0.83, f1=0.80
       • Modelo campeão: XGBoost

4. REGISTRO
   └─▶ MLflow Model Registry
       • ModelXGB-GCP v1 (Staging)
       • ModelRFC-GCP v1 (Production)

5. SCORING
   └─▶ predicoes_inadimplencia.csv
       • ID_Cliente
       • prediction (0 ou 1)
       • probability (0.0 a 1.0)
```

---

## 🚀 Executando o Projeto

### Opção A: Local (Desenvolvimento)

Ideal para testar e desenvolver:

```bash
# 1. Pré-processamento
cd src
python pre_processamento.py

# 2. Treinamento (MLflow salva local em ./mlruns)
python model_registry.py

# 3. Scoring
python scoring_model_final.py \
    --model-name ModelRFC-GCP \
    --model-version 1 \
    --input-csv ../data/base_clientes_inadimplencia_2.csv
```

### Opção B: Cloud Shell (Recomendado)

1. Acesse: https://shell.cloud.google.com
2. Clone o projeto:
   ```bash
   git clone <seu-repo>
   cd Projeto-SPIN
   pip install -r requirements.txt
   ```
3. Execute os scripts

### Opção C: Notebook Interativo

Abra e execute: `notebooks/fluxo_completo_mlops.ipynb`

Este notebook tem todo o fluxo explicado passo a passo!

### Opção D: Pipeline Completo (Produção)

```bash
# No Cloud Shell ou Vertex AI Workbench
cd jobs
python vertex_pipeline.py
```

---

## 🎓 Conceitos para Entrevista

### Perguntas Frequentes e Respostas

---

**❓ "O que é MLOps?"**

> MLOps é a prática de aplicar princípios de DevOps ao ciclo de vida de Machine Learning. 
> Inclui:
> - Versionamento de dados e modelos
> - Automação de pipelines
> - Monitoramento de performance
> - CI/CD para ML

---

**❓ "Como você versiona modelos?"**

> Uso MLflow Model Registry. Cada modelo tem:
> - Nome único (ex: ModelRFC-GCP)
> - Múltiplas versões (1, 2, 3...)
> - Stages (Staging, Production)
> 
> Posso fazer rollback facilmente se uma versão nova tiver problemas.

---

**❓ "Como você sabe se um modelo está degradando?"**

> Monitoro três tipos de métricas:
> 1. **Negócio:** Taxa real de inadimplência vs predita
> 2. **Dados:** Data drift (distribuição das features mudando)
> 3. **Sistema:** Latência, throughput, erros

---

**❓ "O que é Feature Engineering?"**

> É criar novas variáveis a partir dos dados brutos que ajudam o modelo a aprender.
> 
> Exemplo neste projeto:
> - `Data_Nascimento` → `Idade`
> - `Data_Contratacao` → `Tempo_Assinatura_Meses`
> - `Data_Vencimento` + `Status` → `Dias_Atraso`

---

**❓ "Qual a diferença entre batch e real-time?"**

| Tipo | Quando usar | Exemplo | Latência |
|------|-------------|---------|----------|
| **Batch** | Muitos dados de uma vez | Scoring noturno | Minutos/horas |
| **Real-time** | Uma predição por vez | API de crédito | Milissegundos |
| **Streaming** | Dados contínuos | Fraude em tempo real | Segundos |

---

**❓ "Como você escolhe o melhor modelo?"**

> 1. Defino a métrica principal (F1-Score para classes desbalanceadas)
> 2. Treino vários modelos
> 3. Comparo métricas no MLflow
> 4. Considero também: interpretabilidade, custo computacional, latência

---

### Métricas de Avaliação

| Métrica | Fórmula | Quando usar |
|---------|---------|-------------|
| **Accuracy** | (TP+TN)/(Total) | Classes balanceadas |
| **Precision** | TP/(TP+FP) | Evitar falsos positivos |
| **Recall** | TP/(TP+FN) | Não perder positivos reais |
| **F1-Score** | 2*(P*R)/(P+R) | Equilíbrio |
| **ROC-AUC** | Área sob curva | Comparar modelos |

**Para inadimplência:** Priorizamos **Recall** (não queremos deixar passar inadimplentes) e **F1-Score** (equilíbrio geral).

---

### Checklist de MLOps Implementado

- [x] ✅ Dados versionados e armazenados (GCS)
- [x] ✅ Código versionado (Git)
- [x] ✅ Experimentos rastreados (MLflow)
- [x] ✅ Modelos versionados (Model Registry)
- [x] ✅ Pipeline automatizado (Vertex AI)
- [ ] 🔄 Monitoramento em produção (próximo passo)
- [ ] 🔄 CI/CD para retreino automático (próximo passo)

---

## 📚 Recursos Adicionais

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [Google Cloud Storage](https://cloud.google.com/storage/docs)

---

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

**Boa sorte na entrevista!** 🚀🎯


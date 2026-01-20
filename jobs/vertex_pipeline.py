"""
Vertex AI Pipeline - Projeto SPIN
==================================
Este arquivo define o pipeline de ML usando Kubeflow Pipelines (KFP) v2.

O que é um Pipeline de ML?
--------------------------
Um pipeline automatiza o fluxo de trabalho de ML:
1. Pré-processamento de dados
2. Treinamento do modelo
3. Avaliação
4. Deploy (opcional)

Por que usar pipelines?
-----------------------
- Reprodutibilidade: mesmo código, mesmo resultado
- Rastreabilidade: sabe exatamente o que rodou
- Automação: pode agendar execuções
- Escalabilidade: roda em máquinas potentes na cloud

IMPORTANTE PARA ENTREVISTA:
---------------------------
Este é o "ciclo completo de deploy" que a vaga menciona!
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform
import os

# ==================== CONFIGURAÇÕES ====================
PROJECT_ID = "mlops-484912"
REGION = "us-central1"
BUCKET_NAME = "meu-bucket-29061999"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"


# ==================== COMPONENTES DO PIPELINE ====================
# Cada componente é uma etapa isolada que pode rodar em containers diferentes

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "google-cloud-storage"]
)
def preprocessamento(
    input_gcs_path: str,
    output_dataset: Output[Dataset],
):
    """
    Componente 1: Pré-processamento dos dados.
    
    - Carrega dados brutos do GCS
    - Aplica transformações
    - Salva dados processados
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Carrega dados
    df = pd.read_csv(input_gcs_path)
    print(f"Dados carregados: {df.shape}")
    
    # Trata valores nulos
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna('desconhecido')
    
    # Calcula idade
    if "Data_Nascimento" in df.columns:
        df["Data_Nascimento"] = pd.to_datetime(df["Data_Nascimento"])
        ref = datetime.today()
        df["Idade"] = ref.year - df["Data_Nascimento"].dt.year
    
    # Remove colunas desnecessárias
    drop_cols = ["Telefone", "Nome", "Email", "Data_Nascimento"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Salva output
    df.to_csv(output_dataset.path, index=False)
    print(f"Dados processados salvos: {output_dataset.path}")


@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "mlflow", "google-cloud-storage"]
)
def treinamento(
    input_dataset: Input[Dataset],
    model_name: str,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
):
    """
    Componente 2: Treinamento do modelo.
    
    - Carrega dados processados
    - Treina modelo XGBoost
    - Calcula métricas
    - Salva modelo
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from xgboost import XGBClassifier
    import pickle
    import mlflow
    
    # Carrega dados
    df = pd.read_csv(input_dataset.path)
    
    # Prepara features e target
    target = "Status_Pagamento"
    if target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        raise ValueError(f"Coluna target '{target}' não encontrada")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treina modelo
    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    
    # Log métricas no Vertex AI
    for k, v in metrics.items():
        output_metrics.log_metric(k, v)
        print(f"{k}: {v:.4f}")
    
    # Salva modelo
    with open(output_model.path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Modelo salvo: {output_model.path}")


@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "google-cloud-storage"]
)
def scoring(
    input_dataset: Input[Dataset],
    input_model: Input[Model],
    output_predictions: Output[Dataset],
):
    """
    Componente 3: Scoring (predições em batch).
    
    - Carrega modelo treinado
    - Faz predições nos novos dados
    - Salva resultados
    """
    import pandas as pd
    import pickle
    
    # Carrega modelo
    with open(input_model.path, "rb") as f:
        model = pickle.load(f)
    
    # Carrega dados
    df = pd.read_csv(input_dataset.path)
    
    # Remove target se existir
    target = "Status_Pagamento"
    if target in df.columns:
        df_features = df.drop(columns=[target])
    else:
        df_features = df
    
    # Predições
    predictions = model.predict(df_features)
    probabilities = model.predict_proba(df_features)[:, 1]
    
    # Monta output
    df_out = df.copy()
    df_out["prediction"] = predictions
    df_out["probability"] = probabilities
    
    # Salva
    df_out.to_csv(output_predictions.path, index=False)
    print(f"Predições salvas: {output_predictions.path}")


# ==================== PIPELINE ====================
@dsl.pipeline(
    name="pipeline-inadimplencia-spin",
    description="Pipeline completo de ML: pré-processamento → treinamento → scoring"
)
def ml_pipeline(
    input_data_path: str = f"gs://{BUCKET_NAME}/data/base_clientes_inadimplencia.csv",
    model_name: str = "modelo-inadimplencia",
):
    """
    Pipeline principal de Machine Learning.
    
    Este é o fluxo completo que você precisa entender para a entrevista:
    
    1. PREPROCESSAMENTO: limpa e transforma dados brutos
    2. TREINAMENTO: treina modelo e calcula métricas
    3. SCORING: aplica modelo em novos dados
    
    Na vida real, você também teria:
    - Validação do modelo
    - Deploy para endpoint
    - Monitoramento
    """
    # Etapa 1: Pré-processamento
    preprocess_task = preprocessamento(
        input_gcs_path=input_data_path
    )
    
    # Etapa 2: Treinamento (depende do pré-processamento)
    train_task = treinamento(
        input_dataset=preprocess_task.outputs["output_dataset"],
        model_name=model_name,
    )
    
    # Etapa 3: Scoring (depende do treinamento)
    score_task = scoring(
        input_dataset=preprocess_task.outputs["output_dataset"],
        input_model=train_task.outputs["output_model"],
    )


# ==================== EXECUÇÃO ====================
def run_pipeline():
    """
    Executa o pipeline no Vertex AI.
    
    IMPORTANTE: Este código precisa rodar em uma máquina com acesso ao GCP.
    Pode ser:
    - Cloud Shell
    - Vertex AI Workbench (Notebook gerenciado)
    - Sua máquina local com gcloud configurado
    """
    # Inicializa Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}"
    )
    
    # Compila o pipeline
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="pipeline_inadimplencia.json"
    )
    print("Pipeline compilado: pipeline_inadimplencia.json")
    
    # Submete o job
    job = aiplatform.PipelineJob(
        display_name="pipeline-inadimplencia-run",
        template_path="pipeline_inadimplencia.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "input_data_path": f"gs://{BUCKET_NAME}/data/base_clientes_inadimplencia.csv",
            "model_name": "modelo-inadimplencia-v1",
        },
    )
    
    job.submit()
    print(f"Pipeline submetido! Acompanhe em: https://console.cloud.google.com/vertex-ai/pipelines")
    
    return job


if __name__ == "__main__":
    run_pipeline()

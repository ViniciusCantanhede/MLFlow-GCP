import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, recall_score, f1_score
from google.cloud import storage
from google.cloud import aiplatform

# ==================== CONFIGURAÇÕES GCP ====================
PROJECT_ID = "mlops-484912"
REGION = "us-central1"
BUCKET_NAME = "meu-bucket-29061999"

# Detecta diretório do projeto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# MLflow: usa tracking local para desenvolvimento
# Em produção, você usaria um servidor MLflow ou Vertex AI
MLFLOW_TRACKING_URI = os.path.join(PROJECT_DIR, "mlruns")

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_dados():
    logging.info("Carregando os dados pré-processados")
    csv_path = os.path.join(PROJECT_DIR, "df_transformado.csv")
    df_transformado = pd.read_csv(csv_path)
    return df_transformado

def split_dados(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    logging.info("Dividindo em amostras de treino 80% e teste 20%")
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def treinar_modelo_xgb(X_train, y_train, X_test, y_test, params=None):
    logging.info("Treinando modelo XGBoost")
    if params is None:
        params = {
            "objective": "binary:logistic",
            "use_label_encoder": False
        }
    model = xgb.XGBClassifier(**params, enable_categorical=True)
    model.fit(X_train, y_train) #treinamento
    y_pred = model.predict(X_test) #teste

    # Métricas importantes
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")

    return model, metrics

def treinar_modelo_rf(X_train, y_train, X_test, y_test, params=None):
    logging.info("Treinando modelo RandomForestClassifier")
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train) #treinamento
    y_pred = model.predict(X_test)  # predição

    # Métricas importantes
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")

    return model, metrics


def registra_mlflow_gcp(model, metrics, experiment_name="inadimplencia-rfc", tags=None, model_type="sklearn"):
    """
    Registra modelo no MLflow usando Google Cloud Storage como backend.
    
    Este é o padrão de MLOps para rastreabilidade de experimentos:
    - Tracking URI: onde ficam logs de métricas/params
    - Artifact URI: onde ficam os modelos serializados
    """
    logging.info("Registrando modelo no MLflow com GCS")
    
    # Configura MLflow para usar GCS como backend
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Cria/seleciona experimento
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log de parâmetros do modelo
        mlflow.log_param("model_type", "RandomForestClassifier" if model_type == "sklearn" else "XGBoostClassifier")
        mlflow.log_param("project_id", PROJECT_ID)
        mlflow.log_param("bucket", BUCKET_NAME)
        
        # Log de métricas - IMPORTANTE para comparar modelos
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Tags para organização e filtro
        if tags:
            mlflow.set_tags(tags)
        
        # Log do modelo serializado
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, "model_rfc", registered_model_name="ModelRFC-GCP")
        else:
            mlflow.xgboost.log_model(model, "model_xgb", registered_model_name="ModelXGB-GCP")
        
        logging.info(f"Modelo registrado com sucesso no experimento: {experiment_name}")


def registra_mlflow_gcp_xgb(model, metrics, experiment_name="inadimplencia-xgb", tags=None):
    """Registra modelo XGBoost no MLflow com GCS."""
    logging.info("Registrando modelo XGBoost no MLflow com GCS")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoostClassifier")
        mlflow.log_param("project_id", PROJECT_ID)
        
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if tags:
            mlflow.set_tags(tags)

        mlflow.xgboost.log_model(model, "model_xgb", registered_model_name="ModelXGB-GCP")


def upload_to_gcs(local_path: str, gcs_path: str):
    """
    Faz upload de arquivo local para o Google Cloud Storage.
    Útil para salvar datasets processados, modelos, ou outputs.
    """
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logging.info(f"Upload concluído: gs://{BUCKET_NAME}/{gcs_path}")


def download_from_gcs(gcs_path: str, local_path: str):
    """Download de arquivo do GCS para local."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    logging.info(f"Download concluído: {local_path}")

if __name__ == "__main__":
    # ==================== INICIALIZAÇÃO GCP ====================
    # Inicializa Vertex AI (equivalente ao MLClient do Azure)
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}"
    )
    logging.info(f"Vertex AI inicializado - Projeto: {PROJECT_ID}, Região: {REGION}")
    
    # ==================== PIPELINE DE TREINAMENTO ====================
    # 1. Carrega dados pré-processados
    df_transformado = carregar_dados()
    
    # Remove ID_Cliente se existir (não é feature)
    if 'ID_Cliente' in df_transformado.columns:
        df_transformado = df_transformado.drop(columns=['ID_Cliente'])
    
    # 2. Split treino/teste - target agora é "Inadimplente"
    X_train, X_test, y_train, y_test = split_dados(df_transformado, target="Inadimplente")
    
    # Mostra features que serão usadas
    logging.info(f"Features do modelo ({len(X_train.columns)}): {list(X_train.columns)}")
    
    # 3. Treina modelos (experimentos)
    model_xgb, metrics_xgb = treinar_modelo_xgb(X_train, y_train, X_test, y_test)
    model_rf, metrics_rf = treinar_modelo_rf(X_train, y_train, X_test, y_test)
    
    # 4. Registra no MLflow (rastreabilidade)
    registra_mlflow_gcp_xgb(model_xgb, metrics_xgb, experiment_name="inadimplencia-xgb")
    registra_mlflow_gcp(model_rf, metrics_rf, 
                        experiment_name="inadimplencia-rfc",
                        tags={"versao": "2.0", "pipeline": "producao", "cloud": "gcp"})

    logging.info("Pipeline de modelagem e registro concluído!")


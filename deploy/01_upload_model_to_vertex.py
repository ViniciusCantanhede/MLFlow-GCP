"""
PASSO 1: Upload do modelo para Vertex AI Model Registry
=========================================================
Este script pega o modelo treinado e registra no Vertex AI.
"""

import os
import sys
import joblib
import shutil
import pickle
from google.cloud import aiplatform
from google.cloud import storage

# Configurações
PROJECT_ID = "mlops-484912"
REGION = "us-central1"
BUCKET_NAME = "meu-bucket-29061999"
MODEL_DISPLAY_NAME = "modelo-inadimplencia-gcp"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MLRUNS_DIR = os.path.join(PROJECT_DIR, "mlruns")

# Adicionar src ao path
sys.path.append(os.path.join(PROJECT_DIR, "src"))


def find_latest_model():
    """Encontra o modelo mais recente no MLflow"""
    import mlflow
    mlflow.set_tracking_uri(MLRUNS_DIR)
    
    # Buscar todos os runs
    runs = mlflow.search_runs(
        experiment_ids=["0"],  # Default experiment
        order_by=["metrics.accuracy DESC"],
    )
    
    if runs.empty:
        raise Exception("Nenhum modelo encontrado no MLflow!")
    
    # Pegar o melhor run
    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    accuracy = best_run.get("metrics.accuracy", 0)
    f1 = best_run.get("metrics.f1_score", 0)
    
    print(f"✅ Melhor modelo encontrado:")
    print(f"   Run ID: {run_id}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Encontrar o arquivo do modelo
    artifacts_dir = os.path.join(MLRUNS_DIR, "0", run_id, "artifacts")
    
    # Procurar pelo modelo
    model_path = None
    for root, dirs, files in os.walk(artifacts_dir):
        for file in files:
            if file.endswith(('.pkl', '.joblib', 'model.pkl')):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break
    
    if not model_path:
        # Tentar carregar via MLflow
        model_uri = f"runs:/{run_id}/model"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            # Salvar temporariamente
            temp_path = os.path.join(PROJECT_DIR, "temp_model.pkl")
            joblib.dump(model, temp_path)
            model_path = temp_path
            print(f"   Modelo carregado via MLflow URI")
        except:
            raise Exception(f"Não foi possível encontrar o modelo para run {run_id}")
    
    return model_path, run_id, accuracy


def prepare_model_for_vertex(model_path, output_dir):
    """
    Prepara o modelo no formato que o Vertex AI espera.
    Vertex AI precisa de um diretório com model.pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copiar modelo
    dest_path = os.path.join(output_dir, "model.pkl")
    shutil.copy(model_path, dest_path)
    
    print(f"✅ Modelo preparado em: {output_dir}")
    return output_dir


def upload_to_gcs(local_dir, gcs_path):
    """Faz upload do modelo para o GCS"""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            blob_path = f"{gcs_path}/{file}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded: gs://{BUCKET_NAME}/{blob_path}")
    
    return f"gs://{BUCKET_NAME}/{gcs_path}"


def register_model_in_vertex(artifact_uri, model_name, accuracy):
    """Registra o modelo no Vertex AI Model Registry"""
    
    # Inicializar Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Upload do modelo
    # Usando container de sklearn pré-construído
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        description=f"Modelo de predição de inadimplência. Accuracy: {accuracy:.4f}",
        labels={
            "task": "classification",
            "framework": "sklearn",
            "accuracy": str(round(accuracy, 4)).replace(".", "_")
        }
    )
    
    print(f"\n🎉 Modelo registrado no Vertex AI!")
    print(f"   Model Resource Name: {model.resource_name}")
    print(f"   Model ID: {model.name}")
    
    return model


def main():
    print("=" * 60)
    print("🚀 UPLOAD DO MODELO PARA VERTEX AI MODEL REGISTRY")
    print("=" * 60)
    
    # 1. Encontrar o melhor modelo
    print("\n📌 Passo 1: Encontrando o melhor modelo...")
    model_path, run_id, accuracy = find_latest_model()
    print(f"   Modelo encontrado: {model_path}")
    
    # 2. Preparar modelo
    print("\n📌 Passo 2: Preparando modelo para Vertex AI...")
    temp_dir = os.path.join(PROJECT_DIR, "temp_model_vertex")
    prepare_model_for_vertex(model_path, temp_dir)
    
    # 3. Upload para GCS
    print("\n📌 Passo 3: Fazendo upload para GCS...")
    gcs_path = f"models/inadimplencia/{run_id}"
    artifact_uri = upload_to_gcs(temp_dir, gcs_path)
    
    # 4. Registrar no Vertex AI
    print("\n📌 Passo 4: Registrando no Vertex AI Model Registry...")
    model = register_model_in_vertex(artifact_uri, MODEL_DISPLAY_NAME, accuracy)
    
    # Limpar temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.exists(os.path.join(PROJECT_DIR, "temp_model.pkl")):
        os.remove(os.path.join(PROJECT_DIR, "temp_model.pkl"))
    
    print("\n" + "=" * 60)
    print("✅ MODELO REGISTRADO COM SUCESSO!")
    print("=" * 60)
    print(f"\nPróximo passo: Execute 02_deploy_endpoint.py para criar o endpoint")
    
    # Salvar model resource name para o próximo script
    with open(os.path.join(SCRIPT_DIR, ".model_resource_name"), "w") as f:
        f.write(model.resource_name)
    
    return model


if __name__ == "__main__":
    main()

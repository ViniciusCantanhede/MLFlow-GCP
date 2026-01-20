"""
DEPLOY ALTERNATIVO: Cloud Run com FastAPI
==========================================
Mais flexível que Vertex AI Endpoint - você controla as versões das libs.
"""

import os
import subprocess
import shutil

# Configurações
PROJECT_ID = "mlops-484912"
REGION = "us-central1"
SERVICE_NAME = "api-inadimplencia"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/{SERVICE_NAME}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def find_model():
    """Encontra o modelo treinado"""
    import mlflow
    
    mlruns_path = os.path.join(PROJECT_DIR, "mlruns")
    mlflow.set_tracking_uri(mlruns_path)
    
    # Buscar em todos os experiments
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if not runs.empty:
            best_run = runs.sort_values("metrics.accuracy", ascending=False).iloc[0]
            run_id = best_run["run_id"]
            
            # Tentar carregar modelo
            for artifact_name in ["model_rfc", "model_xgb", "model"]:
                try:
                    model_uri = f"runs:/{run_id}/{artifact_name}"
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"✅ Modelo carregado: {artifact_name}")
                    return model
                except:
                    try:
                        model = mlflow.xgboost.load_model(model_uri)
                        print(f"✅ Modelo XGBoost carregado: {artifact_name}")
                        return model
                    except:
                        continue
    
    raise Exception("Modelo não encontrado!")


def save_model_locally(model):
    """Salva modelo como pkl"""
    import joblib
    model_path = os.path.join(SCRIPT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo em: {model_path}")
    return model_path


def build_and_push_image():
    """Builda e envia imagem para Container Registry"""
    print("\n📦 Buildando imagem Docker...")
    
    # Copiar requirements para deploy
    shutil.copy(
        os.path.join(PROJECT_DIR, "requirements.txt"),
        os.path.join(SCRIPT_DIR, "requirements.txt")
    )
    
    # Build
    cmd_build = f"docker build -t {IMAGE_NAME} -f {SCRIPT_DIR}/Dockerfile {PROJECT_DIR}"
    print(f"   $ {cmd_build}")
    result = subprocess.run(cmd_build, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Erro no build: {result.stderr}")
        return False
    
    # Push
    print("\n📤 Enviando imagem para Container Registry...")
    cmd_push = f"docker push {IMAGE_NAME}"
    result = subprocess.run(cmd_push, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Erro no push: {result.stderr}")
        return False
    
    print(f"✅ Imagem enviada: {IMAGE_NAME}")
    return True


def deploy_to_cloud_run():
    """Deploy no Cloud Run"""
    print("\n🚀 Fazendo deploy no Cloud Run...")
    
    cmd = f"""
    gcloud run deploy {SERVICE_NAME} \
        --image {IMAGE_NAME} \
        --platform managed \
        --region {REGION} \
        --allow-unauthenticated \
        --memory 1Gi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 3 \
        --port 8080
    """
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Erro no deploy: {result.stderr}")
        return None
    
    # Pegar URL do serviço
    cmd_url = f"gcloud run services describe {SERVICE_NAME} --region {REGION} --format 'value(status.url)'"
    result = subprocess.run(cmd_url, shell=True, capture_output=True, text=True)
    
    url = result.stdout.strip()
    print(f"✅ Deploy concluído!")
    print(f"🔗 URL: {url}")
    
    return url


def main():
    print("=" * 60)
    print("🚀 DEPLOY VIA CLOUD RUN (FastAPI)")
    print("=" * 60)
    
    # 1. Encontrar e salvar modelo
    print("\n📌 Passo 1: Preparando modelo...")
    model = find_model()
    save_model_locally(model)
    
    # 2. Build Docker
    print("\n📌 Passo 2: Buildando container...")
    if not build_and_push_image():
        print("\n⚠️ Para buildar sem Docker local, use Cloud Build:")
        print(f"   gcloud builds submit --tag {IMAGE_NAME} .")
        return
    
    # 3. Deploy
    print("\n📌 Passo 3: Deploy no Cloud Run...")
    url = deploy_to_cloud_run()
    
    if url:
        print("\n" + "=" * 60)
        print("✅ API ONLINE!")
        print("=" * 60)
        print(f"\n🔗 URL Base: {url}")
        print(f"📚 Documentação: {url}/docs")
        print(f"❤️ Health Check: {url}/health")
        print(f"\n📝 Exemplo de chamada:")
        print(f'''
curl -X POST "{url}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"features": [35, 8000, 5, 15000, 12.5, 24, 750, 1, 1, 2]}}'
''')


if __name__ == "__main__":
    main()

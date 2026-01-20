"""
Script para fazer upload dos dados para o Google Cloud Storage.

COMO USAR:
-----------
Opção 1 - Pelo Console GCP (mais fácil):
    1. Acesse: https://console.cloud.google.com/storage/browser/meu-bucket-29061999
    2. Clique em "Upload" 
    3. Selecione os arquivos da pasta data/

Opção 2 - Pelo Cloud Shell (recomendado):
    1. Acesse: https://shell.cloud.google.com
    2. Clone seu repo ou faça upload dos arquivos
    3. Execute:
        gsutil cp data/*.csv gs://meu-bucket-29061999/data/

Opção 3 - Instalar gcloud CLI local:
    1. Baixe: https://cloud.google.com/sdk/docs/install
    2. Execute: gcloud init
    3. Execute: gsutil cp data/*.csv gs://meu-bucket-29061999/data/

Opção 4 - Via Python (este script):
    1. pip install google-cloud-storage
    2. Configure autenticação (veja abaixo)
    3. Execute este script
"""

from google.cloud import storage
import os

# Configurações
PROJECT_ID = "mlops-484912"
BUCKET_NAME = "meu-bucket-29061999"
LOCAL_DATA_DIR = "data"

def upload_files():
    """Faz upload de todos os CSVs para o GCS."""
    
    # Inicializa cliente
    # Nota: Requer autenticação configurada
    # export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
    # ou: gcloud auth application-default login
    
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        # Lista arquivos locais
        for filename in os.listdir(LOCAL_DATA_DIR):
            if filename.endswith('.csv'):
                local_path = os.path.join(LOCAL_DATA_DIR, filename)
                gcs_path = f"data/{filename}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                
                print(f"✅ Upload: {filename} → gs://{BUCKET_NAME}/{gcs_path}")
        
        print("\n🎉 Upload concluído!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n📋 Instruções de autenticação:")
        print("1. Crie uma Service Account no GCP Console")
        print("2. Baixe a chave JSON")
        print("3. Execute: export GOOGLE_APPLICATION_CREDENTIALS='path/to/key.json'")
        print("4. Rode este script novamente")


if __name__ == "__main__":
    upload_files()

"""
PASSO 2: Deploy do modelo como Endpoint REST
=============================================
Este script cria um endpoint no Vertex AI que serve o modelo como API.
"""

import os
from google.cloud import aiplatform

# Configurações
PROJECT_ID = "mlops-484912"
REGION = "us-central1"
ENDPOINT_DISPLAY_NAME = "endpoint-inadimplencia"

# Configurações de máquina (pode ajustar conforme necessidade)
MACHINE_TYPE = "n1-standard-2"  # 2 vCPUs, 7.5 GB RAM
MIN_REPLICAS = 1
MAX_REPLICAS = 3  # Auto-scaling até 3 réplicas

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model():
    """Busca o modelo registrado no Vertex AI"""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Tentar ler o resource name salvo
    resource_file = os.path.join(SCRIPT_DIR, ".model_resource_name")
    
    if os.path.exists(resource_file):
        with open(resource_file, "r") as f:
            resource_name = f.read().strip()
        model = aiplatform.Model(resource_name)
        print(f"✅ Modelo encontrado: {model.display_name}")
        return model
    
    # Se não existir, buscar pelo nome
    models = aiplatform.Model.list(
        filter=f'display_name="modelo-inadimplencia-gcp"',
        order_by="create_time desc"
    )
    
    if not models:
        raise Exception("Nenhum modelo encontrado! Execute 01_upload_model_to_vertex.py primeiro.")
    
    model = models[0]
    print(f"✅ Modelo encontrado: {model.display_name}")
    return model


def create_endpoint():
    """Cria um novo endpoint"""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Verificar se já existe
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'
    )
    
    if endpoints:
        print(f"⚠️ Endpoint já existe: {endpoints[0].resource_name}")
        return endpoints[0]
    
    # Criar novo endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        description="Endpoint para predição de inadimplência de clientes",
        labels={
            "task": "classification",
            "team": "mlops"
        }
    )
    
    print(f"✅ Endpoint criado: {endpoint.resource_name}")
    return endpoint


def deploy_model(model, endpoint):
    """Faz deploy do modelo no endpoint"""
    
    print(f"\n🚀 Iniciando deploy...")
    print(f"   Isso pode levar 5-15 minutos...")
    
    # Deploy
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="inadimplencia-v1",
        machine_type=MACHINE_TYPE,
        min_replica_count=MIN_REPLICAS,
        max_replica_count=MAX_REPLICAS,
        traffic_percentage=100,  # 100% do tráfego vai para este modelo
        sync=True  # Aguarda o deploy completar
    )
    
    print(f"\n🎉 Deploy concluído!")
    return endpoint


def main():
    print("=" * 60)
    print("🚀 DEPLOY DO MODELO COMO ENDPOINT REST")
    print("=" * 60)
    
    # 1. Buscar modelo
    print("\n📌 Passo 1: Buscando modelo no Vertex AI...")
    model = get_model()
    
    # 2. Criar endpoint
    print("\n📌 Passo 2: Criando endpoint...")
    endpoint = create_endpoint()
    
    # 3. Deploy
    print("\n📌 Passo 3: Fazendo deploy do modelo...")
    endpoint = deploy_model(model, endpoint)
    
    print("\n" + "=" * 60)
    print("✅ ENDPOINT PRONTO PARA USO!")
    print("=" * 60)
    
    print(f"\n📍 Endpoint Resource Name:")
    print(f"   {endpoint.resource_name}")
    
    print(f"\n🔗 Para fazer predições via API REST:")
    print(f"   POST https://{REGION}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict")
    
    print(f"\n📝 Próximo passo: Execute 03_test_endpoint.py para testar")
    
    # Salvar endpoint resource name
    with open(os.path.join(SCRIPT_DIR, ".endpoint_resource_name"), "w") as f:
        f.write(endpoint.resource_name)
    
    return endpoint


if __name__ == "__main__":
    main()

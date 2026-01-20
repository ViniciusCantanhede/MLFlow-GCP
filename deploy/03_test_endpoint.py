"""
PASSO 3: Testar o Endpoint
===========================
Este script testa o endpoint fazendo predições via API.
"""

import os
import json
import pandas as pd
from google.cloud import aiplatform

# Configurações
PROJECT_ID = "mlops-484912"
REGION = "us-central1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def get_endpoint():
    """Busca o endpoint criado"""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Ler resource name salvo
    resource_file = os.path.join(SCRIPT_DIR, ".endpoint_resource_name")
    
    if os.path.exists(resource_file):
        with open(resource_file, "r") as f:
            resource_name = f.read().strip()
        endpoint = aiplatform.Endpoint(resource_name)
        return endpoint
    
    # Buscar pelo nome
    endpoints = aiplatform.Endpoint.list(
        filter='display_name="endpoint-inadimplencia"'
    )
    
    if not endpoints:
        raise Exception("Endpoint não encontrado! Execute 02_deploy_endpoint.py primeiro.")
    
    return endpoints[0]


def prepare_test_data():
    """Prepara dados de teste (mesmas features que o modelo espera)"""
    
    # Features que o modelo usa (baseado no pre_processamento.py)
    # Você precisa ajustar baseado nas features reais do seu modelo
    
    test_samples = [
        # Cliente 1: Perfil de bom pagador
        {
            "idade": 35,
            "renda_mensal": 8000,
            "tempo_emprego": 5,
            "valor_emprestimo": 15000,
            "taxa_juros": 12.5,
            "num_parcelas": 24,
            "score_credito": 750,
            "possui_imovel": 1,
            "possui_veiculo": 1,
            "num_dependentes": 2
        },
        # Cliente 2: Perfil de risco
        {
            "idade": 22,
            "renda_mensal": 2500,
            "tempo_emprego": 0.5,
            "valor_emprestimo": 30000,
            "taxa_juros": 25.0,
            "num_parcelas": 48,
            "score_credito": 450,
            "possui_imovel": 0,
            "possui_veiculo": 0,
            "num_dependentes": 0
        },
        # Cliente 3: Perfil médio
        {
            "idade": 45,
            "renda_mensal": 5000,
            "tempo_emprego": 10,
            "valor_emprestimo": 20000,
            "taxa_juros": 18.0,
            "num_parcelas": 36,
            "score_credito": 600,
            "possui_imovel": 1,
            "possui_veiculo": 0,
            "num_dependentes": 3
        }
    ]
    
    return test_samples


def predict(endpoint, instances):
    """Faz predição via endpoint"""
    
    # Converter para formato que Vertex AI espera
    # Vertex AI espera lista de listas (matriz)
    instances_list = []
    for instance in instances:
        instances_list.append(list(instance.values()))
    
    # Fazer predição
    response = endpoint.predict(instances=instances_list)
    
    return response.predictions


def main():
    print("=" * 60)
    print("🧪 TESTANDO ENDPOINT DE PREDIÇÃO")
    print("=" * 60)
    
    # 1. Buscar endpoint
    print("\n📌 Passo 1: Conectando ao endpoint...")
    endpoint = get_endpoint()
    print(f"✅ Endpoint: {endpoint.display_name}")
    
    # 2. Preparar dados
    print("\n📌 Passo 2: Preparando dados de teste...")
    test_data = prepare_test_data()
    print(f"✅ {len(test_data)} amostras preparadas")
    
    # 3. Fazer predições
    print("\n📌 Passo 3: Fazendo predições...")
    
    try:
        predictions = predict(endpoint, test_data)
        
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DAS PREDIÇÕES")
        print("=" * 60)
        
        labels = ["✅ Adimplente", "⚠️ Inadimplente"]
        
        for i, (data, pred) in enumerate(zip(test_data, predictions)):
            print(f"\n{'='*40}")
            print(f"Cliente {i+1}:")
            print(f"  - Idade: {data['idade']}")
            print(f"  - Renda: R$ {data['renda_mensal']:,.2f}")
            print(f"  - Score Crédito: {data['score_credito']}")
            print(f"  - Valor Empréstimo: R$ {data['valor_emprestimo']:,.2f}")
            
            # Predição pode ser int ou lista de probabilidades
            if isinstance(pred, list):
                pred_class = 1 if pred[1] > 0.5 else 0
                prob = max(pred)
            else:
                pred_class = int(pred)
                prob = None
            
            print(f"\n  🎯 Predição: {labels[pred_class]}")
            if prob:
                print(f"  📊 Confiança: {prob:.2%}")
    
    except Exception as e:
        print(f"\n❌ Erro na predição: {e}")
        print("\n💡 Dica: Verifique se as features correspondem ao modelo treinado.")
        print("   As features precisam estar na mesma ordem e formato do treino.")
        
        # Mostrar como fazer via curl
        print("\n" + "=" * 60)
        print("📝 ALTERNATIVA: Testar via curl")
        print("=" * 60)
        print(f"""
curl -X POST \\
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \\
  -H "Content-Type: application/json" \\
  https://{REGION}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict \\
  -d '{{
    "instances": [[35, 8000, 5, 15000, 12.5, 24, 750, 1, 1, 2]]
  }}'
""")
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 60)


if __name__ == "__main__":
    main()

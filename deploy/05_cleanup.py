"""
PASSO 5: Limpeza de Recursos (para não gastar dinheiro!)
=========================================================
Execute este script quando terminar de testar para evitar custos.
"""

import os
from google.cloud import aiplatform

# Configurações
PROJECT_ID = "mlops-484912"
REGION = "us-central1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cleanup():
    """Remove endpoint e modelo para evitar custos"""
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    print("=" * 60)
    print("🧹 LIMPEZA DE RECURSOS")
    print("=" * 60)
    
    # 1. Remover endpoint (e undeploy modelos)
    print("\n📌 Passo 1: Removendo endpoints...")
    
    endpoints = aiplatform.Endpoint.list(
        filter='display_name="endpoint-inadimplencia"'
    )
    
    for endpoint in endpoints:
        print(f"   Removendo: {endpoint.display_name}")
        
        # Undeploy todos os modelos primeiro
        try:
            endpoint.undeploy_all()
        except:
            pass
        
        # Deletar endpoint
        endpoint.delete()
        print(f"   ✅ Endpoint removido")
    
    if not endpoints:
        print("   Nenhum endpoint encontrado")
    
    # 2. Remover modelos (opcional)
    print("\n📌 Passo 2: Removendo modelos...")
    
    models = aiplatform.Model.list(
        filter='display_name="modelo-inadimplencia-gcp"'
    )
    
    for model in models:
        print(f"   Removendo: {model.display_name}")
        model.delete()
        print(f"   ✅ Modelo removido")
    
    if not models:
        print("   Nenhum modelo encontrado")
    
    # 3. Limpar arquivos locais
    print("\n📌 Passo 3: Limpando arquivos temporários...")
    
    for f in [".model_resource_name", ".endpoint_resource_name"]:
        path = os.path.join(SCRIPT_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"   ✅ Removido: {f}")
    
    print("\n" + "=" * 60)
    print("✅ LIMPEZA CONCLUÍDA!")
    print("=" * 60)
    print("\n💡 Seus custos do GCP para este deploy foram zerados.")


if __name__ == "__main__":
    print("\n⚠️  ATENÇÃO: Este script vai REMOVER:")
    print("    - O endpoint (API de predição)")
    print("    - O modelo registrado no Vertex AI")
    print("\n    Os dados de treino e MLflow local NÃO serão afetados.")
    
    confirm = input("\nDigite 'sim' para confirmar: ").lower()
    
    if confirm == 'sim':
        cleanup()
    else:
        print("\n❌ Operação cancelada.")

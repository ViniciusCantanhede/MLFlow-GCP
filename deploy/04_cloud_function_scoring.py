"""
PASSO 4 (Opcional): Criar Cloud Function para Scoring Automático
=================================================================
Este código cria uma Cloud Function que pode ser chamada via HTTP
para fazer predições usando o modelo do Vertex AI.
"""

# Este é o código da Cloud Function
# Salve como main.py no Cloud Functions

CLOUD_FUNCTION_CODE = '''
import functions_framework
from google.cloud import aiplatform
import json

PROJECT_ID = "mlops-484912"
REGION = "us-central1"
ENDPOINT_ID = "SEU_ENDPOINT_ID"  # Substituir pelo ID real

# Inicializar uma vez (otimização)
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = None

def get_endpoint():
    global endpoint
    if endpoint is None:
        endpoint = aiplatform.Endpoint(
            f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
        )
    return endpoint

@functions_framework.http
def predict(request):
    """
    Cloud Function para predição de inadimplência.
    
    Exemplo de chamada:
    curl -X POST https://REGION-PROJECT_ID.cloudfunctions.net/predict-inadimplencia \\
      -H "Content-Type: application/json" \\
      -d '{"instances": [[35, 8000, 5, 15000, 12.5, 24, 750, 1, 1, 2]]}'
    """
    
    # Permitir CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return ('', 204, headers)
    
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        # Pegar dados do request
        request_json = request.get_json(silent=True)
        
        if not request_json or 'instances' not in request_json:
            return (json.dumps({
                "error": "Formato inválido. Envie: {'instances': [[...]]}"
            }), 400, headers)
        
        instances = request_json['instances']
        
        # Fazer predição
        ep = get_endpoint()
        response = ep.predict(instances=instances)
        
        # Formatar resposta
        predictions = []
        for i, pred in enumerate(response.predictions):
            if isinstance(pred, list):
                pred_class = 1 if pred[1] > 0.5 else 0
                probability = max(pred)
            else:
                pred_class = int(pred)
                probability = None
            
            predictions.append({
                "prediction": "inadimplente" if pred_class == 1 else "adimplente",
                "class": pred_class,
                "probability": probability
            })
        
        return (json.dumps({
            "success": True,
            "predictions": predictions
        }), 200, headers)
        
    except Exception as e:
        return (json.dumps({
            "error": str(e)
        }), 500, headers)
'''

# requirements.txt para Cloud Function
REQUIREMENTS = '''
functions-framework==3.*
google-cloud-aiplatform>=1.38.0
'''


def create_cloud_function_files():
    """Cria os arquivos necessários para deploy da Cloud Function"""
    import os
    
    cf_dir = os.path.join(os.path.dirname(__file__), "cloud_function")
    os.makedirs(cf_dir, exist_ok=True)
    
    # main.py
    with open(os.path.join(cf_dir, "main.py"), "w") as f:
        f.write(CLOUD_FUNCTION_CODE)
    
    # requirements.txt
    with open(os.path.join(cf_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)
    
    print(f"✅ Arquivos criados em: {cf_dir}")
    print(f"\nPara fazer deploy:")
    print(f"""
    cd {cf_dir}
    
    gcloud functions deploy predict-inadimplencia \\
      --gen2 \\
      --runtime=python311 \\
      --region={REGION} \\
      --source=. \\
      --entry-point=predict \\
      --trigger-http \\
      --allow-unauthenticated \\
      --memory=512MB \\
      --timeout=60s
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("☁️ CONFIGURAÇÃO DA CLOUD FUNCTION")
    print("=" * 60)
    
    print("\nEsta é uma opção ADICIONAL para servir o modelo.")
    print("Ela cria uma API HTTP simples que chama o Vertex AI Endpoint.")
    
    create = input("\nDeseja criar os arquivos? (s/n): ").lower()
    
    if create == 's':
        create_cloud_function_files()

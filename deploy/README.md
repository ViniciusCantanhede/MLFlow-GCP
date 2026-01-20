# Deploy de Produção - Vertex AI

Este diretório contém os scripts para fazer deploy do modelo em produção no Google Cloud Platform.

## 🎯 Visão Geral

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MLflow Local  │────▶│  Vertex AI Model │────▶│  Vertex AI      │
│   (mlruns/)     │     │  Registry        │     │  Endpoint       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │   API REST      │
                                                 │   (Predições)   │
                                                 └─────────────────┘
```

## 📋 Passo a Passo

### Pré-requisitos

Certifique-se de ter:
1. ✅ Modelo treinado no MLflow (`mlruns/`)
2. ✅ Autenticação GCP configurada
3. ✅ APIs habilitadas: Vertex AI, Cloud Storage

### Passo 1: Upload do Modelo para Vertex AI

```bash
python deploy/01_upload_model_to_vertex.py
```

Este script:
- Encontra o melhor modelo no MLflow
- Faz upload para o GCS
- Registra no Vertex AI Model Registry

### Passo 2: Deploy como Endpoint REST

```bash
python deploy/02_deploy_endpoint.py
```

Este script:
- Cria um endpoint no Vertex AI
- Faz deploy do modelo
- Configura auto-scaling (1-3 réplicas)

⏱️ **Tempo estimado: 5-15 minutos**

### Passo 3: Testar o Endpoint

```bash
python deploy/03_test_endpoint.py
```

Ou via curl:

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/mlops-484912/locations/us-central1/endpoints/ENDPOINT_ID:predict \
  -d '{
    "instances": [[35, 8000, 5, 15000, 12.5, 24, 750, 1, 1, 2]]
  }'
```

### Passo 4 (Opcional): Cloud Function

```bash
python deploy/04_cloud_function_scoring.py
```

Cria uma Cloud Function para expor o modelo como API HTTP simples.

### Passo 5: Limpeza (IMPORTANTE!)

⚠️ **Para evitar custos, limpe os recursos quando terminar:**

```bash
python deploy/05_cleanup.py
```

## 💰 Custos Estimados

| Recurso | Custo/hora | Notas |
|---------|------------|-------|
| Vertex AI Endpoint (n1-standard-2) | ~$0.10 | Por réplica |
| Cloud Storage | ~$0.02/GB | Armazenamento do modelo |
| Predições | ~$0.0001 | Por predição |

💡 **Dica**: Para testes, use `n1-standard-2` com 1 réplica. Para produção, configure auto-scaling.

## 🔧 Troubleshooting

### Erro: "Model artifact not found"
- Verifique se o modelo está no GCS
- Confirme o path: `gs://meu-bucket-29061999/models/`

### Erro: "Permission denied"
- Execute: `gcloud auth application-default login`
- Verifique as permissões da service account

### Deploy demora mais de 20 minutos
- Normal para primeira vez
- Verifique logs: Console GCP → Vertex AI → Endpoints

## 📚 Referências

- [Vertex AI Model Deployment](https://cloud.google.com/vertex-ai/docs/general/deployment)
- [Pricing](https://cloud.google.com/vertex-ai/pricing)
- [Best Practices](https://cloud.google.com/vertex-ai/docs/predictions/best-practices)

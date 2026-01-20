"""
API REST para servir o modelo de inadimplência usando FastAPI.
Deploy via Cloud Run - mais flexível e comum em produção.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List, Optional

app = FastAPI(
    title="API de Predição de Inadimplência",
    description="Modelo de ML para prever inadimplência de clientes",
    version="1.0.0"
)

# Carregar modelo na inicialização
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modelo carregado de: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        raise e


class ClienteInput(BaseModel):
    """Schema de entrada - features do cliente"""
    features: List[float]  # Lista de features numéricas
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [35, 8000, 5, 15000, 12.5, 24, 750, 1, 1, 2]
            }
        }


class ClienteBatchInput(BaseModel):
    """Schema para predição em batch"""
    instances: List[List[float]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    [35, 8000, 5, 15000, 12.5, 24, 750, 1, 1, 2],
                    [22, 2500, 0.5, 30000, 25.0, 48, 450, 0, 0, 0]
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Schema de saída"""
    prediction: str  # "adimplente" ou "inadimplente"
    class_id: int    # 0 ou 1
    probability: Optional[float] = None


class BatchPredictionOutput(BaseModel):
    """Schema de saída para batch"""
    predictions: List[PredictionOutput]


@app.get("/")
async def root():
    return {
        "message": "API de Predição de Inadimplência",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check para Cloud Run/Kubernetes"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(cliente: ClienteInput):
    """
    Faz predição para um único cliente.
    
    Envie as features do cliente como uma lista de valores numéricos.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # Converter para array 2D
        X = np.array(cliente.features).reshape(1, -1)
        
        # Predição
        pred = model.predict(X)[0]
        
        # Probabilidade (se disponível)
        prob = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            prob = float(max(proba))
        
        return PredictionOutput(
            prediction="inadimplente" if pred == 1 else "adimplente",
            class_id=int(pred),
            probability=prob
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: ClienteBatchInput):
    """
    Faz predição para múltiplos clientes de uma vez.
    
    Ideal para scoring em batch.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # Converter para array 2D
        X = np.array(batch.instances)
        
        # Predições
        preds = model.predict(X)
        
        # Probabilidades (se disponível)
        probs = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
        
        results = []
        for i, pred in enumerate(preds):
            prob = float(max(probs[i])) if probs is not None else None
            results.append(PredictionOutput(
                prediction="inadimplente" if pred == 1 else "adimplente",
                class_id=int(pred),
                probability=prob
            ))
        
        return BatchPredictionOutput(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")


# Para rodar localmente:
# uvicorn api_model:app --reload --port 8080
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

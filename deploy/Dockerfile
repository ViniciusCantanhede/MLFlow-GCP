FROM python:3.11-slim

WORKDIR /app

# Instalar dependências adicionais da API
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn numpy pandas

# Copiar código da API
COPY api_model.py .

# Copiar modelo
COPY model/ ./model/

# Variáveis de ambiente
ENV MODEL_PATH=/app/model/model.pkl
ENV PORT=8080

# Expor porta
EXPOSE 8080

# Comando para iniciar
CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "8080"]

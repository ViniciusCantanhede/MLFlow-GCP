FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do projeto principal
COPY ../requirements.txt ./requirements-main.txt
RUN pip install --no-cache-dir -r requirements-main.txt

# Instalar dependências adicionais da API
RUN pip install --no-cache-dir fastapi uvicorn mlflow

# Copiar código da API
COPY api_model.py .

# Copiar modelo do MLflow (será copiado manualmente antes do build)
COPY model/ ./model/

# Variáveis de ambiente
ENV MODEL_PATH=/app/model
ENV PORT=8080

# Expor porta
EXPOSE 8080

# Comando para iniciar
CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "8080"]

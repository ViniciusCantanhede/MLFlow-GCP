FROM python:3.11-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código e modelo
COPY deploy/api_model.py .
COPY model.pkl .

# Variáveis de ambiente
ENV MODEL_PATH=/app/model.pkl
ENV PORT=8080

# Expor porta
EXPOSE 8080

# Comando para iniciar
CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "8080"]

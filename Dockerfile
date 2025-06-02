# Use uma imagem base oficial do Python, slim para evitar imagem gigante
FROM python:3.9-slim

# Define diretório de trabalho
WORKDIR /app

# Copia o requirements para a imagem
COPY requirements.txt .

# Atualiza o pip e instala dependências
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia o restante do código
COPY . .

# Comando para rodar a API com gunicorn e uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "--bind", "0.0.0.0:8000"]

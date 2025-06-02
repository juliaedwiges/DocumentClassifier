# Use uma imagem base oficial do Python, slim para evitar imagem gigante
FROM python:3.9-slim

# Define diretório de trabalho
WORKDIR /app

# Copia o requirements para a imagem
COPY requirements.txt .

# Atualiza o pip
RUN pip install --upgrade pip

# Instala as dependências, forçando PyTorch CPU
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copia o restante do código
COPY . .

# Comando para rodar a API com gunicorn e uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "--bind", "0.0.0.0:8000"]

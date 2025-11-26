FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema de forma mais eficiente
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para cache
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios necessários
RUN mkdir -p debug_frames rostos

ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 main:app
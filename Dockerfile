FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema de forma mais eficiente
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar requirements primeiro (para melhor cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p debug_frames rostos logs

# Configurar variáveis de ambiente
ENV PYTHONPATH=/app
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health-simple || exit 1

# Comando otimizado para produção
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 --access-logfile - --error-logfile - main:app
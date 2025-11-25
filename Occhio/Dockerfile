# Usa Python 3.9 como base
FROM python:3.9-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema INCLUINDO CMAKE
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código para o container
COPY . .

# Cria pastas que sua aplicação precisa
RUN mkdir -p /app/debug_frames /app/rostos

# Define variáveis de ambiente
ENV PYTHONPATH=/app
ENV PORT=8080

# Expõe a porta 8080
EXPOSE 8080

# Comando para iniciar a aplicação
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
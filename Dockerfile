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

# Instalar dependências Python uma por uma para debug
RUN pip install --no-cache-dir flask==2.3.3
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir ultralytics==8.0.186
RUN pip install --no-cache-dir face-recognition==1.3.0
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir openai==1.3.7
RUN pip install --no-cache-dir pillow==10.0.1
RUN pip install --no-cache-dir pymysql==1.1.0
RUN pip install --no-cache-dir gunicorn==21.2.0

# Copiar código
COPY . .

# Criar diretórios necessários
RUN mkdir -p debug_frames rostos

ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Evitar archivos .pyc y buffering en logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias primero para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Crear directorios necesarios (data se sobreescribe con el volumen de Fly)
RUN mkdir -p data logs

CMD ["python", "main.py", "--mode=live"]

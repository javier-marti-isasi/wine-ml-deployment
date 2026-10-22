FROM python:3.11-slim

# Evita archivos .pyc y muestra los logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code
COPY requirements.txt .
# Instala dependencias y crea un usuario sin privilegios (no-root)
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home appuser

# 1. Copia el código y el modelo (los archivos quedan con propietario root)
COPY app/ ./app/
COPY artifacts/model.cbm artifacts/metadata.json ./artifacts/
# 2. Hace a appuser propietario de /code (debe ir antes de USER, solo root puede chown)
RUN chown -R appuser:appuser /code
# 3. El proceso (uvicorn) corre como appuser: si la app se compromete, no hay root en el contenedor
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Commands to build and run the container
# docker build -t wine-ml-api:1.0 .
# docker run --rm -p 8000:8000 --name wine-api wine-ml-api:1.0

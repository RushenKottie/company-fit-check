FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_SERVER_HOST=0.0.0.0 \
    MLFLOW_SERVER_PORT=5000

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        mlflow \
        psycopg2-binary \
        azure-storage-blob \
        azure-identity

EXPOSE 5000

CMD mlflow server \
    --host "${MLFLOW_SERVER_HOST}" \
    --port "${MLFLOW_SERVER_PORT}" \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION}"

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

COPY . .

RUN mkdir -p /app/.tmp /app/.mlruns

EXPOSE 8000

CMD ["chainlit", "run", "src/interfaces/chainlit/app.py", "--host", "0.0.0.0", "--port", "8000"]

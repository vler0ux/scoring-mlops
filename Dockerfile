FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_URI=/app/mlflow_model \
    LOG_FILE=/app/logs/predictions.jsonl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

COPY api/        ./api/
COPY mlflow_model/ ./mlflow_model/

RUN mkdir -p /app/logs

EXPOSE 7860

CMD ["python", "api/app.py"]
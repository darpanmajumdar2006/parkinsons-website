FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/parkinsons_model.joblib \
    PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY data ./data
COPY models/.gitkeep ./models/.gitkeep

RUN python -m src.train \
    --dataset /app/data/pd_dataset.csv \
    --model-output /app/models/parkinsons_model.joblib \
    --metrics-output /app/models/metrics.json

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

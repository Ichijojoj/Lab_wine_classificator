
FROM python:3.10-slim

LABEL maintainer="your_email@example.com"
LABEL description="Wine Quality Prediction API with FastAPI"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import fastapi; import uvicorn; import sklearn; print('All packages installed')"

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p models data

# Копирование config в корень
COPY configs/config.ini ./config.ini

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV MODEL_PATH=/app/models/wine_model.pkl
ENV SCALER_PATH=/app/models/scaler.pkl
ENV METRICS_PATH=/app/models/metrics.pkl

# Порт
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]
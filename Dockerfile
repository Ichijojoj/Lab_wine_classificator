FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p models data

# Загрузка модели (если используется DVC)
# RUN dvc pull

# Переменные окружения
ENV MODEL_PATH=models/wine_model.pkl
ENV FLASK_APP=api/app.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"]
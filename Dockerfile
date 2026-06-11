FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей (для natasha и некоторых ML-библиотек)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта (алгоритмы, main.py и т.д.)
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Команда для запуска через uvicorn с авто-перезагрузкой (удобно для разработки)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
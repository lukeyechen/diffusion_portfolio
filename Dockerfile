FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY config.py ./config.py
COPY core ./core
COPY api ./api

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}

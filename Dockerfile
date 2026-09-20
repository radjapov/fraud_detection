# Dockerfile
# Multi-stage: сборочный этап ставит билдер-зависимости, затем финальный образ - облегчённый
# python:3.12-slim (тот же Python, что в .venv/uv; версии библиотек закреплены в requirements.txt).

ARG PYTHON_VER=3.12
FROM python:${PYTHON_VER}-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# копируем requirements.txt (создай его в проекте)
COPY requirements.txt /build/requirements.txt

# установим все зависимости в wheel-cache (скорее всего понадобится numba/llvmlite собрать)
RUN python -m pip install --upgrade pip wheel setuptools
RUN pip wheel --wheel-dir=/build/wheels -r /build/requirements.txt

# Финальный образ
FROM python:${PYTHON_VER}-slim

ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# опубликовать порт наружу (локально по умолчанию слушаем только 127.0.0.1)
ENV FRAUD_HOST=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Скопируем сгенерированные wheel-файлы и установим
COPY --from=builder /build/wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
# Устанавливаем сначала колеса (если они есть), затем требования (откат к pip)
RUN if [ -d /wheels ] && [ "$(ls -A /wheels)" ]; then pip install --no-index --find-links=/wheels -r requirements.txt; else pip install -r requirements.txt; fi

# Копируем код
COPY . /app

# expose port
EXPOSE 5001

# рабочая директория и команда запуска
ENV FLASK_APP=fraud_app
CMD ["python", "-m", "fraud_app"]

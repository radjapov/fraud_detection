# какой питон использовать
PYTHON ?= python3
IMAGE_NAME = fraud-detection-ui
TAG = latest
COMPOSE_FILE = docker-compose.yml
.PHONY: run run-bp run-uv test format lint docker-build docker-run
.PHONY: build-image up down logs shell clean

# 1) Запуск приложения напрямую (без uv, просто то, что ты делаешь обычно)
run:
	$(PYTHON) -m fraud_app

run-bp: run

# 2) Запуск через uv, если хочешь использовать зависимости из pyproject.toml
run-uv:
	uv run python -m fraud_app

# tests (needs pytest: uv sync --extra dev)
test:
	$(PYTHON) -m pytest -q

# 3) Автоформатирование кода (isort + black)
format:
	isort .
	black .

# 4) Проверка форматирования (ничего не меняет, только ругается)
lint:
	isort --check-only .
	black --check .

# 5–6) Команды под докер
# Makefile
build-image:
	docker build -t $(IMAGE_NAME):$(TAG) .

up:
	docker-compose -f $(COMPOSE_FILE) up -d --build

down:
	docker-compose -f $(COMPOSE_FILE) down

logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

shell:
	docker-compose -f $(COMPOSE_FILE) exec fraud-ui /bin/sh

clean:
	docker rmi $(IMAGE_NAME):$(TAG) || true
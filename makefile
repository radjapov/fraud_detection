# какой питон использовать
PYTHON := python3.14
IMAGE_NAME = fraud-detection-ui
TAG = latest
COMPOSE_FILE = docker-compose.yml
.PHONY: run run-uv format lint docker-build docker-run
.PHONY: build-image up down logs shell clean

# 1) Запуск приложения напрямую (без uv, просто то, что ты делаешь обычно)
run:
	$(PYTHON) app_ui.py

run-bp:
	python3.14 -m fraud_app

# 2) Запуск через uv, если хочешь использовать зависимости из pyproject.toml
run-uv:
	uv run $(PYTHON) app_ui.py

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
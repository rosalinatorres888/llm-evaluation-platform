.PHONY: help install dev test clean run dashboard api docker lint format

help:
	@echo "LLM Evaluation Platform - Available Commands"
	@echo "============================================"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install development dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black/isort"
	@echo "  make clean      - Clean cache and temporary files"
	@echo "  make run        - Run the demo evaluation"
	@echo "  make dashboard  - Launch Streamlit dashboard"
	@echo "  make api        - Start FastAPI server"
	@echo "  make docker     - Build and run with Docker"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	pylint src/
	mypy src/
	bandit -r src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf data/cache/* logs/*

run:
	python scripts/run_evaluation.py

dashboard:
	streamlit run src/dashboard/app.py

api:
	uvicorn src.api.main:app --reload --port 8000

docker:
	docker-compose up --build

docker-down:
	docker-compose down

docs:
	mkdocs serve

build-docs:
	mkdocs build

.PHONY: help up down migrate ingest-matches validate-data build-features extract-news train-baseline run-experiment evaluate test lint

PYTHON ?= python

help:
	@echo "Comandos disponíveis:"
	@echo "  make up                               Inicia serviços via Docker"
	@echo "  make down                             Para serviços do Docker"
	@echo "  make migrate                          Aplica migrations de banco de dados"
	@echo "  make ingest-matches SEASONS=...       Ingere temporadas do Brasileirão Série A"
	@echo "  make validate-data                    Valida integridade e temporalidade dos dados"
	@echo "  make build-features AS_OF_HOURS=24    Gera snapshots tabulares pre-match"
	@echo "  make extract-news                     Executa extração estruturada de notícias via LLM"
	@echo "  make train-baseline                   Treina modelos baselines B0, B1 e G1 Poisson"
	@echo "  make run-experiment CONFIG=...        Executa experimento completo com M1 e ablações"
	@echo "  make evaluate                         Avalia métricas fora da amostra e gera relatório"
	@echo "  make test                             Executa suíte de testes automatizados"
	@echo "  make lint                             Executa ruff linter e checagem de tipos"

up:
	docker compose up -d

down:
	docker compose down

migrate:
	$(PYTHON) -m src.database

ingest-matches:
	$(PYTHON) -m src.ingestion.openfootball --seasons $(or $(SEASONS),2021,2022,2023,2024)

validate-data:
	$(PYTHON) -m src.data_quality.auditor

build-features:
	$(PYTHON) -m src.features.snapshot_builder --as-of-hours $(or $(AS_OF_HOURS),24)

extract-news:
	$(PYTHON) -m src.llm_extraction.extractor

train-baseline:
	$(PYTHON) -m src.models.train_baseline

run-experiment:
	$(PYTHON) -m src.evaluation.run_experiment --config $(or $(CONFIG),configs/experiment_v1.yaml)

evaluate:
	$(PYTHON) -m src.evaluation.evaluate_models

test:
	pytest tests/unit tests/integration tests/e2e -v

lint:
	ruff check .

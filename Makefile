PYTHON := poetry run python
POETRY := poetry run

.PHONY: all setup data features train format lint test clean acquisition validate_before validate_after visualize predict

all: validate_before train validate_after visualize 

setup:
	pip install poetry poetry-plugin-shell
	rm -f poetry.lock
	poetry lock
	poetry install


acquisition: data/raw/data.csv

data/raw/data.csv:
	$(PYTHON) src/Acquisition/Acquisition.py

validate_before:
	$(PYTHON) src/validation/validation.py \
		--input data/raw/data.csv \
		--output reports/results/data_quality_report_before.csv

# Data cleaning
data: data/cleaned/cleaned_data.csv

data/cleaned/cleaned_data.csv: data/raw/data.csv
	$(PYTHON) src/data/data_cleaning.py \
		--input data/raw/data.csv \
		--output data/cleaned/cleaned_data.csv


validate_after:
	$(PYTHON) src/validation/validation.py \
		--input data/cleaned/cleaned_data.csv \
		--output reports/results/data_quality_report_after.csv

visualize: 
	$(PYTHON) src/visualization/visualization.py \
		--input data/cleaned/cleaned_data.csv \

# Feature transformation
features: data/processed/train.csv data/processed/validation.csv data/processed/test.csv

data/processed/train.csv data/processed/validation.csv data/processed/test.csv: data/cleaned/cleaned_data.csv
	$(PYTHON) src/features/feature_transformation.py

# Training
train: models/best_model_latest.pkl

models/best_model_latest.pkl: data/processed/train.csv data/processed/validation.csv
	$(PYTHON) src/models/train.py


predict: models/best_model_latest.pkl
	$(PYTHON) src/models/test.py \
		--input data/processed/test.csv \
		--model models/best_model_latest.pkl
format:
	$(POETRY) ruff format --diff tests/
	$(POETRY) ruff format --diff src/

lint:
	$(POETRY) ruff check tests/
	$(POETRY) ruff check src/

test:
	$(POETRY) pytest tests/ -v
	$(POETRY) pytest --cov=src 

clean:
	rm -rf data/cleaned/*
	rm -rf data/processed/*
	rm -rf reports/results/*
	rm -rf reports/figures/*
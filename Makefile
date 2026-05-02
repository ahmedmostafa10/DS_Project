PYTHON := PYTHONPATH=. python  # this for conda 

.PHONY: all data feature train test clean


all: train

# clean
data: data/cleaned/cleaned_data.csv

data/cleaned/cleaned_data.csv: data/raw/data.csv
	$(PYTHON) src/data/data_cleaning.py \
		--input  data/raw/data.csv \
		--output data/cleaned/cleaned_data.csv

# feature tarnsformation
features: data/processed/train.csv data/processed/validation.csv data/processed/test.csv

data/processed/train.csv data/processed/validation.csv data/processed/test.csv: data/cleaned/cleaned_data.csv
	$(PYTHON) src/features/feature_transformation.py


# training
train: models/best_model_latest.pkl

models/best_model_latest.pkl: data/processed/train.csv data/processed/validation.csv 
	$(PYTHON) src/models/train.py

#predict 



test:
	pytest tests/ -v


clean:
	rm -f data/cleaned/cleaned_data.csv
	rm -f data/cleaned/cleaning_quarantined_data.csv
	rm -f data/processed/train.csv
	rm -f data/processed/validation.csv
	rm -f data/processed/test.csv
	rm -f reports/results/cleaning_log_report.csv
	rm -f reports/results/transformation_log_report.csv
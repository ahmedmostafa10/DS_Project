# Data Science Project — Apartment Price Prediction

## Team Members

- Mariam Amin
- Karim Mahmoud
- Khalid ElGammal
- Ahmed Mostafa Bakr

---

## Overview

Real estate prices in Egypt are highly volatile and hard to assess consistently. This project classifies apartment listings into three price tiers (**Low**, **Medium**, and **High**) using structural and location-based features such as area, number of rooms, neighbourhood, and proximity to points of interest.

The model targets real estate agencies seeking to automate initial property valuation, reducing reliance on manual, subjective assessment.

---

## Data Sources

| Source | Description |
|---|---|
| Kaggle (`waddahali/real-estate-listings`) | Tabular residential listings scraped from Property Finder |
| Bayut Egypt | Live listings scraped from Egypt's largest real estate portal |
| OpenStreetMap API | Geospatial data used to enrich listings with nearby Points of Interest (POIs) |

The Kaggle and Bayut datasets are row-appended. OpenStreetMap features are merged as additional columns, adding 12 spatial features per listing (distances to and counts of nearby schools, hospitals, supermarkets, malls, transit stations, and cafes).

---

## Target Variable

`price_category` is derived by binning `price_egp` into three quantile-based tiers, ensuring balanced class distribution (~33% each):

- `Low` — bottom third of the price distribution
- `Medium` — middle third
- `High` — top third

---

## Repository Structure

```
├── data/
│   ├── raw/                  # Original source data (Kaggle + Bayut)
│   ├── cleaned/              # Output of the cleaning pipeline
│   └── processed/            # Train / validation / test splits
├── src/
│   ├── Acquisition/
│   │   └── Acquisition.py            # Data acquisition from source
│   ├── data/
│   │   └── data_cleaning.py          # Full cleaning pipeline
│   ├── features/
│   │   └── feature_transformation.py # Scaling, encoding, engineering, selection
│   ├── models/
│   │   ├── train.py                  # Model training and MLflow logging
│   │   └── test.py                   # Inference on test split
│   ├── validation/
│   │   └── validation.py             # Pre- and post-cleaning data quality checks
│   └── visualization/
│       └── visualization.py          # Exploratory plots
├── models/                   # Saved model artifacts
├── reports/
│   ├── figures/              # Generated visualisations
│   └── results/              # Cleaning, transformation, and validation logs
├── tests/                    # Unit tests
├── Makefile                  # Pipeline automation
└── pyproject.toml            # Poetry dependency definition
```

---

## How to Run

This project uses **Poetry** and **Python 3.11+**. All pipeline steps are automated via `make`.

### 1. Environment Setup

```bash
make setup
```

### 2. Run the Full Pipeline

```bash
make all
```

This single command executes the complete pipeline in order — acquisition → cleaning → features → training → validation → visualisation. Each step is only rebuilt if its input files have changed or are missing.

### 3. Run Individual Steps

To run a specific stage in isolation:

```bash
make acquisition      # Fetch raw data → data/raw/data.csv
make validate_before  # Validate raw data quality → reports/results/data_quality_report_before.csv
make data             # Clean raw data → data/cleaned/cleaned_data.csv
make validate_after   # Validate cleaned data quality → reports/results/data_quality_report_after.csv
make visualize        # Generate exploratory plots → reports/figures/
make features         # Produce train / validation / test splits → data/processed/
make train            # Train models and save the best → models/best_model_latest.pkl
make predict          # Run inference on the test split → reports/results/
```

### 4. Code Quality

```bash
make format   # Dry-run formatting check (ruff)
make lint     # Static analysis (ruff)
```

### 5. Tests

```bash
make test
```

### 6. Clean

Removes all generated files (cleaned data, processed splits, model artifacts, log reports, and figures).

```bash
make clean
```

---

## Experiment Tracking

All training runs are logged with **MLflow**, including accuracy, macro F1-score, per-class classification reports, best cross-validation score, and the saved model artifact. Each run is uniquely named `{model_name}_{timestamp}` for reproducibility.

---

## Continuous Integration

The project uses a **GitHub Actions** CI pipeline that runs automatically on every push or pull request to `main`.

**Pipeline steps:**

1. Set up Python 3.11
2. `make setup` — install dependencies via Poetry
3. `make format` — formatting check (fails on any diff)
4. `make lint` — static analysis (fails on any finding)
5. `make test` — run the full pytest suite

All steps must pass for the pipeline to succeed.
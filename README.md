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
│   ├── data/
│   │   └── data_cleaning.py  # Full cleaning pipeline
│   └── features/
│       └── feature_transformation.py  # Scaling, encoding, engineering, selection
│   └── models/
│       └── train.py          # Model training and MLflow logging
├── models/                   # Saved model artifacts
├── reports/
│   └── results/              # Cleaning and transformation logs
├── tests/                    # Unit and integration tests
├── Makefile                  # Pipeline automation
└── pyproject.toml            # Poetry dependency definition
```

---

## How to Run

This project uses **Poetry** and **Python 3.11+**. All pipeline steps are automated via `make`.

### 1. Environment setup

```bash
make setup
```

### 2. Data cleaning

Expects raw data at `data/raw/data.csv`.

```bash
make data
```

### 3. Feature transformation

Produces train, validation, and test splits under `data/processed/`.

```bash
make features
```

### 4. Train

Trains all candidate models, logs experiments to MLflow, and saves the best model.

```bash
make train
```

Running `make` alone (no target) executes the full pipeline from cleaning through training, rebuilding only steps whose inputs have changed.

### 5. Code quality

```bash
make format   # Dry-run formatting check (ruff)
make lint     # Static analysis (ruff)
```

### 6. Tests

```bash
make test
```

### 7. Clean

Removes all generated files (cleaned data, processed splits, model artifacts, log reports).

```bash
make clean
```

---

## Pipeline Overview

```
Raw CSV
  └─► make data      → data/cleaned/cleaned_data.csv
        └─► make features  → data/processed/{train,validation,test}.csv
              └─► make train    → models/best_model_latest.pkl
```

The cleaning pipeline covers relevance filtering, accuracy quarantine, consistency normalisation, type coercion, missing value handling, deduplication, and outlier treatment. The transformation pipeline covers feature scaling, encoding, engineering, and selection (variance threshold → correlation filter → multicollinearity removal → RFECV).

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
# House Price Prediction - Project Documentation

## Team Members

- Karim Mahmoud Ahmed 
- Mariam Amin Amin
- Khalid elgammal
- Ahmed Mostafa Bakr


---

## Project Description

### Overview
This project implements an end-to-end machine learning pipeline for **house price prediction**. It encompasses data collection, cleaning, feature engineering, model training, and evaluation using real estate property data.

### Objective
Build and evaluate predictive models to accurately forecast house prices based on property attributes, geographical features, and location-based information extracted from OpenStreetMap.

### Project Scope
The pipeline includes:

1. **Data Cleaning**: 
   - Removal of irrelevant columns
   - Data validation based on predefined rules (price ranges, area bounds, geographical coordinates)
   - Outlier detection and quarantine
   - Logging of data quality issues

2. **Feature Transformation**:
   - Numerical scaling using RobustScaler and StandardScaler
   - Categorical encoding using OneHotEncoder
   - Feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
   - Train-validation-test split (typically 60-20-20)

3. **Model Training & Evaluation**:
   - Multiple ML algorithms: Random Forest, Logistic Regression, Decision Trees, XGBoost
   - Hyperparameter tuning via Grid Search
   - Experiment tracking with MLflow
   - Performance metrics: Accuracy, F1-Score, Classification Reports, Confusion Matrices
   - Cross-validation for robust model assessment

### Technologies & Dependencies
- **Language**: Python 3.11+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Visualization**: Matplotlib, Seaborn
- **Package Manager**: Poetry
- **Testing**: Pytest
- **Code Quality**: Ruff (linting & formatting)

---

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Poetry package manager
- pip (for poetry installation)

### Installation Steps

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd DS_Project
   ```

2. **Install Poetry and plugins**:
   ```bash
   pip install poetry poetry-plugin-shell
   ```

3. **Install project dependencies**:
   ```bash
   make setup
   ```
   
   Or manually:
   ```bash
   rm -f poetry.lock
   poetry lock
   poetry install
   ```

4. **Verify installation**:
   ```bash
   poetry run python --version
   ```

---

## Run Instructions

### Quick Start (Run Full Pipeline)
Execute the complete pipeline from data cleaning to model training:
```bash
make all
```

### Run Individual Stages

#### 1. Data Cleaning
Cleans raw data and generates validation reports:
```bash
make data
```
Or directly:
```bash
poetry run python src/data/data_cleaning.py \
  --input data/raw/data.csv \
  --output data/cleaned/cleaned_data.csv
```
**Output**: 
- `data/cleaned/cleaned_data.csv` - Cleaned dataset
- `data/cleaned/cleaning_quarantined_data.csv` - Quarantined (invalid) records
- `reports/results/cleaning_log_report.csv` - Cleaning process logs

#### 2. Feature Transformation
Transforms and engineers features from cleaned data:
```bash
make features
```
Or directly:
```bash
poetry run python src/features/feature_transformation.py
```
**Output**: 
- `data/processed/train.csv` - Training set (60%)
- `data/processed/validation.csv` - Validation set (20%)
- `data/processed/test.csv` - Test set (20%)
- `reports/results/transformation_log_report.csv` - Transformation logs

#### 3. Model Training
Trains and evaluates machine learning models:
```bash
make train
```
Or directly:
```bash
poetry run python src/models/train.py
```
**Output**: 
- `models/best_model_latest.pkl` - Best trained model
- MLflow experiment tracking data in `mlruns/`
- Confusion matrices and performance reports in `reports/`

### Code Quality

#### Run Tests
Execute all unit tests:
```bash
make test
```
Or:
```bash
poetry run pytest tests/ -v
```

#### Format Code
Format code according to project standards:
```bash
make format
```

#### Lint Code
Check code for style issues:
```bash
make lint
```

### View MLflow Experiments
Track and compare experiment results:
```bash
poetry run mlflow ui
```
Then open `http://localhost:5000` in your browser to view:
- All training experiments
- Model performance metrics
- Hyperparameter configurations
- Artifacts and reports

### Clean Generated Files
Remove all generated data and model files:
```bash
make clean
```

---

## Project Pipeline Overview

```
Raw Data (data/raw/data.csv)
    ↓
[Data Cleaning Stage]
    - Validate records based on rules
    - Remove irrelevant columns
    - Handle outliers
    - Generate cleaning reports
    ↓
Cleaned Data (data/cleaned/cleaned_data.csv)
    ↓
[Feature Transformation Stage]
    - Scale numerical features
    - Encode categorical features
    - Apply feature selection (RFECV)
    - Split into train/validation/test
    ↓
Processed Data (data/processed/)
    ├── train.csv
    ├── validation.csv
    └── test.csv
    ↓
[Model Training Stage]
    - Train multiple models (RF, LR, DT, XGBoost)
    - Hyperparameter tuning with GridSearch
    - Evaluate on validation set
    - Select best model
    - Test on test set
    ↓
Best Model (models/best_model_latest.pkl)
    + Performance Reports + MLflow Tracking
```

---

## Project Structure

```
DS_Project/                          # Root project directory
├── configs/                         # Configuration files
│   ├── __init__.py
│   ├── config.toml                 # Main configuration file
│   └── models_grid_search_params.py # Hyperparameter configurations
│
├── data/                           # Data directory (cookiecutter standard)
│   ├── __init__.py
│   ├── raw/                        # Original raw data
│   │   └── data.csv               # Raw input data
│   ├── cleaned/                    # Data after cleaning stage
│   │   ├── cleaned_data.csv       # Main cleaned dataset
│   │   └── cleaning_quarantined_data.csv  # Invalid/rejected records
│   └── processed/                  # Final processed data (train/val/test split)
│       ├── train.csv              # Training set (60%)
│       ├── validation.csv         # Validation set (20%)
│       └── test.csv               # Test set (20%)
│
├── src/                           # Source code module
│   ├── __init__.py
│   ├── data/                      # Data processing module
│   │   ├── __init__.py
│   │   └── data_cleaning.py       # Data cleaning implementation
│   ├── features/                  # Feature engineering module
│   │   ├── __init__.py
│   │   └── feature_transformation.py  # Feature transformation implementation
│   ├── models/                    # Model training module
│   │   ├── __init__.py
│   │   ├── train.py               # Model training script
│   │   └── test.py                # Model testing/evaluation
│   └── visualization/             # Visualization utilities
│       └── __init__.py
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── __init__.py
│   ├── 02_data_cleaning.ipynb     # Data cleaning exploration
│   ├── 03_feature_transformation.ipynb  # Feature engineering exploration
│   └── 04_modeling.ipynb          # Model experimentation
│
├── models/                        # Trained model artifacts
│   └── __init__.py
│
├── reports/                       # Generated reports and outputs
│   ├── __init__.py
│   ├── cleaning_log_report.csv    # Cleaning process report
│   ├── transformation_log_report.csv  # Feature transformation report
│   ├── figures/                   # Generated visualizations
│   └── results/                   # Final results and metrics
│       └── cleaning_log_report.csv
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_cleaning.py           # Data cleaning tests
│   └── test_feature_transformation.py  # Feature transformation tests
│
├── scripts/                       # Utility scripts
│   └── data/
│
├── mlruns/                        # MLflow experiment tracking (auto-generated)
│   └── 1/                         # Experiment runs and artifacts
│
├── Makefile                       # Build automation targets
├── pyproject.toml                 # Poetry dependencies and project config
├── pyproject.txt                  # Additional project metadata
├── requirements.txt               # Alternative dependency file
├── README.md                      # Original project README
└── PROJECT_DOCUMENTATION.md       # This comprehensive documentation
```

### Directory Structure Explanation (Cookiecutter Data Science Pattern)

This project follows the **cookiecutter-data-science** structure:

- **`configs/`** - Centralized configuration management
- **`data/`** - Data at various processing stages (raw → cleaned → processed)
- **`src/`** - Production-quality source code organized by function
- **`notebooks/`** - Numbered Jupyter notebooks for exploration (02, 03, 04 convention)
- **`models/`** - Serialized trained model artifacts
- **`reports/`** - Generated analysis outputs, logs, and figures
- **`tests/`** - Automated test suite
- **`mlruns/`** - MLflow experiment tracking artifacts

---

## Configuration

All project settings are centralized in `configs/config.toml`:
- Data paths
- Validation rules (min/max area, price, coordinates)
- Columns for different transformations
- Model parameters for grid search

Modify this file to adjust pipeline behavior without changing code.

---

## Common Tasks

### Add a New Team Member
Edit the **Team Members** section at the top of this file with the actual names.

### Retrain Model
```bash
make clean
make all
```

### Debug Data Quality Issues
1. Check `data/cleaned/cleaning_quarantined_data.csv` for rejected records
2. Review `reports/results/cleaning_log_report.csv` for validation failures
3. Adjust rules in `configs/config.toml` if needed

### Tune Hyperparameters
1. Edit `configs/models_grid_search_params.py`
2. Run training again: `make train`
3. Compare results in MLflow UI

### Run Specific Tests
```bash
poetry run pytest tests/test_cleaning.py -v
poetry run pytest tests/test_feature_transformation.py -v
```

---

## Notes

- All data processing is **reproducible** - same input produces same output
- **Experiment tracking** via MLflow allows comparing multiple model configurations
- **Modular design** - each stage can be run independently
- **Comprehensive logging** - all stages produce detailed reports
- Code follows **PEP 8** standards with Ruff linter

---

## Support & Troubleshooting

- **Poetry issues**: Reinstall with `make setup`
- **Import errors**: Ensure you're using `poetry run` prefix
- **Missing data files**: Run data cleaning stage first with `make data`
- **Model not improving**: Adjust hyperparameters in `configs/models_grid_search_params.py`

---

*Last Updated: May 3, 2026*

"""
Pytest configuration and shared fixtures for the test suite.

This file contains common fixtures and pytest configuration that are used
across all test modules.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Data directory path
DATA_DIR = project_root / "data"


@pytest.fixture(scope="session")
def project_root_fixture():
    """Fixture providing project root path."""
    return project_root


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_data():
    """Load sample CSV data from the project's cleaned_data.csv file."""
    csv_file = DATA_DIR / "cleaned_data.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file).head(5)  # First 5 rows for sample
    else:
        # Fallback to minimal data if file doesn't exist
        return pd.DataFrame(
            {
                "city": ["Cairo", "Giza", "Alexandria", "Cairo", "Giza"],
                "bedrooms": [2, 3, 1, 4, 2],
            }
        )


@pytest.fixture
def sample_numeric_data():
    """Load numeric data from the project's cleaned_data.csv file."""
    csv_file = DATA_DIR / "cleaned_data.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file).head(5)
        # Keep only numeric columns
        return df.select_dtypes(include=[np.number])
    else:
        # Fallback to minimal numeric data if file doesn't exist
        return pd.DataFrame(
            {
                "bedrooms": [1, 2, 3, 4, 5],
                "bathroom": [1.0, 1.5, 2.0, 2.5, 3.0],
                "area_value": [50.0, 100.0, 150.0, 200.0, 250.0],
            }
        )


@pytest.fixture
def sample_categorical_data():
    """Load categorical data from the project's merged_data.csv file."""
    csv_file = DATA_DIR / "merged_data.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file).head(5)  # First 5 rows for sample
    else:
        # Fallback to minimal data if file doesn't exist
        return pd.DataFrame(
            {
                "city": ["Cairo", "Giza", "Alexandria", "Cairo", "Giza"],
                "price_currency": ["EGP", "EGP", "EGP", "EGP", "EGP"],
            }
        )


@pytest.fixture
def sample_mixed_data():
    """Load mixed data from the project's merged_data.csv file."""
    csv_file = DATA_DIR / "merged_data.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file).head(5)
    else:
        # Fallback to basic mixed data if file doesn't exist
        return pd.DataFrame(
            {
                "city": ["Cairo", "Giza", "Alexandria", "Cairo", "Giza"],
                "bedrooms": [2, 3, 1, 4, 2],
                "price": [5000000, 7500000, 4000000, 9500000, 6500000],
            }
        )


@pytest.fixture
def sample_invalid_data():
    """Load data from merged_data.csv for validation testing."""
    csv_file = DATA_DIR / "merged_data.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file).head(5)  # First 5 rows
    else:
        # Fallback to minimal data if file doesn't exist
        return pd.DataFrame(
            {
                "city": ["Cairo", "Giza", None, "Cairo", "Giza"],
                "bedrooms": [2, -1, 1, 20, 2],
            }
        )


@pytest.fixture
def sample_transformation_data():
    """Load sample data from the project's cleaned_data.csv for transformation testing."""
    csv_file = DATA_DIR / "cleaned_data.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file).head(5)  # First 5 rows for sample
    else:
        # Fallback to minimal data if file doesn't exist
        return pd.DataFrame(
            {
                "city": ["Cairo", "Giza", "Alexandria", "Cairo", "Giza"],
                "bedrooms": [2, 3, 1, 4, 2],
                "amenities": [
                    '["WiFi", "Parking"]',
                    '["WiFi", "Gym", "Pool"]',
                    '["Parking"]',
                    '["WiFi", "Parking", "Elevator"]',
                    '["WiFi"]',
                ],
            }
        )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "acquisition: mark test as part of acquisition module")
    config.addinivalue_line("markers", "validation: mark test as part of validation module")
    config.addinivalue_line("markers", "transformation: mark test as part of transformation module")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add module markers based on test file location
        if "acquisition" in str(item.fspath):
            item.add_marker(pytest.mark.acquisition)
        if "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        if "transformation" in str(item.fspath):
            item.add_marker(pytest.mark.transformation)

        # Add integration marker for classes starting with TestIntegration or Test*Integration
        if "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "Test" in item.nodeid and "Test" not in item.parent.name:
            # Tests in Test* classes are unit tests by default
            item.add_marker(pytest.mark.unit)

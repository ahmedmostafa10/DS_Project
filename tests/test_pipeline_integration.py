import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.data_cleaning import (
    IRRELEVANT_COLUMNS_TO_REMOVE,
    DataCleaningPipeline,
    accuracy_qurantine_based_fixing,
    accuracy_rule_based_correction,
    drop_duplicates,
    fix_consistency,
    fix_missingness,
    high_missingness_removal,
    remove_irrelevant_columns,
)
from src.features.feature_transformation import FeatureTransformation
from src.validation.validation import DataValidationPipeline


@pytest.fixture(scope="module")
def temp_pipeline_dir():
    """Create a temporary directory structure mimicking the project data directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "data", "raw"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "data", "cleaned"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "reports", "results"), exist_ok=True)
        yield tmpdir


@pytest.fixture
def mock_raw_data():
    """Create a mock raw dataset similar to what Acquisition would produce."""
    import numpy as np

    n_samples = 100
    np.random.seed(42)

    data = {
        "city": np.random.choice(["Cairo", "Giza", "Alexandria"], n_samples),
        "town": np.random.choice(
            ["Maadi", "Sheikh Zayed", "Smouha", "New Cairo", "6th of October"], n_samples
        ),
        "district": np.random.choice(
            [
                "Zahraa El Maadi",
                "Beverly Hills",
                "Smouha Heights",
                "Fifth Settlement",
                "West Somid",
            ],
            n_samples,
        ),
        "bedrooms": np.random.choice(["3", "studio", "2", "4", "3"], n_samples),
        "bathroom": np.random.choice(["2", "1", "none", "3", "7+"], n_samples),
        "area_value": np.random.uniform(50.0, 300.0, n_samples),
        "price_egp": np.random.uniform(500000, 10000000, n_samples),
        "lon": np.random.uniform(29.0, 32.0, n_samples),
        "lat": np.random.uniform(29.0, 32.0, n_samples),
        "property_type": np.random.choice(["Apartments", "Apartment", "Villa"], n_samples),
        "offering_type": np.random.choice(["Residential for Sale", "for-sale"], n_samples),
        "completion_status": np.random.choice(["completed", "off_plan"], n_samples),
        "furnished": np.random.choice(["NO", "YES", "PARTLY"], n_samples),
        "dist_nearest_school_km": np.random.uniform(0.1, 5.0, n_samples),
        "dist_nearest_hospital_km": np.random.uniform(0.1, 5.0, n_samples),
        "dist_nearest_mall_km": np.random.uniform(0.1, 5.0, n_samples),
        "dist_nearest_transit_station_km": np.random.uniform(0.1, 5.0, n_samples),
        "dist_nearest_cafe_restaurant_km": np.random.uniform(0.1, 5.0, n_samples),
        "dist_nearest_supermarket_km": np.random.uniform(0.1, 5.0, n_samples),
        "listing_level": np.random.choice(["premium", "basic", "standard"], n_samples),
        "is_exclusive": np.random.choice([True, False], n_samples),
        "amenities": np.random.choice(
            ["['WiFi']", None, "['Pool']", "['WiFi', 'Pool']"], n_samples
        ),
        "school_count_within_3km": np.random.randint(0, 10, n_samples),
        "hospital_count_within_3km": np.random.randint(0, 10, n_samples),
        "mall_count_within_3km": np.random.randint(0, 10, n_samples),
        "transit_station_count_within_3km": np.random.randint(0, 10, n_samples),
        "cafe_restaurant_count_within_3km": np.random.randint(0, 20, n_samples),
        "supermarket_count_within_3km": np.random.randint(0, 15, n_samples),
        "is_premium": np.random.choice([True, False], n_samples),
        "is_featured": np.random.choice([True, False], n_samples),
    }
    return pd.DataFrame(data)


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration test that chains Data Cleaning, Validation, and Feature Transformation."""

    def test_cleaning_to_transformation_pipeline(self, temp_pipeline_dir, mock_raw_data):
        # 1. Setup paths
        raw_path = os.path.join(temp_pipeline_dir, "data", "raw", "data.csv")
        cleaned_path = os.path.join(temp_pipeline_dir, "data", "cleaned", "cleaned_data.csv")
        train_path = os.path.join(temp_pipeline_dir, "data", "processed", "train.csv")
        val_path = os.path.join(temp_pipeline_dir, "data", "processed", "validation.csv")
        test_path = os.path.join(temp_pipeline_dir, "data", "processed", "test.csv")
        report_before_path = os.path.join(
            temp_pipeline_dir, "reports", "results", "report_before.csv"
        )
        report_after_path = os.path.join(
            temp_pipeline_dir, "reports", "results", "report_after.csv"
        )

        # Save raw data
        mock_raw_data.to_csv(raw_path, index=False)

        # 2. Run Validation BEFORE Cleaning
        validator_before = DataValidationPipeline()
        validator_before.run_validation(raw_path, report_before_path)
        assert os.path.exists(report_before_path), (
            "Validation before cleaning should produce a report"
        )

        # 3. Run Data Cleaning
        cleaning_pipeline = DataCleaningPipeline()
        cleaning_pipeline.add_step(
            remove_irrelevant_columns, columns_to_remove=IRRELEVANT_COLUMNS_TO_REMOVE
        )
        cleaning_pipeline.add_step(accuracy_rule_based_correction)
        cleaning_pipeline.add_step(accuracy_qurantine_based_fixing)
        cleaning_pipeline.add_step(fix_consistency)
        cleaning_pipeline.add_step(high_missingness_removal)
        cleaning_pipeline.add_step(fix_missingness)
        cleaning_pipeline.add_step(drop_duplicates)

        clean_df = cleaning_pipeline.fit_transform(mock_raw_data)
        clean_df.to_csv(cleaned_path, index=False)

        assert os.path.exists(cleaned_path), "Cleaning pipeline should output a cleaned dataset"
        assert not clean_df.empty, "Cleaned dataset should not be empty"
        assert "amenities" not in clean_df.columns, (
            "High missingness removal should drop 'amenities'"
        )

        # 4. Run Validation AFTER Cleaning
        validator_after = DataValidationPipeline()
        validator_after.run_validation(cleaned_path, report_after_path)
        assert os.path.exists(report_after_path), (
            "Validation after cleaning should produce a report"
        )

        # 5. Run Feature Transformation
        with (
            patch("src.features.feature_transformation.CLEANED_DATA_PATH", cleaned_path),
            patch("src.features.feature_transformation.config") as mock_config,
        ):
            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {
                    "cleaned_data_path": cleaned_path,
                    "transformation_log_report_path": os.path.join(
                        temp_pipeline_dir, "reports", "transf_log.csv"
                    ),
                    "transformation_logging_path": os.path.join(
                        temp_pipeline_dir, "reports", "transf.log"
                    ),
                    "processed_data_dir": os.path.join(temp_pipeline_dir, "data", "processed"),
                    "train_path": train_path,
                    "validation_path": val_path,
                    "test_path": test_path,
                },
                "split": {"test_size": 0.2, "val_size": 0.25, "random_state": 42},
                "encoding": {
                    "listing_level": {"basic": 1, "standard": 2, "premium": 3},
                    "cat_col": ["property_type", "completion_status", "furnished"],
                    "freq_col": ["city", "town", "district"],
                },
                "data": {
                    "distance_columns": [
                        "dist_nearest_school_km",
                        "dist_nearest_hospital_km",
                        "dist_nearest_mall_km",
                        "dist_nearest_transit_station_km",
                        "dist_nearest_cafe_restaurant_km",
                        "dist_nearest_supermarket_km",
                    ],
                    "count_columns": [
                        "school_count_within_3km",
                        "hospital_count_within_3km",
                        "mall_count_within_3km",
                        "transit_station_count_within_3km",
                        "cafe_restaurant_count_within_3km",
                        "supermarket_count_within_3km",
                    ],
                },
                "feature_selection": {
                    "variance_threshold": 0.0,
                    "correlation_threshold": 0.0,
                    "multicollinearity_threshold": 0.99,
                    "rfecv_step": 1,
                    "rfecv_cv": 2,
                    "rfecv_scoring": "f1_macro",
                },
            }.get(key)

            ft = FeatureTransformation()

            df_ft = ft.load_data()
            df_ft = ft.create_target(df_ft)
            X_train, X_val, X_test, y_train, y_val, y_test = ft.split_data(df_ft)
            X_train, X_val, X_test = ft.scale_features(X_train, X_val, X_test)
            X_train, X_val, X_test = ft.encode_features(X_train, X_val, X_test)
            X_train, X_val, X_test = ft.add_feature_interactions(X_train, X_val, X_test)

            def mock_select_features(X_t, X_v, X_te, y_t):
                return X_t, X_v, X_te

            ft.select_features = mock_select_features

            X_train, X_val, X_test = ft.select_features(X_train, X_val, X_test, y_train)
            ft.save_outputs(X_train, X_val, X_test, y_train, y_val, y_test)

            assert os.path.exists(train_path), "Feature transformation should output train.csv"
            assert os.path.exists(val_path), "Feature transformation should output validation.csv"
            assert os.path.exists(test_path), "Feature transformation should output test.csv"

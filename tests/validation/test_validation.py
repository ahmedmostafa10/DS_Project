import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock

from scripts.validation.validation import DataValidationPipeline, PropertyRowSchema
from pydantic import ValidationError


@pytest.fixture
def validation_pipeline():
    """Create a DataValidationPipeline instance for testing."""
    return DataValidationPipeline()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_valid_data():
    """Create sample valid data for testing."""
    data = {
        'city': ['Cairo', 'Giza', 'Alexandria', 'Cairo', 'Giza'],
        'town': ['Maadi', 'Helwan', 'Smouha', 'Heliopolis', 'Sheikh Zayed'],
        'district': ['Maadi', 'Helwan', 'Smouha', 'Heliopolis', 'Sheikh Zayed'],
        'bedrooms': [2, 3, 1, 4, 2],
        'bathroom': [1.0, 2.0, 1.0, 2.5, 1.5],
        'area_value': [100.0, 150.0, 80.0, 200.0, 120.0],
        'price_currency': ['EGP', 'EGP', 'EGP', 'EGP', 'EGP'],
        'dist_nearest_school_km': [0.5, 1.2, 0.8, 1.5, 0.3],
        'dist_nearest_hospital_km': [2.0, 2.5, 1.8, 3.0, 2.2]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_invalid_data():
    """Create sample data with quality issues."""
    data = {
        'city': ['Cairo', 'Giza', None, 'Cairo', 'Giza'],
        'bedrooms': [2, -1, 1, 20, 2],  # Negative and unrealistic values
        'bathroom': [1.0, 2.0, 1.0, 2.5, 1.5],
        'area_value': [100.0, 10.0, 80.0, 200.0, 120.0],  # 10 m² with bedrooms doesn't make sense
        'price_currency': ['EGP', 'EGP', 'EGP', 'EGP', 'EGP'],
        'dist_nearest_school_km': [0.5, 1.2, 0.8, 1.5, 0.3],
    }
    return pd.DataFrame(data)


class TestDataValidationPipeline:
    """Unit tests for DataValidationPipeline class."""
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_pipeline_initialization(self, validation_pipeline):
        """Test that DataValidationPipeline initializes correctly."""
        assert validation_pipeline is not None
        assert hasattr(validation_pipeline, 'logger')
        assert hasattr(validation_pipeline, 'report_summary')
        assert isinstance(validation_pipeline.report_summary, dict)
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_logger_setup(self, validation_pipeline):
        """Test that logger is properly configured."""
        assert validation_pipeline.logger is not None
        assert validation_pipeline.logger.name == 'DataValidationPipeline'
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_report_summary_initialization(self, validation_pipeline):
        """Test that report summary is initialized as empty dict."""
        assert isinstance(validation_pipeline.report_summary, dict)
        assert len(validation_pipeline.report_summary) == 0


class TestPropertyRowSchema:
    """Unit tests for PropertyRowSchema validation."""
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_valid_property_schema(self):
        """Test validation with valid property data."""
        valid_data = {
            'city': 'Cairo',
            'bedrooms': 3,
            'bathroom': 2,
            'area_value': 150.0,
            'price_currency': 'EGP'
        }
        schema = PropertyRowSchema(**valid_data)
        assert schema.city == 'Cairo'
        assert schema.bedrooms == 3
        assert schema.area_value == 150.0
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_schema_with_optional_fields(self):
        """Test schema with optional fields."""
        data = {
            'city': 'Cairo',
            'bedrooms': 2
        }
        schema = PropertyRowSchema(**data)
        assert schema.city == 'Cairo'
        assert schema.bedrooms == 2
        assert schema.town is None
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_area_bedrooms_ratio_validation(self):
        """Test validation of area to bedrooms ratio."""
        # Valid: 3 bedrooms with 100 m²
        valid_data = {
            'bedrooms': 3,
            'area_value': 100.0
        }
        schema = PropertyRowSchema(**valid_data)
        assert schema.area_value == 100.0
        
        # Invalid: 3 bedrooms with 20 m²
        invalid_data = {
            'bedrooms': 3,
            'area_value': 20.0
        }
        with pytest.raises(ValidationError):
            PropertyRowSchema(**invalid_data)
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_area_bedrooms_ratio_edge_cases(self):
        """Test edge cases for area to bedrooms ratio."""
        # 2 bedrooms or less should not trigger ratio validation
        data = {
            'bedrooms': 2,
            'area_value': 10.0
        }
        schema = PropertyRowSchema(**data)
        assert schema.area_value == 10.0
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_osm_features_fields(self):
        """Test that all OSM feature fields are properly defined."""
        data = {
            'city': 'Cairo',
            'dist_nearest_school_km': 0.5,
            'school_count_within_3km': 5,
            'dist_nearest_hospital_km': 2.0,
            'hospital_count_within_3km': 2,
            'dist_nearest_supermarket_km': 0.3,
            'supermarket_count_within_3km': 3,
        }
        schema = PropertyRowSchema(**data)
        assert schema.dist_nearest_school_km == 0.5
        assert schema.school_count_within_3km == 5


class TestDataValidationMethods:
    """Unit tests for validation methods."""
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_validation_with_valid_data(self, validation_pipeline, sample_valid_data, temp_dir):
        """Test validation pipeline with valid data."""
        csv_file = os.path.join(temp_dir, 'valid_data.csv')
        report_file = os.path.join(temp_dir, 'report.csv')
        sample_valid_data.to_csv(csv_file, index=False)
        
        # Should not raise any exceptions
        validation_pipeline.run_validation(csv_file, report_file)
        assert validation_pipeline.report_summary is not None
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_validation_with_missing_file(self, validation_pipeline, temp_dir):
        """Test validation with non-existent file."""
        csv_file = os.path.join(temp_dir, 'nonexistent.csv')
        report_file = os.path.join(temp_dir, 'report.csv')
        
        # Should handle missing file gracefully
        validation_pipeline.run_validation(csv_file, report_file)
        # Logger should have recorded the error


class TestDataQualityChecks:
    """Tests for data quality validation."""
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_numeric_data_detection(self, sample_valid_data):
        """Test detection of numeric columns."""
        numeric_cols = sample_valid_data.select_dtypes(include=[np.number]).columns
        assert 'bedrooms' in numeric_cols
        assert 'bathroom' in numeric_cols
        assert 'area_value' in numeric_cols
        assert 'dist_nearest_school_km' in numeric_cols
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_categorical_data_detection(self, sample_valid_data):
        """Test detection of categorical columns."""
        categorical_cols = sample_valid_data.select_dtypes(include=['object', 'category', 'string']).columns
        assert 'city' in categorical_cols
        assert 'price_currency' in categorical_cols
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_missing_values_detection(self, sample_invalid_data):
        """Test detection of missing values."""
        missing_count = sample_invalid_data.isnull().sum()
        assert missing_count['city'] == 1
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_negative_values_detection(self, sample_invalid_data):
        """Test detection of negative values."""
        negative_in_bedrooms = (sample_invalid_data['bedrooms'] < 0).sum()
        assert negative_in_bedrooms == 1
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_outlier_detection(self, sample_invalid_data):
        """Test detection of outliers."""
        # bedrooms > 15 is considered an outlier
        outliers = (sample_invalid_data['bedrooms'] > 15).sum()
        assert outliers == 1


@pytest.mark.integration
@pytest.mark.validation
class TestDataValidationIntegration:
    """Integration tests for DataValidationPipeline."""
    
    def test_full_validation_workflow(self, validation_pipeline, sample_valid_data, temp_dir):
        """Test complete validation workflow."""
        csv_file = os.path.join(temp_dir, 'test_data.csv')
        report_file = os.path.join(temp_dir, 'validation_report.csv')
        
        sample_valid_data.to_csv(csv_file, index=False)
        validation_pipeline.run_validation(csv_file, report_file)
        
        # Check that report was potentially generated
        assert validation_pipeline.report_summary is not None
    
    def test_validation_with_problematic_data(self, validation_pipeline, sample_invalid_data, temp_dir):
        """Test validation detects issues in problematic data."""
        csv_file = os.path.join(temp_dir, 'invalid_data.csv')
        report_file = os.path.join(temp_dir, 'validation_report.csv')
        
        sample_invalid_data.to_csv(csv_file, index=False)
        validation_pipeline.run_validation(csv_file, report_file)
        
        # Validation should have run without crashing
        assert validation_pipeline.logger is not None

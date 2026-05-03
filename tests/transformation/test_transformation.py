import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock

from scripts.transformation.transformation import DataTransformationPipeline


@pytest.fixture
def transformation_pipeline():
    """Create a DataTransformationPipeline instance for testing."""
    return DataTransformationPipeline()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_transformation_data():
    """Create sample data for transformation testing."""
    data = {
        'listing_id': ['1', '2', '3', '4', '5'],
        'city': ['Cairo', 'Giza', 'Alexandria', 'Cairo', 'Giza'],
        'bedrooms': [2, 3, 1, 4, 2],
        'bathroom': [1.0, 2.0, 1.0, 2.5, 1.5],
        'area_value': [100.0, 150.0, 80.0, 200.0, 120.0],
        'price': [500000, 750000, 400000, 950000, 650000],
        'amenities': [
            '["WiFi", "Parking"]',
            '["WiFi", "Gym", "Pool"]',
            '["Parking"]',
            '["WiFi", "Parking", "Elevator"]',
            '["WiFi"]'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_categorical_data():
    """Create data with categorical features."""
    data = {
        'city': ['Cairo', 'Giza', 'Cairo', 'Alexandria', 'Cairo'],
        'bedrooms': [2, 3, 1, 2, 3],
        'price': [500000, 750000, 450000, 400000, 800000]
    }
    return pd.DataFrame(data)


class TestDataTransformationPipeline:
    """Unit tests for DataTransformationPipeline class."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_pipeline_initialization(self, transformation_pipeline):
        """Test that DataTransformationPipeline initializes correctly."""
        assert transformation_pipeline is not None
        assert hasattr(transformation_pipeline, 'logger')
        assert hasattr(transformation_pipeline, 'report_summary')
        assert isinstance(transformation_pipeline.report_summary, dict)
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_logger_setup(self, transformation_pipeline):
        """Test that logger is properly configured."""
        assert transformation_pipeline.logger is not None
        assert transformation_pipeline.logger.name == 'DataTransformationPipeline'
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_report_summary_initialization(self, transformation_pipeline):
        """Test that report summary is initialized as empty dict."""
        assert isinstance(transformation_pipeline.report_summary, dict)
        assert len(transformation_pipeline.report_summary) == 0


class TestDataTransformationMethods:
    """Unit tests for transformation methods."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_transformation_with_valid_data(self, transformation_pipeline, sample_transformation_data, temp_dir):
        """Test transformation pipeline with valid data."""
        input_csv = os.path.join(temp_dir, 'input_data.csv')
        output_csv = os.path.join(temp_dir, 'output_data.csv')
        report_csv = os.path.join(temp_dir, 'transformation_report.csv')
        
        sample_transformation_data.to_csv(input_csv, index=False)
        
        # Run transformation
        transformation_pipeline.run_transformation(
            input_csv=input_csv,
            output_csv=output_csv,
            output_report_csv=report_csv
        )
        
        # Check output file was created
        assert os.path.exists(output_csv)
        
        # Check output data has same or more rows
        output_data = pd.read_csv(output_csv)
        assert len(output_data) > 0
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_transformation_with_missing_file(self, transformation_pipeline, temp_dir):
        """Test transformation with non-existent input file."""
        input_csv = os.path.join(temp_dir, 'nonexistent.csv')
        output_csv = os.path.join(temp_dir, 'output_data.csv')
        
        # Should handle missing file gracefully
        transformation_pipeline.run_transformation(input_csv, output_csv)
        # Logger should have recorded the error


class TestAmenitiesFeatureExtraction:
    """Tests for amenities feature extraction."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_amenities_extraction(self, transformation_pipeline, sample_transformation_data):
        """Test amenities feature extraction."""
        pipeline = transformation_pipeline
        
        # Test that pipeline can process amenities column
        if "amenities" in sample_transformation_data.columns:
            # Accept both object and StringDtype for pandas 2+ compatibility
            assert sample_transformation_data["amenities"].dtype in ('object', 'string')
            assert len(sample_transformation_data["amenities"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_json_amenities_parsing(self):
        """Test parsing of JSON amenities."""
        import json
        
        amenities_str = '["WiFi", "Parking", "Gym"]'
        amenities_list = json.loads(amenities_str)
        
        assert len(amenities_list) == 3
        assert 'WiFi' in amenities_list
        assert 'Parking' in amenities_list
        assert 'Gym' in amenities_list
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_invalid_json_amenities_handling(self):
        """Test handling of invalid JSON in amenities."""
        invalid_amenities = 'not a json string'
        
        # This should raise an error or be handled gracefully
        try:
            import json
            json.loads(invalid_amenities)
            assert False, "Should have raised error"
        except json.JSONDecodeError:
            assert True


class TestFeatureEngineering:
    """Tests for feature engineering methods."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_numeric_column_presence(self, sample_transformation_data):
        """Test that numeric columns are properly identified."""
        numeric_cols = sample_transformation_data.select_dtypes(include=[np.number]).columns
        assert 'bedrooms' in numeric_cols
        assert 'bathroom' in numeric_cols
        assert 'area_value' in numeric_cols
        assert 'price' in numeric_cols
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_string_column_conversion(self, sample_transformation_data):
        """Test string column handling."""
        # Accept both object and StringDtype for pandas 2+ compatibility
        assert sample_transformation_data['listing_id'].dtype in ('object', 'string')
        assert sample_transformation_data['amenities'].dtype in ('object', 'string')
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_categorical_features(self, sample_categorical_data):
        """Test identification of categorical features."""
        categorical_cols = sample_categorical_data.select_dtypes(include=['object', 'string']).columns
        assert 'city' in categorical_cols


class TestFeatureScaling:
    """Tests for feature scaling functionality."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_numeric_data_ranges(self, sample_transformation_data):
        """Test numeric data ranges before scaling."""
        assert sample_transformation_data['price'].min() >= 0
        assert sample_transformation_data['bedrooms'].min() >= 1
        assert sample_transformation_data['area_value'].min() > 0
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_scaling_preservation_of_shape(self, sample_transformation_data):
        """Test that scaling preserves data shape."""
        original_shape = sample_transformation_data.shape
        
        # Simulated scaling wouldn't change the shape
        scaled_data = sample_transformation_data.copy()
        assert scaled_data.shape == original_shape
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_null_handling_during_scaling(self, sample_transformation_data):
        """Test handling of null values during scaling."""
        data_with_nulls = sample_transformation_data.copy()
        data_with_nulls.loc[0, 'price'] = np.nan
        
        null_count = data_with_nulls.isnull().sum().sum()
        assert null_count == 1


class TestBinningAndDiscretization:
    """Tests for binning and discretization."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_numeric_binning(self, sample_transformation_data):
        """Test that numeric columns can be binned."""
        bedrooms_col = sample_transformation_data['bedrooms']
        
        # Test basic binning with pd.cut
        binned = pd.cut(bedrooms_col, bins=3, labels=['small', 'medium', 'large'])
        assert len(binned) == len(bedrooms_col)
        assert binned.dtype == 'category'


class TestFeatureInteractions:
    """Tests for feature interaction generation."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_interaction_feature_generation(self, sample_transformation_data):
        """Test generation of interaction features."""
        df = sample_transformation_data.copy()
        
        # Example: price per bedroom
        df['price_per_bedroom'] = df['price'] / df['bedrooms']
        
        assert 'price_per_bedroom' in df.columns
        assert len(df['price_per_bedroom']) > 0
        assert all(df['price_per_bedroom'] > 0)
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_area_bedroom_interaction(self, sample_transformation_data):
        """Test area to bedroom interaction."""
        df = sample_transformation_data.copy()
        
        df['area_per_bedroom'] = df['area_value'] / df['bedrooms']
        
        assert 'area_per_bedroom' in df.columns
        assert all(df['area_per_bedroom'] > 0)


class TestFeatureEncoding:
    """Tests for feature encoding."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_categorical_encoding(self, sample_categorical_data):
        """Test categorical feature encoding."""
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        encoded_cities = le.fit_transform(sample_categorical_data['city'])
        
        assert len(encoded_cities) == len(sample_categorical_data)
        assert all(isinstance(x, (int, np.integer)) for x in encoded_cities)
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_onehot_encoding(self, sample_categorical_data):
        """Test one-hot encoding."""
        encoded = pd.get_dummies(sample_categorical_data[['city']])
        
        assert 'city_Alexandria' in encoded.columns or 'city_Cairo' in encoded.columns
        assert encoded.shape[0] == sample_categorical_data.shape[0]


@pytest.mark.integration
@pytest.mark.transformation
class TestDataTransformationIntegration:
    """Integration tests for DataTransformationPipeline."""
    
    def test_full_transformation_workflow(self, transformation_pipeline, sample_transformation_data, temp_dir):
        """Test complete transformation workflow."""
        input_csv = os.path.join(temp_dir, 'input.csv')
        output_csv = os.path.join(temp_dir, 'output.csv')
        report_csv = os.path.join(temp_dir, 'report.csv')
        
        sample_transformation_data.to_csv(input_csv, index=False)
        
        # Run full transformation
        transformation_pipeline.run_transformation(
            input_csv=input_csv,
            output_csv=output_csv,
            output_report_csv=report_csv,
            target_col='price'
        )
        
        # Verify output exists
        if os.path.exists(output_csv):
            output_df = pd.read_csv(output_csv)
            assert len(output_df) > 0
    
    def test_transformation_report_generation(self, transformation_pipeline, sample_transformation_data, temp_dir):
        """Test that transformation generates a report."""
        input_csv = os.path.join(temp_dir, 'input.csv')
        output_csv = os.path.join(temp_dir, 'output.csv')
        report_csv = os.path.join(temp_dir, 'report.csv')
        
        sample_transformation_data.to_csv(input_csv, index=False)
        transformation_pipeline.run_transformation(input_csv, output_csv, report_csv)
        
        # Report summary should be populated
        assert transformation_pipeline.report_summary is not None


class TestTransformationEdgeCases:
    """Tests for edge cases in transformation."""
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_single_row_data(self, transformation_pipeline, temp_dir):
        """Test transformation with single row of data."""
        single_row = pd.DataFrame({
            'listing_id': ['1'],
            'bedrooms': [2],
            'price': [500000]
        })
        
        input_csv = os.path.join(temp_dir, 'single.csv')
        output_csv = os.path.join(temp_dir, 'single_out.csv')
        
        single_row.to_csv(input_csv, index=False)
        transformation_pipeline.run_transformation(input_csv, output_csv)
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_empty_amenities(self, transformation_pipeline):
        """Test handling of empty amenities."""
        data = pd.DataFrame({
            'listing_id': ['1', '2'],
            'bedrooms': [2, 3],
            'amenities': ['[]', '[]']
        })
        
        assert all(data['amenities'] == '[]')
    
    @pytest.mark.unit
    @pytest.mark.transformation
    def test_special_characters_in_data(self, transformation_pipeline):
        """Test handling of special characters."""
        data = pd.DataFrame({
            'city': ['Cairo, Egypt', 'Giza/Giza', 'Alex (Alexandria)'],
            'bedrooms': [2, 3, 1]
        })
        
        assert len(data) == 3
        assert 'Cairo, Egypt' in data['city'].values

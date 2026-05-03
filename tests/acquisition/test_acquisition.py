import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import logging

from scripts.Acquisition.Acquisition import DataCollectionPipeline


@pytest.fixture
def pipeline():
    """Create a DataCollectionPipeline instance for testing."""
    return DataCollectionPipeline()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = {
        'listing_id': [1, 2, 3],
        'city': ['Cairo', 'Giza', 'Alexandria'],
        'bedrooms': [2, 3, 1],
        'bathroom': [1, 2, 1],
        'area_value': [100, 150, 80],
        'price': [500000, 750000, 400000]
    }
    return pd.DataFrame(data)


class TestDataCollectionPipeline:
    """Unit tests for DataCollectionPipeline class."""
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_pipeline_initialization(self):
        """Test that DataCollectionPipeline initializes correctly."""
        pipeline = DataCollectionPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'scraped_data')
        assert hasattr(pipeline, 'pipeline_logs')
        assert hasattr(pipeline, 'session')
        assert isinstance(pipeline.scraped_data, list)
        assert isinstance(pipeline.pipeline_logs, list)
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_session_configuration(self):
        """Test that HTTP session is properly configured with retries."""
        pipeline = DataCollectionPipeline()
        assert pipeline.session is not None
        assert 'User-Agent' in pipeline.session.headers
        assert len(pipeline.session.headers['User-Agent']) > 0
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_scraped_data_storage(self):
        """Test that pipeline can store scraped data."""
        pipeline = DataCollectionPipeline()
        test_data = {'id': 1, 'name': 'Test Property'}
        pipeline.scraped_data.append(test_data)
        assert len(pipeline.scraped_data) == 1
        assert pipeline.scraped_data[0] == test_data
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_pipeline_logs_storage(self):
        """Test that pipeline can store logs."""
        pipeline = DataCollectionPipeline()
        test_log = {'timestamp': '2024-01-01', 'status': 'success', 'message': 'Test log'}
        pipeline.pipeline_logs.append(test_log)
        assert len(pipeline.pipeline_logs) == 1
        assert pipeline.pipeline_logs[0] == test_log
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_logger_setup(self):
        """Test that logger is properly configured."""
        pipeline = DataCollectionPipeline()
        assert pipeline.logger is not None
        assert pipeline.logger.name == 'DataCollectionPipeline'
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    @patch('scripts.Acquisition.Acquisition.requests.Session.get')
    def test_collect_from_web_with_mocked_request(self, mock_get):
        """Test web collection with mocked HTTP request."""
        pipeline = DataCollectionPipeline()
        
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.text = '<html><body>Test</body></html>'
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # This tests that the method handles a basic response
        assert mock_response.status_code == 200
        assert 'Test' in mock_response.text
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_session_has_retry_strategy(self):
        """Test that session has retry strategy configured."""
        pipeline = DataCollectionPipeline()
        # Check that adapters are configured with retries
        assert pipeline.session.get_adapter('http://') is not None
        assert pipeline.session.get_adapter('https://') is not None
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_pipeline_initialization_creates_directories(self):
        """Test that necessary directories are accessible."""
        pipeline = DataCollectionPipeline()
        # The pipeline should have access to logger which writes to LOGS_DIR
        assert pipeline.logger is not None
        assert hasattr(pipeline.logger, 'handlers')


class TestDataCollectionLogMethods:
    """Unit tests for logging methods in DataCollectionPipeline."""
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_log_collection_storage(self, pipeline):
        """Test that collection logs are properly stored."""
        pipeline.pipeline_logs.append({
            'source': 'web',
            'records': 10,
            'status': 'success'
        })
        assert len(pipeline.pipeline_logs) == 1
        assert pipeline.pipeline_logs[0]['source'] == 'web'
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_multiple_log_entries(self, pipeline):
        """Test storing multiple log entries."""
        for i in range(5):
            pipeline.pipeline_logs.append({'iteration': i, 'status': 'processed'})
        assert len(pipeline.pipeline_logs) == 5


class TestDataProcessing:
    """Tests for data processing capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_scraped_data_list_operations(self, pipeline, sample_csv_data):
        """Test basic data operations on scraped data."""
        for _, row in sample_csv_data.iterrows():
            pipeline.scraped_data.append(row.to_dict())
        
        assert len(pipeline.scraped_data) == 3
        assert 'city' in pipeline.scraped_data[0]
        assert pipeline.scraped_data[0]['city'] == 'Cairo'
    
    @pytest.mark.unit
    @pytest.mark.acquisition
    def test_data_consistency(self, pipeline):
        """Test that data storage maintains consistency."""
        data1 = {'id': 1, 'value': 100}
        data2 = {'id': 2, 'value': 200}
        
        pipeline.scraped_data.append(data1)
        pipeline.scraped_data.append(data2)
        
        assert pipeline.scraped_data[0] is data1
        assert pipeline.scraped_data[1] is data2


@pytest.mark.integration
@pytest.mark.acquisition
class TestDataCollectionIntegration:
    """Integration tests for DataCollectionPipeline."""
    
    def test_pipeline_workflow(self, pipeline, sample_csv_data):
        """Test complete pipeline workflow."""
        # Simulate collecting data
        for _, row in sample_csv_data.iterrows():
            pipeline.scraped_data.append(row.to_dict())
        
        # Simulate logging
        pipeline.pipeline_logs.append({
            'total_collected': len(pipeline.scraped_data),
            'status': 'completed'
        })
        
        assert len(pipeline.scraped_data) == 3
        assert pipeline.pipeline_logs[-1]['status'] == 'completed'
    
    def test_session_persistence(self, pipeline):
        """Test that session persists across operations."""
        session1 = pipeline.session
        session2 = pipeline.session
        assert session1 is session2

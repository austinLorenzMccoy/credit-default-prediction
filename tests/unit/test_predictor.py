"""
Unit tests for the CreditPredictor class.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.predictor import CreditPredictor
from src.config import settings

@pytest.fixture
def mock_models():
    """Fixture to create mock models and scalers."""
    with patch('src.models.predictor.load_model_safely') as mock_model_loader, \
         patch('src.models.predictor.load_scaler_safely') as mock_scaler_loader:
        
        # Configure mocks
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.25]])
        mock_model_loader.return_value = mock_model
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_scaler_loader.return_value = mock_scaler
        
        yield mock_model, mock_scaler

def test_predictor_initialization(mock_models):
    """Test that the predictor initializes correctly."""
    mock_model, mock_scaler = mock_models
    
    # Create predictor
    predictor = CreditPredictor()
    
    # Assert
    assert predictor.is_ready is True
    assert predictor.classification_model is not None
    assert predictor.regression_model is not None
    assert predictor.classification_scaler is not None
    assert predictor.regression_scaler is not None

def test_predict_default(mock_models, sample_input_data):
    """Test the default prediction functionality."""
    mock_model, mock_scaler = mock_models
    
    # Create predictor
    predictor = CreditPredictor()
    
    # Make prediction
    success, result = predictor.predict_default(sample_input_data)
    
    # Assert
    assert success is True
    assert "prediction" in result
    assert "probability" in result
    assert "is_high_risk" in result
    assert result["probability"] == 0.25
    assert result["is_high_risk"] is False
    assert "Low Risk of Default" in result["prediction"]

def test_predict_credit_limit(mock_models, sample_input_data):
    """Test the credit limit prediction functionality."""
    mock_model, mock_scaler = mock_models
    
    # Create predictor
    predictor = CreditPredictor()
    
    # Make prediction
    success, result = predictor.predict_credit_limit(sample_input_data)
    
    # Assert
    assert success is True
    assert "predicted_credit_limit" in result
    assert result["predicted_credit_limit"] == 0.25

def test_predictor_not_ready():
    """Test behavior when models are not loaded properly."""
    with patch('src.models.predictor.load_model_safely') as mock_model_loader:
        # Configure mock to return None (failed to load model)
        mock_model_loader.return_value = None
        
        # Create predictor
        predictor = CreditPredictor()
        
        # Assert
        assert predictor.is_ready is False
        
        # Test predictions when not ready
        success, result = predictor.predict_default({})
        assert success is False
        assert "error" in result
        
        success, result = predictor.predict_credit_limit({})
        assert success is False
        assert "error" in result

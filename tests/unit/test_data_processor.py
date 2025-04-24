"""
Unit tests for data processor utilities.
"""
import pytest
import numpy as np
from src.utils.data_processor import prepare_input_data, format_default_prediction

def test_prepare_input_data():
    """Test that input data is prepared correctly."""
    # Test data
    input_dict = {"feature1": 1, "feature2": 2, "feature3": 3}
    feature_order = ["feature1", "feature3", "feature2"]
    
    # Expected output
    expected = np.array([[1, 3, 2]])
    
    # Actual output
    result = prepare_input_data(input_dict, feature_order)
    
    # Assert
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_prepare_input_data_missing_features():
    """Test that missing features are handled correctly."""
    # Test data
    input_dict = {"feature1": 1, "feature3": 3}
    feature_order = ["feature1", "feature2", "feature3"]
    
    # Expected output (missing feature2 should be 0)
    expected = np.array([[1, 0, 3]])
    
    # Actual output
    result = prepare_input_data(input_dict, feature_order)
    
    # Assert
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_format_default_prediction_low_risk():
    """Test formatting for low risk predictions."""
    # Test data
    prediction = 0.25
    
    # Expected output
    expected = "Low Risk of Default (Probability: 25.00%)"
    
    # Actual output
    result = format_default_prediction(prediction)
    
    # Assert
    assert result == expected

def test_format_default_prediction_high_risk():
    """Test formatting for high risk predictions."""
    # Test data
    prediction = 0.75
    
    # Expected output
    expected = "High Risk of Default (Probability: 75.00%)"
    
    # Actual output
    result = format_default_prediction(prediction)
    
    # Assert
    assert result == expected

"""
Unit tests for model loader utilities.
"""
import os
import pytest
import tempfile
import pickle
import numpy as np
from unittest.mock import patch, MagicMock

from src.utils.model_loader import load_model_safely, load_scaler_safely

def test_load_model_safely_file_not_found():
    """Test handling of non-existent model files."""
    # Test with non-existent file
    result = load_model_safely("non_existent_file.keras")
    
    # Assert
    assert result is None

@patch('src.utils.model_loader.load_model')
def test_load_model_safely_exception(mock_load_model):
    """Test handling of exceptions during model loading."""
    # Setup mock to raise exception
    mock_load_model.side_effect = Exception("Test exception")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.keras') as temp_file:
        # Test with file that exists but causes exception
        result = load_model_safely(temp_file.name)
        
        # Assert
        assert result is None
        mock_load_model.assert_called_once_with(temp_file.name)

def test_load_scaler_safely_file_not_found():
    """Test handling of non-existent scaler files."""
    # Test with non-existent file
    result = load_scaler_safely("non_existent_file.pkl")
    
    # Assert
    assert result is None

def test_load_scaler_safely_valid_file():
    """Test loading a valid scaler file."""
    # Create a temporary file with a valid pickle
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
        # Create a simple object to pickle
        test_data = {"test": "data"}
        pickle.dump(test_data, temp_file)
    
    try:
        # Test with valid file
        result = load_scaler_safely(temp_file.name)
        
        # Assert
        assert result == test_data
    finally:
        # Clean up
        os.unlink(temp_file.name)

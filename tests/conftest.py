"""
Pytest configuration and fixtures.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.models.predictor import CreditPredictor
from src.api.app import app
from fastapi.testclient import TestClient

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing."""
    predictor = MagicMock(spec=CreditPredictor)
    predictor.is_ready = True
    predictor.classification_model = MagicMock()
    predictor.regression_model = MagicMock()
    predictor.classification_scaler = MagicMock()
    predictor.regression_scaler = MagicMock()
    return predictor

@pytest.fixture
def sample_input_data():
    """Sample input data for testing."""
    return {
        "LIMIT_BAL": 100000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 25,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 10000,
        "BILL_AMT2": 9000,
        "BILL_AMT3": 8000,
        "BILL_AMT4": 7000,
        "BILL_AMT5": 6000,
        "BILL_AMT6": 5000,
        "PAY_AMT1": 2000,
        "PAY_AMT2": 2000,
        "PAY_AMT3": 2000,
        "PAY_AMT4": 2000,
        "PAY_AMT5": 2000,
        "PAY_AMT6": 2000
    }

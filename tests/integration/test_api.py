"""
Integration tests for the FastAPI application.
"""
import pytest
pytest_plugins = ["pytest_asyncio"]
from httpx import AsyncClient
from unittest.mock import patch, MagicMock

from src.api.app import app

@pytest.fixture
def mock_predictor():
    """Fixture to create a mock predictor."""
    with patch('src.api.router.CreditPredictor') as mock:
        # Configure the mock
        instance = mock.return_value
        instance.is_ready = True
        
        # Return the mock
        yield instance

@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root endpoint returns correct information."""
    async with AsyncClient(base_url="http://test") as client:
        response = await client.get("/")
        
        assert response.status_code == 200
        assert "Welcome to" in response.json()["message"]
        assert "documentation" in response.json()
        assert "endpoints" in response.json()

@pytest.mark.asyncio
async def test_health_endpoint(mock_predictor):
    """Test the health endpoint."""
    # Configure mock
    mock_predictor.classification_model = MagicMock()
    mock_predictor.regression_model = MagicMock()
    mock_predictor.classification_scaler = MagicMock()
    mock_predictor.regression_scaler = MagicMock()
    
    # Make request
    async with AsyncClient(base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        
        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert all(response.json()["model_status"].values())

@pytest.mark.asyncio
async def test_predict_default_endpoint(mock_predictor):
    """Test the default prediction endpoint."""
    # Configure mock
    mock_predictor.predict_default.return_value = (
        True, 
        {
            "prediction": "Low Risk of Default (Probability: 25.00%)",
            "probability": 0.25,
            "is_high_risk": False
        }
    )
    
    # Test data
    test_data = {
        "data": {
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
    }
    
    # Make request
    async with AsyncClient(base_url="http://test") as client:
        response = await client.post("/api/v1/predict/default", json=test_data)
        
        # Assert
        assert response.status_code == 200
        assert response.json()["prediction"] == "Low Risk of Default (Probability: 25.00%)"
        assert response.json()["probability"] == 0.25
        assert response.json()["is_high_risk"] is False
        
        # Verify mock was called with correct data
        mock_predictor.predict_default.assert_called_once_with(test_data["data"])

@pytest.mark.asyncio
async def test_predict_credit_limit_endpoint(mock_predictor):
    """Test the credit limit prediction endpoint."""
    # Configure mock
    mock_predictor.predict_credit_limit.return_value = (
        True, 
        {
            "predicted_credit_limit": 150000.0
        }
    )
    
    # Test data
    test_data = {
        "data": {
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
    }
    
    # Make request
    async with AsyncClient(base_url="http://test") as client:
        response = await client.post("/api/v1/predict/credit-limit", json=test_data)
        
        # Assert
        assert response.status_code == 200
        assert response.json()["predicted_credit_limit"] == 150000.0
        
        # Verify mock was called with correct data
        mock_predictor.predict_credit_limit.assert_called_once_with(test_data["data"])

"""
API router for credit prediction endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

# Use the mock predictor for now due to model loading issues
from app.models.mock_predictor import MockPredictor
from app.schemas.request_models import (
    SimplifiedPredictionRequest, 
    DefaultPredictionResponse, 
    CreditLimitPredictionResponse,
    ErrorResponse,
    HealthResponse
)

router = APIRouter()

def get_predictor():
    """Dependency to get the predictor instance."""
    # Using MockPredictor instead of CreditPredictor due to model loading issues
    predictor = MockPredictor()
    if not predictor.is_ready:
        raise HTTPException(status_code=500, detail="Predictor not initialized properly")
    return predictor

@router.post(
    "/predict/default", 
    response_model=DefaultPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict credit card default risk",
    description="Predicts the probability of credit card default based on simplified customer data"
)
def predict_default(request: SimplifiedPredictionRequest, predictor = Depends(get_predictor)):
    """Predict credit default risk based on simplified customer data."""
    success, result = predictor.predict_default(request)
    
    if not success:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    
    return result

@router.post(
    "/predict/credit-limit", 
    response_model=CreditLimitPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict credit limit",
    description="Predicts the appropriate credit limit based on simplified customer data"
)
def predict_credit_limit(request: SimplifiedPredictionRequest, predictor = Depends(get_predictor)):
    """Predict credit limit based on simplified customer data."""
    success, result = predictor.predict_credit_limit(request)
    
    if not success:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    
    return result

@router.get(
    "/health", 
    response_model=HealthResponse,
    summary="Check API health",
    description="Checks the health of the API"
)
def health_check(predictor = Depends(get_predictor)):
    """Check the health of the API."""
    # For MockPredictor, we just check if it's ready
    model_status = {
        "predictor_ready": predictor.is_ready
    }
    
    return {
        "status": "healthy" if predictor.is_ready else "degraded",
        "model_status": model_status
    }

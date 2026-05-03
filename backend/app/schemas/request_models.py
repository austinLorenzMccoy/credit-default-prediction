"""
Pydantic models for API request and response validation.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator

class SimplifiedPredictionRequest(BaseModel):
    """Simplified model for prediction requests with fewer parameters."""
    credit_limit: float = Field(..., description="Current credit limit")
    age: int = Field(..., description="Customer age")
    gender: int = Field(..., description="Gender (1=male, 2=female)")
    education: int = Field(..., description="Education level (1=graduate, 2=university, 3=high school, 4=others)")
    marital_status: int = Field(..., description="Marital status (1=married, 2=single, 3=others)")
    payment_status: int = Field(..., description="Last month's payment status (0=paid duly, 1=1 month delay, 2=2 months delay, etc.)")
    bill_amount: float = Field(..., description="Last month's bill amount")
    payment_amount: float = Field(..., description="Last month's payment amount")

class DefaultPredictionResponse(BaseModel):
    """Response model for default prediction."""
    prediction: str = Field(..., description="Human-readable prediction result")
    probability: float = Field(..., description="Raw probability value (0-1)")
    is_high_risk: bool = Field(..., description="Whether the prediction indicates high risk")
    risk_factors: List[str] = Field(default=[], description="Factors contributing to the risk assessment")
    
class CreditLimitPredictionResponse(BaseModel):
    """Response model for credit limit prediction."""
    predicted_credit_limit: float = Field(..., description="Predicted credit limit value")
    adjustment_factor: float = Field(..., description="Factor by which the current limit was adjusted")
    recommendation_factors: List[str] = Field(default=[], description="Factors contributing to the recommendation")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="API health status: 'healthy' or 'degraded'")
    model_status: Dict[str, bool] = Field(..., description="Status of individual models and scalers")

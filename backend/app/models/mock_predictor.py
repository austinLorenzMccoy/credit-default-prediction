"""
Mock predictor for testing the API.
"""
import logging
import random
from typing import Dict, Any, Tuple, List
from app.schemas.request_models import SimplifiedPredictionRequest

logger = logging.getLogger(__name__)

class MockPredictor:
    """Mock predictor class for testing the API."""
    
    def __init__(self):
        """Initialize the mock predictor."""
        self.is_ready = True
        logger.info("Mock predictor initialized")
    
    def predict_default(self, input_data: SimplifiedPredictionRequest) -> Tuple[bool, Dict[str, Any]]:
        """
        Intelligent prediction for credit default risk based on simplified parameters.
        
        Args:
            input_data: SimplifiedPredictionRequest with customer data
            
        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Calculate risk score based on meaningful parameters
            risk_score = 0.0
            risk_factors = []
            
            # Payment status is a strong indicator
            if input_data.payment_status > 0:
                risk_score += 0.2 * input_data.payment_status
                risk_factors.append(f"Payment delay of {input_data.payment_status} month(s)")
            
            # Bill to payment ratio
            if input_data.bill_amount > 0:
                payment_ratio = input_data.payment_amount / input_data.bill_amount
                if payment_ratio < 0.1:
                    risk_score += 0.3
                    risk_factors.append("Very low payment to bill ratio")
                elif payment_ratio < 0.3:
                    risk_score += 0.15
                    risk_factors.append("Low payment to bill ratio")
            
            # Credit utilization (assuming bill_amount represents utilization)
            if input_data.bill_amount > input_data.credit_limit * 0.8:
                risk_score += 0.15
                risk_factors.append("High credit utilization")
            
            # Age factor - younger customers statistically higher risk
            if input_data.age < 25:
                risk_score += 0.1
                risk_factors.append("Age under 25")
            
            # Cap the risk score at 0.95
            prediction = min(risk_score, 0.95)
            # Ensure minimum risk is 0.05
            prediction = max(prediction, 0.05)
            
            # Format result
            if prediction < 0.5:
                result = f"Low Risk of Default (Probability: {prediction:.2%})"
            else:
                result = f"High Risk of Default (Probability: {prediction:.2%})"
            
            return True, {
                "prediction": result,
                "probability": float(prediction),
                "is_high_risk": prediction >= 0.5,
                "risk_factors": risk_factors
            }
        
        except Exception as e:
            logger.error(f"Error in default prediction: {str(e)}")
            return False, {"error": str(e)}
    
    def predict_credit_limit(self, input_data: SimplifiedPredictionRequest) -> Tuple[bool, Dict[str, Any]]:
        """
        Intelligent prediction for credit limit based on simplified parameters.
        
        Args:
            input_data: SimplifiedPredictionRequest with customer data
            
        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Start with the current credit limit as base
            base_limit = input_data.credit_limit
            adjustment_factor = 1.0
            recommendation_factors = []
            
            # Payment history affects credit limit
            if input_data.payment_status == 0:
                # Good payment history
                adjustment_factor += 0.2
                recommendation_factors.append("Good payment history")
            else:
                # Payment delays reduce potential increase
                adjustment_factor -= 0.1 * input_data.payment_status
                recommendation_factors.append(f"Payment delay affects potential increase")
            
            # Payment to bill ratio
            if input_data.bill_amount > 0:
                payment_ratio = input_data.payment_amount / input_data.bill_amount
                if payment_ratio > 0.5:
                    adjustment_factor += 0.15
                    recommendation_factors.append("High payment to bill ratio")
            
            # Age factor - older customers often qualify for higher limits
            if input_data.age > 30:
                adjustment_factor += 0.1
                recommendation_factors.append("Age factor positive")
            
            # Education level can affect credit worthiness
            if input_data.education <= 2:  # Graduate or university
                adjustment_factor += 0.05
                recommendation_factors.append("Education level positive")
            
            # Ensure adjustment is reasonable
            adjustment_factor = max(0.8, min(adjustment_factor, 1.5))
            
            # Calculate new limit
            predicted_limit = base_limit * adjustment_factor
            
            # Round to nearest 1000
            predicted_limit = round(predicted_limit / 1000) * 1000
            
            return True, {
                "predicted_credit_limit": float(predicted_limit),
                "adjustment_factor": float(adjustment_factor),
                "recommendation_factors": recommendation_factors
            }
        
        except Exception as e:
            logger.error(f"Error in credit limit prediction: {str(e)}")
            return False, {"error": str(e)}

"""
Prediction models for credit default and credit limit.
"""
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.utils.data_processor import prepare_input_data, format_default_prediction
from src.utils.model_loader import load_model_safely, load_scaler_safely

logger = logging.getLogger(__name__)

class CreditPredictor:
    """Class for handling credit default and credit limit predictions."""
    
    def __init__(self):
        """Initialize the predictor by loading models and scalers."""
        self.classification_model = None
        self.regression_model = None
        self.classification_scaler = None
        self.regression_scaler = None
        self.is_ready = False
        self.load_models()
    
    def load_models(self) -> None:
        """Load all required models and scalers."""
        self.classification_model = load_model_safely(settings.CLASSIFICATION_MODEL_PATH)
        self.regression_model = load_model_safely(settings.REGRESSION_MODEL_PATH)
        self.classification_scaler = load_scaler_safely(settings.CLASSIFICATION_SCALER_PATH)
        self.regression_scaler = load_scaler_safely(settings.REGRESSION_SCALER_PATH)
        
        # Check if all models and scalers are loaded
        self.is_ready = all([
            self.classification_model is not None,
            self.regression_model is not None,
            self.classification_scaler is not None,
            self.regression_scaler is not None
        ])
        
        if self.is_ready:
            logger.info("All models and scalers loaded successfully")
        else:
            logger.error("Failed to load all models and scalers")
    
    def predict_default(self, input_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Predict credit default risk.
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            Tuple of (success, result_dict)
        """
        if not self.is_ready:
            return False, {"error": "Models not loaded properly"}
        
        try:
            # Prepare input data
            features = prepare_input_data(input_data, settings.CLASSIFICATION_FEATURES)
            
            # Scale input data
            scaled_features = self.classification_scaler.transform(features)
            
            # Make prediction
            prediction = float(self.classification_model.predict(scaled_features)[0][0])
            
            # Format result
            result = format_default_prediction(prediction)
            
            return True, {
                "prediction": result,
                "probability": float(prediction),
                "is_high_risk": prediction >= 0.5
            }
        
        except Exception as e:
            logger.error(f"Error in default prediction: {str(e)}")
            return False, {"error": str(e)}
    
    def predict_credit_limit(self, input_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Predict credit limit.
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            Tuple of (success, result_dict)
        """
        if not self.is_ready:
            return False, {"error": "Models not loaded properly"}
        
        try:
            # Prepare input data
            features = prepare_input_data(input_data, settings.CLASSIFICATION_FEATURES)
            
            # Scale input data
            scaled_features = self.regression_scaler.transform(features)
            
            # Make prediction
            prediction = float(self.regression_model.predict(scaled_features)[0][0])
            
            return True, {
                "predicted_credit_limit": prediction
            }
        
        except Exception as e:
            logger.error(f"Error in credit limit prediction: {str(e)}")
            return False, {"error": str(e)}

"""
Test script to diagnose model loading issues.
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from app.utils.model_loader import load_model_safely, load_scaler_safely
from app.config import settings

def test_model_loading():
    """Test loading models and scalers."""
    logger.info(f"Project root: {project_root}")
    
    # Check if model files exist
    logger.info(f"Checking if model files exist:")
    logger.info(f"Classification model path: {settings.CLASSIFICATION_MODEL_PATH}")
    logger.info(f"Regression model path: {settings.REGRESSION_MODEL_PATH}")
    logger.info(f"Classification scaler path: {settings.CLASSIFICATION_SCALER_PATH}")
    logger.info(f"Regression scaler path: {settings.REGRESSION_SCALER_PATH}")
    
    logger.info(f"Classification model exists: {os.path.exists(settings.CLASSIFICATION_MODEL_PATH)}")
    logger.info(f"Regression model exists: {os.path.exists(settings.REGRESSION_MODEL_PATH)}")
    logger.info(f"Classification scaler exists: {os.path.exists(settings.CLASSIFICATION_SCALER_PATH)}")
    logger.info(f"Regression scaler exists: {os.path.exists(settings.REGRESSION_SCALER_PATH)}")
    
    # Try loading models
    logger.info("Attempting to load models:")
    classification_model = load_model_safely(settings.CLASSIFICATION_MODEL_PATH)
    regression_model = load_model_safely(settings.REGRESSION_MODEL_PATH)
    
    logger.info(f"Classification model loaded: {classification_model is not None}")
    logger.info(f"Regression model loaded: {regression_model is not None}")
    
    # Try loading scalers
    logger.info("Attempting to load scalers:")
    classification_scaler = load_scaler_safely(settings.CLASSIFICATION_SCALER_PATH)
    regression_scaler = load_scaler_safely(settings.REGRESSION_SCALER_PATH)
    
    logger.info(f"Classification scaler loaded: {classification_scaler is not None}")
    logger.info(f"Regression scaler loaded: {regression_scaler is not None}")
    
    # Check if all models and scalers are loaded
    all_loaded = all([
        classification_model is not None,
        regression_model is not None,
        classification_scaler is not None,
        regression_scaler is not None
    ])
    
    logger.info(f"All models and scalers loaded: {all_loaded}")
    
    return all_loaded

if __name__ == "__main__":
    test_model_loading()

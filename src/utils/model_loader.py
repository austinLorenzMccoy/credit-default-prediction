"""
Utility functions for loading ML models and scalers.
"""
import os
import pickle
import logging
from typing import Any, Optional

from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

def load_model_safely(model_path: str) -> Optional[Any]:
    """
    Safely load a Keras model with error handling.
    
    Args:
        model_path (str): Path to the .keras model file
    
    Returns:
        Loaded Keras model or None if loading fails
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Error: Model file {model_path} not found.")
            return None
        return load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

def load_scaler_safely(scaler_path: str) -> Optional[Any]:
    """
    Safely load a pickle scaler with error handling.
    
    Args:
        scaler_path (str): Path to the .pkl scaler file
    
    Returns:
        Loaded scaler or None if loading fails
    """
    try:
        if not os.path.exists(scaler_path):
            logger.error(f"Error: Scaler file {scaler_path} not found.")
            return None
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading scaler {scaler_path}: {e}")
        return None

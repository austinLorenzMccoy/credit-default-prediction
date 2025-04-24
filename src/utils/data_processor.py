"""
Utility functions for data processing and preparation.
"""
import numpy as np
from typing import Dict, List, Any

def prepare_input_data(input_dict: Dict[str, Any], feature_order: List[str]) -> np.ndarray:
    """
    Prepare input data in the correct order for the model
    
    Args:
        input_dict (Dict): Input data dictionary
        feature_order (List): Ordered list of features
    
    Returns:
        numpy array of features
    """
    # Ensure all features are present
    input_data = [input_dict.get(feature, 0) for feature in feature_order]
    return np.array(input_data).reshape(1, -1)

def format_default_prediction(prediction: float) -> str:
    """
    Format the default prediction into a human-readable string
    
    Args:
        prediction (float): Raw prediction value (0-1)
    
    Returns:
        str: Formatted prediction result
    """
    if prediction < 0.5:
        return f"Low Risk of Default (Probability: {prediction:.2%})"
    else:
        return f"High Risk of Default (Probability: {prediction:.2%})"

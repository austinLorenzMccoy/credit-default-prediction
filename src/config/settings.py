"""
Configuration settings for the Credit Default Prediction application.
"""
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Model file paths
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "classification_model.keras")
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "regression_model.keras")
CLASSIFICATION_SCALER_PATH = os.path.join(BASE_DIR, "models", "classification_scaler.pkl")
REGRESSION_SCALER_PATH = os.path.join(BASE_DIR, "models", "regression_scaler.pkl")

# API settings
API_TITLE = "Credit Default Prediction API"
API_DESCRIPTION = "API for predicting credit card default risk and credit limits"
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# Feature lists
CLASSIFICATION_FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

"""
Model training script for credit default prediction and credit limit recommendation.
"""
import os
import logging
import datetime
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# TensorFlow removed - using scikit-learn only for Python 3.14 compatibility
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import RandomOverSampler

from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training credit prediction models."""
    
    def __init__(self, data_path=None):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to the dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.classification_features = settings.CLASSIFICATION_FEATURES
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(settings.BASE_DIR, "models"), exist_ok=True)
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.join(settings.BASE_DIR, "logs"), exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        
        # Strip column names
        self.data.columns = self.data.columns.str.strip()
        
        # Ensure numeric conversion for all columns
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        
        # Data Cleaning
        self.data = self.data[self.data['default_payment_next_month'].isin([0, 1])]  # Remove invalid labels
        self.data.fillna(self.data.median(), inplace=True)  # Impute missing values
        
        logger.info(f"Data loaded successfully with {len(self.data)} records")
        
        return self.data
    
    def train_classification_model(self):
        """Train the default prediction classification model."""
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
        
        logger.info("Training classification model...")
        
        # Prepare data
        X_class = self.data[self.classification_features]
        y_class = self.data['default_payment_next_month']
        
        # Handle Class Imbalance
        logger.info("Applying random oversampling to balance classes...")
        oversampler = RandomOverSampler(random_state=42)
        X_class_balanced, y_class_balanced = oversampler.fit_resample(X_class, y_class)
        
        # Scale Features
        logger.info("Scaling features...")
        scaler_class = StandardScaler()
        X_class_balanced_scaled = scaler_class.fit_transform(X_class_balanced)
        
        # Save Classification Scaler
        scaler_path = os.path.join(settings.BASE_DIR, "models", "classification_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_class, f)
        logger.info(f"Classification scaler saved to {scaler_path}")
        
        # Split Dataset
        X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
            X_class_balanced_scaled, y_class_balanced, test_size=0.2, random_state=42, stratify=y_class_balanced
        )
        
        # Build Classification Model
        logger.info("Building classification model...")
        classification_model = Sequential([
            Dense(64, activation='relu', input_dim=X_class_train.shape[1]),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        log_dir_class = os.path.join(settings.BASE_DIR, "logs", f"classification_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir_class, exist_ok=True)
        
        early_stopping_class = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        tensorboard_callback_class = TensorBoard(log_dir=log_dir_class, histogram_freq=1)
        
        # Train Classification Model
        logger.info("Training classification model...")
        classification_history = classification_model.fit(
            X_class_train, y_class_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping_class, tensorboard_callback_class],
            verbose=1
        )
        
        # Evaluate Classification Model
        class_test_loss, class_test_accuracy = classification_model.evaluate(X_class_test, y_class_test, verbose=0)
        logger.info(f"Classification Test Accuracy: {class_test_accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(settings.BASE_DIR, "models", "classification_model.keras")
        classification_model.save(model_path)
        logger.info(f"Classification model saved to {model_path}")
        
        return classification_model
    
    def train_regression_model(self):
        """Train the credit limit regression model."""
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
        
        logger.info("Training regression model...")
        
        # Prepare data
        regression_features = [
            col for col in self.classification_features 
            if col not in ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']  # Adjust features as needed
        ]
        
        X_reg = self.data[regression_features]
        y_reg = self.data['LIMIT_BAL']
        
        # Split Dataset
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Scale Features
        logger.info("Scaling features...")
        scaler_reg = StandardScaler()
        X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
        X_reg_test_scaled = scaler_reg.transform(X_reg_test)
        
        # Save Regression Scaler
        scaler_path = os.path.join(settings.BASE_DIR, "models", "regression_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_reg, f)
        logger.info(f"Regression scaler saved to {scaler_path}")
        
        # Build Regression Model
        logger.info("Building regression model...")
        regression_model = Sequential([
            Dense(128, activation='relu', input_dim=X_reg_train_scaled.shape[1], kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        regression_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Callbacks
        log_dir_reg = os.path.join(settings.BASE_DIR, "logs", f"regression_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir_reg, exist_ok=True)
        
        early_stopping_reg = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        tensorboard_callback_reg = TensorBoard(log_dir=log_dir_reg, histogram_freq=1)
        
        # Train Regression Model
        logger.info("Training regression model...")
        regression_history = regression_model.fit(
            X_reg_train_scaled, y_reg_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping_reg, tensorboard_callback_reg],
            verbose=1
        )
        
        # Evaluate Regression Model
        y_reg_pred = regression_model.predict(X_reg_test_scaled)
        reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
        reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
        reg_r2 = r2_score(y_reg_test, y_reg_pred)
        
        logger.info(f"Regression Metrics:\nMSE: {reg_mse:.4f}\nMAE: {reg_mae:.4f}\nR²: {reg_r2:.4f}")
        
        # Save model
        model_path = os.path.join(settings.BASE_DIR, "models", "regression_model.keras")
        regression_model.save(model_path)
        logger.info(f"Regression model saved to {model_path}")
        
        return regression_model
    
    def train_all_models(self):
        """Train both classification and regression models."""
        if self.data is None:
            self.load_data()
        
        logger.info("Training all models...")
        classification_model = self.train_classification_model()
        regression_model = self.train_regression_model()
        
        logger.info("All models trained successfully!")
        
        return classification_model, regression_model


if __name__ == "__main__":
    # Path to the dataset
    data_path = os.path.join(settings.BASE_DIR, "notebook", "data", "default_of_credit_card.csv")
    
    # Initialize and run the model trainer
    trainer = ModelTrainer(data_path)
    trainer.train_all_models()

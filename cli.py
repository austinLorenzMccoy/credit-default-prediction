#!/usr/bin/env python
"""
Command-line interface for the Credit Default Prediction API.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.models.train import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)

def run_api(host="0.0.0.0", port=8000, reload=True):
    """Run the FastAPI application."""
    import uvicorn
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def train_models(data_path=None):
    """Train the machine learning models."""
    if data_path is None:
        data_path = os.path.join(settings.BASE_DIR, "notebook", "data", "default_of_credit_card.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)
    
    logger.info(f"Training models using data from {data_path}")
    trainer = ModelTrainer(data_path)
    trainer.train_all_models()

def run_tests():
    """Run the test suite."""
    import pytest
    logger.info("Running test suite")
    pytest.main(["-xvs", "tests"])

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Credit Default Prediction API CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    api_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the machine learning models")
    train_parser.add_argument("--data", help="Path to the data file")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run the test suite")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api(args.host, args.port, not args.no_reload)
    elif args.command == "train":
        train_models(args.data)
    elif args.command == "test":
        run_tests()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

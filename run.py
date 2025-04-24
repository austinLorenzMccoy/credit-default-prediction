"""
Entry point for the Credit Default Prediction API.
"""
import os
import logging
import uvicorn
from src.api.app import app

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI application
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

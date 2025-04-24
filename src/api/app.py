"""
Main FastAPI application.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.router import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix=settings.API_PREFIX)

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return JSONResponse({
        "message": f"Welcome to {settings.API_TITLE}",
        "version": settings.API_VERSION,
        "documentation": "/docs",
        "endpoints": {
            f"{settings.API_PREFIX}/predict/default": "Predict credit card default probability",
            f"{settings.API_PREFIX}/predict/credit-limit": "Predict credit limit",
            f"{settings.API_PREFIX}/health": "Check API and model health status"
        },
        "input_format": {
            "example": {
                "credit_limit": 100000,
                "age": 25,
                "gender": 2,
                "education": 2,
                "marital_status": 1,
                "payment_status": 0,
                "bill_amount": 10000,
                "payment_amount": 2000
            }
        }
    })

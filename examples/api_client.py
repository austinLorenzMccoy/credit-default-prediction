#!/usr/bin/env python3
"""
Example client for the Credit Default Prediction API.
This script demonstrates how to interact with the API using Python requests.
"""

import requests
import json
import argparse
from typing import Dict, Any, Optional


class CreditPredictionClient:
    """Client for interacting with the Credit Default Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip('/')
        self.api_prefix = "/api/v1"
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the API."""
        url = f"{self.base_url}{self.api_prefix}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def predict_default(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the probability of credit default."""
        url = f"{self.base_url}{self.api_prefix}/predict/default"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def predict_credit_limit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the recommended credit limit."""
        url = f"{self.base_url}{self.api_prefix}/predict/credit-limit"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()


def create_sample_data(low_risk: bool = True) -> Dict[str, Any]:
    """Create sample data for API requests.
    
    Args:
        low_risk: If True, create data for a low-risk customer.
                 If False, create data for a high-risk customer.
    
    Returns:
        Dictionary with sample customer data.
    """
    if low_risk:
        return {
            "credit_limit": 100000,
            "age": 35,
            "gender": 2,  # female
            "education": 1,  # graduate
            "marital_status": 1,  # married
            "payment_status": 0,  # paid duly
            "bill_amount": 10000,
            "payment_amount": 8000  # high payment ratio
        }
    else:
        return {
            "credit_limit": 50000,
            "age": 22,
            "gender": 1,  # male
            "education": 3,  # high school
            "marital_status": 2,  # single
            "payment_status": 2,  # 2 months delay
            "bill_amount": 30000,
            "payment_amount": 1000  # low payment ratio
        }


def main():
    """Main function to demonstrate API usage."""
    parser = argparse.ArgumentParser(description="Credit Prediction API Client")
    parser.add_argument("--url", default="http://localhost:8000", 
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--endpoint", choices=["health", "default", "credit-limit", "all"],
                        default="all", help="API endpoint to call (default: all)")
    parser.add_argument("--risk", choices=["low", "high"], default="low",
                        help="Risk profile for sample data (default: low)")
    
    args = parser.parse_args()
    client = CreditPredictionClient(args.url)
    
    # Determine risk profile for sample data
    low_risk = args.risk == "low"
    sample_data = create_sample_data(low_risk)
    
    # Call the requested endpoint(s)
    if args.endpoint in ["health", "all"]:
        print("\n=== Health Check ===")
        health_result = client.check_health()
        print(json.dumps(health_result, indent=2))
    
    if args.endpoint in ["default", "all"]:
        print("\n=== Default Prediction ===")
        print(f"Input data ({args.risk} risk profile):")
        print(json.dumps(sample_data, indent=2))
        default_result = client.predict_default(sample_data)
        print("\nResult:")
        print(json.dumps(default_result, indent=2))
    
    if args.endpoint in ["credit-limit", "all"]:
        print("\n=== Credit Limit Prediction ===")
        print(f"Input data ({args.risk} risk profile):")
        print(json.dumps(sample_data, indent=2))
        limit_result = client.predict_credit_limit(sample_data)
        print("\nResult:")
        print(json.dumps(limit_result, indent=2))


if __name__ == "__main__":
    main()

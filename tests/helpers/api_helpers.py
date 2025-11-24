"""
API test helpers for testing REST endpoints.

Provides utilities for making API requests, validating responses,
and managing test authentication.
"""

from typing import Dict, Any, Optional
from fastapi.testclient import TestClient
import json


class APITestHelper:
    """Helper class for API testing."""
    
    def __init__(self, client: TestClient):
        self.client = client
        self.default_headers = {
            "Content-Type": "application/json"
        }
        self._auth_token: Optional[str] = None
        
    def set_auth_token(self, token: str):
        """Set authentication token for requests."""
        self._auth_token = token
        
    def get_headers(self, additional_headers: Optional[Dict] = None) -> Dict:
        """Get headers with authentication."""
        headers = self.default_headers.copy()
        
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
            
        if additional_headers:
            headers.update(additional_headers)
            
        return headers
        
    def get(self, url: str, **kwargs) -> Any:
        """Make GET request."""
        headers = self.get_headers(kwargs.pop("headers", None))
        response = self.client.get(url, headers=headers, **kwargs)
        return response
        
    def post(self, url: str, data: Optional[Dict] = None, **kwargs) -> Any:
        """Make POST request."""
        headers = self.get_headers(kwargs.pop("headers", None))
        response = self.client.post(
            url,
            json=data,
            headers=headers,
            **kwargs
        )
        return response
        
    def put(self, url: str, data: Optional[Dict] = None, **kwargs) -> Any:
        """Make PUT request."""
        headers = self.get_headers(kwargs.pop("headers", None))
        response = self.client.put(
            url,
            json=data,
            headers=headers,
            **kwargs
        )
        return response
        
    def delete(self, url: str, **kwargs) -> Any:
        """Make DELETE request."""
        headers = self.get_headers(kwargs.pop("headers", None))
        response = self.client.delete(url, headers=headers, **kwargs)
        return response
        
    def assert_success(self, response, status_code: int = 200):
        """Assert response is successful."""
        assert response.status_code == status_code, (
            f"Expected {status_code}, got {response.status_code}: "
            f"{response.text}"
        )
        
    def assert_error(self, response, status_code: int = 400):
        """Assert response is an error."""
        assert response.status_code == status_code, (
            f"Expected error {status_code}, got {response.status_code}"
        )
        
    def assert_json_response(self, response):
        """Assert response is valid JSON."""
        try:
            response.json()
        except json.JSONDecodeError:
            raise AssertionError(f"Response is not valid JSON: {response.text}")
            
    def assert_has_fields(self, data: Dict, fields: list):
        """Assert dictionary has required fields."""
        missing = [f for f in fields if f not in data]
        assert not missing, f"Missing required fields: {missing}"
        
    def assert_prediction_response(self, response):
        """Assert response is a valid prediction response."""
        self.assert_success(response)
        self.assert_json_response(response)
        
        data = response.json()
        required_fields = [
            "symbol",
            "predictions",
            "confidence",
            "timestamp"
        ]
        self.assert_has_fields(data, required_fields)
        
        # Validate predictions structure
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        
        for pred in data["predictions"]:
            pred_fields = ["day", "predicted_price", "confidence"]
            self.assert_has_fields(pred, pred_fields)


def create_test_prediction_request(
    symbol: str = "AAPL",
    days: int = 5,
    include_analysis: bool = False
) -> Dict:
    """Create a test prediction request."""
    return {
        "symbol": symbol,
        "days": days,
        "include_analysis": include_analysis
    }


def create_test_backtest_request(
    symbol: str = "AAPL",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31"
) -> Dict:
    """Create a test backtest request."""
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date
    }


def create_test_portfolio_request(
    assets: list = None,
    risk_tolerance: str = "moderate"
) -> Dict:
    """Create a test portfolio optimization request."""
    if assets is None:
        assets = ["AAPL", "MSFT", "GOOGL"]
        
    return {
        "assets": assets,
        "risk_tolerance": risk_tolerance
    }


def validate_error_response(response, expected_code: int = 400):
    """Validate error response structure."""
    assert response.status_code == expected_code
    
    data = response.json()
    assert "detail" in data or "error" in data
    
    return data


def validate_pagination(data: Dict):
    """Validate pagination fields in response."""
    required_fields = ["items", "total", "page", "page_size"]
    for field in required_fields:
        assert field in data, f"Missing pagination field: {field}"
        
    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)
    assert isinstance(data["page"], int)
    assert isinstance(data["page_size"], int)

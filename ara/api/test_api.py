"""
Simple test script for API endpoints
Run with: python -m ara.api.test_api
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient

from ara.api.app import create_app


def test_api():
    """Test basic API functionality"""
    print("Creating FastAPI application...")
    app = create_app()

    print("Creating test client...")
    client = TestClient(app)

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200

    # Test root endpoint
    print("\n2. Testing root endpoint...")
    response = client.get("/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200

    # Test OpenAPI docs
    print("\n3. Testing OpenAPI docs...")
    response = client.get("/openapi.json")
    print(f"   Status: {response.status_code}")
    assert response.status_code == 200

    print("\n✅ All basic tests passed!")
    print("\nAPI is ready to use. Start the server with:")
    print("   uvicorn ara.api.app:app --reload")
    print("\nThen visit:")
    print("   http://localhost:8000/docs - Interactive API documentation")
    print("   http://localhost:8000/redoc - Alternative API documentation")
    print("   http://localhost:8000/health - Health check")


if __name__ == "__main__":
    test_api()

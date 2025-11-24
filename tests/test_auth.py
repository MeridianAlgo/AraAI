"""
Tests for authentication and authorization system
"""

import pytest
from fastapi.testclient import TestClient
from ara.api.app import app
from ara.api.auth.models import UserRole, AccessTier
from ara.api.auth.jwt_handler import create_access_token, verify_token, get_password_hash
from ara.api.auth.api_key_manager import APIKeyManager
from ara.api.auth.user_manager import UserManager
from ara.api.auth.rate_limiter import RateLimiter, TokenBucket


client = TestClient(app)


class TestJWTAuthentication:
    """Test JWT token authentication"""
    
    def test_login_success(self):
        """Test successful login"""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "pro123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "pro@ara.ai"
        assert data["user"]["tier"] == "pro"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "wrongpassword"}
        )
        assert response.status_code == 401
    
    def test_get_current_user(self):
        """Test getting current user info"""
        # Login first
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "pro123"}
        )
        token = login_response.json()["access_token"]
        
        # Get user info
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "pro@ara.ai"
        assert "quotas" in data
        assert data["quotas"]["requests_per_minute"] == 60


class TestAPIKeyAuthentication:
    """Test API key authentication"""
    
    def test_create_api_key(self):
        """Test creating API key"""
        # Login first
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "pro123"}
        )
        token = login_response.json()["access_token"]
        
        # Create API key
        response = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Test Key", "expires_in_days": 30}
        )
        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert data["key"].startswith("ara_")
        assert data["name"] == "Test Key"
    
    def test_list_api_keys(self):
        """Test listing API keys"""
        # Login first
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "pro123"}
        )
        token = login_response.json()["access_token"]
        
        # List keys
        response = client.get(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestRoleBasedAccess:
    """Test role-based access control"""
    
    def test_admin_access(self):
        """Test admin can access admin endpoints"""
        # Login as admin
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "admin@ara.ai", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        
        # Access admin endpoint
        response = client.get(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
    
    def test_user_cannot_access_admin(self):
        """Test regular user cannot access admin endpoints"""
        # Login as regular user
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "pro123"}
        )
        token = login_response.json()["access_token"]
        
        # Try to access admin endpoint
        response = client.get(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403


class TestTieredAccess:
    """Test tiered access system"""
    
    def test_free_tier_quotas(self):
        """Test free tier has correct quotas"""
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "free@ara.ai", "password": "free123"}
        )
        token = login_response.json()["access_token"]
        
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        data = response.json()
        assert data["tier"] == "free"
        assert data["quotas"]["requests_per_minute"] == 10
        assert data["quotas"]["requests_per_day"] == 1000
    
    def test_pro_tier_quotas(self):
        """Test pro tier has correct quotas"""
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "pro@ara.ai", "password": "pro123"}
        )
        token = login_response.json()["access_token"]
        
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        data = response.json()
        assert data["tier"] == "pro"
        assert data["quotas"]["requests_per_minute"] == 60
        assert data["quotas"]["requests_per_day"] == 10000


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_token_bucket(self):
        """Test token bucket algorithm"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should be able to consume 10 tokens
        for _ in range(10):
            assert bucket.consume(1) is True
        
        # 11th token should fail
        assert bucket.consume(1) is False
    
    def test_rate_limiter_check(self):
        """Test rate limiter check"""
        rate_limiter = RateLimiter(use_redis=False)
        user_manager = UserManager()
        
        # Get a test user
        user = user_manager.get_user_by_email("free@ara.ai")
        
        # Should allow first request
        allowed, retry_after = rate_limiter.check_rate_limit(user)
        assert allowed is True
        assert retry_after is None


class TestAPIKeyManager:
    """Test API key manager"""
    
    def test_generate_key(self):
        """Test key generation"""
        manager = APIKeyManager()
        key = manager.generate_key()
        assert key.startswith("ara_")
        assert len(key) > 10
    
    def test_hash_key(self):
        """Test key hashing"""
        manager = APIKeyManager()
        key = "ara_test123"
        hashed = manager.hash_key(key)
        assert hashed != key
        assert len(hashed) == 64  # SHA256 hex
    
    def test_create_and_validate_key(self):
        """Test creating and validating API key"""
        manager = APIKeyManager()
        user_manager = UserManager()
        user = user_manager.get_user_by_email("pro@ara.ai")
        
        # Create key
        api_key = manager.create_api_key(user, "Test Key")
        plain_key = api_key.key
        
        # Validate key
        user_id = manager.validate_api_key(plain_key)
        assert user_id == user.id


class TestUserManager:
    """Test user manager"""
    
    def test_get_user_by_email(self):
        """Test getting user by email"""
        manager = UserManager()
        user = manager.get_user_by_email("pro@ara.ai")
        assert user is not None
        assert user.email == "pro@ara.ai"
        assert user.tier == AccessTier.PRO
    
    def test_get_user_by_id(self):
        """Test getting user by ID"""
        manager = UserManager()
        user = manager.get_user_by_email("pro@ara.ai")
        
        # Get by ID
        user_by_id = manager.get_user_by_id(user.id)
        assert user_by_id is not None
        assert user_by_id.id == user.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

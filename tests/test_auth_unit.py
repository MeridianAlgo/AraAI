"""
Unit tests for authentication components (without FastAPI app)
"""

import pytest
from datetime import datetime, timedelta
from ara.api.auth.models import User, UserRole, AccessTier, TIER_QUOTAS
from ara.api.auth.jwt_handler import (
    create_access_token,
    verify_token,
    get_password_hash,
    verify_password
)
from ara.api.auth.api_key_manager import APIKeyManager
from ara.api.auth.user_manager import UserManager
from ara.api.auth.rate_limiter import RateLimiter, TokenBucket


class TestUserModel:
    """Test User model"""
    
    def test_user_creation(self):
        """Test creating a user"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            role=UserRole.USER,
            tier=AccessTier.FREE
        )
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.tier == AccessTier.FREE
    
    def test_user_quotas(self):
        """Test getting user quotas"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            tier=AccessTier.PRO
        )
        quotas = user.get_quotas()
        assert quotas.requests_per_minute == 60
        assert quotas.requests_per_day == 10000
    
    def test_user_has_feature(self):
        """Test checking feature access"""
        free_user = User(
            email="free@example.com",
            username="freeuser",
            hashed_password="hashed",
            tier=AccessTier.FREE
        )
        pro_user = User(
            email="pro@example.com",
            username="prouser",
            hashed_password="hashed",
            tier=AccessTier.PRO
        )
        
        assert free_user.has_feature("predictions") is True
        assert free_user.has_feature("sentiment_analysis") is False
        assert pro_user.has_feature("sentiment_analysis") is True


class TestJWTHandler:
    """Test JWT token handling"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False
    
    def test_create_token(self):
        """Test creating JWT token"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            role=UserRole.USER,
            tier=AccessTier.PRO
        )
        
        token = create_access_token(user)
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self):
        """Test verifying JWT token"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            role=UserRole.USER,
            tier=AccessTier.PRO
        )
        
        token = create_access_token(user)
        token_data = verify_token(token)
        
        assert token_data.user_id == user.id
        assert token_data.email == user.email
        assert token_data.role == user.role
        assert token_data.tier == user.tier


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
    
    def test_create_api_key(self):
        """Test creating API key"""
        manager = APIKeyManager()
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed"
        )
        
        api_key = manager.create_api_key(user, "Test Key")
        
        assert api_key.key.startswith("ara_")
        assert api_key.name == "Test Key"
        assert api_key.user_id == user.id
        assert api_key.is_active is True
    
    def test_validate_api_key(self):
        """Test validating API key"""
        manager = APIKeyManager()
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed"
        )
        
        # Create key
        api_key = manager.create_api_key(user, "Test Key")
        plain_key = api_key.key
        
        # Validate key
        user_id = manager.validate_api_key(plain_key)
        assert user_id == user.id
    
    def test_validate_invalid_key(self):
        """Test validating invalid API key"""
        manager = APIKeyManager()
        user_id = manager.validate_api_key("ara_invalid_key")
        assert user_id is None
    
    def test_list_user_keys(self):
        """Test listing user keys"""
        manager = APIKeyManager()
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed"
        )
        
        # Create multiple keys
        manager.create_api_key(user, "Key 1")
        manager.create_api_key(user, "Key 2")
        
        keys = manager.list_user_keys(user.id)
        assert len(keys) == 2
    
    def test_revoke_key(self):
        """Test revoking API key"""
        manager = APIKeyManager()
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed"
        )
        
        api_key = manager.create_api_key(user, "Test Key")
        plain_key = api_key.key
        
        # Revoke key
        manager.revoke_key(api_key.id, user.id)
        
        # Should not validate after revocation
        user_id = manager.validate_api_key(plain_key)
        assert user_id is None


class TestUserManager:
    """Test user manager"""
    
    def test_get_demo_users(self):
        """Test demo users are created"""
        manager = UserManager()
        
        admin = manager.get_user_by_email("admin@ara.ai")
        assert admin is not None
        assert admin.role == UserRole.ADMIN
        assert admin.tier == AccessTier.ENTERPRISE
        
        pro = manager.get_user_by_email("pro@ara.ai")
        assert pro is not None
        assert pro.tier == AccessTier.PRO
        
        free = manager.get_user_by_email("free@ara.ai")
        assert free is not None
        assert free.tier == AccessTier.FREE
    
    def test_create_user(self):
        """Test creating new user"""
        manager = UserManager()
        
        user = manager.create_user(
            email="newuser@example.com",
            username="newuser",
            password="password123",
            tier=AccessTier.PRO
        )
        
        assert user.email == "newuser@example.com"
        assert user.tier == AccessTier.PRO
    
    def test_get_user_by_id(self):
        """Test getting user by ID"""
        manager = UserManager()
        user = manager.get_user_by_email("pro@ara.ai")
        
        user_by_id = manager.get_user_by_id(user.id)
        assert user_by_id is not None
        assert user_by_id.id == user.id
    
    def test_update_user_tier(self):
        """Test updating user tier"""
        manager = UserManager()
        user = manager.get_user_by_email("free@ara.ai")
        
        updated = manager.update_user_tier(user.id, AccessTier.PRO)
        assert updated.tier == AccessTier.PRO


class TestTokenBucket:
    """Test token bucket algorithm"""
    
    def test_consume_tokens(self):
        """Test consuming tokens"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should be able to consume 10 tokens
        for _ in range(10):
            assert bucket.consume(1) is True
        
        # 11th token should fail
        assert bucket.consume(1) is False
    
    def test_get_wait_time(self):
        """Test getting wait time"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Consume all tokens
        for _ in range(10):
            bucket.consume(1)
        
        # Should need to wait
        wait_time = bucket.get_wait_time(1)
        assert wait_time > 0


class TestRateLimiter:
    """Test rate limiter"""
    
    def test_check_rate_limit(self):
        """Test checking rate limit"""
        limiter = RateLimiter(use_redis=False)
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            tier=AccessTier.FREE
        )
        
        # First request should be allowed
        allowed, retry_after = limiter.check_rate_limit(user)
        assert allowed is True
        assert retry_after is None
    
    def test_rate_limit_headers(self):
        """Test getting rate limit headers"""
        limiter = RateLimiter(use_redis=False)
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            tier=AccessTier.PRO
        )
        
        headers = limiter.get_rate_limit_headers(user)
        
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Limit"] == "60"


class TestTierQuotas:
    """Test tier quotas"""
    
    def test_free_tier_quotas(self):
        """Test free tier quotas"""
        quotas = TIER_QUOTAS[AccessTier.FREE]
        assert quotas.requests_per_minute == 10
        assert quotas.requests_per_day == 1000
        assert quotas.max_batch_size == 10
        assert quotas.features_enabled["predictions"] is True
        assert quotas.features_enabled["sentiment_analysis"] is False
    
    def test_pro_tier_quotas(self):
        """Test pro tier quotas"""
        quotas = TIER_QUOTAS[AccessTier.PRO]
        assert quotas.requests_per_minute == 60
        assert quotas.requests_per_day == 10000
        assert quotas.max_batch_size == 50
        assert quotas.features_enabled["sentiment_analysis"] is True
    
    def test_enterprise_tier_quotas(self):
        """Test enterprise tier quotas"""
        quotas = TIER_QUOTAS[AccessTier.ENTERPRISE]
        assert quotas.requests_per_minute == 300
        assert quotas.requests_per_day == 100000
        assert quotas.max_batch_size == 100
        assert quotas.features_enabled["webhooks"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

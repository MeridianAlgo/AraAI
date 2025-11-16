"""
Standalone test for authentication system (without pytest)
Tests the core authentication functionality independently
"""

import sys
sys.path.insert(0, '.')

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


def test_password_hashing():
    """Test password hashing"""
    print("Testing password hashing...")
    password = "testpassword123"
    hashed = get_password_hash(password)
    
    assert hashed != password, "Password should be hashed"
    assert verify_password(password, hashed), "Password verification should succeed"
    assert not verify_password("wrongpassword", hashed), "Wrong password should fail"
    print("✓ Password hashing works")


def test_jwt_tokens():
    """Test JWT token creation and verification"""
    print("\nTesting JWT tokens...")
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed",
        role=UserRole.USER,
        tier=AccessTier.PRO
    )
    
    token = create_access_token(user)
    assert isinstance(token, str), "Token should be a string"
    assert len(token) > 0, "Token should not be empty"
    
    token_data = verify_token(token)
    assert token_data.user_id == user.id, "User ID should match"
    assert token_data.email == user.email, "Email should match"
    assert token_data.role == user.role, "Role should match"
    assert token_data.tier == user.tier, "Tier should match"
    print("✓ JWT tokens work")


def test_api_keys():
    """Test API key management"""
    print("\nTesting API keys...")
    manager = APIKeyManager()
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed"
    )
    
    # Generate key
    key = manager.generate_key()
    assert key.startswith("ara_"), "Key should have ara_ prefix"
    print(f"  Generated key: {key[:20]}...")
    
    # Create API key
    api_key = manager.create_api_key(user, "Test Key")
    plain_key = api_key.key
    assert plain_key.startswith("ara_"), "API key should have ara_ prefix"
    assert api_key.name == "Test Key", "Name should match"
    print(f"  Created API key: {plain_key[:20]}...")
    
    # Validate key
    user_id = manager.validate_api_key(plain_key)
    assert user_id == user.id, "Validation should return correct user ID"
    print("  ✓ API key validated successfully")
    
    # List keys
    keys = manager.list_user_keys(user.id)
    assert len(keys) == 1, "Should have 1 key"
    print(f"  ✓ Listed {len(keys)} key(s)")
    
    # Revoke key
    manager.revoke_key(api_key.id, user.id)
    user_id = manager.validate_api_key(plain_key)
    assert user_id is None, "Revoked key should not validate"
    print("  ✓ Key revoked successfully")
    
    print("✓ API key management works")


def test_user_manager():
    """Test user management"""
    print("\nTesting user manager...")
    manager = UserManager()
    
    # Check demo users
    admin = manager.get_user_by_email("admin@ara.ai")
    assert admin is not None, "Admin user should exist"
    assert admin.role == UserRole.ADMIN, "Admin should have admin role"
    assert admin.tier == AccessTier.ENTERPRISE, "Admin should have enterprise tier"
    print("  ✓ Admin user exists")
    
    pro = manager.get_user_by_email("pro@ara.ai")
    assert pro is not None, "Pro user should exist"
    assert pro.tier == AccessTier.PRO, "Pro user should have pro tier"
    print("  ✓ Pro user exists")
    
    free = manager.get_user_by_email("free@ara.ai")
    assert free is not None, "Free user should exist"
    assert free.tier == AccessTier.FREE, "Free user should have free tier"
    print("  ✓ Free user exists")
    
    # Create new user
    new_user = manager.create_user(
        email="newuser@example.com",
        username="newuser",
        password="password123",
        tier=AccessTier.PRO
    )
    assert new_user.email == "newuser@example.com", "Email should match"
    print("  ✓ Created new user")
    
    # Get by ID
    user_by_id = manager.get_user_by_id(new_user.id)
    assert user_by_id is not None, "Should find user by ID"
    assert user_by_id.id == new_user.id, "IDs should match"
    print("  ✓ Retrieved user by ID")
    
    print("✓ User manager works")


def test_tier_quotas():
    """Test tier quotas"""
    print("\nTesting tier quotas...")
    
    # Free tier
    free_quotas = TIER_QUOTAS[AccessTier.FREE]
    assert free_quotas.requests_per_minute == 10, "Free tier should have 10 req/min"
    assert free_quotas.requests_per_day == 1000, "Free tier should have 1000 req/day"
    assert free_quotas.features_enabled["predictions"] is True, "Free tier should have predictions"
    assert free_quotas.features_enabled["sentiment_analysis"] is False, "Free tier should not have sentiment"
    print("  ✓ Free tier: 10 req/min, 1000 req/day")
    
    # Pro tier
    pro_quotas = TIER_QUOTAS[AccessTier.PRO]
    assert pro_quotas.requests_per_minute == 60, "Pro tier should have 60 req/min"
    assert pro_quotas.requests_per_day == 10000, "Pro tier should have 10000 req/day"
    assert pro_quotas.features_enabled["sentiment_analysis"] is True, "Pro tier should have sentiment"
    print("  ✓ Pro tier: 60 req/min, 10000 req/day")
    
    # Enterprise tier
    ent_quotas = TIER_QUOTAS[AccessTier.ENTERPRISE]
    assert ent_quotas.requests_per_minute == 300, "Enterprise tier should have 300 req/min"
    assert ent_quotas.requests_per_day == 100000, "Enterprise tier should have 100000 req/day"
    assert ent_quotas.features_enabled["webhooks"] is True, "Enterprise tier should have webhooks"
    print("  ✓ Enterprise tier: 300 req/min, 100000 req/day")
    
    print("✓ Tier quotas work")


def test_token_bucket():
    """Test token bucket algorithm"""
    print("\nTesting token bucket...")
    bucket = TokenBucket(capacity=10, refill_rate=1.0)
    
    # Consume tokens
    consumed = 0
    for i in range(15):
        if bucket.consume(1):
            consumed += 1
    
    assert consumed == 10, f"Should consume exactly 10 tokens, consumed {consumed}"
    print(f"  ✓ Consumed {consumed}/10 tokens correctly")
    
    # Check wait time
    wait_time = bucket.get_wait_time(1)
    assert wait_time > 0, "Should need to wait for more tokens"
    print(f"  ✓ Wait time: {wait_time:.2f} seconds")
    
    print("✓ Token bucket works")


def test_rate_limiter():
    """Test rate limiter"""
    print("\nTesting rate limiter...")
    limiter = RateLimiter(use_redis=False)
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed",
        tier=AccessTier.FREE
    )
    
    # Check rate limit
    allowed, retry_after = limiter.check_rate_limit(user)
    assert allowed is True, "First request should be allowed"
    assert retry_after is None, "No retry needed"
    print("  ✓ Rate limit check passed")
    
    # Get headers
    headers = limiter.get_rate_limit_headers(user)
    assert "X-RateLimit-Limit" in headers, "Should have limit header"
    assert "X-RateLimit-Remaining" in headers, "Should have remaining header"
    assert headers["X-RateLimit-Limit"] == "10", "Free tier should have 10 req/min"
    print(f"  ✓ Rate limit headers: {headers['X-RateLimit-Limit']} req/min")
    
    print("✓ Rate limiter works")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Authentication System Tests")
    print("=" * 70)
    
    try:
        test_password_hashing()
        test_jwt_tokens()
        test_api_keys()
        test_user_manager()
        test_tier_quotas()
        test_token_bucket()
        test_rate_limiter()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

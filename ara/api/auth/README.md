# Authentication and Authorization System

This module implements JWT token-based authentication, API key management, role-based access control (RBAC), tiered access, and rate limiting for the ARA AI API.

## Features

### 1. JWT Token Authentication
- Secure JWT tokens with configurable expiration
- Password hashing using bcrypt
- Token-based session management

### 2. API Key Management
- Generate secure API keys with `ara_` prefix
- API key rotation and revocation
- Expiration support
- Usage tracking

### 3. Role-Based Access Control (RBAC)
- **Admin**: Full system access
- **User**: Standard user access
- **Readonly**: Read-only access

### 4. Tiered Access System
- **Free Tier**:
  - 10 requests/minute
  - 1,000 requests/day
  - Basic predictions and backtesting
  - Max 5 portfolio assets
  
- **Pro Tier**:
  - 60 requests/minute
  - 10,000 requests/day
  - All features including sentiment analysis
  - Max 20 portfolio assets
  
- **Enterprise Tier**:
  - 300 requests/minute
  - 100,000 requests/day
  - All features including webhooks
  - Max 100 portfolio assets

### 5. Rate Limiting
- Token bucket algorithm
- Per-user and per-endpoint limits
- Redis support for distributed systems
- Rate limit headers in responses
- Automatic daily reset

## Quick Start

### 1. Login with JWT Token

```bash
# Login to get JWT token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "pro@ara.ai",
    "password": "pro123"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "...",
    "email": "pro@ara.ai",
    "username": "prouser",
    "role": "user",
    "tier": "pro"
  }
}
```

### 2. Use JWT Token

```bash
# Make authenticated request with JWT token
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "days": 5
  }'
```

### 3. Create API Key

```bash
# Create API key (requires JWT token)
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key",
    "expires_in_days": 90
  }'

# Response:
{
  "id": "...",
  "key": "ara_1234567890abcdef...",  # Only shown once!
  "name": "My API Key",
  "created_at": "2024-01-01T00:00:00",
  "expires_at": "2024-04-01T00:00:00",
  "is_active": true
}
```

### 4. Use API Key

```bash
# Make authenticated request with API key
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: ara_1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "days": 5
  }'
```

## Demo Credentials

The system comes with pre-configured demo users:

| Email | Password | Role | Tier | Requests/Min | Requests/Day |
|-------|----------|------|------|--------------|--------------|
| admin@ara.ai | admin123 | admin | enterprise | 300 | 100,000 |
| pro@ara.ai | pro123 | user | pro | 60 | 10,000 |
| free@ara.ai | free123 | user | free | 10 | 1,000 |

## API Endpoints

### Authentication

#### POST /api/v1/auth/login
Login with email and password to receive JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user-id",
    "email": "user@example.com",
    "username": "username",
    "role": "user",
    "tier": "pro"
  }
}
```

#### GET /api/v1/auth/me
Get current user information and quotas.

**Headers:**
- `Authorization: Bearer YOUR_JWT_TOKEN` or
- `X-API-Key: YOUR_API_KEY`

**Response:**
```json
{
  "id": "user-id",
  "email": "user@example.com",
  "username": "username",
  "role": "user",
  "tier": "pro",
  "is_active": true,
  "quotas": {
    "requests_per_minute": 60,
    "requests_per_day": 10000,
    "max_batch_size": 50,
    "max_backtest_days": 1825,
    "max_portfolio_assets": 20,
    "features_enabled": {
      "predictions": true,
      "backtesting": true,
      "portfolio_optimization": true,
      "sentiment_analysis": true,
      "real_time_data": true,
      "advanced_models": true,
      "webhooks": false
    }
  }
}
```

### API Key Management

#### POST /api/v1/auth/api-keys
Create a new API key.

**Request:**
```json
{
  "name": "My API Key",
  "expires_in_days": 90
}
```

**Response:**
```json
{
  "id": "key-id",
  "key": "ara_1234567890abcdef...",
  "name": "My API Key",
  "created_at": "2024-01-01T00:00:00",
  "expires_at": "2024-04-01T00:00:00",
  "is_active": true
}
```

#### GET /api/v1/auth/api-keys
List all API keys for current user.

#### DELETE /api/v1/auth/api-keys/{key_id}
Delete an API key.

#### POST /api/v1/auth/api-keys/{key_id}/revoke
Revoke an API key (can be reactivated).

#### POST /api/v1/auth/api-keys/{key_id}/rotate
Rotate an API key (create new, revoke old).

### Admin Endpoints

#### GET /api/v1/auth/admin/users
List all users (admin only).

#### PUT /api/v1/auth/admin/users/{user_id}/tier
Update user's access tier (admin only).

**Request:**
```json
{
  "tier": "pro"
}
```

#### PUT /api/v1/auth/admin/users/{user_id}/role
Update user's role (admin only).

**Request:**
```json
{
  "role": "admin"
}
```

## Rate Limiting

### Rate Limit Headers

All authenticated requests include rate limit headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1704067200
X-RateLimit-Daily-Limit: 10000
X-RateLimit-Daily-Remaining: 9500
```

### Rate Limit Exceeded Response

When rate limit is exceeded:

**Status:** 429 Too Many Requests

**Response:**
```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Please try again later.",
  "details": {
    "retry_after": 30,
    "tier": "free"
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

**Headers:**
```
Retry-After: 30
```

## Using Dependencies in Routes

### Require Authentication

```python
from fastapi import APIRouter, Depends
from ara.api.auth.dependencies import get_current_user
from ara.api.auth.models import User

router = APIRouter()

@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello {current_user.username}"}
```

### Require Specific Role

```python
from ara.api.auth.dependencies import require_role
from ara.api.auth.models import UserRole

@router.get("/admin-only")
async def admin_route(
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    return {"message": "Admin access granted"}
```

### Require Minimum Tier

```python
from ara.api.auth.dependencies import require_tier
from ara.api.auth.models import AccessTier

@router.get("/pro-feature")
async def pro_route(
    current_user: User = Depends(require_tier(AccessTier.PRO))
):
    return {"message": "Pro feature access granted"}
```

### Require Specific Feature

```python
from ara.api.auth.dependencies import require_feature

@router.get("/sentiment")
async def sentiment_route(
    current_user: User = Depends(require_feature("sentiment_analysis"))
):
    return {"message": "Sentiment analysis available"}
```

## Configuration

### JWT Settings

Edit `ara/api/auth/jwt_handler.py`:

```python
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
```

**Important:** Change `SECRET_KEY` in production! Use environment variable:

```python
import os
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default-secret-key")
```

### Rate Limiting with Redis

To enable Redis-based rate limiting:

1. Install Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Windows
# Download from https://redis.io/download
```

2. Start Redis:
```bash
redis-server
```

3. Enable Redis in code:
```python
from ara.api.auth.rate_limiter import RateLimiter

rate_limiter = RateLimiter(use_redis=True)
```

## Security Best Practices

1. **Change Default Secret Key**: Never use the default JWT secret key in production
2. **Use HTTPS**: Always use HTTPS in production to protect tokens
3. **Rotate API Keys**: Implement automatic key rotation every 90 days
4. **Monitor Failed Attempts**: Track and alert on failed authentication attempts
5. **Use Strong Passwords**: Enforce password complexity requirements
6. **Enable Redis**: Use Redis for distributed rate limiting in production
7. **Set Expiration**: Always set expiration for API keys
8. **Audit Logs**: Log all authentication and authorization events

## Testing

### Test Authentication

```python
import pytest
from fastapi.testclient import TestClient
from ara.api.app import app

client = TestClient(app)

def test_login():
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "pro@ara.ai", "password": "pro123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_protected_endpoint():
    # Login first
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "pro@ara.ai", "password": "pro123"}
    )
    token = login_response.json()["access_token"]
    
    # Access protected endpoint
    response = client.post(
        "/api/v1/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={"symbol": "AAPL", "days": 5}
    )
    assert response.status_code == 200
```

## Troubleshooting

### "Could not validate credentials"
- Check that JWT token is valid and not expired
- Verify Authorization header format: `Bearer YOUR_TOKEN`
- Ensure API key is active and not expired

### "Rate limit exceeded"
- Wait for the time specified in `Retry-After` header
- Upgrade to higher tier for more requests
- Check rate limit headers to monitor usage

### "Insufficient permissions"
- Verify user has required role or tier
- Contact admin to upgrade account
- Check feature availability in tier

### Redis Connection Error
- Ensure Redis server is running
- Check Redis connection settings
- Falls back to in-memory if Redis unavailable

## Architecture

```
ara/api/auth/
├── __init__.py              # Module exports
├── models.py                # Data models (User, APIKey, etc.)
├── jwt_handler.py           # JWT token creation/verification
├── api_key_manager.py       # API key management
├── user_manager.py          # User account management
├── dependencies.py          # FastAPI dependencies
├── rate_limiter.py          # Rate limiting implementation
└── README.md                # This file

ara/api/middleware/
├── __init__.py
└── rate_limit.py            # Rate limiting middleware

ara/api/routes/
└── auth.py                  # Authentication endpoints
```

## Future Enhancements

- [ ] OAuth2 integration (Google, GitHub)
- [ ] Two-factor authentication (2FA)
- [ ] IP whitelisting
- [ ] Webhook authentication
- [ ] API key scopes and permissions
- [ ] Usage analytics dashboard
- [ ] Automatic tier upgrades
- [ ] Payment integration

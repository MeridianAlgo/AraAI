"""
Authentication and authorization data models
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User roles for RBAC"""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class AccessTier(str, Enum):
    """Access tiers for resource quotas"""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TierQuotas(BaseModel):
    """Resource quotas per tier"""

    requests_per_minute: int
    requests_per_day: int
    max_batch_size: int
    max_backtest_days: int
    max_portfolio_assets: int
    features_enabled: Dict[str, bool]


# Define quotas for each tier
TIER_QUOTAS: Dict[AccessTier, TierQuotas] = {
    AccessTier.FREE: TierQuotas(
        requests_per_minute=10,
        requests_per_day=1000,
        max_batch_size=10,
        max_backtest_days=365,
        max_portfolio_assets=5,
        features_enabled={
            "predictions": True,
            "backtesting": True,
            "portfolio_optimization": False,
            "sentiment_analysis": False,
            "real_time_data": False,
            "advanced_models": False,
            "webhooks": False,
        },
    ),
    AccessTier.PRO: TierQuotas(
        requests_per_minute=60,
        requests_per_day=10000,
        max_batch_size=50,
        max_backtest_days=1825,  # 5 years
        max_portfolio_assets=20,
        features_enabled={
            "predictions": True,
            "backtesting": True,
            "portfolio_optimization": True,
            "sentiment_analysis": True,
            "real_time_data": True,
            "advanced_models": True,
            "webhooks": False,
        },
    ),
    AccessTier.ENTERPRISE: TierQuotas(
        requests_per_minute=300,
        requests_per_day=100000,
        max_batch_size=100,
        max_backtest_days=3650,  # 10 years
        max_portfolio_assets=100,
        features_enabled={
            "predictions": True,
            "backtesting": True,
            "portfolio_optimization": True,
            "sentiment_analysis": True,
            "real_time_data": True,
            "advanced_models": True,
            "webhooks": True,
        },
    ),
}


class User(BaseModel):
    """User model"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    username: str
    hashed_password: str
    role: UserRole = UserRole.USER
    tier: AccessTier = AccessTier.FREE
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_quotas(self) -> TierQuotas:
        """Get resource quotas for user's tier"""
        return TIER_QUOTAS[self.tier]

    def has_feature(self, feature: str) -> bool:
        """Check if user has access to a feature"""
        quotas = self.get_quotas()
        return quotas.features_enabled.get(feature, False)


class APIKey(BaseModel):
    """API Key model"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str  # The actual API key (hashed in storage)
    user_id: str
    name: str  # Friendly name for the key
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Rate limiting tracking
    requests_today: int = 0
    last_reset_date: datetime = Field(default_factory=datetime.now)


class TokenData(BaseModel):
    """JWT token payload data"""

    user_id: str
    email: str
    role: UserRole
    tier: AccessTier


class LoginRequest(BaseModel):
    """Login request model"""

    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response model"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: Dict[str, Any]


class APIKeyCreateRequest(BaseModel):
    """API key creation request"""

    name: str = Field(..., description="Friendly name for the API key")
    expires_in_days: Optional[int] = Field(None, description="Days until expiration (None = never)")


class APIKeyResponse(BaseModel):
    """API key response"""

    id: str
    key: str  # Only returned on creation
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool


class APIKeyListResponse(BaseModel):
    """API key list response (without actual key)"""

    id: str
    name: str
    created_at: datetime
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    is_active: bool

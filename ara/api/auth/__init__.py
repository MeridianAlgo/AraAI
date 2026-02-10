"""
Authentication and authorization module
"""

from ara.api.auth.api_key_manager import APIKeyManager
from ara.api.auth.dependencies import get_current_user, require_role, require_tier
from ara.api.auth.jwt_handler import create_access_token, verify_token
from ara.api.auth.models import AccessTier, APIKey, User, UserRole

__all__ = [
    "User",
    "APIKey",
    "UserRole",
    "AccessTier",
    "create_access_token",
    "verify_token",
    "APIKeyManager",
    "get_current_user",
    "require_role",
    "require_tier",
]

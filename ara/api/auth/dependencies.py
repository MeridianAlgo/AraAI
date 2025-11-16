"""
FastAPI dependencies for authentication and authorization
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from ara.api.auth.models import User, UserRole, AccessTier
from ara.api.auth.jwt_handler import verify_token, AuthenticationError
from ara.api.auth.api_key_manager import APIKeyManager
from ara.api.auth.user_manager import UserManager


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global instances (should be dependency injected in production)
api_key_manager = APIKeyManager()
user_manager = UserManager()


async def get_current_user(
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> User:
    """
    Get current authenticated user from JWT token or API key
    
    Args:
        bearer_token: JWT bearer token
        api_key: API key from header
        
    Returns:
        User object
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    user = None
    
    # Try JWT token first
    if bearer_token:
        try:
            token_data = verify_token(bearer_token.credentials)
            user = user_manager.get_user_by_id(token_data.user_id)
        except AuthenticationError:
            raise credentials_exception
    
    # Try API key if no JWT token
    elif api_key:
        user_id = api_key_manager.validate_api_key(api_key)
        if user_id:
            user = user_manager.get_user_by_id(user_id)
            # Track request for rate limiting
            api_key_manager.track_request(api_key)
    
    if not user:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


def require_role(required_role: UserRole):
    """
    Dependency to require a specific role
    
    Args:
        required_role: Required user role
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        """Check if user has required role"""
        # Admin has access to everything
        if current_user.role == UserRole.ADMIN:
            return current_user
        
        # Check specific role
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )
        
        return current_user
    
    return role_checker


def require_tier(required_tier: AccessTier):
    """
    Dependency to require a minimum access tier
    
    Args:
        required_tier: Required access tier
        
    Returns:
        Dependency function
    """
    # Define tier hierarchy
    tier_hierarchy = {
        AccessTier.FREE: 0,
        AccessTier.PRO: 1,
        AccessTier.ENTERPRISE: 2
    }
    
    async def tier_checker(current_user: User = Depends(get_current_user)) -> User:
        """Check if user has required tier"""
        user_tier_level = tier_hierarchy.get(current_user.tier, 0)
        required_tier_level = tier_hierarchy.get(required_tier, 0)
        
        if user_tier_level < required_tier_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient access tier. Required: {required_tier.value}, Current: {current_user.tier.value}"
            )
        
        return current_user
    
    return tier_checker


def require_feature(feature: str):
    """
    Dependency to require access to a specific feature
    
    Args:
        feature: Feature name
        
    Returns:
        Dependency function
    """
    async def feature_checker(current_user: User = Depends(get_current_user)) -> User:
        """Check if user has access to feature"""
        if not current_user.has_feature(feature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature '{feature}' not available in your tier. Upgrade to access this feature."
            )
        
        return current_user
    
    return feature_checker


async def get_optional_user(
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise (for optional auth)
    
    Args:
        bearer_token: JWT bearer token
        api_key: API key from header
        
    Returns:
        User object or None
    """
    try:
        return await get_current_user(bearer_token, api_key)
    except HTTPException:
        return None


async def get_current_user_ws(token: str) -> Optional[str]:
    """
    Get current user ID from WebSocket token
    
    Args:
        token: JWT token or API key
        
    Returns:
        User ID or None if authentication fails
    """
    try:
        # Try as JWT token first
        token_data = verify_token(token)
        return token_data.user_id
    except AuthenticationError:
        # Try as API key
        user_id = api_key_manager.validate_api_key(token)
        if user_id:
            api_key_manager.track_request(token)
            return user_id
    
    return None

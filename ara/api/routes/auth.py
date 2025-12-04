"""
Authentication and authorization routes
"""

from datetime import timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from ara.api.auth.models import (
    LoginRequest,
    LoginResponse,
    APIKeyCreateRequest,
    APIKeyResponse,
    APIKeyListResponse,
    User,
    UserRole,
    AccessTier
)
from ara.api.auth.jwt_handler import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ara.api.auth.api_key_manager import APIKeyManager, APIKeyError
from ara.api.auth.user_manager import UserManager
from ara.api.auth.dependencies import get_current_user, require_role


router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# Global instances
api_key_manager = APIKeyManager()
user_manager = UserManager()


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login with email and password",
    description="Authenticate with email and password to receive a JWT token"
)
async def login(request: LoginRequest):
    """
    Login endpoint
    
    Demo credentials:
    - admin@ara.ai / admin123 (Enterprise tier)
    - pro@ara.ai / pro123 (Pro tier)
    - free@ara.ai / free123 (Free tier)
    """
    # Get user database (in-memory for demo)
    user_db = {
        user.email: user.model_dump()
        for user in user_manager._users.values()
    }
    
    user = authenticate_user(request.email, request.password, user_db)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(user, expires_delta=access_token_expires)
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "tier": user.tier.value
        }
    )


@router.get(
    "/me",
    response_model=dict,
    summary="Get current user",
    description="Get information about the currently authenticated user"
)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    quotas = current_user.get_quotas()
    
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "role": current_user.role.value,
        "tier": current_user.tier.value,
        "is_active": current_user.is_active,
        "quotas": {
            "requests_per_minute": quotas.requests_per_minute,
            "requests_per_day": quotas.requests_per_day,
            "max_batch_size": quotas.max_batch_size,
            "max_backtest_days": quotas.max_backtest_days,
            "max_portfolio_assets": quotas.max_portfolio_assets,
            "features_enabled": quotas.features_enabled
        }
    }


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    summary="Create API key",
    description="Create a new API key for programmatic access"
)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new API key"""
    api_key = api_key_manager.create_api_key(
        user=current_user,
        name=request.name,
        expires_in_days=request.expires_in_days
    )
    
    return APIKeyResponse(
        id=api_key.id,
        key=api_key.key,  # Only returned on creation
        name=api_key.name,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        is_active=api_key.is_active
    )


@router.get(
    "/api-keys",
    response_model=List[APIKeyListResponse],
    summary="List API keys",
    description="List all API keys for the current user"
)
async def list_api_keys(current_user: User = Depends(get_current_user)):
    """List user's API keys"""
    keys = api_key_manager.list_user_keys(current_user.id)
    
    return [
        APIKeyListResponse(
            id=key.id,
            name=key.name,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            expires_at=key.expires_at,
            is_active=key.is_active
        )
        for key in keys
    ]


@router.delete(
    "/api-keys/{key_id}",
    summary="Delete API key",
    description="Delete an API key"
)
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an API key"""
    try:
        api_key_manager.delete_key(key_id, current_user.id)
        return {"message": "API key deleted successfully"}
    except APIKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post(
    "/api-keys/{key_id}/revoke",
    summary="Revoke API key",
    description="Revoke an API key (can be reactivated)"
)
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Revoke an API key"""
    try:
        api_key_manager.revoke_key(key_id, current_user.id)
        return {"message": "API key revoked successfully"}
    except APIKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post(
    "/api-keys/{key_id}/rotate",
    response_model=APIKeyResponse,
    summary="Rotate API key",
    description="Rotate an API key (create new, revoke old)"
)
async def rotate_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Rotate an API key"""
    try:
        new_key = api_key_manager.rotate_key(key_id, current_user.id)
        return APIKeyResponse(
            id=new_key.id,
            key=new_key.key,  # Only returned on creation
            name=new_key.name,
            created_at=new_key.created_at,
            expires_at=new_key.expires_at,
            is_active=new_key.is_active
        )
    except APIKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# Admin endpoints
@router.get(
    "/admin/users",
    summary="List all users (Admin only)",
    description="List all users in the system",
    dependencies=[Depends(require_role(UserRole.ADMIN))]
)
async def list_users(current_user: User = Depends(get_current_user)):
    """List all users (admin only)"""
    users = user_manager._users.values()
    
    return [
        {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "tier": user.tier.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat()
        }
        for user in users
    ]


@router.put(
    "/admin/users/{user_id}/tier",
    summary="Update user tier (Admin only)",
    description="Update a user's access tier",
    dependencies=[Depends(require_role(UserRole.ADMIN))]
)
async def update_user_tier(
    user_id: str,
    tier: AccessTier,
    current_user: User = Depends(get_current_user)
):
    """Update user tier (admin only)"""
    try:
        user = user_manager.update_user_tier(user_id, tier)
        return {
            "message": "User tier updated successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "tier": user.tier.value
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.put(
    "/admin/users/{user_id}/role",
    summary="Update user role (Admin only)",
    description="Update a user's role",
    dependencies=[Depends(require_role(UserRole.ADMIN))]
)
async def update_user_role(
    user_id: str,
    role: UserRole,
    current_user: User = Depends(get_current_user)
):
    """Update user role (admin only)"""
    try:
        user = user_manager.update_user_role(user_id, role)
        return {
            "message": "User role updated successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "role": user.role.value
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

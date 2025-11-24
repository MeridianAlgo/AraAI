"""
API Key management
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from ara.api.auth.models import APIKey, User
from ara.core.exceptions import AraAIException


class APIKeyError(AraAIException):
    """API key error"""
    def __init__(self, message: str = "API key error"):
        super().__init__(message, {"type": "api_key_error"})


class APIKeyManager:
    """
    Manages API key generation, validation, and storage
    """
    
    def __init__(self):
        """Initialize API key manager"""
        # In-memory storage for demo (should use database in production)
        self._keys: Dict[str, APIKey] = {}
        self._key_to_user: Dict[str, str] = {}  # hashed_key -> user_id
    
    def generate_key(self) -> str:
        """
        Generate a secure random API key
        
        Returns:
            API key string (ara_...)
        """
        # Generate 32 bytes of random data
        random_bytes = secrets.token_bytes(32)
        # Convert to hex and add prefix
        key = f"ara_{random_bytes.hex()}"
        return key
    
    def hash_key(self, key: str) -> str:
        """
        Hash an API key for secure storage
        
        Args:
            key: Plain API key
            
        Returns:
            Hashed key
        """
        return hashlib.sha256(key.encode()).hexdigest()
    
    def create_api_key(
        self,
        user: User,
        name: str,
        expires_in_days: Optional[int] = None
    ) -> APIKey:
        """
        Create a new API key for a user
        
        Args:
            user: User object
            name: Friendly name for the key
            expires_in_days: Days until expiration (None = never)
            
        Returns:
            APIKey object with plain key (only time it's visible)
        """
        # Generate key
        plain_key = self.generate_key()
        hashed_key = self.hash_key(plain_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key=hashed_key,  # Store hashed version
            user_id=user.id,
            name=name,
            expires_at=expires_at
        )
        
        # Store in memory
        self._keys[api_key.id] = api_key
        self._key_to_user[hashed_key] = user.id
        
        # Return with plain key (only time it's visible)
        api_key_response = api_key.model_copy()
        api_key_response.key = plain_key  # Return plain key
        return api_key_response
    
    def validate_api_key(self, key: str) -> Optional[str]:
        """
        Validate an API key and return user_id
        
        Args:
            key: Plain API key
            
        Returns:
            User ID if valid, None otherwise
        """
        hashed_key = self.hash_key(key)
        user_id = self._key_to_user.get(hashed_key)
        
        if not user_id:
            return None
        
        # Find the key object
        api_key = None
        for key_obj in self._keys.values():
            if key_obj.key == hashed_key and key_obj.user_id == user_id:
                api_key = key_obj
                break
        
        if not api_key:
            return None
        
        # Check if key is active
        if not api_key.is_active:
            return None
        
        # Check if key is expired
        if api_key.expires_at and api_key.expires_at < datetime.now():
            return None
        
        # Update last used timestamp
        api_key.last_used_at = datetime.now()
        
        return user_id
    
    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """
        List all API keys for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of APIKey objects (without plain keys)
        """
        return [
            key for key in self._keys.values()
            if key.user_id == user_id
        ]
    
    def revoke_key(self, key_id: str, user_id: str) -> bool:
        """
        Revoke an API key
        
        Args:
            key_id: API key ID
            user_id: User ID (for authorization)
            
        Returns:
            True if revoked successfully
            
        Raises:
            APIKeyError: If key not found or unauthorized
        """
        api_key = self._keys.get(key_id)
        
        if not api_key:
            raise APIKeyError("API key not found")
        
        if api_key.user_id != user_id:
            raise APIKeyError("Unauthorized to revoke this key")
        
        api_key.is_active = False
        return True
    
    def delete_key(self, key_id: str, user_id: str) -> bool:
        """
        Delete an API key
        
        Args:
            key_id: API key ID
            user_id: User ID (for authorization)
            
        Returns:
            True if deleted successfully
            
        Raises:
            APIKeyError: If key not found or unauthorized
        """
        api_key = self._keys.get(key_id)
        
        if not api_key:
            raise APIKeyError("API key not found")
        
        if api_key.user_id != user_id:
            raise APIKeyError("Unauthorized to delete this key")
        
        # Remove from storage
        hashed_key = api_key.key
        del self._keys[key_id]
        if hashed_key in self._key_to_user:
            del self._key_to_user[hashed_key]
        
        return True
    
    def rotate_key(self, key_id: str, user_id: str) -> APIKey:
        """
        Rotate an API key (create new, revoke old)
        
        Args:
            key_id: API key ID to rotate
            user_id: User ID (for authorization)
            
        Returns:
            New APIKey object with plain key
            
        Raises:
            APIKeyError: If key not found or unauthorized
        """
        old_key = self._keys.get(key_id)
        
        if not old_key:
            raise APIKeyError("API key not found")
        
        if old_key.user_id != user_id:
            raise APIKeyError("Unauthorized to rotate this key")
        
        # Create new key with same name and expiration
        from ara.api.auth.user_manager import UserManager
        user_manager = UserManager()
        user = user_manager.get_user_by_id(user_id)
        
        if not user:
            raise APIKeyError("User not found")
        
        expires_in_days = None
        if old_key.expires_at:
            expires_in_days = (old_key.expires_at - datetime.now()).days
        
        new_key = self.create_api_key(user, old_key.name, expires_in_days)
        
        # Revoke old key
        old_key.is_active = False
        
        return new_key
    
    def track_request(self, key: str) -> None:
        """
        Track a request for rate limiting
        
        Args:
            key: Plain API key
        """
        hashed_key = self.hash_key(key)
        
        # Find the key object
        for key_obj in self._keys.values():
            if key_obj.key == hashed_key:
                # Reset counter if new day
                today = datetime.now().date()
                if key_obj.last_reset_date.date() != today:
                    key_obj.requests_today = 0
                    key_obj.last_reset_date = datetime.now()
                
                key_obj.requests_today += 1
                break

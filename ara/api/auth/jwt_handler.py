"""
JWT token handling
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt

from ara.api.auth.models import TokenData, User
from ara.core.exceptions import AraAIException

# JWT settings (should be in config)
SECRET_KEY = "your-secret-key-change-this-in-production"  # TODO: Move to config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


class AuthenticationError(AraAIException):
    """Authentication error"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, {"type": "authentication_error"})


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        (hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password),
    )


def get_password_hash(password: str) -> str:
    """
    Hash a password

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token

    Args:
        user: User object
        expires_delta: Token expiration time

    Returns:
        JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "tier": user.tier.value,
        "exp": expire,
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token

    Args:
        token: JWT token string

    Returns:
        TokenData object

    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        tier: str = payload.get("tier")

        if user_id is None or email is None:
            raise AuthenticationError("Invalid token payload")

        return TokenData(user_id=user_id, email=email, role=role, tier=tier)
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def authenticate_user(email: str, password: str, user_db: dict) -> Optional[User]:
    """
    Authenticate a user with email and password

    Args:
        email: User email
        password: Plain text password
        user_db: User database (dict for now)

    Returns:
        User object if authentication successful, None otherwise
    """
    user_data = user_db.get(email)
    if not user_data:
        return None

    user = User(**user_data)
    if not verify_password(password, user.hashed_password):
        return None

    return user

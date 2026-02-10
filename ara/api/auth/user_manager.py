"""
User management
"""

from typing import Dict, Optional

from ara.api.auth.jwt_handler import get_password_hash
from ara.api.auth.models import AccessTier, User, UserRole
from ara.core.exceptions import AraAIException


class UserError(AraAIException):
    """User management error"""

    def __init__(self, message: str = "User error"):
        super().__init__(message, {"type": "user_error"})


class UserManager:
    """
    Manages user accounts
    """

    def __init__(self):
        """Initialize user manager"""
        # In-memory storage for demo (should use database in production)
        self._users: Dict[str, User] = {}
        self._email_to_id: Dict[str, str] = {}

        # Create demo users
        self._create_demo_users()

    def _create_demo_users(self):
        """Create demo users for testing"""
        # Admin user
        admin = User(
            email="admin@ara.ai",
            username="admin",
            hashed_password=get_password_hash("admin123"),
            role=UserRole.ADMIN,
            tier=AccessTier.ENTERPRISE,
        )
        self._users[admin.id] = admin
        self._email_to_id[admin.email] = admin.id

        # Pro user
        pro_user = User(
            email="pro@ara.ai",
            username="prouser",
            hashed_password=get_password_hash("pro123"),
            role=UserRole.USER,
            tier=AccessTier.PRO,
        )
        self._users[pro_user.id] = pro_user
        self._email_to_id[pro_user.email] = pro_user.id

        # Free user
        free_user = User(
            email="free@ara.ai",
            username="freeuser",
            hashed_password=get_password_hash("free123"),
            role=UserRole.USER,
            tier=AccessTier.FREE,
        )
        self._users[free_user.id] = free_user
        self._email_to_id[free_user.email] = free_user.id

    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.USER,
        tier: AccessTier = AccessTier.FREE,
    ) -> User:
        """
        Create a new user

        Args:
            email: User email
            username: Username
            password: Plain text password
            role: User role
            tier: Access tier

        Returns:
            User object

        Raises:
            UserError: If user already exists
        """
        if email in self._email_to_id:
            raise UserError("User with this email already exists")

        user = User(
            email=email,
            username=username,
            hashed_password=get_password_hash(password),
            role=role,
            tier=tier,
        )

        self._users[user.id] = user
        self._email_to_id[email] = user.id

        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email

        Args:
            email: User email

        Returns:
            User object or None
        """
        user_id = self._email_to_id.get(email)
        if user_id:
            return self._users.get(user_id)
        return None

    def update_user_tier(self, user_id: str, tier: AccessTier) -> User:
        """
        Update user's access tier

        Args:
            user_id: User ID
            tier: New access tier

        Returns:
            Updated user object

        Raises:
            UserError: If user not found
        """
        user = self._users.get(user_id)
        if not user:
            raise UserError("User not found")

        user.tier = tier
        return user

    def update_user_role(self, user_id: str, role: UserRole) -> User:
        """
        Update user's role

        Args:
            user_id: User ID
            role: New role

        Returns:
            Updated user object

        Raises:
            UserError: If user not found
        """
        user = self._users.get(user_id)
        if not user:
            raise UserError("User not found")

        user.role = role
        return user

    def deactivate_user(self, user_id: str) -> User:
        """
        Deactivate a user

        Args:
            user_id: User ID

        Returns:
            Updated user object

        Raises:
            UserError: If user not found
        """
        user = self._users.get(user_id)
        if not user:
            raise UserError("User not found")

        user.is_active = False
        return user

    def activate_user(self, user_id: str) -> User:
        """
        Activate a user

        Args:
            user_id: User ID

        Returns:
            Updated user object

        Raises:
            UserError: If user not found
        """
        user = self._users.get(user_id)
        if not user:
            raise UserError("User not found")

        user.is_active = True
        return user

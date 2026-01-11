"""
Input Sanitization Module

Provides comprehensive input validation and sanitization to prevent
injection attacks and ensure data integrity.
"""

import re
from typing import Any, Dict, List, Optional
from datetime import datetime


class InputSanitizer:
    """Sanitizes and validates user inputs"""

    # Regex patterns for validation
    SYMBOL_PATTERN = re.compile(r"^[A-Z0-9\-\.]{1,20}$")
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    API_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{32,128}$")

    # Dangerous characters and patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\/\*|\*\/|xp_|sp_)",
        r"(\bOR\b.*=.*|1=1|'=')",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]

    @classmethod
    def sanitize_symbol(cls, symbol: str) -> str:
        """
        Sanitize and validate trading symbol

        Args:
            symbol: Trading symbol (e.g., AAPL, BTC-USD)

        Returns:
            Sanitized symbol

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        # Convert to uppercase and strip whitespace
        symbol = symbol.strip().upper()

        # Validate format
        if not cls.SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"Invalid symbol format: {symbol}. "
                "Must contain only letters, numbers, hyphens, and dots (1-20 chars)"
            )

        return symbol

    @classmethod
    def sanitize_integer(
        cls,
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        field_name: str = "value",
    ) -> int:
        """
        Sanitize and validate integer input

        Args:
            value: Input value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Name of field for error messages

        Returns:
            Validated integer

        Raises:
            ValueError: If value is invalid
        """
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be an integer")

        if min_value is not None and int_value < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}")

        if max_value is not None and int_value > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}")

        return int_value

    @classmethod
    def sanitize_float(
        cls,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        field_name: str = "value",
    ) -> float:
        """
        Sanitize and validate float input

        Args:
            value: Input value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Name of field for error messages

        Returns:
            Validated float

        Raises:
            ValueError: If value is invalid
        """
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a number")

        if not (-1e308 < float_value < 1e308):  # Check for infinity
            raise ValueError(f"{field_name} must be a finite number")

        if min_value is not None and float_value < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}")

        if max_value is not None and float_value > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}")

        return float_value

    @classmethod
    def sanitize_string(
        cls,
        value: Any,
        max_length: int = 1000,
        allow_empty: bool = False,
        field_name: str = "value",
    ) -> str:
        """
        Sanitize and validate string input

        Args:
            value: Input value
            max_length: Maximum allowed length
            allow_empty: Whether empty strings are allowed
            field_name: Name of field for error messages

        Returns:
            Sanitized string

        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        # Strip whitespace
        sanitized = value.strip()

        if not allow_empty and not sanitized:
            raise ValueError(f"{field_name} cannot be empty")

        if len(sanitized) > max_length:
            raise ValueError(f"{field_name} must be <= {max_length} characters")

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError(f"{field_name} contains potentially dangerous content")

        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError(f"{field_name} contains potentially dangerous content")

        return sanitized

    @classmethod
    def sanitize_email(cls, email: str) -> str:
        """
        Sanitize and validate email address

        Args:
            email: Email address

        Returns:
            Sanitized email

        Raises:
            ValueError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise ValueError("Email must be a non-empty string")

        email = email.strip().lower()

        if not cls.EMAIL_PATTERN.match(email):
            raise ValueError("Invalid email format")

        if len(email) > 254:  # RFC 5321
            raise ValueError("Email address too long")

        return email

    @classmethod
    def sanitize_api_key(cls, api_key: str) -> str:
        """
        Sanitize and validate API key

        Args:
            api_key: API key

        Returns:
            Sanitized API key

        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("API key must be a non-empty string")

        api_key = api_key.strip()

        if not cls.API_KEY_PATTERN.match(api_key):
            raise ValueError(
                "Invalid API key format. "
                "Must contain only alphanumeric characters, hyphens, and underscores (32-128 chars)"
            )

        return api_key

    @classmethod
    def sanitize_date(cls, date_str: str) -> datetime:
        """
        Sanitize and validate date string

        Args:
            date_str: Date string in ISO format (YYYY-MM-DD)

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date is invalid
        """
        if not date_str or not isinstance(date_str, str):
            raise ValueError("Date must be a non-empty string")

        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.strip())
        except ValueError:
            pass

        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        raise ValueError(
            "Invalid date format. Use ISO format (YYYY-MM-DD) or common formats"
        )

    @classmethod
    def sanitize_list(
        cls, value: Any, item_type: type, max_items: int = 100, field_name: str = "list"
    ) -> List[Any]:
        """
        Sanitize and validate list input

        Args:
            value: Input value
            item_type: Expected type of list items
            max_items: Maximum number of items
            field_name: Name of field for error messages

        Returns:
            Validated list

        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list")

        if len(value) > max_items:
            raise ValueError(f"{field_name} must contain <= {max_items} items")

        # Validate each item
        sanitized = []
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                raise ValueError(
                    f"{field_name}[{i}] must be of type {item_type.__name__}"
                )
            sanitized.append(item)

        return sanitized

    @classmethod
    def sanitize_dict(
        cls,
        value: Any,
        allowed_keys: Optional[List[str]] = None,
        field_name: str = "dict",
    ) -> Dict[str, Any]:
        """
        Sanitize and validate dictionary input

        Args:
            value: Input value
            allowed_keys: List of allowed keys (None = all allowed)
            field_name: Name of field for error messages

        Returns:
            Validated dictionary

        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a dictionary")

        if allowed_keys is not None:
            for key in value.keys():
                if key not in allowed_keys:
                    raise ValueError(
                        f"Invalid key '{key}' in {field_name}. "
                        f"Allowed keys: {', '.join(allowed_keys)}"
                    )

        return value

    @classmethod
    def sanitize_enum(
        cls, value: Any, allowed_values: List[str], field_name: str = "value"
    ) -> str:
        """
        Sanitize and validate enum input

        Args:
            value: Input value
            allowed_values: List of allowed values
            field_name: Name of field for error messages

        Returns:
            Validated value

        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        value = value.strip().lower()

        if value not in [v.lower() for v in allowed_values]:
            raise ValueError(
                f"Invalid {field_name}: {value}. "
                f"Allowed values: {', '.join(allowed_values)}"
            )

        return value

"""
Deprecation Warning System

Provides decorators and utilities for managing deprecated features
with clear migration paths and timelines.
"""

import warnings
import functools
from enum import Enum
from typing import Optional, Callable
from datetime import datetime


class DeprecationLevel(Enum):
    """Deprecation severity levels"""

    INFO = "info"  # Feature will be deprecated in future
    WARNING = "warning"  # Feature is deprecated, still works
    ERROR = "error"  # Feature is deprecated, may not work correctly
    REMOVED = "removed"  # Feature has been removed


class DeprecationTimeline:
    """Deprecation timeline for features"""

    # Version 4.0.0 (Current) - Compatibility layer introduced
    V4_0_0 = "4.0.0"
    V4_0_0_DATE = datetime(2024, 1, 1)

    # Version 4.5.0 - Deprecation warnings become more prominent
    V4_5_0 = "4.5.0"
    V4_5_0_DATE = datetime(2024, 6, 1)

    # Version 5.0.0 - Compatibility layer removed
    V5_0_0 = "5.0.0"
    V5_0_0_DATE = datetime(2025, 1, 1)

    @classmethod
    def get_current_version(cls) -> str:
        """Get current version"""
        return cls.V4_0_0

    @classmethod
    def get_removal_version(cls) -> str:
        """Get version when compatibility layer will be removed"""
        return cls.V5_0_0

    @classmethod
    def get_removal_date(cls) -> datetime:
        """Get estimated date when compatibility layer will be removed"""
        return cls.V5_0_0_DATE


def deprecated(
    reason: str,
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
    level: DeprecationLevel = DeprecationLevel.WARNING,
) -> Callable:
    """
    Decorator to mark functions/classes as deprecated

    Args:
        reason: Why the feature is deprecated
        version: Version when deprecation started
        removal_version: Version when feature will be removed
        alternative: Suggested alternative to use
        level: Deprecation severity level

    Example:
        @deprecated(
            reason="Old API structure",
            version="4.0.0",
            removal_version="5.0.0",
            alternative="ara.api.PredictionEngine.predict()",
            level=DeprecationLevel.WARNING
        )
        def old_predict(symbol):
            pass
    """
    if removal_version is None:
        removal_version = DeprecationTimeline.get_removal_version()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build deprecation message
            msg_parts = [
                f"{func.__name__} is deprecated since version {version}.",
                reason,
            ]

            if alternative:
                msg_parts.append(f"Use {alternative} instead.")

            if removal_version:
                msg_parts.append(f"This feature will be removed in version {removal_version}.")

            message = " ".join(msg_parts)

            # Issue warning based on level
            if level == DeprecationLevel.INFO:
                warnings.warn(message, FutureWarning, stacklevel=2)
            elif level == DeprecationLevel.WARNING:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            elif level == DeprecationLevel.ERROR:
                warnings.warn(message, PendingDeprecationWarning, stacklevel=2)
            elif level == DeprecationLevel.REMOVED:
                raise RuntimeError(
                    f"{func.__name__} has been removed in version {removal_version}. "
                    f"{alternative if alternative else 'No alternative available.'}"
                )

            # Call original function
            return func(*args, **kwargs)

        # Add deprecation metadata
        wrapper.__deprecated__ = True
        wrapper.__deprecation_info__ = {
            "reason": reason,
            "version": version,
            "removal_version": removal_version,
            "alternative": alternative,
            "level": level.value,
        }

        return wrapper

    return decorator


class DeprecationWarningManager:
    """Manages deprecation warnings and provides migration guidance"""

    def __init__(self):
        self._warnings_issued = set()
        self._migration_guide_shown = False

    def warn_once(self, feature: str, message: str):
        """Issue a deprecation warning only once per feature"""
        if feature not in self._warnings_issued:
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            self._warnings_issued.add(feature)

    def show_migration_guide(self):
        """Show migration guide to user"""
        if not self._migration_guide_shown:
            print("\n" + "=" * 70)
            print("MIGRATION GUIDE: ARA AI 4.0")
            print("=" * 70)
            print("\nYou are using the backward compatibility layer.")
            print("Please migrate to the new API for better performance and features.")
            print("\nOld API (deprecated):")
            print("  from meridianalgo.core import predict_stock")
            print("  result = predict_stock('AAPL', days=5)")
            print("\nNew API (recommended):")
            print("  from ara.api.prediction_engine import PredictionEngine")
            print("  engine = PredictionEngine()")
            print("  result = await engine.predict('AAPL', days=5)")
            print("\nFor more information, see:")
            print("  https://docs.ara-ai.com/migration-guide")
            print("\nCompatibility layer will be removed in version 5.0.0")
            print("=" * 70 + "\n")

            self._migration_guide_shown = True

    def get_deprecation_summary(self) -> dict:
        """Get summary of deprecation warnings issued"""
        return {
            "warnings_issued": len(self._warnings_issued),
            "features": list(self._warnings_issued),
            "migration_guide_shown": self._migration_guide_shown,
            "current_version": DeprecationTimeline.get_current_version(),
            "removal_version": DeprecationTimeline.get_removal_version(),
            "removal_date": DeprecationTimeline.get_removal_date().isoformat(),
        }


# Global warning manager
_warning_manager = DeprecationWarningManager()


def get_warning_manager() -> DeprecationWarningManager:
    """Get global deprecation warning manager"""
    return _warning_manager

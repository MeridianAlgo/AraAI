"""
Error tracking integration (Sentry-compatible)
Provides error tracking and reporting capabilities
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.asyncio import AsyncioIntegration

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    sentry_sdk = None

from ara.utils.logging import get_logger

logger = get_logger(__name__)


class ErrorTracker:
    """
    Error tracking and reporting
    Compatible with Sentry and similar services
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        environment: str = "development",
        release: Optional[str] = None,
        sample_rate: float = 1.0,
        traces_sample_rate: float = 0.1,
        enabled: bool = True,
    ):
        """
        Initialize error tracker

        Args:
            dsn: Sentry DSN or similar endpoint
            environment: Environment name (development, staging, production)
            release: Release version
            sample_rate: Error sampling rate (0.0 to 1.0)
            traces_sample_rate: Trace sampling rate (0.0 to 1.0)
            enabled: Whether error tracking is enabled
        """
        self.enabled = enabled and SENTRY_AVAILABLE and dsn is not None
        self.dsn = dsn
        self.environment = environment

        if not SENTRY_AVAILABLE and enabled and dsn:
            logger.warning(
                "Sentry SDK not available. Install with: pip install sentry-sdk"
            )
            self.enabled = False
            return

        if self.enabled:
            try:
                sentry_sdk.init(
                    dsn=dsn,
                    environment=environment,
                    release=release,
                    sample_rate=sample_rate,
                    traces_sample_rate=traces_sample_rate,
                    integrations=[FastApiIntegration(), AsyncioIntegration()],
                    # Send default PII (Personally Identifiable Information)
                    send_default_pii=False,
                    # Attach stacktrace
                    attach_stacktrace=True,
                    # Max breadcrumbs
                    max_breadcrumbs=50,
                )
                logger.info(f"Error tracking initialized: {environment}")
            except Exception as e:
                logger.error(f"Failed to initialize error tracking: {e}")
                self.enabled = False

    def capture_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        level: str = "error",
    ) -> Optional[str]:
        """
        Capture an exception

        Args:
            exception: Exception to capture
            context: Additional context
            level: Error level (error, warning, info)

        Returns:
            Event ID if captured, None otherwise
        """
        if not self.enabled:
            # Log locally if Sentry not available
            logger.error(
                f"Exception captured: {exception}",
                exception_type=type(exception).__name__,
                context=context or {},
            )
            return None

        try:
            with sentry_sdk.push_scope() as scope:
                # Set level
                scope.level = level

                # Add context
                if context:
                    for key, value in context.items():
                        scope.set_context(key, value)

                # Capture exception
                event_id = sentry_sdk.capture_exception(exception)
                return event_id
        except Exception as e:
            logger.error(f"Failed to capture exception: {e}")
            return None

    def capture_message(
        self,
        message: str,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Capture a message

        Args:
            message: Message to capture
            level: Message level (error, warning, info)
            context: Additional context

        Returns:
            Event ID if captured, None otherwise
        """
        if not self.enabled:
            logger.info(message, context=context or {})
            return None

        try:
            with sentry_sdk.push_scope() as scope:
                # Set level
                scope.level = level

                # Add context
                if context:
                    for key, value in context.items():
                        scope.set_context(key, value)

                # Capture message
                event_id = sentry_sdk.capture_message(message, level=level)
                return event_id
        except Exception as e:
            logger.error(f"Failed to capture message: {e}")
            return None

    def set_user(self, user_id: str, **kwargs):
        """
        Set user context

        Args:
            user_id: User identifier
            **kwargs: Additional user attributes (email, username, etc.)
        """
        if not self.enabled:
            return

        try:
            sentry_sdk.set_user({"id": user_id, **kwargs})
        except Exception as e:
            logger.error(f"Failed to set user context: {e}")

    def set_tag(self, key: str, value: str):
        """
        Set a tag

        Args:
            key: Tag key
            value: Tag value
        """
        if not self.enabled:
            return

        try:
            sentry_sdk.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag: {e}")

    def set_context(self, name: str, context: Dict[str, Any]):
        """
        Set context

        Args:
            name: Context name
            context: Context data
        """
        if not self.enabled:
            return

        try:
            sentry_sdk.set_context(name, context)
        except Exception as e:
            logger.error(f"Failed to set context: {e}")

    def add_breadcrumb(
        self,
        message: str,
        category: str = "default",
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a breadcrumb

        Args:
            message: Breadcrumb message
            category: Breadcrumb category
            level: Breadcrumb level
            data: Additional data
        """
        if not self.enabled:
            return

        try:
            sentry_sdk.add_breadcrumb(
                message=message, category=category, level=level, data=data or {}
            )
        except Exception as e:
            logger.error(f"Failed to add breadcrumb: {e}")


# Global error tracker
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """
    Get global error tracker

    Returns:
        ErrorTracker instance
    """
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(enabled=False)
    return _error_tracker


def init_error_tracking(
    dsn: Optional[str] = None,
    environment: str = "development",
    release: Optional[str] = None,
    sample_rate: float = 1.0,
    traces_sample_rate: float = 0.1,
):
    """
    Initialize error tracking

    Args:
        dsn: Sentry DSN or similar endpoint
        environment: Environment name
        release: Release version
        sample_rate: Error sampling rate
        traces_sample_rate: Trace sampling rate
    """
    global _error_tracker
    _error_tracker = ErrorTracker(
        dsn=dsn,
        environment=environment,
        release=release,
        sample_rate=sample_rate,
        traces_sample_rate=traces_sample_rate,
        enabled=dsn is not None,
    )


def capture_exception(
    exception: Exception, context: Optional[Dict[str, Any]] = None, level: str = "error"
) -> Optional[str]:
    """
    Capture an exception

    Args:
        exception: Exception to capture
        context: Additional context
        level: Error level

    Returns:
        Event ID if captured
    """
    tracker = get_error_tracker()
    return tracker.capture_exception(exception, context, level)


def capture_message(
    message: str, level: str = "info", context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture a message

    Args:
        message: Message to capture
        level: Message level
        context: Additional context

    Returns:
        Event ID if captured
    """
    tracker = get_error_tracker()
    return tracker.capture_message(message, level, context)


def track_errors(func: Callable) -> Callable:
    """
    Decorator to automatically track errors

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            capture_exception(
                e,
                context={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args)[:200],  # Limit size
                    "kwargs": str(kwargs)[:200],
                },
            )
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            capture_exception(
                e,
                context={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200],
                },
            )
            raise

    import inspect

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

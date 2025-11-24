"""
ARA AI REST API
FastAPI-based REST API for predictions, backtesting, and portfolio management
"""

# Lazy import to avoid circular dependencies
def create_app():
    """Create and return FastAPI application"""
    from ara.api.app import create_app as _create_app
    return _create_app()

__all__ = ["create_app"]

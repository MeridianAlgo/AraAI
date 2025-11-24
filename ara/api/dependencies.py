"""
FastAPI dependencies for dependency injection
"""

from typing import Optional
from fastapi import Header, HTTPException, status
from functools import lru_cache
import uuid


# Simple in-memory cache for request tracking
_request_cache = {}


def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())


def get_request_id(x_request_id: Optional[str] = Header(None)) -> str:
    """Get or generate request ID from header"""
    if x_request_id:
        return x_request_id
    return generate_request_id()


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify API key (placeholder for future authentication)
    Currently allows all requests
    """
    # TODO: Implement proper API key verification in task 17
    return x_api_key or "anonymous"


def get_cache_key(symbol: str, days: int, analysis_level: str) -> str:
    """Generate cache key for predictions"""
    return f"pred:{symbol}:{days}:{analysis_level}"


def get_cached_prediction(cache_key: str) -> Optional[dict]:
    """Get cached prediction if available and not expired"""
    # TODO: Implement Redis caching in future
    return _request_cache.get(cache_key)


def cache_prediction(cache_key: str, prediction: dict, ttl: int = 60):
    """Cache prediction result"""
    # TODO: Implement Redis caching with TTL in future
    _request_cache[cache_key] = prediction


def clear_cache():
    """Clear prediction cache"""
    _request_cache.clear()

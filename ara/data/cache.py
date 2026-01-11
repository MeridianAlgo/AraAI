"""
Multi-level caching system for ARA AI
Implements L1 (in-memory) and L2 (Redis) caching with LRU eviction
"""

import hashlib
import pickle
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
import asyncio

from ara.utils import get_logger

logger = get_logger(__name__)


class LRUCache:
    """
    Thread-safe in-memory LRU cache (L1)
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 60):
        """
        Initialize LRU cache

        Args:
            max_size: Maximum number of items (in MB)
            default_ttl: Default time-to-live in seconds
        """
        self.max_size_mb = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "size_bytes": 0}

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            # Check expiration
            entry = self._cache[key]
            if datetime.now() > entry["expires_at"]:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1

            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = default)
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)

            # Calculate size
            value_size = len(pickle.dumps(value))

            # Evict if necessary
            while self._should_evict(value_size):
                self._evict_oldest()

            # Store entry
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "size": value_size,
                "created_at": datetime.now(),
            }

            # Move to end
            self._cache.move_to_end(key)
            self._stats["size_bytes"] += value_size

    def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats["size_bytes"] -= entry["size"]
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats["size_bytes"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "size_mb": self._stats["size_bytes"] / (1024 * 1024),
                "items": len(self._cache),
            }

    def _should_evict(self, new_size: int) -> bool:
        """Check if eviction is needed"""
        current_mb = self._stats["size_bytes"] / (1024 * 1024)
        new_mb = new_size / (1024 * 1024)
        return (current_mb + new_mb) > self.max_size_mb and len(self._cache) > 0

    def _evict_oldest(self) -> None:
        """Evict least recently used item"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats["size_bytes"] -= entry["size"]
            self._stats["evictions"] += 1
            logger.debug(f"Evicted cache key: {key}")


class RedisCache:
    """
    Redis-based cache (L2)
    Requires redis-py library
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default time-to-live in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self._client = None
        self._stats = {"hits": 0, "misses": 0}

        # Try to import redis
        try:
            import redis

            self._redis_module = redis
        except ImportError:
            logger.warning(
                "redis-py not installed. Redis cache will be disabled. "
                "Install with: pip install redis"
            )
            self._redis_module = None

    def _get_client(self):
        """Get or create Redis client"""
        if self._redis_module is None:
            return None

        if self._client is None:
            try:
                self._client = self._redis_module.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                # Test connection
                self._client.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._client = None

        return self._client

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            value = client.get(key)
            if value is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return pickle.loads(value)

        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in Redis

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            ttl = ttl or self.default_ttl
            serialized = pickle.dumps(value)
            client.setex(key, ttl, serialized)
            return True

        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from Redis

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            return client.delete(key) > 0
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all keys in current database

        Returns:
            True if successful
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            client.flushdb()
            return True
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        client = self._get_client()
        info = {}
        if client:
            try:
                info = client.info("stats")
            except:
                pass

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "connected": client is not None,
            "redis_info": info,
        }


class CacheManager:
    """
    Multi-level cache manager
    Coordinates L1 (in-memory) and L2 (Redis) caches
    """

    def __init__(
        self,
        l1_size_mb: int = 100,
        l1_ttl: int = 60,
        l2_enabled: bool = True,
        l2_host: str = "localhost",
        l2_port: int = 6379,
        l2_ttl: int = 3600,
        l2_password: Optional[str] = None,
    ):
        """
        Initialize cache manager

        Args:
            l1_size_mb: L1 cache size in MB
            l1_ttl: L1 cache TTL in seconds
            l2_enabled: Enable L2 (Redis) cache
            l2_host: Redis host
            l2_port: Redis port
            l2_ttl: L2 cache TTL in seconds
            l2_password: Redis password
        """
        self.l1 = LRUCache(max_size=l1_size_mb, default_ttl=l1_ttl)

        self.l2 = None
        if l2_enabled:
            self.l2 = RedisCache(
                host=l2_host, port=l2_port, password=l2_password, default_ttl=l2_ttl
            )

        # Cache warming configuration
        self._warm_cache_enabled = False
        self._warm_cache_keys: Dict[str, Callable] = {}

        logger.info(
            "Initialized CacheManager", l1_size_mb=l1_size_mb, l2_enabled=l2_enabled
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (L1 -> L2)

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            logger.debug(f"L1 cache hit: {key}")
            return value

        # Try L2
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                logger.debug(f"L2 cache hit: {key}")
                # Promote to L1
                self.l1.set(key, value)
                return value

        logger.debug(f"Cache miss: {key}")
        return None

    def set(
        self,
        key: str,
        value: Any,
        l1_ttl: Optional[int] = None,
        l2_ttl: Optional[int] = None,
    ) -> None:
        """
        Set value in cache (both L1 and L2)

        Args:
            key: Cache key
            value: Value to cache
            l1_ttl: L1 TTL in seconds
            l2_ttl: L2 TTL in seconds
        """
        # Set in L1
        self.l1.set(key, value, ttl=l1_ttl)

        # Set in L2
        if self.l2:
            self.l2.set(key, value, ttl=l2_ttl)

        logger.debug(f"Cached: {key}")

    def delete(self, key: str) -> None:
        """
        Delete key from all cache levels

        Args:
            key: Cache key
        """
        self.l1.delete(key)
        if self.l2:
            self.l2.delete(key)

        logger.debug(f"Deleted from cache: {key}")

    def clear(self) -> None:
        """Clear all cache levels"""
        self.l1.clear()
        if self.l2:
            self.l2.clear()

        logger.info("Cleared all caches")

    def invalidate_pattern(self, pattern: str) -> None:
        """
        Invalidate all keys matching pattern

        Args:
            pattern: Key pattern (e.g., 'market_data:*')
        """
        # L1 doesn't support pattern matching, so clear all
        # In production, you'd want to track keys by pattern
        logger.warning(
            f"Pattern invalidation not fully supported, clearing L1. Pattern: {pattern}"
        )
        self.l1.clear()

        # L2 (Redis) supports pattern matching
        if self.l2 and self.l2._get_client():
            try:
                client = self.l2._get_client()
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} keys matching: {pattern}")
            except Exception as e:
                logger.warning(f"Failed to invalidate pattern: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all cache levels

        Returns:
            Dict with cache stats
        """
        stats = {
            "l1": self.l1.get_stats(),
            "l2": self.l2.get_stats() if self.l2 else None,
        }

        # Calculate combined hit rate
        total_hits = stats["l1"]["hits"]
        total_misses = stats["l1"]["misses"]

        if stats["l2"]:
            total_hits += stats["l2"]["hits"]
            total_misses += stats["l2"]["misses"]

        total_requests = total_hits + total_misses
        combined_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        stats["combined_hit_rate"] = combined_hit_rate

        return stats

    def enable_cache_warming(
        self, keys: Dict[str, Callable], interval_seconds: int = 300
    ) -> None:
        """
        Enable cache warming for frequently accessed data

        Args:
            keys: Dict mapping cache keys to functions that fetch the data
            interval_seconds: How often to refresh (default: 5 minutes)
        """
        self._warm_cache_enabled = True
        self._warm_cache_keys = keys

        # Start background task
        asyncio.create_task(self._cache_warming_loop(interval_seconds))

        logger.info(
            f"Enabled cache warming for {len(keys)} keys", interval=interval_seconds
        )

    async def _cache_warming_loop(self, interval: int) -> None:
        """Background task for cache warming"""
        while self._warm_cache_enabled:
            try:
                for key, fetch_func in self._warm_cache_keys.items():
                    try:
                        value = await fetch_func()
                        self.set(key, value)
                        logger.debug(f"Warmed cache: {key}")
                    except Exception as e:
                        logger.warning(f"Failed to warm cache for {key}: {e}")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Cache warming loop error: {e}")
                await asyncio.sleep(interval)

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create deterministic key from arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)

        # Hash for consistent length
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"ara:{key_hash}"


def cached(cache_manager: CacheManager, ttl: int = 60, key_prefix: str = ""):
    """
    Decorator for caching function results

    Args:
        cache_manager: CacheManager instance
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys

    Returns:
        Decorated function
    """

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{CacheManager.generate_key(*args, **kwargs)}"

            # Try cache
            result = cache_manager.get(key)
            if result is not None:
                return result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            cache_manager.set(key, result, l1_ttl=ttl, l2_ttl=ttl * 10)

            return result

        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{CacheManager.generate_key(*args, **kwargs)}"

            # Try cache
            result = cache_manager.get(key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.set(key, result, l1_ttl=ttl, l2_ttl=ttl * 10)

            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

"""
Cache Optimization Module
Implements intelligent cache warming, hit rate monitoring, and optimization strategies
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading

from ara.utils import get_logger

logger = get_logger(__name__)


class CacheHitRateMonitor:
    """
    Monitors cache hit rates and provides optimization recommendations
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize hit rate monitor

        Args:
            window_size: Number of recent requests to track
        """
        self.window_size = window_size
        self._hits = deque(maxlen=window_size)
        self._misses = deque(maxlen=window_size)
        self._key_stats = defaultdict(lambda: {"hits": 0, "misses": 0, "last_access": None})
        self._lock = threading.RLock()

        # Thresholds for optimization
        self.low_hit_rate_threshold = 0.7  # 70%
        self.high_miss_rate_threshold = 0.3  # 30%

    def record_hit(self, key: str) -> None:
        """Record a cache hit"""
        with self._lock:
            self._hits.append((time.time(), key))
            self._key_stats[key]["hits"] += 1
            self._key_stats[key]["last_access"] = datetime.now()

    def record_miss(self, key: str) -> None:
        """Record a cache miss"""
        with self._lock:
            self._misses.append((time.time(), key))
            self._key_stats[key]["misses"] += 1
            self._key_stats[key]["last_access"] = datetime.now()

    def get_hit_rate(self, time_window_seconds: Optional[int] = None) -> float:
        """
        Calculate hit rate

        Args:
            time_window_seconds: Calculate for recent time window (None = all)

        Returns:
            Hit rate (0.0 to 1.0)
        """
        with self._lock:
            if time_window_seconds:
                cutoff = time.time() - time_window_seconds
                hits = sum(1 for t, _ in self._hits if t >= cutoff)
                misses = sum(1 for t, _ in self._misses if t >= cutoff)
            else:
                hits = len(self._hits)
                misses = len(self._misses)

            total = hits + misses
            return hits / total if total > 0 else 0.0

    def get_key_stats(self, key: str) -> Dict[str, Any]:
        """Get statistics for a specific key"""
        with self._lock:
            stats = self._key_stats[key]
            total = stats["hits"] + stats["misses"]
            hit_rate = stats["hits"] / total if total > 0 else 0.0

            return {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": hit_rate,
                "last_access": stats["last_access"],
            }

    def get_top_keys(self, n: int = 10, by: str = "hits") -> List[Tuple[str, Dict]]:
        """
        Get top N keys by hits or misses

        Args:
            n: Number of keys to return
            by: Sort by 'hits' or 'misses'

        Returns:
            List of (key, stats) tuples
        """
        with self._lock:
            sorted_keys = sorted(self._key_stats.items(), key=lambda x: x[1][by], reverse=True)
            return [(k, self.get_key_stats(k)) for k, _ in sorted_keys[:n]]

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Analyze cache performance and provide recommendations

        Returns:
            List of recommendation dicts
        """
        recommendations = []

        # Check overall hit rate
        hit_rate = self.get_hit_rate()
        if hit_rate < self.low_hit_rate_threshold:
            recommendations.append(
                {
                    "type": "low_hit_rate",
                    "severity": "high",
                    "message": f"Overall hit rate is low ({hit_rate:.2%})",
                    "suggestion": "Consider increasing cache size or TTL",
                }
            )

        # Find frequently missed keys (candidates for cache warming)
        with self._lock:
            high_miss_keys = [
                (key, stats)
                for key, stats in self._key_stats.items()
                if stats["misses"] > 10
                and stats["misses"] / (stats["hits"] + stats["misses"])
                > self.high_miss_rate_threshold
            ]

        if high_miss_keys:
            recommendations.append(
                {
                    "type": "high_miss_keys",
                    "severity": "medium",
                    "message": f"Found {len(high_miss_keys)} keys with high miss rates",
                    "keys": [k for k, _ in high_miss_keys[:5]],
                    "suggestion": "Consider cache warming for these keys",
                }
            )

        # Find stale keys (not accessed recently)
        with self._lock:
            stale_cutoff = datetime.now() - timedelta(hours=1)
            stale_keys = [
                key
                for key, stats in self._key_stats.items()
                if stats["last_access"] and stats["last_access"] < stale_cutoff
            ]

        if len(stale_keys) > 100:
            recommendations.append(
                {
                    "type": "stale_keys",
                    "severity": "low",
                    "message": f"Found {len(stale_keys)} stale keys",
                    "suggestion": "Consider reducing cache size or implementing eviction",
                }
            )

        return recommendations

    def reset_stats(self) -> None:
        """Reset all statistics"""
        with self._lock:
            self._hits.clear()
            self._misses.clear()
            self._key_stats.clear()


class IntelligentCacheWarmer:
    """
    Intelligent cache warming system that learns access patterns
    """

    def __init__(self, cache_manager, monitor: CacheHitRateMonitor):
        """
        Initialize cache warmer

        Args:
            cache_manager: CacheManager instance
            monitor: CacheHitRateMonitor instance
        """
        self.cache_manager = cache_manager
        self.monitor = monitor
        self._warming_tasks: Dict[str, Callable] = {}
        self._warming_enabled = False
        self._warming_interval = 300  # 5 minutes default
        self._lock = threading.RLock()

        logger.info("Initialized IntelligentCacheWarmer")

    def register_warming_task(self, key: str, fetch_func: Callable, priority: int = 1) -> None:
        """
        Register a cache warming task

        Args:
            key: Cache key to warm
            fetch_func: Async function to fetch data
            priority: Priority (higher = more important)
        """
        with self._lock:
            self._warming_tasks[key] = {
                "fetch_func": fetch_func,
                "priority": priority,
                "last_warmed": None,
                "warm_count": 0,
                "errors": 0,
            }

        logger.info(f"Registered warming task: {key} (priority={priority})")

    def unregister_warming_task(self, key: str) -> None:
        """Unregister a cache warming task"""
        with self._lock:
            if key in self._warming_tasks:
                del self._warming_tasks[key]
                logger.info(f"Unregistered warming task: {key}")

    async def start_warming(self, interval_seconds: int = 300) -> None:
        """
        Start cache warming loop

        Args:
            interval_seconds: Warming interval in seconds
        """
        self._warming_enabled = True
        self._warming_interval = interval_seconds

        logger.info(f"Starting cache warming (interval={interval_seconds}s)")

        while self._warming_enabled:
            try:
                await self._warm_cache()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_warming(self) -> None:
        """Stop cache warming"""
        self._warming_enabled = False
        logger.info("Stopped cache warming")

    async def _warm_cache(self) -> None:
        """Execute cache warming for all registered tasks"""
        with self._lock:
            tasks = list(self._warming_tasks.items())

        # Sort by priority
        tasks.sort(key=lambda x: x[1]["priority"], reverse=True)

        for key, task_info in tasks:
            try:
                # Fetch data
                fetch_func = task_info["fetch_func"]
                if asyncio.iscoroutinefunction(fetch_func):
                    data = await fetch_func()
                else:
                    data = fetch_func()

                # Store in cache
                self.cache_manager.set(key, data)

                # Update task info
                with self._lock:
                    self._warming_tasks[key]["last_warmed"] = datetime.now()
                    self._warming_tasks[key]["warm_count"] += 1

                logger.debug(f"Warmed cache: {key}")

            except Exception as e:
                logger.warning(f"Failed to warm cache for {key}: {e}")
                with self._lock:
                    self._warming_tasks[key]["errors"] += 1

    def get_warming_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics"""
        with self._lock:
            return {
                "enabled": self._warming_enabled,
                "interval": self._warming_interval,
                "task_count": len(self._warming_tasks),
                "tasks": {
                    key: {
                        "priority": info["priority"],
                        "last_warmed": (
                            info["last_warmed"].isoformat() if info["last_warmed"] else None
                        ),
                        "warm_count": info["warm_count"],
                        "errors": info["errors"],
                    }
                    for key, info in self._warming_tasks.items()
                },
            }

    def auto_register_popular_keys(self, min_hits: int = 100) -> int:
        """
        Automatically register popular keys for warming

        Args:
            min_hits: Minimum hits to consider a key popular

        Returns:
            Number of keys registered
        """
        top_keys = self.monitor.get_top_keys(n=20, by="hits")
        registered = 0

        for key, stats in top_keys:
            if stats["hits"] >= min_hits and key not in self._warming_tasks:
                # We can't automatically create fetch functions,
                # so this is a placeholder for manual registration
                logger.info(f"Popular key identified for warming: {key} ({stats['hits']} hits)")
                registered += 1

        return registered


class ConnectionPool:
    """
    Generic connection pool for database connections
    """

    def __init__(
        self,
        create_connection: Callable,
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time: int = 300,
    ):
        """
        Initialize connection pool

        Args:
            create_connection: Function to create new connections
            max_connections: Maximum number of connections
            min_connections: Minimum number of connections to maintain
            max_idle_time: Maximum idle time before closing connection (seconds)
        """
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time

        self._pool: deque = deque()
        self._active_connections = 0
        self._lock = threading.RLock()
        self._stats = {"created": 0, "reused": 0, "closed": 0, "errors": 0}

        # Initialize minimum connections
        self._initialize_pool()

        logger.info(
            "Initialized ConnectionPool",
            max_connections=max_connections,
            min_connections=min_connections,
        )

    def _initialize_pool(self) -> None:
        """Initialize pool with minimum connections"""
        for _ in range(self.min_connections):
            try:
                conn = self.create_connection()
                self._pool.append(
                    {
                        "connection": conn,
                        "created_at": time.time(),
                        "last_used": time.time(),
                    }
                )
                self._stats["created"] += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")

    def get_connection(self):
        """
        Get a connection from the pool

        Returns:
            Database connection
        """
        with self._lock:
            # Try to reuse existing connection
            while self._pool:
                conn_info = self._pool.popleft()

                # Check if connection is still valid
                idle_time = time.time() - conn_info["last_used"]
                if idle_time < self.max_idle_time:
                    self._active_connections += 1
                    self._stats["reused"] += 1
                    return conn_info["connection"]
                else:
                    # Close stale connection
                    try:
                        conn_info["connection"].close()
                        self._stats["closed"] += 1
                    except:
                        pass

            # Create new connection if under limit
            if self._active_connections < self.max_connections:
                try:
                    conn = self.create_connection()
                    self._active_connections += 1
                    self._stats["created"] += 1
                    return conn
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(f"Failed to create connection: {e}")
                    raise
            else:
                raise Exception("Connection pool exhausted")

    def return_connection(self, connection) -> None:
        """
        Return a connection to the pool

        Args:
            connection: Database connection to return
        """
        with self._lock:
            self._active_connections -= 1
            self._pool.append(
                {
                    "connection": connection,
                    "created_at": time.time(),
                    "last_used": time.time(),
                }
            )

    def close_all(self) -> None:
        """Close all connections in the pool"""
        with self._lock:
            while self._pool:
                conn_info = self._pool.popleft()
                try:
                    conn_info["connection"].close()
                    self._stats["closed"] += 1
                except:
                    pass

            self._active_connections = 0

        logger.info("Closed all connections in pool")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "active_connections": self._active_connections,
                "max_connections": self.max_connections,
                "created": self._stats["created"],
                "reused": self._stats["reused"],
                "closed": self._stats["closed"],
                "errors": self._stats["errors"],
            }


class LazyLoader:
    """
    Lazy loading utility for models and data
    """

    def __init__(self):
        """Initialize lazy loader"""
        self._loaded_items: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable] = {}
        self._lock = threading.RLock()

        logger.info("Initialized LazyLoader")

    def register(self, name: str, loader: Callable) -> None:
        """
        Register a lazy-loadable item

        Args:
            name: Item name
            loader: Function to load the item
        """
        with self._lock:
            self._loaders[name] = loader

        logger.debug(f"Registered lazy loader: {name}")

    def get(self, name: str) -> Any:
        """
        Get an item (load if not already loaded)

        Args:
            name: Item name

        Returns:
            Loaded item
        """
        with self._lock:
            # Return if already loaded
            if name in self._loaded_items:
                return self._loaded_items[name]

            # Load if loader exists
            if name in self._loaders:
                logger.info(f"Lazy loading: {name}")
                item = self._loaders[name]()
                self._loaded_items[name] = item
                return item

            raise KeyError(f"No loader registered for: {name}")

    def unload(self, name: str) -> None:
        """
        Unload an item to free memory

        Args:
            name: Item name
        """
        with self._lock:
            if name in self._loaded_items:
                del self._loaded_items[name]
                logger.info(f"Unloaded: {name}")

    def unload_all(self) -> None:
        """Unload all items"""
        with self._lock:
            count = len(self._loaded_items)
            self._loaded_items.clear()
            logger.info(f"Unloaded {count} items")

    def is_loaded(self, name: str) -> bool:
        """Check if an item is loaded"""
        with self._lock:
            return name in self._loaded_items

    def get_stats(self) -> Dict[str, Any]:
        """Get lazy loader statistics"""
        with self._lock:
            return {
                "loaded_count": len(self._loaded_items),
                "registered_count": len(self._loaders),
                "loaded_items": list(self._loaded_items.keys()),
            }

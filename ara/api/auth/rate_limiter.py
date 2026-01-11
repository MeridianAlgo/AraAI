"""
Redis-based token bucket rate limiter
"""

import time
from typing import Optional, Tuple
from datetime import datetime, timedelta
from ara.api.auth.models import User, TierQuotas
from ara.core.exceptions import AraAIException


class RateLimitExceeded(AraAIException):
    """Rate limit exceeded error"""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None
    ):
        details = {"type": "rate_limit_exceeded"}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, details)
        self.retry_after = retry_after


class TokenBucket:
    """
    Token bucket algorithm for rate limiting
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient tokens
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait until tokens available

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter with per-user and per-endpoint limits
    """

    def __init__(self, use_redis: bool = False):
        """
        Initialize rate limiter

        Args:
            use_redis: Whether to use Redis (False = in-memory)
        """
        self.use_redis = use_redis

        # In-memory storage (should use Redis in production)
        self._user_buckets = {}  # user_id -> TokenBucket (per-minute)
        self._user_daily_counts = {}  # user_id -> (count, reset_time)
        self._endpoint_buckets = {}  # (user_id, endpoint) -> TokenBucket

        # Redis client (if enabled)
        self.redis_client = None
        if use_redis:
            try:
                import redis

                self.redis_client = redis.Redis(
                    host="localhost", port=6379, db=0, decode_responses=True
                )
            except ImportError:
                print("Warning: redis-py not installed, using in-memory rate limiting")
                self.use_redis = False

    def check_rate_limit(
        self, user: User, endpoint: Optional[str] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is within rate limits

        Args:
            user: User object
            endpoint: Optional endpoint name for per-endpoint limits

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        quotas = user.get_quotas()

        # Check per-minute limit
        allowed, retry_after = self._check_per_minute_limit(user, quotas)
        if not allowed:
            return False, retry_after

        # Check daily limit
        allowed, retry_after = self._check_daily_limit(user, quotas)
        if not allowed:
            return False, retry_after

        # Check per-endpoint limit (if specified)
        if endpoint:
            allowed, retry_after = self._check_endpoint_limit(user, endpoint, quotas)
            if not allowed:
                return False, retry_after

        return True, None

    def _check_per_minute_limit(
        self, user: User, quotas: TierQuotas
    ) -> Tuple[bool, Optional[int]]:
        """Check per-minute rate limit"""
        bucket_key = f"user:{user.id}:minute"

        if self.use_redis and self.redis_client:
            return self._check_redis_limit(
                bucket_key, quotas.requests_per_minute, 60  # 60 seconds window
            )
        else:
            return self._check_memory_limit(
                user.id,
                quotas.requests_per_minute,
                quotas.requests_per_minute / 60.0,  # refill rate per second
            )

    def _check_daily_limit(
        self, user: User, quotas: TierQuotas
    ) -> Tuple[bool, Optional[int]]:
        """Check daily rate limit"""
        user_id = user.id

        # Get or initialize daily count
        if user_id not in self._user_daily_counts:
            reset_time = datetime.now() + timedelta(days=1)
            reset_time = reset_time.replace(hour=0, minute=0, second=0, microsecond=0)
            self._user_daily_counts[user_id] = (0, reset_time)

        count, reset_time = self._user_daily_counts[user_id]

        # Reset if new day
        if datetime.now() >= reset_time:
            reset_time = datetime.now() + timedelta(days=1)
            reset_time = reset_time.replace(hour=0, minute=0, second=0, microsecond=0)
            count = 0

        # Check limit
        if count >= quotas.requests_per_day:
            retry_after = int((reset_time - datetime.now()).total_seconds())
            return False, retry_after

        # Increment count
        self._user_daily_counts[user_id] = (count + 1, reset_time)
        return True, None

    def _check_endpoint_limit(
        self, user: User, endpoint: str, quotas: TierQuotas
    ) -> Tuple[bool, Optional[int]]:
        """Check per-endpoint rate limit"""
        # Use same per-minute limit for endpoints
        bucket_key = f"{user.id}:{endpoint}"

        return self._check_memory_limit(
            bucket_key, quotas.requests_per_minute, quotas.requests_per_minute / 60.0
        )

    def _check_memory_limit(
        self, key: str, capacity: int, refill_rate: float
    ) -> Tuple[bool, Optional[int]]:
        """Check rate limit using in-memory token bucket"""
        # Get or create bucket
        if key not in self._user_buckets:
            self._user_buckets[key] = TokenBucket(capacity, refill_rate)

        bucket = self._user_buckets[key]

        # Try to consume token
        if bucket.consume(1):
            return True, None

        # Calculate retry after
        wait_time = bucket.get_wait_time(1)
        retry_after = int(wait_time) + 1

        return False, retry_after

    def _check_redis_limit(
        self, key: str, limit: int, window: int
    ) -> Tuple[bool, Optional[int]]:
        """Check rate limit using Redis"""
        if not self.redis_client:
            return True, None

        try:
            # Use Redis sliding window counter
            now = time.time()
            window_start = now - window

            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            count = self.redis_client.zcard(key)

            if count >= limit:
                # Get oldest entry to calculate retry_after
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = oldest[0][1]
                    retry_after = int(oldest_time + window - now) + 1
                    return False, retry_after
                return False, window

            # Add current request
            self.redis_client.zadd(key, {str(now): now})
            self.redis_client.expire(key, window)

            return True, None

        except Exception as e:
            # Fallback to allowing request if Redis fails
            print(f"Redis error: {e}")
            return True, None

    def get_rate_limit_headers(
        self, user: User, endpoint: Optional[str] = None
    ) -> dict:
        """
        Get rate limit headers for response

        Args:
            user: User object
            endpoint: Optional endpoint name

        Returns:
            Dictionary of headers
        """
        quotas = user.get_quotas()

        # Get current usage
        user_id = user.id
        bucket_key = user_id

        remaining = quotas.requests_per_minute
        if bucket_key in self._user_buckets:
            bucket = self._user_buckets[bucket_key]
            bucket._refill()
            remaining = int(bucket.tokens)

        # Get daily remaining
        daily_remaining = quotas.requests_per_day
        if user_id in self._user_daily_counts:
            count, _ = self._user_daily_counts[user_id]
            daily_remaining = max(0, quotas.requests_per_day - count)

        # Calculate reset time (next minute)
        reset_time = int(time.time()) + 60

        return {
            "X-RateLimit-Limit": str(quotas.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-Daily-Limit": str(quotas.requests_per_day),
            "X-RateLimit-Daily-Remaining": str(daily_remaining),
        }

    def reset_user_limits(self, user_id: str):
        """
        Reset rate limits for a user (admin function)

        Args:
            user_id: User ID
        """
        # Clear in-memory buckets
        if user_id in self._user_buckets:
            del self._user_buckets[user_id]

        if user_id in self._user_daily_counts:
            del self._user_daily_counts[user_id]

        # Clear endpoint buckets
        keys_to_remove = [
            key
            for key in self._endpoint_buckets.keys()
            if key.startswith(f"{user_id}:")
        ]
        for key in keys_to_remove:
            del self._endpoint_buckets[key]

        # Clear Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                pattern = f"user:{user_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Redis error: {e}")


# Global rate limiter instance
rate_limiter = RateLimiter(use_redis=False)  # Set to True to use Redis

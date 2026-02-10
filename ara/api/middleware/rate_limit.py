"""
Rate limiting middleware
"""

from datetime import datetime

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ara.api.auth.rate_limiter import rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply rate limiting to all requests
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Skip rate limiting for certain paths
        skip_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/",
            "/api/v1/auth/login",  # Don't rate limit login
        ]

        if request.url.path in skip_paths:
            return await call_next(request)

        # Get current user (if authenticated)
        try:
            # Extract auth from request
            bearer_token = None
            api_key = None

            # Check for bearer token
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                from fastapi.security import HTTPAuthorizationCredentials

                bearer_token = HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=auth_header.replace("Bearer ", "")
                )

            # Check for API key
            api_key = request.headers.get("X-API-Key")

            # Get user
            from ara.api.auth.dependencies import get_current_user

            user = await get_current_user(bearer_token, api_key)

            if user:
                # Check rate limit
                endpoint = request.url.path
                allowed, retry_after = rate_limiter.check_rate_limit(user, endpoint)

                if not allowed:
                    # Rate limit exceeded
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": "RateLimitExceeded",
                            "message": "Rate limit exceeded. Please try again later.",
                            "details": {
                                "retry_after": retry_after,
                                "tier": user.tier.value,
                            },
                            "timestamp": datetime.now().isoformat(),
                        },
                        headers={
                            "Retry-After": str(retry_after) if retry_after else "60",
                            **rate_limiter.get_rate_limit_headers(user, endpoint),
                        },
                    )

                # Process request
                response = await call_next(request)

                # Add rate limit headers to response
                headers = rate_limiter.get_rate_limit_headers(user, endpoint)
                for key, value in headers.items():
                    response.headers[key] = value

                return response

        except Exception:
            # If authentication fails, continue without rate limiting
            # (auth will be handled by endpoint dependencies)
            pass

        # Process request without rate limiting
        return await call_next(request)

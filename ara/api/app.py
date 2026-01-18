"""
FastAPI application factory
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from ara.api.models import HealthResponse
from ara.api.routes import (
    predictions,
    backtesting,
    portfolio,
    models,
    market,
    auth,
    websocket,
    webhooks,
    health,
)
from ara.api.middleware import RateLimitMiddleware
from ara.api.openapi_config import get_openapi_config, TAGS_METADATA
from ara.core.exceptions import AraAIException
from ara import __version__


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
        Configured FastAPI application
    """
    # Get OpenAPI configuration
    openapi_config = get_openapi_config()

    app = FastAPI(
        title=openapi_config["title"],
        description=openapi_config["description"],
        version=openapi_config["version"],
        contact=openapi_config["contact"],
        license_info=openapi_config["license"],
        servers=openapi_config["servers"],
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Exception handlers
    @app.exception_handler(AraAIException)
    async def ara_exception_handler(request: Request, exc: AraAIException):
        """Handle ARA AI exceptions"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "message": str(exc),
                "details": exc.details,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)},
                "timestamp": datetime.now().isoformat(),
            },
        )

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Health check",
        description="Check API health and service status",
    )
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            version=__version__,
            timestamp=datetime.now(),
            services={
                "api": "operational",
                "prediction_engine": "operational",
                "data_providers": "operational",
            },
        )

    # Root endpoint
    @app.get(
        "/",
        tags=["root"],
        summary="API information",
        description="Get API information and available endpoints",
    )
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "ARA AI Prediction API",
            "version": __version__,
            "description": "World-class financial prediction system",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "predictions": "/api/v1/predict",
                "batch_predictions": "/api/v1/predict/batch",
                "prediction_status": "/api/v1/predictions/{id}",
            },
        }

    # Favicon endpoint (prevents 404 errors)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Return empty response for favicon requests"""
        return JSONResponse(content={}, status_code=204)

    # Include routers
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(predictions.router)
    app.include_router(backtesting.router)
    app.include_router(portfolio.router)
    app.include_router(models.router)
    app.include_router(market.router)
    app.include_router(websocket.router)
    app.include_router(webhooks.router)

    # Initialize monitoring (optional, based on environment variables)
    # init_error_tracking(dsn=os.getenv("SENTRY_DSN"))
    # init_tracing(otlp_endpoint=os.getenv("OTLP_ENDPOINT"))

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Prediction API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict
from datetime import datetime

from ara.api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionStatus,
    ErrorResponse
)
from ara.api.dependencies import (
    get_request_id,
    verify_api_key,
    get_cache_key,
    get_cached_prediction,
    cache_prediction
)
from ara.api.prediction_engine import PredictionEngine
from ara.core.interfaces import AssetType
from ara.core.exceptions import AraAIException


router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global prediction engine instance
_prediction_engine = None


def get_prediction_engine() -> PredictionEngine:
    """Get or create prediction engine instance"""
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine()
    return _prediction_engine


# In-memory storage for async predictions (TODO: Replace with Redis/DB)
_prediction_jobs: Dict[str, PredictionStatus] = {}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate prediction for a single asset",
    description="Generate price predictions for a stock, cryptocurrency, or forex pair"
)
async def predict(
    request: PredictionRequest,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key)
):
    """
    Generate prediction for a single asset
    
    - **symbol**: Asset symbol (e.g., AAPL, BTC-USD, EURUSD)
    - **days**: Number of days to predict (1-30)
    - **asset_type**: Asset type (auto-detected if not provided)
    - **analysis_level**: Level of analysis detail
    - **include_explanations**: Include prediction explanations
    """
    try:
        # Check cache first
        cache_key = get_cache_key(
            request.symbol,
            request.days,
            request.analysis_level.value
        )
        
        cached = get_cached_prediction(cache_key)
        if cached:
            cached["request_id"] = request_id
            return cached
        
        # Get prediction engine
        engine = get_prediction_engine()
        
        # Convert asset type
        asset_type = None
        if request.asset_type:
            asset_type = AssetType(request.asset_type.value)
        
        # Generate prediction
        result = await engine.predict(
            symbol=request.symbol,
            days=request.days,
            asset_type=asset_type,
            include_explanations=request.include_explanations
        )
        
        # Add request ID
        result["request_id"] = request_id
        
        # Cache result
        cache_prediction(cache_key, result, ttl=60)
        
        return result
        
    except AraAIException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.__class__.__name__,
                "message": str(e),
                "details": e.details,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"An unexpected error occurred: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate predictions for multiple assets",
    description="Generate price predictions for multiple stocks, cryptocurrencies, or forex pairs"
)
async def batch_predict(
    request: BatchPredictionRequest,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key)
):
    """
    Generate predictions for multiple assets (max 100)
    
    - **symbols**: List of asset symbols
    - **days**: Number of days to predict
    - **asset_type**: Asset type for all symbols
    - **analysis_level**: Level of analysis detail
    """
    try:
        # Get prediction engine
        engine = get_prediction_engine()
        
        # Convert asset type
        asset_type = None
        if request.asset_type:
            asset_type = AssetType(request.asset_type.value)
        
        # Generate batch predictions
        result = await engine.batch_predict(
            symbols=request.symbols,
            days=request.days,
            asset_type=asset_type
        )
        
        # Add request ID
        result["request_id"] = request_id
        
        return result
        
    except AraAIException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.__class__.__name__,
                "message": str(e),
                "details": e.details,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"An unexpected error occurred: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionStatus,
    status_code=status.HTTP_200_OK,
    summary="Get prediction status",
    description="Retrieve the status and result of a prediction request"
)
async def get_prediction_status(
    prediction_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get prediction status by ID
    
    - **prediction_id**: Unique prediction request ID
    """
    if prediction_id not in _prediction_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NotFound",
                "message": f"Prediction with ID {prediction_id} not found",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return _prediction_jobs[prediction_id]

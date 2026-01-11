"""
Model management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field

from ara.api.dependencies import get_request_id, verify_api_key
from ara.models.model_registry import ModelRegistry
from ara.models.retraining_scheduler import ModelRetrainingScheduler
from ara.core.exceptions import AraAIException


router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelInfo(BaseModel):
    """Model information"""

    model_id: str
    symbol: str
    version: str
    model_type: str
    accuracy: float
    training_date: datetime
    data_period: str
    status: str  # active, archived, training


class ModelStatusResponse(BaseModel):
    """Response for model status"""

    models: List[ModelInfo]
    total_count: int
    active_count: int
    timestamp: datetime
    request_id: str


class TrainingRequest(BaseModel):
    """Request for model training"""

    symbol: str = Field(..., description="Asset symbol")
    data_period: str = Field("2y", description="Training data period")
    model_types: Optional[List[str]] = Field(None, description="Model types to train")
    force_retrain: bool = Field(
        False, description="Force retraining even if model exists"
    )


class TrainingJobResponse(BaseModel):
    """Response for training job"""

    job_id: str
    symbol: str
    status: str
    message: str
    timestamp: datetime


class ComparisonRequest(BaseModel):
    """Request for model comparison"""

    symbol: str = Field(..., description="Asset symbol")
    model_versions: Optional[List[str]] = Field(
        None, description="Model versions to compare"
    )
    metric: str = Field("accuracy", description="Comparison metric")


class ModelComparison(BaseModel):
    """Model comparison result"""

    model_id: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mae: float
    rmse: float
    training_date: datetime


class ComparisonResponse(BaseModel):
    """Response for model comparison"""

    symbol: str
    comparisons: List[ModelComparison]
    best_model: str
    metric: str
    timestamp: datetime
    request_id: str


class DeployRequest(BaseModel):
    """Request for model deployment"""

    model_id: str = Field(..., description="Model ID to deploy")
    environment: str = Field("production", description="Deployment environment")


class DeployResponse(BaseModel):
    """Response for model deployment"""

    model_id: str
    environment: str
    status: str
    message: str
    timestamp: datetime
    request_id: str


# In-memory storage (TODO: Replace with database)
_training_jobs: Dict[str, Dict] = {}


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model status",
    description="Get status of all models in the registry",
)
async def get_model_status(
    symbol: Optional[str] = None,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Get model status

    - **symbol**: Optional symbol filter
    """
    try:
        # Get model registry
        registry = ModelRegistry()

        # Get all models
        all_models = registry.list_models(symbol=symbol)

        # Convert to response format
        models = []
        for model_data in all_models:
            models.append(
                ModelInfo(
                    model_id=model_data["model_id"],
                    symbol=model_data["symbol"],
                    version=model_data["version"],
                    model_type=model_data["model_type"],
                    accuracy=model_data.get("accuracy", 0.0),
                    training_date=model_data["training_date"],
                    data_period=model_data.get("data_period", "unknown"),
                    status=model_data.get("status", "active"),
                )
            )

        active_count = sum(1 for m in models if m.status == "active")

        return ModelStatusResponse(
            models=models,
            total_count=len(models),
            active_count=active_count,
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to get model status: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.post(
    "/train",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Train model",
    description="Start model training job (async operation)",
)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Train model for a symbol

    This is an async operation. Use the job_id to check status.

    - **symbol**: Asset symbol
    - **data_period**: Training data period
    - **model_types**: Model types to train (default: all)
    - **force_retrain**: Force retraining even if model exists
    """
    try:
        # Create job entry
        job_id = request_id
        _training_jobs[job_id] = {
            "symbol": request.symbol,
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        # Add background task
        background_tasks.add_task(_train_model_task, job_id, request)

        return TrainingJobResponse(
            job_id=job_id,
            symbol=request.symbol,
            status="pending",
            message="Training job created and will be processed",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to create training job: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


async def _train_model_task(job_id: str, request: TrainingRequest):
    """Background task to train model"""
    try:
        # Update status
        _training_jobs[job_id]["status"] = "training"
        _training_jobs[job_id]["updated_at"] = datetime.now()

        # Create scheduler
        scheduler = ModelRetrainingScheduler()

        # Train model
        result = scheduler.train_model(
            symbol=request.symbol,
            data_period=request.data_period,
            force_retrain=request.force_retrain,
        )

        # Update job with result
        _training_jobs[job_id]["status"] = "completed"
        _training_jobs[job_id]["progress"] = 1.0
        _training_jobs[job_id]["result"] = result
        _training_jobs[job_id]["updated_at"] = datetime.now()

    except Exception as e:
        # Update job with error
        _training_jobs[job_id]["status"] = "failed"
        _training_jobs[job_id]["error"] = str(e)
        _training_jobs[job_id]["updated_at"] = datetime.now()


@router.get(
    "/train/{job_id}",
    summary="Get training job status",
    description="Get the status and result of a training job",
)
async def get_training_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get training job status

    - **job_id**: Training job ID
    """
    if job_id not in _training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NotFound",
                "message": f"Training job {job_id} not found",
                "timestamp": datetime.now().isoformat(),
            },
        )

    return _training_jobs[job_id]


@router.get(
    "/compare",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare models",
    description="Compare performance of different model versions",
)
async def compare_models(
    symbol: str,
    metric: str = "accuracy",
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Compare model performance

    - **symbol**: Asset symbol
    - **metric**: Comparison metric (accuracy, precision, recall, f1_score, mae, rmse)
    """
    try:
        # Get model registry
        registry = ModelRegistry()

        # Get all models for symbol
        models = registry.list_models(symbol=symbol)

        if not models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "NotFound",
                    "message": f"No models found for symbol {symbol}",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                },
            )

        # Build comparisons
        comparisons = []
        for model_data in models:
            comparisons.append(
                ModelComparison(
                    model_id=model_data["model_id"],
                    version=model_data["version"],
                    accuracy=model_data.get("accuracy", 0.0),
                    precision=model_data.get("precision", 0.0),
                    recall=model_data.get("recall", 0.0),
                    f1_score=model_data.get("f1_score", 0.0),
                    mae=model_data.get("mae", 0.0),
                    rmse=model_data.get("rmse", 0.0),
                    training_date=model_data["training_date"],
                )
            )

        # Find best model based on metric
        best_model = max(comparisons, key=lambda x: getattr(x, metric))

        return ComparisonResponse(
            symbol=symbol,
            comparisons=comparisons,
            best_model=best_model.model_id,
            metric=metric,
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Model comparison failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.post(
    "/deploy",
    response_model=DeployResponse,
    status_code=status.HTTP_200_OK,
    summary="Deploy model",
    description="Deploy a model to production or staging environment",
)
async def deploy_model(
    request: DeployRequest,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Deploy model to environment

    - **model_id**: Model ID to deploy
    - **environment**: Target environment (production, staging)
    """
    try:
        # Get model registry
        registry = ModelRegistry()

        # Deploy model
        result = registry.deploy_model(
            model_id=request.model_id, environment=request.environment
        )

        return DeployResponse(
            model_id=request.model_id,
            environment=request.environment,
            status="deployed",
            message=f"Model {request.model_id} deployed to {request.environment}",
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except AraAIException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.__class__.__name__,
                "message": str(e),
                "details": e.details,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Model deployment failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete model",
    description="Delete a model from the registry",
)
async def delete_model(
    model_id: str,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Delete model

    - **model_id**: Model ID to delete
    """
    try:
        # Get model registry
        registry = ModelRegistry()

        # Delete model
        registry.delete_model(model_id)

        return {
            "model_id": model_id,
            "status": "deleted",
            "message": f"Model {model_id} deleted successfully",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }

    except AraAIException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.__class__.__name__,
                "message": str(e),
                "details": e.details,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Model deletion failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )

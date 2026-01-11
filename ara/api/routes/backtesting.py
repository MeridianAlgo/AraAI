"""
Backtesting API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict
from datetime import datetime, date
from pydantic import BaseModel, Field

from ara.api.dependencies import get_request_id, verify_api_key
from ara.backtesting.engine import BacktestEngine


router = APIRouter(prefix="/api/v1", tags=["backtesting"])


class BacktestRequest(BaseModel):
    """Request model for backtesting"""

    symbol: str = Field(..., description="Asset symbol")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(10000.0, gt=0, description="Initial capital")
    strategy: str = Field("buy_and_hold", description="Trading strategy")
    walk_forward_window: int = Field(
        252, ge=30, description="Walk-forward window in days"
    )
    retraining_frequency: int = Field(
        30, ge=1, description="Model retraining frequency in days"
    )


class BacktestResponse(BaseModel):
    """Response model for backtesting"""

    symbol: str
    start_date: date
    end_date: date
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    directional_accuracy: float
    mae: float
    rmse: float
    mape: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    final_capital: float
    total_return: float
    timestamp: datetime
    request_id: str


class BacktestJobResponse(BaseModel):
    """Response for async backtest job"""

    job_id: str
    status: str
    message: str
    timestamp: datetime


# In-memory job storage (TODO: Replace with Redis/DB)
_backtest_jobs: Dict[str, Dict] = {}


@router.post(
    "/backtest",
    response_model=BacktestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run backtest",
    description="Run a backtest on historical data (async operation)",
)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Run backtest on historical data

    This is an async operation. Use the job_id to check status.

    - **symbol**: Asset symbol
    - **start_date**: Backtest start date
    - **end_date**: Backtest end date
    - **initial_capital**: Initial capital for simulation
    - **strategy**: Trading strategy to use
    """
    try:
        # Create job entry
        job_id = request_id
        _backtest_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        # Add background task
        background_tasks.add_task(_run_backtest_task, job_id, request)

        return BacktestJobResponse(
            job_id=job_id,
            status="pending",
            message="Backtest job created and will be processed",
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to create backtest job: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


async def _run_backtest_task(job_id: str, request: BacktestRequest):
    """Background task to run backtest"""
    try:
        # Update status
        _backtest_jobs[job_id]["status"] = "processing"
        _backtest_jobs[job_id]["updated_at"] = datetime.now()

        # Create backtest engine
        engine = BacktestEngine()

        # Run backtest
        result = engine.run_backtest(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            walk_forward_window=request.walk_forward_window,
            retraining_frequency=request.retraining_frequency,
        )

        # Update job with result
        _backtest_jobs[job_id]["status"] = "completed"
        _backtest_jobs[job_id]["progress"] = 1.0
        _backtest_jobs[job_id]["result"] = result
        _backtest_jobs[job_id]["updated_at"] = datetime.now()

    except Exception as e:
        # Update job with error
        _backtest_jobs[job_id]["status"] = "failed"
        _backtest_jobs[job_id]["error"] = str(e)
        _backtest_jobs[job_id]["updated_at"] = datetime.now()


@router.get(
    "/backtest/{job_id}",
    summary="Get backtest status",
    description="Get the status and result of a backtest job",
)
async def get_backtest_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get backtest job status

    - **job_id**: Backtest job ID
    """
    if job_id not in _backtest_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NotFound",
                "message": f"Backtest job {job_id} not found",
                "timestamp": datetime.now().isoformat(),
            },
        )

    job = _backtest_jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }

    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]

    return response

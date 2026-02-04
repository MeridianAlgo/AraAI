"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AssetTypeEnum(str, Enum):
    """Asset type enumeration"""

    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"


class AnalysisLevel(str, Enum):
    """Analysis detail level"""

    BASIC = "basic"
    STANDARD = "standard"
    FULL = "full"


class PredictionRequest(BaseModel):
    """Request model for single prediction"""

    symbol: str = Field(..., description="Asset symbol (e.g., AAPL, BTC-USD, EURUSD)")
    days: int = Field(5, ge=1, le=30, description="Number of days to predict (1-30)")
    asset_type: Optional[AssetTypeEnum] = Field(
        None, description="Asset type (auto-detected if not provided)"
    )
    analysis_level: AnalysisLevel = Field(
        AnalysisLevel.STANDARD, description="Level of analysis detail"
    )
    include_explanations: bool = Field(True, description="Include prediction explanations")

    @validator("symbol")
    def validate_symbol(cls, v):
        if not v or len(v) < 1:
            raise ValueError("Symbol cannot be empty")
        return v.upper().strip()


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    symbols: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of symbols (max 100)"
    )
    days: int = Field(5, ge=1, le=30, description="Number of days to predict")
    asset_type: Optional[AssetTypeEnum] = Field(None, description="Asset type for all symbols")
    analysis_level: AnalysisLevel = Field(
        AnalysisLevel.BASIC, description="Level of analysis detail"
    )

    @validator("symbols")
    def validate_symbols(cls, v):
        return [s.upper().strip() for s in v if s]


class DailyPrediction(BaseModel):
    """Single day prediction"""

    day: int
    date: datetime
    predicted_price: float
    predicted_return: float
    confidence: float
    lower_bound: float
    upper_bound: float


class ConfidenceScore(BaseModel):
    """Confidence score breakdown"""

    overall: float = Field(..., ge=0, le=1)
    model_agreement: float = Field(..., ge=0, le=1)
    data_quality: float = Field(..., ge=0, le=1)
    regime_stability: float = Field(..., ge=0, le=1)
    historical_accuracy: float = Field(..., ge=0, le=1)


class Factor(BaseModel):
    """Contributing factor"""

    name: str
    value: float
    contribution: float = Field(..., ge=-1, le=1)
    description: str


class Explanations(BaseModel):
    """Prediction explanations"""

    top_factors: List[Factor]
    feature_importance: Dict[str, float]
    natural_language: str


class MarketRegime(BaseModel):
    """Market regime information"""

    current_regime: str
    confidence: float
    transition_probabilities: Dict[str, float]
    duration_in_regime: int
    expected_duration: int


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    symbol: str
    asset_type: str
    current_price: float
    predictions: List[DailyPrediction]
    confidence: ConfidenceScore
    explanations: Optional[Explanations]
    regime: Optional[MarketRegime]
    timestamp: datetime
    model_version: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[PredictionResponse]
    total_count: int
    successful_count: int
    failed_count: int
    failed_symbols: List[str]
    timestamp: datetime
    request_id: str


class PredictionStatus(BaseModel):
    """Status of a prediction request"""

    request_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = Field(None, ge=0, le=1)
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]

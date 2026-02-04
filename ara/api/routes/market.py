"""
Market analysis API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel

from ara.api.dependencies import get_request_id, verify_api_key
from ara.models.regime_detector import RegimeDetector
from ara.sentiment.aggregator import SentimentAggregator
from ara.correlation.analyzer import CorrelationAnalyzer
from ara.features.calculator import IndicatorCalculator
from ara.data.base_provider import BaseDataProvider
from ara.core.exceptions import AraAIException


router = APIRouter(prefix="/api/v1/market", tags=["market"])


class RegimeResponse(BaseModel):
    """Response for market regime"""

    symbol: str
    current_regime: str
    confidence: float
    transition_probabilities: Dict[str, float]
    duration_in_regime: int
    expected_duration: int
    regime_features: Dict[str, float]
    timestamp: datetime
    request_id: str


class SentimentResponse(BaseModel):
    """Response for sentiment analysis"""

    symbol: str
    overall_sentiment: float
    sentiment_breakdown: Dict[str, float]
    sources: List[str]
    sentiment_momentum: float
    sentiment_divergence: Optional[float]
    timestamp: datetime
    request_id: str


class CorrelationPair(BaseModel):
    """Correlation between two assets"""

    asset1: str
    asset2: str
    correlation: float
    rolling_correlation: List[float]
    correlation_breakdown: bool


class CorrelationResponse(BaseModel):
    """Response for correlation analysis"""

    assets: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    correlation_pairs: List[CorrelationPair]
    high_correlations: List[CorrelationPair]
    low_correlations: List[CorrelationPair]
    timestamp: datetime
    request_id: str


class Indicator(BaseModel):
    """Technical indicator value"""

    name: str
    value: float
    signal: str  # bullish, bearish, neutral
    description: str


class IndicatorResponse(BaseModel):
    """Response for technical indicators"""

    symbol: str
    indicators: List[Indicator]
    trend_indicators: List[Indicator]
    momentum_indicators: List[Indicator]
    volatility_indicators: List[Indicator]
    volume_indicators: List[Indicator]
    timestamp: datetime
    request_id: str


@router.get(
    "/regime",
    response_model=RegimeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get market regime",
    description="Detect current market regime for an asset",
)
async def get_market_regime(
    symbol: str,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Get market regime for a symbol

    - **symbol**: Asset symbol
    """
    try:
        # Fetch data
        provider = BaseDataProvider()
        data = await provider.fetch_historical(symbol, period="1y", interval="1d")

        if data is None or len(data) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "InsufficientData",
                    "message": f"Insufficient data for {symbol}",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                },
            )

        # Detect regime
        detector = RegimeDetector()
        regime_info = detector.detect_regime(data)

        return RegimeResponse(
            symbol=symbol,
            current_regime=regime_info.get("regime", "unknown"),
            confidence=regime_info.get("confidence", 0.5),
            transition_probabilities=regime_info.get("transition_probabilities", {}),
            duration_in_regime=regime_info.get("duration", 0),
            expected_duration=regime_info.get("expected_duration", 0),
            regime_features=regime_info.get("features", {}),
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except HTTPException:
        raise
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
                "message": f"Regime detection failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.get(
    "/sentiment",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get market sentiment",
    description="Analyze market sentiment from multiple sources",
)
async def get_market_sentiment(
    symbol: str,
    sources: Optional[str] = None,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Get market sentiment for a symbol

    - **symbol**: Asset symbol
    - **sources**: Comma-separated list of sources (twitter, reddit, news)
    """
    try:
        # Parse sources
        source_list = sources.split(",") if sources else ["twitter", "reddit", "news"]
        source_list = [s.strip().lower() for s in source_list]

        # Analyze sentiment
        aggregator = SentimentAggregator()
        sentiment_data = aggregator.aggregate_sentiment(symbol=symbol, sources=source_list)

        return SentimentResponse(
            symbol=symbol,
            overall_sentiment=sentiment_data.get("overall_sentiment", 0.0),
            sentiment_breakdown=sentiment_data.get("breakdown", {}),
            sources=source_list,
            sentiment_momentum=sentiment_data.get("momentum", 0.0),
            sentiment_divergence=sentiment_data.get("divergence"),
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
                "message": f"Sentiment analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.get(
    "/correlations",
    response_model=CorrelationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get asset correlations",
    description="Calculate correlations between multiple assets",
)
async def get_correlations(
    assets: str,
    period: str = "1y",
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Get correlations between assets

    - **assets**: Comma-separated list of asset symbols (2-20)
    - **period**: Analysis period (1mo, 3mo, 6mo, 1y, 2y)
    """
    try:
        # Parse assets
        asset_list = [a.strip().upper() for a in assets.split(",")]

        if len(asset_list) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": "At least 2 assets required for correlation analysis",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                },
            )

        if len(asset_list) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": "Maximum 20 assets allowed",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                },
            )

        # Analyze correlations
        analyzer = CorrelationAnalyzer()
        correlation_data = analyzer.calculate_correlations(assets=asset_list, period=period)

        # Build correlation pairs
        correlation_pairs = []
        high_correlations = []
        low_correlations = []

        for i, asset1 in enumerate(asset_list):
            for asset2 in asset_list[i + 1 :]:
                corr = correlation_data["matrix"].get(asset1, {}).get(asset2, 0.0)
                pair = CorrelationPair(
                    asset1=asset1,
                    asset2=asset2,
                    correlation=corr,
                    rolling_correlation=[],
                    correlation_breakdown=abs(corr) > 0.3,
                )
                correlation_pairs.append(pair)

                if corr > 0.8:
                    high_correlations.append(pair)
                elif corr < -0.8:
                    low_correlations.append(pair)

        return CorrelationResponse(
            assets=asset_list,
            correlation_matrix=correlation_data["matrix"],
            correlation_pairs=correlation_pairs,
            high_correlations=high_correlations,
            low_correlations=low_correlations,
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except HTTPException:
        raise
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
                "message": f"Correlation analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.get(
    "/indicators",
    response_model=IndicatorResponse,
    status_code=status.HTTP_200_OK,
    summary="Get technical indicators",
    description="Calculate technical indicators for an asset",
)
async def get_indicators(
    symbol: str,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Get technical indicators for a symbol

    - **symbol**: Asset symbol
    """
    try:
        # Fetch data
        provider = BaseDataProvider()
        data = await provider.fetch_historical(symbol, period="1y", interval="1d")

        if data is None or len(data) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "InsufficientData",
                    "message": f"Insufficient data for {symbol}",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                },
            )

        # Calculate indicators
        calculator = IndicatorCalculator()
        features = calculator.calculate(data, ["rsi", "macd", "bb"])

        # Get latest values
        latest = features.iloc[-1]

        # Helper function to determine signal
        def get_signal(value: float, threshold_high: float = 70, threshold_low: float = 30) -> str:
            if value > threshold_high:
                return "overbought"
            elif value < threshold_low:
                return "oversold"
            return "neutral"

        # Build indicator lists
        all_indicators = []
        trend_indicators = []
        momentum_indicators = []
        volatility_indicators = []
        volume_indicators = []

        # Add some key indicators (simplified for demo)
        if "SMA_20" in latest:
            ind = Indicator(
                name="SMA_20",
                value=float(latest["SMA_20"]),
                signal=("bullish" if data["Close"].iloc[-1] > latest["SMA_20"] else "bearish"),
                description="20-day Simple Moving Average",
            )
            all_indicators.append(ind)
            trend_indicators.append(ind)

        if "RSI_14" in latest:
            rsi_val = float(latest["RSI_14"])
            ind = Indicator(
                name="RSI_14",
                value=rsi_val,
                signal=get_signal(rsi_val),
                description="14-day Relative Strength Index",
            )
            all_indicators.append(ind)
            momentum_indicators.append(ind)

        if "BB_upper" in latest and "BB_lower" in latest:
            ind = Indicator(
                name="Bollinger_Bands",
                value=float(latest["BB_upper"]),
                signal="neutral",
                description="Bollinger Bands (20, 2)",
            )
            all_indicators.append(ind)
            volatility_indicators.append(ind)

        return IndicatorResponse(
            symbol=symbol,
            indicators=all_indicators,
            trend_indicators=trend_indicators,
            momentum_indicators=momentum_indicators,
            volatility_indicators=volatility_indicators,
            volume_indicators=volume_indicators,
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except HTTPException:
        raise
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
                "message": f"Indicator calculation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )

"""
Portfolio management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field

from ara.api.dependencies import get_request_id, verify_api_key
from ara.risk.optimizer import PortfolioOptimizer
from ara.risk.portfolio_analysis import PortfolioAnalyzer
from ara.risk.constraints import PortfolioConstraints
from ara.core.exceptions import AraAIException


router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])


class OptimizationRequest(BaseModel):
    """Request model for portfolio optimization"""

    assets: List[str] = Field(..., min_items=2, max_items=50, description="List of assets")
    risk_tolerance: str = Field(
        "moderate", description="Risk tolerance: conservative, moderate, aggressive"
    )
    optimization_method: str = Field(
        "mpt",
        description="Optimization method: mpt, black_litterman, risk_parity, kelly",
    )
    constraints: Optional[Dict] = Field(None, description="Portfolio constraints")
    expected_returns: Optional[Dict[str, float]] = Field(
        None, description="Expected returns for each asset"
    )


class OptimizationResponse(BaseModel):
    """Response model for portfolio optimization"""

    assets: List[str]
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    optimization_method: str
    timestamp: datetime
    request_id: str


class AnalysisRequest(BaseModel):
    """Request model for portfolio analysis"""

    assets: List[str] = Field(..., min_items=2, description="List of assets")
    weights: Dict[str, float] = Field(..., description="Asset weights (must sum to 1.0)")
    period: str = Field("1y", description="Analysis period")


class AnalysisResponse(BaseModel):
    """Response model for portfolio analysis"""

    assets: List[str]
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    correlation_matrix: Dict
    risk_contribution: Dict[str, float]
    timestamp: datetime
    request_id: str


class RebalanceRequest(BaseModel):
    """Request model for portfolio rebalancing"""

    current_weights: Dict[str, float] = Field(..., description="Current portfolio weights")
    target_weights: Dict[str, float] = Field(..., description="Target portfolio weights")
    current_prices: Dict[str, float] = Field(..., description="Current asset prices")
    portfolio_value: float = Field(..., gt=0, description="Total portfolio value")


class Trade(BaseModel):
    """Trade recommendation"""

    symbol: str
    action: str  # BUY or SELL
    quantity: float
    current_price: float
    target_weight: float
    current_weight: float
    value: float


class RebalanceResponse(BaseModel):
    """Response model for rebalancing"""

    trades: List[Trade]
    total_transaction_cost: float
    estimated_slippage: float
    timestamp: datetime
    request_id: str


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize portfolio allocation",
    description="Calculate optimal portfolio weights based on risk tolerance and constraints",
)
async def optimize_portfolio(
    request: OptimizationRequest,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Optimize portfolio allocation

    - **assets**: List of asset symbols (2-50)
    - **risk_tolerance**: Risk tolerance level
    - **optimization_method**: Optimization algorithm to use
    - **constraints**: Optional portfolio constraints
    - **expected_returns**: Optional expected returns (uses predictions if not provided)
    """
    try:
        # Create optimizer
        optimizer = PortfolioOptimizer()

        # Set constraints if provided
        if request.constraints:
            constraints = PortfolioConstraints(**request.constraints)
        else:
            constraints = PortfolioConstraints()

        # Run optimization
        result = optimizer.optimize(
            assets=request.assets,
            method=request.optimization_method,
            risk_tolerance=request.risk_tolerance,
            constraints=constraints,
            expected_returns=request.expected_returns,
        )

        # Build response
        response = OptimizationResponse(
            assets=request.assets,
            optimal_weights=result["weights"],
            expected_return=result["expected_return"],
            expected_volatility=result["volatility"],
            sharpe_ratio=result["sharpe_ratio"],
            var_95=result.get("var_95", 0.0),
            cvar_95=result.get("cvar_95", 0.0),
            optimization_method=request.optimization_method,
            timestamp=datetime.now(),
            request_id=request_id,
        )

        return response

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
                "message": f"Portfolio optimization failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.get(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze portfolio",
    description="Analyze portfolio risk and performance metrics",
)
async def analyze_portfolio(
    request: AnalysisRequest,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Analyze portfolio metrics

    - **assets**: List of asset symbols
    - **weights**: Asset weights (must sum to 1.0)
    - **period**: Analysis period
    """
    try:
        # Validate weights sum to 1.0
        weight_sum = sum(request.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": f"Weights must sum to 1.0, got {weight_sum}",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                },
            )

        # Create analyzer
        analyzer = PortfolioAnalyzer()

        # Run analysis
        result = analyzer.analyze(
            assets=request.assets, weights=request.weights, period=request.period
        )

        # Build response
        response = AnalysisResponse(
            assets=request.assets,
            weights=request.weights,
            expected_return=result["expected_return"],
            volatility=result["volatility"],
            sharpe_ratio=result["sharpe_ratio"],
            sortino_ratio=result["sortino_ratio"],
            max_drawdown=result["max_drawdown"],
            var_95=result["var_95"],
            cvar_95=result["cvar_95"],
            correlation_matrix=result["correlation_matrix"],
            risk_contribution=result["risk_contribution"],
            timestamp=datetime.now(),
            request_id=request_id,
        )

        return response

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
                "message": f"Portfolio analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )


@router.post(
    "/rebalance",
    response_model=RebalanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate rebalancing trades",
    description="Calculate trades needed to rebalance portfolio to target weights",
)
async def rebalance_portfolio(
    request: RebalanceRequest,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
):
    """
    Calculate rebalancing trades

    - **current_weights**: Current portfolio weights
    - **target_weights**: Target portfolio weights
    - **current_prices**: Current asset prices
    - **portfolio_value**: Total portfolio value
    """
    try:
        trades = []
        total_cost = 0.0

        # Calculate trades for each asset
        for symbol in set(
            list(request.current_weights.keys()) + list(request.target_weights.keys())
        ):
            current_weight = request.current_weights.get(symbol, 0.0)
            target_weight = request.target_weights.get(symbol, 0.0)

            if abs(current_weight - target_weight) < 0.001:
                continue  # Skip if difference is negligible

            current_value = request.portfolio_value * current_weight
            target_value = request.portfolio_value * target_weight
            trade_value = target_value - current_value

            if symbol not in request.current_prices:
                continue  # Skip if price not available

            price = request.current_prices[symbol]
            quantity = trade_value / price

            # Transaction cost (0.1% of trade value)
            cost = abs(trade_value) * 0.001
            total_cost += cost

            trades.append(
                Trade(
                    symbol=symbol,
                    action="BUY" if quantity > 0 else "SELL",
                    quantity=abs(quantity),
                    current_price=price,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    value=abs(trade_value),
                )
            )

        # Estimated slippage (0.05% of total trade value)
        total_trade_value = sum(t.value for t in trades)
        estimated_slippage = total_trade_value * 0.0005

        return RebalanceResponse(
            trades=trades,
            total_transaction_cost=round(total_cost, 2),
            estimated_slippage=round(estimated_slippage, 2),
            timestamp=datetime.now(),
            request_id=request_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": f"Rebalancing calculation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
            },
        )

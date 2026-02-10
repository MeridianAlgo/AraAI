"""
Portfolio Analysis and Reporting

Implements portfolio analysis, visualization, and reporting capabilities including:
- Efficient frontier visualization
- Portfolio composition charts
- Risk contribution analysis
- Scenario analysis and stress testing
- Portfolio comparison reports
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PortfolioAnalyzer:
    """
    Analyze and visualize portfolio characteristics.

    This class provides methods for:
    - Efficient frontier visualization
    - Portfolio composition analysis
    - Risk contribution breakdown
    - Scenario analysis
    - Stress testing
    """

    def __init__(self):
        """Initialize PortfolioAnalyzer."""
        pass

    def plot_efficient_frontier(
        self,
        frontier_data: pd.DataFrame,
        max_sharpe_portfolio: Optional[Dict] = None,
        min_risk_portfolio: Optional[Dict] = None,
        current_portfolio: Optional[Dict] = None,
        title: str = "Efficient Frontier",
    ) -> go.Figure:
        """
        Create interactive efficient frontier visualization.

        Args:
            frontier_data: DataFrame with 'return', 'volatility', 'sharpe_ratio' columns
            max_sharpe_portfolio: Portfolio with maximum Sharpe ratio
            min_risk_portfolio: Portfolio with minimum risk
            current_portfolio: Current portfolio for comparison
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> frontier_df = pd.DataFrame({
            ...     'return': [0.08, 0.10, 0.12],
            ...     'volatility': [0.15, 0.18, 0.22],
            ...     'sharpe_ratio': [0.4, 0.45, 0.43]
            ... })
            >>> fig = analyzer.plot_efficient_frontier(frontier_df)
            >>> fig.show()
        """
        fig = go.Figure()

        # Plot efficient frontier
        fig.add_trace(
            go.Scatter(
                x=frontier_data["volatility"],
                y=frontier_data["return"],
                mode="lines+markers",
                name="Efficient Frontier",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
                hovertemplate=("Return: %{y:.2%}<br>Volatility: %{x:.2%}<br><extra></extra>"),
            )
        )

        # Add max Sharpe portfolio
        if max_sharpe_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[max_sharpe_portfolio["volatility"]],
                    y=[max_sharpe_portfolio["expected_return"]],
                    mode="markers",
                    name="Max Sharpe Ratio",
                    marker=dict(
                        size=15,
                        color="green",
                        symbol="star",
                        line=dict(color="darkgreen", width=2),
                    ),
                    hovertemplate=(
                        "Max Sharpe Portfolio<br>"
                        "Return: %{y:.2%}<br>"
                        "Volatility: %{x:.2%}<br>"
                        f"Sharpe: {max_sharpe_portfolio['sharpe_ratio']:.3f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Add min risk portfolio
        if min_risk_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[min_risk_portfolio["volatility"]],
                    y=[min_risk_portfolio["expected_return"]],
                    mode="markers",
                    name="Min Risk",
                    marker=dict(
                        size=15,
                        color="orange",
                        symbol="diamond",
                        line=dict(color="darkorange", width=2),
                    ),
                    hovertemplate=(
                        "Min Risk Portfolio<br>"
                        "Return: %{y:.2%}<br>"
                        "Volatility: %{x:.2%}<br>"
                        f"Sharpe: {min_risk_portfolio['sharpe_ratio']:.3f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Add current portfolio
        if current_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[current_portfolio["volatility"]],
                    y=[current_portfolio["expected_return"]],
                    mode="markers",
                    name="Current Portfolio",
                    marker=dict(
                        size=15,
                        color="red",
                        symbol="circle",
                        line=dict(color="darkred", width=2),
                    ),
                    hovertemplate=(
                        "Current Portfolio<br>"
                        "Return: %{y:.2%}<br>"
                        "Volatility: %{x:.2%}<br>"
                        f"Sharpe: {current_portfolio.get('sharpe_ratio', 0):.3f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
            hovermode="closest",
            template="plotly_white",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig

    def plot_portfolio_composition(
        self,
        weights: Dict[str, float],
        chart_type: str = "pie",
        title: str = "Portfolio Composition",
    ) -> go.Figure:
        """
        Create portfolio composition visualization.

        Args:
            weights: Dictionary mapping asset names to weights
            chart_type: Type of chart ('pie' or 'bar')
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> weights = {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3}
            >>> fig = analyzer.plot_portfolio_composition(weights, chart_type='pie')
            >>> fig.show()
        """
        # Sort by weight (descending)
        sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

        assets = list(sorted_weights.keys())
        values = list(sorted_weights.values())

        if chart_type == "pie":
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=assets,
                        values=values,
                        textinfo="label+percent",
                        hovertemplate=("%{label}<br>Weight: %{percent}<br><extra></extra>"),
                    )
                ]
            )

        elif chart_type == "bar":
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=assets,
                        y=values,
                        text=[f"{v:.1%}" for v in values],
                        textposition="auto",
                        hovertemplate=("%{x}<br>Weight: %{y:.2%}<br><extra></extra>"),
                    )
                ]
            )

            fig.update_layout(
                xaxis_title="Asset", yaxis_title="Weight", yaxis=dict(tickformat=".0%")
            )

        else:
            raise ValueError(f"Unknown chart type: {chart_type}. Use 'pie' or 'bar'")

        fig.update_layout(title=title, template="plotly_white")

        return fig

    def plot_risk_contribution(
        self,
        risk_decomposition: Dict[str, Dict[str, float]],
        title: str = "Risk Contribution Analysis",
    ) -> go.Figure:
        """
        Visualize risk contribution by asset.

        Args:
            risk_decomposition: Risk decomposition data from RiskCalculator
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> risk_data = {
            ...     'assets': {
            ...         'AAPL': {'weight': 0.3, 'percent_contribution': 35.0},
            ...         'MSFT': {'weight': 0.4, 'percent_contribution': 45.0},
            ...         'GOOGL': {'weight': 0.3, 'percent_contribution': 20.0}
            ...     }
            ... }
            >>> fig = analyzer.plot_risk_contribution(risk_data)
            >>> fig.show()
        """
        assets = list(risk_decomposition["assets"].keys())
        weights = [risk_decomposition["assets"][asset]["weight"] for asset in assets]
        risk_contrib = [
            risk_decomposition["assets"][asset]["percent_contribution"] for asset in assets
        ]

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Portfolio Weights", "Risk Contributions"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        # Add weight bars
        fig.add_trace(
            go.Bar(
                x=assets,
                y=weights,
                name="Weight",
                text=[f"{w:.1%}" for w in weights],
                textposition="auto",
                marker_color="lightblue",
                hovertemplate="%{x}<br>Weight: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add risk contribution bars
        fig.add_trace(
            go.Bar(
                x=assets,
                y=risk_contrib,
                name="Risk Contribution",
                text=[f"{r:.1f}%" for r in risk_contrib],
                textposition="auto",
                marker_color="salmon",
                hovertemplate="%{x}<br>Risk: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Asset", row=1, col=1)
        fig.update_xaxes(title_text="Asset", row=1, col=2)
        fig.update_yaxes(title_text="Weight", tickformat=".0%", row=1, col=1)
        fig.update_yaxes(title_text="Risk Contribution (%)", row=1, col=2)

        fig.update_layout(title=title, template="plotly_white", showlegend=False, height=400)

        return fig

    def perform_scenario_analysis(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        weights: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Perform scenario analysis on portfolio.

        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            weights: Portfolio weights
            scenarios: Dictionary mapping scenario names to asset return changes
                      e.g., {'Market Crash': {'AAPL': -0.30, 'MSFT': -0.25}}

        Returns:
            DataFrame with scenario analysis results

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> weights = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> scenarios = {
            ...     'Market Crash': {'AAPL': -0.30, 'MSFT': -0.25},
            ...     'Bull Market': {'AAPL': 0.20, 'MSFT': 0.15}
            ... }
            >>> results = analyzer.perform_scenario_analysis(returns, weights, scenarios)
        """
        results = []

        # Calculate baseline portfolio return
        df = pd.DataFrame(returns_dict)
        expected_returns = df.mean().values
        weight_array = np.array([weights.get(asset, 0.0) for asset in df.columns])
        baseline_return = np.dot(weight_array, expected_returns) * 252  # Annualized

        # Add baseline scenario
        results.append(
            {
                "scenario": "Baseline",
                "portfolio_return": baseline_return,
                "return_change": 0.0,
            }
        )

        # Analyze each scenario
        for scenario_name, asset_changes in scenarios.items():
            # Calculate scenario return
            scenario_return = 0.0
            for asset, weight in weights.items():
                asset_change = asset_changes.get(asset, 0.0)
                scenario_return += weight * asset_change

            # Calculate change from baseline
            return_change = scenario_return - baseline_return

            results.append(
                {
                    "scenario": scenario_name,
                    "portfolio_return": scenario_return,
                    "return_change": return_change,
                }
            )

        # Create DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def perform_stress_test(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        weights: Dict[str, float],
        stress_scenarios: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perform stress testing on portfolio using historical scenarios.

        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            weights: Portfolio weights
            stress_scenarios: List of predefined stress scenarios
                            (default: common market stress events)

        Returns:
            DataFrame with stress test results

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> weights = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> results = analyzer.perform_stress_test(returns, weights)
        """
        if stress_scenarios is None:
            # Default stress scenarios
            stress_scenarios = [
                "Market Crash (-30%)",
                "Severe Recession (-20%)",
                "Moderate Correction (-10%)",
                "Flash Crash (-15%)",
                "Volatility Spike (2x vol)",
            ]

        # Define stress scenario parameters
        scenario_params = {
            "Market Crash (-30%)": {"return_shock": -0.30, "vol_multiplier": 2.0},
            "Severe Recession (-20%)": {"return_shock": -0.20, "vol_multiplier": 1.5},
            "Moderate Correction (-10%)": {
                "return_shock": -0.10,
                "vol_multiplier": 1.3,
            },
            "Flash Crash (-15%)": {"return_shock": -0.15, "vol_multiplier": 3.0},
            "Volatility Spike (2x vol)": {"return_shock": 0.0, "vol_multiplier": 2.0},
        }

        results = []

        # Calculate baseline metrics
        df = pd.DataFrame(returns_dict)
        expected_returns = df.mean().values
        cov_matrix = df.cov().values
        weight_array = np.array([weights.get(asset, 0.0) for asset in df.columns])

        baseline_return = np.dot(weight_array, expected_returns) * 252
        baseline_vol = np.sqrt(np.dot(weight_array, np.dot(cov_matrix, weight_array))) * np.sqrt(
            252
        )

        # Add baseline
        results.append(
            {
                "scenario": "Baseline",
                "portfolio_return": baseline_return,
                "portfolio_volatility": baseline_vol,
                "portfolio_value_change": 0.0,
            }
        )

        # Analyze each stress scenario
        for scenario_name in stress_scenarios:
            if scenario_name not in scenario_params:
                continue

            params = scenario_params[scenario_name]
            return_shock = params["return_shock"]
            vol_multiplier = params["vol_multiplier"]

            # Apply shocks
            stressed_return = baseline_return + return_shock
            stressed_vol = baseline_vol * vol_multiplier

            # Calculate portfolio value change
            value_change = return_shock

            results.append(
                {
                    "scenario": scenario_name,
                    "portfolio_return": stressed_return,
                    "portfolio_volatility": stressed_vol,
                    "portfolio_value_change": value_change,
                }
            )

        # Create DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def plot_stress_test_results(
        self, stress_test_results: pd.DataFrame, title: str = "Stress Test Results"
    ) -> go.Figure:
        """
        Visualize stress test results.

        Args:
            stress_test_results: DataFrame from perform_stress_test
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> # ... perform stress test ...
            >>> fig = analyzer.plot_stress_test_results(stress_results)
            >>> fig.show()
        """
        fig = go.Figure()

        # Add bar chart for portfolio value changes
        fig.add_trace(
            go.Bar(
                x=stress_test_results["scenario"],
                y=stress_test_results["portfolio_value_change"],
                text=[f"{v:.1%}" for v in stress_test_results["portfolio_value_change"]],
                textposition="auto",
                marker=dict(
                    color=stress_test_results["portfolio_value_change"],
                    colorscale="RdYlGn",
                    cmin=-0.3,
                    cmax=0.1,
                    showscale=True,
                    colorbar=dict(title="Return"),
                ),
                hovertemplate=("%{x}<br>Value Change: %{y:.2%}<br><extra></extra>"),
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Scenario",
            yaxis_title="Portfolio Value Change",
            yaxis=dict(tickformat=".0%"),
            template="plotly_white",
            showlegend=False,
        )

        return fig

    def compare_portfolios(
        self,
        portfolios: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple portfolios side-by-side.

        Args:
            portfolios: Dictionary mapping portfolio names to portfolio data
                       Each portfolio should have 'expected_return', 'volatility',
                       'sharpe_ratio', and optionally other metrics
            metrics: List of metrics to compare (default: all available)

        Returns:
            DataFrame comparing portfolios

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolios = {
            ...     'Conservative': {
            ...         'expected_return': 0.08,
            ...         'volatility': 0.12,
            ...         'sharpe_ratio': 0.5
            ...     },
            ...     'Aggressive': {
            ...         'expected_return': 0.15,
            ...         'volatility': 0.25,
            ...         'sharpe_ratio': 0.52
            ...     }
            ... }
            >>> comparison = analyzer.compare_portfolios(portfolios)
        """
        if not portfolios:
            raise ValueError("Portfolios dictionary cannot be empty")

        # Default metrics
        if metrics is None:
            metrics = ["expected_return", "volatility", "sharpe_ratio"]

        # Build comparison data
        comparison_data = {}

        for metric in metrics:
            comparison_data[metric] = {}
            for portfolio_name, portfolio_data in portfolios.items():
                if metric in portfolio_data:
                    comparison_data[metric][portfolio_name] = portfolio_data[metric]

        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data).T

        return comparison_df

    def plot_portfolio_comparison(
        self, comparison_df: pd.DataFrame, title: str = "Portfolio Comparison"
    ) -> go.Figure:
        """
        Visualize portfolio comparison.

        Args:
            comparison_df: DataFrame from compare_portfolios
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> # ... create comparison_df ...
            >>> fig = analyzer.plot_portfolio_comparison(comparison_df)
            >>> fig.show()
        """
        # Create grouped bar chart
        fig = go.Figure()

        for portfolio_name in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    name=portfolio_name,
                    x=comparison_df.index,
                    y=comparison_df[portfolio_name],
                    text=[
                        f"{v:.2%}" if abs(v) < 10 else f"{v:.2f}"
                        for v in comparison_df[portfolio_name]
                    ],
                    textposition="auto",
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Value",
            barmode="group",
            template="plotly_white",
            showlegend=True,
        )

        return fig

    def generate_portfolio_report(
        self,
        weights: Dict[str, float],
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        portfolio_metrics: Dict[str, float],
        risk_decomposition: Optional[Dict] = None,
        stress_test_results: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Union[str, pd.DataFrame, go.Figure]]:
        """
        Generate comprehensive portfolio report.

        Args:
            weights: Portfolio weights
            returns_dict: Historical returns data
            portfolio_metrics: Portfolio performance metrics
            risk_decomposition: Optional risk decomposition data
            stress_test_results: Optional stress test results

        Returns:
            Dictionary containing report components (text, tables, charts)

        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> weights = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> metrics = {
            ...     'expected_return': 0.12,
            ...     'volatility': 0.18,
            ...     'sharpe_ratio': 0.56
            ... }
            >>> report = analyzer.generate_portfolio_report(weights, returns, metrics)
        """
        report = {}

        # Summary text
        summary = f"""
Portfolio Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Portfolio Composition:
{len(weights)} assets

Performance Metrics:
- Expected Return: {portfolio_metrics.get("expected_return", 0):.2%}
- Volatility: {portfolio_metrics.get("volatility", 0):.2%}
- Sharpe Ratio: {portfolio_metrics.get("sharpe_ratio", 0):.3f}
"""
        report["summary"] = summary

        # Weights table
        weights_df = pd.DataFrame(
            {"Asset": list(weights.keys()), "Weight": list(weights.values())}
        ).sort_values("Weight", ascending=False)
        report["weights_table"] = weights_df

        # Composition chart
        report["composition_chart"] = self.plot_portfolio_composition(weights)

        # Risk decomposition
        if risk_decomposition:
            report["risk_chart"] = self.plot_risk_contribution(risk_decomposition)

        # Stress test results
        if stress_test_results is not None:
            report["stress_test_table"] = stress_test_results
            report["stress_test_chart"] = self.plot_stress_test_results(stress_test_results)

        return report

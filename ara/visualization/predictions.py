"""
Prediction visualization with confidence intervals.

This module provides visualization for predictions with confidence bounds.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PredictionChart:
    """Create prediction visualizations with confidence intervals."""
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize prediction chart.
        
        Args:
            theme: Plotly theme
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required. Install with: pip install plotly")
        
        self.theme = theme
    
    def create_prediction_chart(
        self,
        historical_data: pd.DataFrame,
        predictions: pd.DataFrame,
        symbol: str,
        confidence_level: float = 0.95,
        height: int = 600
    ) -> go.Figure:
        """
        Create prediction chart with confidence intervals.
        
        Args:
            historical_data: DataFrame with columns: date, close
            predictions: DataFrame with columns: date, predicted_price, 
                        lower_bound, upper_bound, confidence
            symbol: Asset symbol
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(go.Scatter(
            x=historical_data['date'] if 'date' in historical_data.columns else historical_data.index,
            y=historical_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=predictions['date'] if 'date' in predictions.columns else predictions.index,
            y=predictions['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Add confidence interval
        if 'upper_bound' in predictions.columns and 'lower_bound' in predictions.columns:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=predictions['date'] if 'date' in predictions.columns else predictions.index,
                y=predictions['upper_bound'],
                mode='lines',
                name=f'{int(confidence_level*100)}% Upper Bound',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                showlegend=True
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=predictions['date'] if 'date' in predictions.columns else predictions.index,
                y=predictions['lower_bound'],
                mode='lines',
                name=f'{int(confidence_level*100)}% Lower Bound',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Predictions with {int(confidence_level*100)}% Confidence Interval',
            xaxis_title='Date',
            yaxis_title='Price',
            template=self.theme,
            height=height,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_multi_horizon_chart(
        self,
        historical_data: pd.DataFrame,
        predictions_dict: Dict[str, pd.DataFrame],
        symbol: str,
        height: int = 800
    ) -> go.Figure:
        """
        Create chart with multiple prediction horizons.
        
        Args:
            historical_data: DataFrame with historical prices
            predictions_dict: Dictionary of horizon -> predictions DataFrame
            symbol: Asset symbol
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(go.Scatter(
            x=historical_data['date'] if 'date' in historical_data.columns else historical_data.index,
            y=historical_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions for each horizon
        colors = ['red', 'orange', 'purple', 'green', 'brown']
        for i, (horizon, predictions) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=predictions['date'] if 'date' in predictions.columns else predictions.index,
                y=predictions['predicted_price'],
                mode='lines+markers',
                name=f'{horizon} Prediction',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f'{symbol} Multi-Horizon Predictions',
            xaxis_title='Date',
            yaxis_title='Price',
            template=self.theme,
            height=height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_prediction_accuracy_chart(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        symbol: str,
        height: int = 800
    ) -> go.Figure:
        """
        Create chart comparing predictions vs actuals.
        
        Args:
            predictions: DataFrame with predicted values
            actuals: DataFrame with actual values
            symbol: Asset symbol
            height: Chart height in pixels
            
        Returns:
            Plotly figure with comparison and error analysis
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'{symbol} Predictions vs Actuals',
                'Prediction Error'
            ),
            row_heights=[0.7, 0.3]
        )
        
        # Predictions vs Actuals
        fig.add_trace(
            go.Scatter(
                x=actuals['date'] if 'date' in actuals.columns else actuals.index,
                y=actuals['close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions['date'] if 'date' in predictions.columns else predictions.index,
                y=predictions['predicted_price'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Calculate error
        error = predictions['predicted_price'].values - actuals['close'].values
        error_pct = (error / actuals['close'].values) * 100
        
        # Error plot
        colors = ['green' if e >= 0 else 'red' for e in error_pct]
        fig.add_trace(
            go.Bar(
                x=actuals['date'] if 'date' in actuals.columns else actuals.index,
                y=error_pct,
                name='Error %',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Error (%)", row=2, col=1)
        
        fig.update_layout(
            template=self.theme,
            height=height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_confidence_evolution_chart(
        self,
        predictions: pd.DataFrame,
        symbol: str,
        height: int = 600
    ) -> go.Figure:
        """
        Create chart showing how confidence evolves over prediction horizon.
        
        Args:
            predictions: DataFrame with confidence scores
            symbol: Asset symbol
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'{symbol} Predictions',
                'Confidence Score'
            ),
            row_heights=[0.6, 0.4]
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=predictions['date'] if 'date' in predictions.columns else predictions.index,
                y=predictions['predicted_price'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Confidence
        if 'confidence' in predictions.columns:
            # Color code by confidence level
            colors = []
            for conf in predictions['confidence']:
                if conf >= 0.8:
                    colors.append('green')
                elif conf >= 0.6:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            fig.add_trace(
                go.Bar(
                    x=predictions['date'] if 'date' in predictions.columns else predictions.index,
                    y=predictions['confidence'],
                    name='Confidence',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add confidence threshold lines
            fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                         annotation_text="High", row=2, col=1)
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                         annotation_text="Medium", row=2, col=1)
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])
        
        fig.update_layout(
            template=self.theme,
            height=height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_feature_contribution_chart(
        self,
        feature_contributions: Dict[str, float],
        prediction_date: datetime,
        symbol: str,
        top_n: int = 10,
        height: int = 500
    ) -> go.Figure:
        """
        Create chart showing top feature contributions to prediction.
        
        Args:
            feature_contributions: Dictionary of feature -> contribution
            prediction_date: Date of prediction
            symbol: Asset symbol
            top_n: Number of top features to show
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        features, contributions = zip(*sorted_features)
        
        # Color code by positive/negative
        colors = ['green' if c > 0 else 'red' for c in contributions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=list(features),
            x=list(contributions),
            orientation='h',
            marker_color=colors,
            text=[f'{c:+.3f}' for c in contributions],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'{symbol} Top {top_n} Feature Contributions<br>{prediction_date.strftime("%Y-%m-%d")}',
            xaxis_title='Contribution to Prediction',
            yaxis_title='Feature',
            template=self.theme,
            height=height,
            showlegend=False
        )
        
        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        return fig

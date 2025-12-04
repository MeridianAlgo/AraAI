"""
Correlation visualization including heatmaps and network graphs.

This module provides correlation analysis visualizations.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class CorrelationChart:
    """Create correlation visualizations."""
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize correlation chart.
        
        Args:
            theme: Plotly theme
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required. Install with: pip install plotly")
        
        self.theme = theme
    
    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix",
        height: int = 700,
        show_values: bool = True
    ) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: DataFrame with correlation values
            title: Chart title
            height: Chart height in pixels
            show_values: Whether to show correlation values in cells
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Prepare text
        text = None
        if show_values:
            text = [[f'{val:.2f}' for val in row] 
                   for row in correlation_matrix.values]
        
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate='%{text}' if show_values else None,
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{y} - %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=height,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def create_rolling_correlation_chart(
        self,
        dates: List,
        correlations: Dict[str, np.ndarray],
        title: str = "Rolling Correlations",
        height: int = 600
    ) -> go.Figure:
        """
        Create rolling correlation chart.
        
        Args:
            dates: List of dates
            correlations: Dictionary of pair_name -> correlation values
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for i, (pair_name, values) in enumerate(correlations.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name=pair_name,
                line=dict(color=color, width=2)
            ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.3,
                     annotation_text="Strong Positive")
        fig.add_hline(y=-0.7, line_dash="dot", line_color="red", opacity=0.3,
                     annotation_text="Strong Negative")
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Correlation',
            template=self.theme,
            height=height,
            hovermode='x unified',
            yaxis_range=[-1, 1]
        )
        
        return fig
    
    def create_correlation_breakdown_chart(
        self,
        asset_pair: str,
        dates: List,
        correlation: np.ndarray,
        price1: np.ndarray,
        price2: np.ndarray,
        asset1_name: str,
        asset2_name: str,
        height: int = 900
    ) -> go.Figure:
        """
        Create detailed correlation breakdown chart.
        
        Args:
            asset_pair: Name of asset pair
            dates: List of dates
            correlation: Correlation values over time
            price1: Price series for first asset
            price2: Price series for second asset
            asset1_name: Name of first asset
            asset2_name: Name of second asset
            height: Chart height in pixels
            
        Returns:
            Plotly figure with multiple subplots
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f'{asset_pair} Correlation',
                f'{asset1_name} Price',
                f'{asset2_name} Price'
            ),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Correlation
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=correlation,
                mode='lines',
                name='Correlation',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 200, 0.2)'
            ),
            row=1, col=1
        )
        
        # Add correlation threshold lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.3, row=1, col=1)
        fig.add_hline(y=-0.7, line_dash="dot", line_color="red", opacity=0.3, row=1, col=1)
        
        # Asset 1 price
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=price1,
                mode='lines',
                name=asset1_name,
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Asset 2 price
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=price2,
                mode='lines',
                name=asset2_name,
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=1, range=[-1, 1])
        fig.update_yaxes(title_text="Price", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=3, col=1)
        
        fig.update_layout(
            title=f'{asset_pair} Correlation Analysis',
            template=self.theme,
            height=height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_scatter_matrix(
        self,
        data: pd.DataFrame,
        assets: List[str],
        title: str = "Asset Scatter Matrix",
        height: int = 800
    ) -> go.Figure:
        """
        Create scatter matrix for multiple assets.
        
        Args:
            data: DataFrame with asset returns
            assets: List of asset column names
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        # Create scatter matrix using plotly figure factory
        fig = ff.create_scatterplotmatrix(
            data[assets],
            diag='histogram',
            height=height,
            width=height,
            title=title
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    def create_correlation_network(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5,
        title: str = "Correlation Network",
        height: int = 700
    ) -> go.Figure:
        """
        Create network graph of correlations above threshold.
        
        Args:
            correlation_matrix: DataFrame with correlation values
            threshold: Minimum correlation to show edge
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        # Get assets
        assets = list(correlation_matrix.columns)
        n = len(assets)
        
        # Create circular layout
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        fig = go.Figure()
        
        # Add edges for correlations above threshold
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for i in range(n):
            for j in range(i+1, n):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    edge_x.extend([x_pos[i], x_pos[j], None])
                    edge_y.extend([y_pos[i], y_pos[j], None])
                    # Color by correlation strength
                    color = f'rgba({"255,0,0" if corr < 0 else "0,255,0"},{abs(corr)})'
                    edge_colors.append(color)
        
        # Add edges
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(color='darkblue', width=2)
            ),
            text=assets,
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'{title}<br>(Showing correlations > {threshold})',
            template=self.theme,
            height=height,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest'
        )
        
        return fig
    
    def create_hierarchical_clustering(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Hierarchical Clustering",
        height: int = 600
    ) -> go.Figure:
        """
        Create hierarchical clustering dendrogram.
        
        Args:
            correlation_matrix: DataFrame with correlation values
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        # Convert correlation to distance
        distance_matrix = 1 - correlation_matrix.abs()
        
        # Create dendrogram using figure factory
        fig = ff.create_dendrogram(
            distance_matrix.values,
            labels=list(correlation_matrix.columns),
            orientation='bottom'
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=height,
            xaxis_title='Assets',
            yaxis_title='Distance'
        )
        
        return fig
    
    def create_lead_lag_chart(
        self,
        dates: List,
        lead_lag_scores: Dict[str, np.ndarray],
        title: str = "Lead-Lag Relationships",
        height: int = 600
    ) -> go.Figure:
        """
        Create chart showing lead-lag relationships between assets.
        
        Args:
            dates: List of dates
            lead_lag_scores: Dictionary of pair -> lead-lag scores
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for pair_name, scores in lead_lag_scores.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines',
                name=pair_name,
                line=dict(width=2)
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Lead-Lag Score (positive = first asset leads)',
            template=self.theme,
            height=height,
            hovermode='x unified'
        )
        
        return fig

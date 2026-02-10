"""
Chart export and report generation.

This module provides functionality to export charts to various formats
and generate comprehensive PDF reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ChartExporter:
    """Export charts to various formats."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize chart exporter.

        Args:
            output_dir: Directory for saving exports
        """
        self.output_dir = output_dir or Path("exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_html(
        self,
        fig: go.Figure,
        filename: str,
        include_plotlyjs: Union[bool, str] = "cdn",
        auto_open: bool = False,
    ) -> Path:
        """
        Export chart to HTML.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            include_plotlyjs: How to include plotly.js ('cdn', True, False)
            auto_open: Whether to open in browser

        Returns:
            Path to saved file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for HTML export")

        filepath = self.output_dir / f"{filename}.html"
        fig.write_html(str(filepath), include_plotlyjs=include_plotlyjs, auto_open=auto_open)
        return filepath

    def export_to_png(
        self,
        fig: go.Figure,
        filename: str,
        width: int = 1200,
        height: int = 800,
        scale: float = 2.0,
    ) -> Path:
        """
        Export chart to PNG.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for higher resolution

        Returns:
            Path to saved file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PNG export")

        filepath = self.output_dir / f"{filename}.png"

        try:
            fig.write_image(str(filepath), format="png", width=width, height=height, scale=scale)
        except Exception as e:
            print("Warning: PNG export failed. Install kaleido: pip install kaleido")
            print(f"Error: {e}")
            return None

        return filepath

    def export_to_svg(
        self, fig: go.Figure, filename: str, width: int = 1200, height: int = 800
    ) -> Path:
        """
        Export chart to SVG.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Path to saved file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for SVG export")

        filepath = self.output_dir / f"{filename}.svg"

        try:
            fig.write_image(str(filepath), format="svg", width=width, height=height)
        except Exception as e:
            print("Warning: SVG export failed. Install kaleido: pip install kaleido")
            print(f"Error: {e}")
            return None

        return filepath

    def export_to_pdf(
        self, fig: go.Figure, filename: str, width: int = 1200, height: int = 800
    ) -> Path:
        """
        Export chart to PDF.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Path to saved file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PDF export")

        filepath = self.output_dir / f"{filename}.pdf"

        try:
            fig.write_image(str(filepath), format="pdf", width=width, height=height)
        except Exception as e:
            print("Warning: PDF export failed. Install kaleido: pip install kaleido")
            print(f"Error: {e}")
            return None

        return filepath

    def export_to_json(self, fig: go.Figure, filename: str, pretty: bool = True) -> Path:
        """
        Export chart data to JSON.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            pretty: Whether to pretty-print JSON

        Returns:
            Path to saved file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for JSON export")

        filepath = self.output_dir / f"{filename}.json"

        fig_json = fig.to_json()

        with open(filepath, "w") as f:
            if pretty:
                json.dump(json.loads(fig_json), f, indent=2)
            else:
                f.write(fig_json)

        return filepath

    def export_data_to_csv(self, data: pd.DataFrame, filename: str) -> Path:
        """
        Export data to CSV.

        Args:
            data: DataFrame to export
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.csv"
        data.to_csv(filepath, index=False)
        return filepath

    def export_data_to_excel(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str
    ) -> Path:
        """
        Export data to Excel.

        Args:
            data: DataFrame or dict of sheet_name -> DataFrame
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.xlsx"

        try:
            if isinstance(data, dict):
                with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                    for sheet_name, df in data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                data.to_excel(filepath, index=False)
        except ImportError:
            print("Warning: Excel export requires openpyxl. Install with: pip install openpyxl")
            return None

        return filepath


class ReportGenerator:
    """Generate comprehensive PDF reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for saving reports
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF reports. Install with: pip install reportlab"
            )

        self.output_dir = output_dir or Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1f77b4"),
                spaceAfter=30,
                alignment=TA_CENTER,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="CustomHeading",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#2ca02c"),
                spaceAfter=12,
                spaceBefore=12,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="MetricLabel",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.grey,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="MetricValue",
                parent=self.styles["Normal"],
                fontSize=14,
                textColor=colors.black,
                fontName="Helvetica-Bold",
            )
        )

    def generate_prediction_report(
        self,
        symbol: str,
        prediction_data: Dict[str, Any],
        charts: Dict[str, go.Figure],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate comprehensive prediction report.

        Args:
            symbol: Asset symbol
            prediction_data: Dictionary with prediction data
            charts: Dictionary of chart_name -> figure
            filename: Optional output filename

        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_report_{symbol}_{timestamp}.pdf"

        filepath = self.output_dir / filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        title = Paragraph(f"Prediction Report: {symbol}", self.styles["CustomTitle"])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Report metadata
        metadata_text = f"""
        <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Symbol:</b> {symbol}<br/>
        <b>Report Type:</b> Price Prediction Analysis
        """
        story.append(Paragraph(metadata_text, self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Summary metrics
        if "metrics" in prediction_data:
            story.append(Paragraph("Summary Metrics", self.styles["CustomHeading"]))
            metrics_table = self._create_metrics_table(prediction_data["metrics"])
            story.append(metrics_table)
            story.append(Spacer(1, 0.3 * inch))

        # Predictions table
        if "predictions" in prediction_data:
            story.append(Paragraph("Predictions", self.styles["CustomHeading"]))
            pred_table = self._create_predictions_table(prediction_data["predictions"])
            story.append(pred_table)
            story.append(Spacer(1, 0.3 * inch))

        # Charts
        if charts:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", self.styles["CustomHeading"]))

            # Export charts as images and add to report
            exporter = ChartExporter(self.output_dir / "temp_charts")

            for chart_name, fig in charts.items():
                # Export to PNG
                img_path = exporter.export_to_png(fig, f"temp_{chart_name}")

                if img_path and img_path.exists():
                    # Add chart title
                    story.append(
                        Paragraph(
                            chart_name.replace("_", " ").title(),
                            self.styles["Heading3"],
                        )
                    )
                    story.append(Spacer(1, 0.1 * inch))

                    # Add image
                    img = Image(str(img_path), width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3 * inch))

        # Build PDF
        doc.build(story)

        # Cleanup temp charts
        temp_dir = self.output_dir / "temp_charts"
        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)

        return filepath

    def generate_backtest_report(
        self,
        symbol: str,
        backtest_results: Dict[str, Any],
        charts: Dict[str, go.Figure],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate comprehensive backtest report.

        Args:
            symbol: Asset symbol
            backtest_results: Dictionary with backtest results
            charts: Dictionary of chart_name -> figure
            filename: Optional output filename

        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{symbol}_{timestamp}.pdf"

        filepath = self.output_dir / filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        title = Paragraph(f"Backtest Report: {symbol}", self.styles["CustomTitle"])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Report metadata
        metadata_text = f"""
        <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Symbol:</b> {symbol}<br/>
        <b>Period:</b> {backtest_results.get("start_date", "N/A")} to {backtest_results.get("end_date", "N/A")}<br/>
        <b>Report Type:</b> Backtest Analysis
        """
        story.append(Paragraph(metadata_text, self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Performance metrics
        if "metrics" in backtest_results:
            story.append(Paragraph("Performance Metrics", self.styles["CustomHeading"]))
            metrics_table = self._create_metrics_table(backtest_results["metrics"])
            story.append(metrics_table)
            story.append(Spacer(1, 0.3 * inch))

        # Trade statistics
        if "trade_statistics" in backtest_results:
            story.append(Paragraph("Trade Statistics", self.styles["CustomHeading"]))
            trade_table = self._create_metrics_table(backtest_results["trade_statistics"])
            story.append(trade_table)
            story.append(Spacer(1, 0.3 * inch))

        # Charts
        if charts:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", self.styles["CustomHeading"]))

            exporter = ChartExporter(self.output_dir / "temp_charts")

            for chart_name, fig in charts.items():
                img_path = exporter.export_to_png(fig, f"temp_{chart_name}")

                if img_path and img_path.exists():
                    story.append(
                        Paragraph(
                            chart_name.replace("_", " ").title(),
                            self.styles["Heading3"],
                        )
                    )
                    story.append(Spacer(1, 0.1 * inch))

                    img = Image(str(img_path), width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3 * inch))

        # Build PDF
        doc.build(story)

        # Cleanup
        temp_dir = self.output_dir / "temp_charts"
        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)

        return filepath

    def generate_portfolio_report(
        self,
        portfolio_data: Dict[str, Any],
        charts: Dict[str, go.Figure],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate comprehensive portfolio report.

        Args:
            portfolio_data: Dictionary with portfolio data
            charts: Dictionary of chart_name -> figure
            filename: Optional output filename

        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_report_{timestamp}.pdf"

        filepath = self.output_dir / filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        title = Paragraph("Portfolio Analysis Report", self.styles["CustomTitle"])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Metadata
        metadata_text = f"""
        <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Report Type:</b> Portfolio Analysis
        """
        story.append(Paragraph(metadata_text, self.styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Portfolio allocation
        if "allocation" in portfolio_data:
            story.append(Paragraph("Portfolio Allocation", self.styles["CustomHeading"]))
            alloc_table = self._create_allocation_table(portfolio_data["allocation"])
            story.append(alloc_table)
            story.append(Spacer(1, 0.3 * inch))

        # Performance metrics
        if "metrics" in portfolio_data:
            story.append(Paragraph("Performance Metrics", self.styles["CustomHeading"]))
            metrics_table = self._create_metrics_table(portfolio_data["metrics"])
            story.append(metrics_table)
            story.append(Spacer(1, 0.3 * inch))

        # Risk metrics
        if "risk_metrics" in portfolio_data:
            story.append(Paragraph("Risk Metrics", self.styles["CustomHeading"]))
            risk_table = self._create_metrics_table(portfolio_data["risk_metrics"])
            story.append(risk_table)
            story.append(Spacer(1, 0.3 * inch))

        # Charts
        if charts:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", self.styles["CustomHeading"]))

            exporter = ChartExporter(self.output_dir / "temp_charts")

            for chart_name, fig in charts.items():
                img_path = exporter.export_to_png(fig, f"temp_{chart_name}")

                if img_path and img_path.exists():
                    story.append(
                        Paragraph(
                            chart_name.replace("_", " ").title(),
                            self.styles["Heading3"],
                        )
                    )
                    story.append(Spacer(1, 0.1 * inch))

                    img = Image(str(img_path), width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3 * inch))

        # Build PDF
        doc.build(story)

        # Cleanup
        temp_dir = self.output_dir / "temp_charts"
        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)

        return filepath

    def _create_metrics_table(self, metrics: Dict[str, Any]) -> Table:
        """Create a formatted metrics table."""
        data = [["Metric", "Value"]]

        for key, value in metrics.items():
            # Format key
            formatted_key = key.replace("_", " ").title()

            # Format value
            if isinstance(value, float):
                if abs(value) < 1:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            data.append([formatted_key, formatted_value])

        table = Table(data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.lightgrey],
                    ),
                ]
            )
        )

        return table

    def _create_predictions_table(self, predictions: List[Dict[str, Any]]) -> Table:
        """Create a formatted predictions table."""
        data = [["Date", "Predicted Price", "Confidence", "Lower Bound", "Upper Bound"]]

        for pred in predictions[:10]:  # Limit to first 10
            data.append(
                [
                    pred.get("date", "N/A"),
                    f"${pred.get('predicted_price', 0):.2f}",
                    f"{pred.get('confidence', 0):.2%}",
                    f"${pred.get('lower_bound', 0):.2f}",
                    f"${pred.get('upper_bound', 0):.2f}",
                ]
            )

        table = Table(data, colWidths=[1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.lightgrey],
                    ),
                ]
            )
        )

        return table

    def _create_allocation_table(self, allocation: Dict[str, float]) -> Table:
        """Create a formatted allocation table."""
        data = [["Asset", "Weight"]]

        for asset, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
            data.append([asset, f"{weight:.2%}"])

        table = Table(data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.lightgrey],
                    ),
                ]
            )
        )

        return table

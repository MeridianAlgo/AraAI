"""
Console management and rich output formatting
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import warnings

warnings.filterwarnings("ignore")


class ConsoleManager:
    """Enhanced console manager with rich formatting"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = Console()

    def print_system_info(self):
        """Print system initialization information"""
        if not self.verbose:
            return

        self.console.print(
            Panel(
                "[bold green] Ara AI Stock Analysis Platform[/]\n"
                "[white]Enhanced with ensemble ML models and intelligent caching[/]",
                title="[bold blue]System Initialization[/]",
                border_style="blue",
            )
        )

    def print_gpu_info(self, gpu_info):
        """Print GPU information"""
        if not self.verbose:
            return

        if gpu_info["details"]:
            gpu_text = "\n".join([f" {detail}" for detail in gpu_info["details"]])
        else:
            gpu_text = " No GPU acceleration available"

        self.console.print(
            Panel(gpu_text, title="[bold cyan]GPU Acceleration[/]", border_style="cyan")
        )

    def print_prediction_results(self, result):
        """Print enhanced prediction results with patterns and analysis"""
        try:
            if not result:
                self.console.print("[red] No prediction results to display[/]")
                return

            symbol = result.get("symbol", "Unknown")
            current_price = result.get("current_price", 0)
            predictions = result.get("predictions", [])

            # Print company header if analysis available
            company_analysis = result.get("company_analysis", {})
            if company_analysis and "error" not in company_analysis:
                company_name = company_analysis.get("company_name", symbol)
                sector = company_analysis.get("sector", "Unknown")
                self.console.print(
                    Panel(
                        f"[bold white]{company_name}[/] ([cyan]{symbol}[/])\n"
                        f"[white]Sector: {sector}[/]",
                        title="[bold blue]Company Information[/]",
                        border_style="blue",
                    )
                )

            # Create main predictions table with confidence
            table = Table(
                title=f" {symbol} Stock Predictions",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold blue",
            )

            table.add_column("Day", style="cyan", no_wrap=True)
            table.add_column("Date", style="white")
            table.add_column("Predicted Price", style="green", justify="right")
            table.add_column("Change", style="yellow", justify="right")
            table.add_column("Change %", style="magenta", justify="right")
            table.add_column("Confidence", style="blue", justify="right")

            for pred in predictions:
                change_color = "green" if pred.get("change", 0) >= 0 else "red"
                change_pct_color = "green" if pred.get("change_pct", 0) >= 0 else "red"
                confidence = pred.get("confidence", 0.75) * 100

                table.add_row(
                    f"Day {pred.get('day', 1)}",
                    pred.get("date", "").split("T")[0],  # Remove time part
                    f"${pred.get('predicted_price', 0):.2f}",
                    f"[{change_color}]${pred.get('change', 0):+.2f}[/]",
                    f"[{change_pct_color}]{pred.get('change_pct', 0):+.1f}%[/]",
                    f"{confidence:.0f}%",
                )

            self.console.print(table)

            # Print current price info
            self.console.print(
                f"\n[bold white] Current Price: [green]${current_price:.2f}[/]"
            )

            # Print pattern analysis
            pattern_summary = result.get("pattern_summary", {})
            if pattern_summary.get("primary_pattern"):
                pattern_name = (
                    pattern_summary["primary_pattern"].replace("_", " ").title()
                )
                signal = pattern_summary["signal_direction"].upper()
                confidence = pattern_summary["pattern_confidence"] * 100

                signal_color = (
                    "green"
                    if signal == "BULLISH"
                    else "red" if signal == "BEARISH" else "yellow"
                )

                self.console.print(
                    Panel(
                        f"[white]Primary Pattern: [bold]{pattern_name}[/]\n"
                        f"Signal: [{signal_color}]{signal}[/]\n"
                        f"Pattern Confidence: {confidence:.0f}%\n"
                        f"Total Patterns Detected: {pattern_summary.get('total_patterns_detected', 0)}",
                        title="[bold magenta] Chart Pattern Analysis[/]",
                        border_style="magenta",
                    )
                )

            # Print company analysis summary
            analysis_summary = result.get("analysis_summary", {})
            if analysis_summary:
                overall_score = analysis_summary.get("overall_score", 50)
                recommendation = analysis_summary.get("recommendation", "HOLD")

                # Color code recommendation
                rec_color = (
                    "green"
                    if "BUY" in recommendation
                    else "red" if "SELL" in recommendation else "yellow"
                )

                self.console.print(
                    Panel(
                        f"[white]Overall Score: [bold]{overall_score:.1f}/100[/]\n"
                        f"Recommendation: [{rec_color}]{recommendation}[/]\n"
                        f"Financial Health: {analysis_summary.get('financial_grade', 'C')}\n"
                        f"Risk Assessment: {analysis_summary.get('risk_grade', 'C')}\n"
                        f"Valuation: {analysis_summary.get('valuation_summary', 'Fair Value')}\n"
                        f"Market Sentiment: {analysis_summary.get('market_sentiment', 'Neutral')}",
                        title="[bold green] Investment Analysis[/]",
                        border_style="green",
                    )
                )

            # Print detailed company analysis if verbose
            if self.verbose and company_analysis and "error" not in company_analysis:
                self._print_detailed_analysis(company_analysis)

            # Print cache info if available
            if result.get("cached"):
                cache_age = result.get("cache_age", "Unknown")
                self.console.print(
                    f"[yellow] Using cached predictions (Age: {cache_age})[/]"
                )

        except Exception as e:
            self.console.print(f"[red] Error displaying results: {e}[/]")

    def _print_detailed_analysis(self, company_analysis):
        """Print detailed company analysis in verbose mode"""
        try:
            # Financial Health Details
            financial = company_analysis.get("financial_health", {})
            if financial and "error" not in financial:
                fin_table = Table(title=" Financial Health", box=box.SIMPLE)
                fin_table.add_column("Metric", style="white")
                fin_table.add_column("Value", style="cyan", justify="right")

                fin_table.add_row(
                    "Health Score", f"{financial.get('health_score', 0):.1f}/100"
                )
                fin_table.add_row(
                    "Debt to Equity", f"{financial.get('debt_to_equity', 0):.2f}"
                )
                fin_table.add_row(
                    "Current Ratio", f"{financial.get('current_ratio', 0):.2f}"
                )
                fin_table.add_row(
                    "ROE", f"{financial.get('return_on_equity', 0)*100:.1f}%"
                )
                fin_table.add_row(
                    "Profit Margin", f"{financial.get('profit_margin', 0)*100:.1f}%"
                )

                self.console.print(fin_table)

            # Risk Assessment Details
            risk = company_analysis.get("risk_assessment", {})
            if risk and "error" not in risk:
                risk_table = Table(title=" Risk Assessment", box=box.SIMPLE)
                risk_table.add_column("Risk Type", style="white")
                risk_table.add_column("Level", style="yellow", justify="right")

                risk_table.add_row("Market Risk", risk.get("market_risk", "Medium"))
                risk_table.add_row(
                    "Volatility Risk", risk.get("volatility_risk", "Medium")
                )
                risk_table.add_row(
                    "Liquidity Risk", risk.get("liquidity_risk", "Medium")
                )
                risk_table.add_row("Credit Risk", risk.get("credit_risk", "Medium"))
                risk_table.add_row("Beta", f"{risk.get('beta', 1.0):.2f}")

                self.console.print(risk_table)

            # Technical Analysis Details
            technical = company_analysis.get("technical_analysis", {})
            if technical and "error" not in technical:
                tech_table = Table(title=" Technical Analysis", box=box.SIMPLE)
                tech_table.add_column("Indicator", style="white")
                tech_table.add_column("Value", style="green", justify="right")

                tech_table.add_row(
                    "Trend Direction", technical.get("trend_direction", "Neutral")
                )
                tech_table.add_row(
                    "Trend Strength", f"{technical.get('trend_strength', 0)}/100"
                )
                tech_table.add_row(
                    "Momentum Score", f"{technical.get('momentum_score', 0)}/100"
                )

                indicators = technical.get("indicators", {})
                if indicators:
                    tech_table.add_row("RSI", f"{indicators.get('rsi', 50):.1f}")
                    tech_table.add_row("MACD", f"{indicators.get('macd', 0):.3f}")

                self.console.print(tech_table)

        except Exception as e:
            self.console.print(f"[red] Error displaying detailed analysis: {e}[/]")

    def print_accuracy_summary(self, accuracy_stats):
        """Print accuracy summary"""
        try:
            if not accuracy_stats:
                self.console.print("[yellow]  No accuracy data available[/]")
                return

            symbol = accuracy_stats.get("symbol", "All")
            total = accuracy_stats.get("total_predictions", 0)
            accuracy_rate = accuracy_stats.get("accuracy_rate", 0)
            excellent_rate = accuracy_stats.get("excellent_rate", 0)
            good_rate = accuracy_stats.get("good_rate", 0)
            avg_error = accuracy_stats.get("avg_error", 0)

            # Create accuracy table
            table = Table(
                title=f" Accuracy Statistics - {symbol}",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold blue",
            )

            table.add_column("Metric", style="white")
            table.add_column("Value", style="green", justify="right")

            table.add_row("Total Predictions", str(total))
            table.add_row("Overall Accuracy", f"{accuracy_rate:.1f}%")
            table.add_row("Excellent (<1% error)", f"{excellent_rate:.1f}%")
            table.add_row("Good (<2% error)", f"{good_rate:.1f}%")
            table.add_row("Average Error", f"{avg_error:.2f}%")

            # Add recent stats if available
            recent_stats = accuracy_stats.get("recent_stats", {})
            if recent_stats:
                table.add_row("", "")  # Separator
                table.add_row("Recent (30d) Total", str(recent_stats.get("total", 0)))
                table.add_row(
                    "Recent (30d) Accuracy",
                    f"{recent_stats.get('accuracy_rate', 0):.1f}%",
                )
                table.add_row(
                    "Recent (30d) Avg Error", f"{recent_stats.get('avg_error', 0):.2f}%"
                )

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red] Error displaying accuracy: {e}[/]")

    def print_validation_summary(self, validation_result):
        """Print validation summary"""
        try:
            if not validation_result:
                self.console.print("[yellow]  No validation results available[/]")
                return

            validated = validation_result.get("validated", 0)
            accuracy_rate = validation_result.get("accuracy_rate", 0)
            excellent_rate = validation_result.get("excellent_rate", 0)
            good_rate = validation_result.get("good_rate", 0)
            avg_error = validation_result.get("avg_error", 0)

            self.console.print(
                Panel(
                    f"[green] Validated: {validated} predictions[/]\n"
                    f"[cyan] Accuracy Rate: {accuracy_rate:.1f}%[/]\n"
                    f"[bright_green] Excellent (<1%): {excellent_rate:.1f}%[/]\n"
                    f"[green] Good (<2%): {good_rate:.1f}%[/]\n"
                    f"[white] Average Error: {avg_error:.2f}%[/]",
                    title="[bold blue]Validation Summary[/]",
                    border_style="blue",
                )
            )

        except Exception as e:
            self.console.print(f"[red] Error displaying validation: {e}[/]")

    def print_error(self, message):
        """Print error message"""
        self.console.print(f"[red] {message}[/]")

    def print_warning(self, message):
        """Print warning message"""
        self.console.print(f"[yellow]  {message}[/]")

    def print_success(self, message):
        """Print success message"""
        self.console.print(f"[green] {message}[/]")

    def print_info(self, message):
        """Print info message"""
        self.console.print(f"[cyan]ℹ  {message}[/]")

    def print_company_analysis(self, analysis):
        """Print comprehensive company analysis"""
        try:
            if not analysis or "error" in analysis:
                self.console.print("[red] No company analysis available[/]")
                return

            symbol = analysis.get("symbol", "Unknown")
            company_name = analysis.get("company_name", symbol)

            # Company Header
            self.console.print(
                Panel(
                    f"[bold white]{company_name}[/] ([cyan]{symbol}[/])\n"
                    f"[white]Sector: {analysis.get('sector', 'Unknown')} | Industry: {analysis.get('industry', 'Unknown')}[/]\n"
                    f"[white]Market Cap: ${analysis.get('market_cap', 0):,.0f}[/]",
                    title="[bold blue] Company Overview[/]",
                    border_style="blue",
                )
            )

            # Overall Score and Recommendation
            overall_score = analysis.get("overall_score", 50)
            recommendation = analysis.get("recommendation", "HOLD")
            rec_color = (
                "green"
                if "BUY" in recommendation
                else "red" if "SELL" in recommendation else "yellow"
            )

            self.console.print(
                Panel(
                    f"[bold white]Overall Investment Score: {overall_score:.1f}/100[/]\n"
                    f"[bold white]Recommendation: [{rec_color}]{recommendation}[/]",
                    title="[bold green] Investment Summary[/]",
                    border_style="green",
                )
            )

            # Create summary table
            summary_table = Table(
                title=" Analysis Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold blue",
            )

            summary_table.add_column("Category", style="white")
            summary_table.add_column("Score/Grade", style="cyan", justify="center")
            summary_table.add_column("Status", style="green", justify="center")

            # Financial Health
            financial = analysis.get("financial_health", {})
            if financial and "error" not in financial:
                health_score = financial.get("health_score", 0)
                health_grade = financial.get("health_grade", "C")
                summary_table.add_row(
                    "Financial Health",
                    f"{health_score:.1f} ({health_grade})",
                    self._get_status_emoji(health_score),
                )

            # Risk Assessment
            risk = analysis.get("risk_assessment", {})
            if risk and "error" not in risk:
                risk_score = risk.get("risk_score", 0)
                risk_grade = risk.get("risk_grade", "C")
                summary_table.add_row(
                    "Risk Management",
                    f"{risk_score:.1f} ({risk_grade})",
                    self._get_status_emoji(risk_score),
                )

            # Valuation
            valuation = analysis.get("valuation_metrics", {})
            if valuation and "error" not in valuation:
                val_score = valuation.get("valuation_score", 0)
                val_grade = valuation.get("valuation_grade", "C")
                val_summary = valuation.get("summary", "Fair Value")
                summary_table.add_row(
                    "Valuation", f"{val_score:.1f} ({val_grade})", val_summary
                )

            # Growth Potential
            growth = analysis.get("growth_analysis", {})
            if growth and "error" not in growth:
                growth_score = growth.get("growth_score", 0)
                growth_grade = growth.get("growth_grade", "C")
                growth_category = growth.get("growth_category", "Moderate Growth")
                summary_table.add_row(
                    "Growth Potential",
                    f"{growth_score:.1f} ({growth_grade})",
                    growth_category,
                )

            # Technical Analysis
            technical = analysis.get("technical_analysis", {})
            if technical and "error" not in technical:
                momentum_score = technical.get("momentum_score", 0)
                trend_direction = technical.get("trend_direction", "Neutral")
                summary_table.add_row(
                    "Technical Momentum", f"{momentum_score:.1f}/100", trend_direction
                )

            # Market Intelligence
            market_intel = analysis.get("market_intelligence", {})
            if market_intel and "error" not in market_intel:
                sentiment_score = market_intel.get("sentiment_score", 0)
                market_sentiment = market_intel.get("market_sentiment", "Neutral")
                summary_table.add_row(
                    "Market Sentiment", f"{sentiment_score:.1f}/100", market_sentiment
                )

            self.console.print(summary_table)

            # Detailed sections if verbose
            if self.verbose:
                self._print_detailed_company_sections(analysis)

        except Exception as e:
            self.console.print(f"[red] Error displaying company analysis: {e}[/]")

    def _print_detailed_company_sections(self, analysis):
        """Print detailed company analysis sections"""
        try:
            # Financial Health Details
            financial = analysis.get("financial_health", {})
            if financial and "error" not in financial:
                fin_table = Table(title=" Financial Health Details", box=box.SIMPLE)
                fin_table.add_column("Metric", style="white")
                fin_table.add_column("Value", style="cyan", justify="right")
                fin_table.add_column("Assessment", style="green")

                debt_equity = financial.get("debt_to_equity", 0)
                fin_table.add_row(
                    "Debt to Equity",
                    f"{debt_equity:.2f}",
                    (
                        "Good"
                        if debt_equity < 0.5
                        else "Fair" if debt_equity < 1.0 else "High"
                    ),
                )

                current_ratio = financial.get("current_ratio", 0)
                fin_table.add_row(
                    "Current Ratio",
                    f"{current_ratio:.2f}",
                    (
                        "Strong"
                        if current_ratio > 1.5
                        else "Fair" if current_ratio > 1.0 else "Weak"
                    ),
                )

                roe = financial.get("return_on_equity", 0)
                fin_table.add_row(
                    "Return on Equity",
                    f"{roe*100:.1f}%",
                    (
                        "Excellent"
                        if roe > 0.2
                        else "Good" if roe > 0.15 else "Fair" if roe > 0.1 else "Poor"
                    ),
                )

                profit_margin = financial.get("profit_margin", 0)
                fin_table.add_row(
                    "Profit Margin",
                    f"{profit_margin*100:.1f}%",
                    (
                        "High"
                        if profit_margin > 0.2
                        else (
                            "Good"
                            if profit_margin > 0.1
                            else "Fair" if profit_margin > 0.05 else "Low"
                        )
                    ),
                )

                self.console.print(fin_table)

            # Risk Assessment Details
            risk = analysis.get("risk_assessment", {})
            if risk and "error" not in risk:
                risk_table = Table(title=" Risk Assessment Details", box=box.SIMPLE)
                risk_table.add_column("Risk Factor", style="white")
                risk_table.add_column("Level", style="yellow")
                risk_table.add_column("Value", style="cyan", justify="right")

                risk_table.add_row(
                    "Market Risk (Beta)",
                    risk.get("market_risk", "Medium"),
                    f"{risk.get('beta', 1.0):.2f}",
                )
                risk_table.add_row(
                    "Volatility Risk",
                    risk.get("volatility_risk", "Medium"),
                    f"{risk.get('volatility_52w', 0)*100:.1f}%",
                )
                risk_table.add_row(
                    "Liquidity Risk", risk.get("liquidity_risk", "Medium"), "-"
                )
                risk_table.add_row(
                    "Credit Risk", risk.get("credit_risk", "Medium"), "-"
                )
                risk_table.add_row(
                    "Max Drawdown",
                    "Historical",
                    f"{risk.get('max_drawdown', 0)*100:.1f}%",
                )

                self.console.print(risk_table)

            # Technical Analysis Details
            technical = analysis.get("technical_analysis", {})
            if technical and "error" not in technical:
                tech_table = Table(title=" Technical Analysis Details", box=box.SIMPLE)
                tech_table.add_column("Indicator", style="white")
                tech_table.add_column("Value", style="green", justify="right")
                tech_table.add_column("Signal", style="cyan")

                indicators = technical.get("indicators", {})
                if indicators:
                    rsi = indicators.get("rsi", 50)
                    tech_table.add_row(
                        "RSI",
                        f"{rsi:.1f}",
                        (
                            "Overbought"
                            if rsi > 70
                            else "Oversold" if rsi < 30 else "Neutral"
                        ),
                    )

                    macd = indicators.get("macd", 0)
                    macd_signal = indicators.get("macd_signal", 0)
                    tech_table.add_row(
                        "MACD",
                        f"{macd:.3f}",
                        "Bullish" if macd > macd_signal else "Bearish",
                    )

                tech_table.add_row(
                    "Trend Direction", technical.get("trend_direction", "Neutral"), "-"
                )
                tech_table.add_row(
                    "Momentum Score", f"{technical.get('momentum_score', 0)}/100", "-"
                )

                # Support and Resistance
                support_levels = technical.get("support_levels", [])
                resistance_levels = technical.get("resistance_levels", [])

                if support_levels:
                    tech_table.add_row(
                        "Support Levels",
                        ", ".join([f"${level:.2f}" for level in support_levels[-3:]]),
                        "-",
                    )
                if resistance_levels:
                    tech_table.add_row(
                        "Resistance Levels",
                        ", ".join(
                            [f"${level:.2f}" for level in resistance_levels[-3:]]
                        ),
                        "-",
                    )

                self.console.print(tech_table)

            # Market Intelligence
            market_intel = analysis.get("market_intelligence", {})
            if market_intel and "error" not in market_intel:
                intel_table = Table(title=" Market Intelligence", box=box.SIMPLE)
                intel_table.add_column("Factor", style="white")
                intel_table.add_column("Value", style="cyan")

                intel_table.add_row(
                    "Market Cap Category",
                    market_intel.get("market_cap_category", "Unknown"),
                )
                intel_table.add_row(
                    "Market Sentiment", market_intel.get("market_sentiment", "Neutral")
                )
                intel_table.add_row(
                    "Institutional Ownership",
                    f"{market_intel.get('institutional_ownership', 0)*100:.1f}%",
                )
                intel_table.add_row(
                    "Insider Ownership",
                    f"{market_intel.get('insider_ownership', 0)*100:.1f}%",
                )

                price_targets = market_intel.get("price_targets", {})
                if price_targets.get("mean", 0) > 0:
                    intel_table.add_row(
                        "Analyst Target (Mean)", f"${price_targets['mean']:.2f}"
                    )
                    intel_table.add_row(
                        "Target Range",
                        f"${price_targets.get('low', 0):.2f} - ${price_targets.get('high', 0):.2f}",
                    )

                self.console.print(intel_table)

        except Exception as e:
            self.console.print(f"[red] Error displaying detailed sections: {e}[/]")

    def _get_status_emoji(self, score):
        """Get status emoji based on score"""
        if score >= 80:
            return " Excellent"
        elif score >= 70:
            return " Good"
        elif score >= 60:
            return " Fair"
        elif score >= 50:
            return " Poor"
        else:
            return " Critical"

    def print_ai_analysis(self, analysis):
        """Print AI-powered company analysis"""
        try:
            if not analysis or "error" in analysis:
                self.console.print("[red] No AI analysis available[/]")
                if "error" in analysis:
                    self.console.print(f"[yellow]Error: {analysis['error']}[/]")
                return

            symbol = analysis.get("symbol", "Unknown")
            company_name = analysis.get("company_name", symbol)

            # Company Header with AI badge
            self.console.print(
                Panel(
                    f"[bold white]{company_name}[/] ([cyan]{symbol}[/]) [bright_magenta] AI-Powered[/]\n"
                    f"[white]Sector: {analysis.get('sector', 'Unknown')} | Industry: {analysis.get('industry', 'Unknown')}[/]",
                    title="[bold blue] AI Company Analysis[/]",
                    border_style="blue",
                )
            )

            # Overall AI Score and Recommendation
            overall_score = analysis.get("overall_score", 50)
            recommendation = analysis.get("recommendation", "HOLD")
            ai_confidence = analysis.get("ai_confidence", 0.85)
            rec_color = (
                "green"
                if "BUY" in recommendation
                else "red" if "SELL" in recommendation else "yellow"
            )

            self.console.print(
                Panel(
                    f"[bold white]AI Investment Score: {overall_score:.1f}/100[/]\n"
                    f"[bold white]AI Recommendation: [{rec_color}]{recommendation}[/]\n"
                    f"[white]AI Confidence: {ai_confidence*100:.0f}%[/]",
                    title="[bold green] AI Investment Summary[/]",
                    border_style="green",
                )
            )

            # AI Sentiment Analysis
            ai_sentiment = analysis.get("ai_sentiment", {})
            if ai_sentiment:
                sentiment = ai_sentiment.get("sentiment", "neutral")
                confidence = ai_sentiment.get("confidence", 0.5)
                reasoning = ai_sentiment.get("reasoning", "No reasoning available")

                sentiment_color = (
                    "green"
                    if sentiment == "positive"
                    else "red" if sentiment == "negative" else "yellow"
                )

                self.console.print(
                    Panel(
                        f"[white]Market Sentiment: [{sentiment_color}]{sentiment.upper()}[/]\n"
                        f"[white]Confidence: {confidence*100:.0f}%[/]\n"
                        f"[white]Analysis: {reasoning}[/]",
                        title="[bold magenta] AI Sentiment Analysis[/]",
                        border_style="magenta",
                    )
                )

            # AI Analysis
            ai_analysis = analysis.get("ai_analysis", {})
            if ai_analysis:
                # AI Insights
                ai_insights = ai_analysis.get("insights", [])
                if ai_insights:
                    insights_text = "\n".join(
                        [f"• {insight}" for insight in ai_insights[:3]]
                    )
                    self.console.print(
                        Panel(
                            insights_text,
                            title="[bold cyan] AI Investment Insights[/]",
                            border_style="cyan",
                        )
                    )

                # AI Risks
                ai_risks = ai_analysis.get("risks", [])
                if ai_risks:
                    risks_text = "\n".join([f"• {risk}" for risk in ai_risks[:3]])
                    self.console.print(
                        Panel(
                            risks_text,
                            title="[bold red] AI Risk Analysis[/]",
                            border_style="red",
                        )
                    )

                # AI Opportunities
                ai_opportunities = ai_analysis.get("opportunities", [])
                if ai_opportunities:
                    opps_text = "\n".join([f"• {opp}" for opp in ai_opportunities[:3]])
                    self.console.print(
                        Panel(
                            opps_text,
                            title="[bold green] AI Growth Opportunities[/]",
                            border_style="green",
                        )
                    )

            # AI Risk Assessment
            ai_risk = analysis.get("ai_risk_assessment", {})
            if ai_risk and "error" not in ai_risk:
                risk_category = ai_risk.get("primary_risk_category", "moderate risk")
                risk_confidence = ai_risk.get("risk_confidence", 0.5)
                risk_summary = ai_risk.get("risk_summary", "Risk assessment completed")

                risk_color = (
                    "red"
                    if "high" in risk_category
                    else "yellow" if "moderate" in risk_category else "green"
                )

                self.console.print(
                    Panel(
                        f"[white]Risk Category: [{risk_color}]{risk_category.upper()}[/]\n"
                        f"[white]Confidence: {risk_confidence*100:.0f}%[/]\n"
                        f"[white]Summary: {risk_summary}[/]",
                        title="[bold red] AI Risk Assessment[/]",
                        border_style="red",
                    )
                )

            # AI Growth Analysis
            ai_growth = analysis.get("ai_growth_potential", {})
            if ai_growth and "error" not in ai_growth:
                growth_category = ai_growth.get("growth_category", "moderate growth")
                growth_confidence = ai_growth.get("growth_confidence", 0.5)
                growth_reasoning = ai_growth.get(
                    "growth_reasoning", "Growth analysis completed"
                )

                growth_color = (
                    "green"
                    if "high" in growth_category
                    else "yellow" if "moderate" in growth_category else "red"
                )

                self.console.print(
                    Panel(
                        f"[white]Growth Potential: [{growth_color}]{growth_category.upper()}[/]\n"
                        f"[white]Confidence: {growth_confidence*100:.0f}%[/]\n"
                        f"[white]Analysis: {growth_reasoning}[/]",
                        title="[bold green] AI Growth Analysis[/]",
                        border_style="green",
                    )
                )

            # Financial Metrics Summary
            financial_metrics = analysis.get("financial_metrics", {})
            if financial_metrics and "error" not in financial_metrics:
                metrics_table = Table(title=" Key Financial Metrics", box=box.SIMPLE)
                metrics_table.add_column("Metric", style="white")
                metrics_table.add_column("Value", style="cyan", justify="right")
                metrics_table.add_column("Trend", style="green")

                current_price = financial_metrics.get("current_price", 0)
                price_1y = financial_metrics.get("price_change_1y", 0)
                volatility = financial_metrics.get("volatility_annual", 0)
                pe_ratio = financial_metrics.get("pe_ratio", 0)

                metrics_table.add_row("Current Price", f"${current_price:.2f}", "-")
                metrics_table.add_row(
                    "1-Year Return",
                    f"{price_1y:.1f}%",
                    "" if price_1y > 0 else "" if price_1y < 0 else "",
                )
                metrics_table.add_row(
                    "Annual Volatility",
                    f"{volatility:.1f}%",
                    (
                        "High"
                        if volatility > 30
                        else "Moderate" if volatility > 15 else "Low"
                    ),
                )
                if pe_ratio > 0:
                    metrics_table.add_row(
                        "P/E Ratio",
                        f"{pe_ratio:.1f}",
                        "High" if pe_ratio > 25 else "Fair" if pe_ratio > 15 else "Low",
                    )

                self.console.print(metrics_table)

            # Technical Signals
            technical_signals = analysis.get("technical_signals", {})
            if technical_signals and "error" not in technical_signals:
                tech_table = Table(title=" Technical Signals", box=box.SIMPLE)
                tech_table.add_column("Signal", style="white")
                tech_table.add_column("Value", style="cyan")
                tech_table.add_column("Status", style="green")

                trend = technical_signals.get("trend", "neutral")
                rsi = technical_signals.get("rsi", 50)
                momentum = technical_signals.get("momentum", "neutral")

                trend_color = (
                    "green"
                    if trend == "bullish"
                    else "red" if trend == "bearish" else "yellow"
                )
                tech_table.add_row(
                    "Trend Direction",
                    f"[{trend_color}]{trend.upper()}[/]",
                    "" if trend == "bullish" else "" if trend == "bearish" else "",
                )
                tech_table.add_row("RSI", f"{rsi:.1f}", momentum.upper())

                self.console.print(tech_table)

            # Market Position
            market_position = analysis.get("market_position", {})
            if market_position and "error" not in market_position:
                self.console.print(
                    Panel(
                        f"[white]Size Category: {market_position.get('size_category', 'Unknown')}[/]\n"
                        f"[white]Market Position: {market_position.get('position_strength', 'Unknown')}[/]\n"
                        f"[white]Competitive Position: {market_position.get('competitive_position', 'Unknown')}[/]",
                        title="[bold blue] Market Position[/]",
                        border_style="blue",
                    )
                )

            # AI Recommendation Details
            ai_analysis = analysis.get("ai_analysis", {})
            if ai_analysis:
                ai_recommendation = ai_analysis.get("recommendation", {})
                if ai_recommendation:
                    rec_action = ai_recommendation.get("action", "HOLD")
                    rec_reasoning = ai_recommendation.get(
                        "reasoning", "No detailed reasoning available"
                    )
                    rec_confidence = ai_recommendation.get("confidence", 0.5)

                    rec_color = (
                        "green"
                        if "BUY" in rec_action
                        else "red" if "SELL" in rec_action else "yellow"
                    )

                    self.console.print(
                        Panel(
                            f"[white]AI Recommendation: [{rec_color}]{rec_action}[/]\n"
                            f"[white]Confidence: {rec_confidence*100:.0f}%[/]\n"
                            f"[white]Reasoning: {rec_reasoning}[/]",
                            title="[bold yellow] AI Recommendation Details[/]",
                            border_style="yellow",
                        )
                    )

                # AI Price Targets
                ai_price_targets = ai_analysis.get("price_targets", {})
                if ai_price_targets:
                    self.console.print(
                        Panel(
                            f"[white]Bull Case: ${ai_price_targets.get('bull_case', 0):.2f}[/]\n"
                            f"[white]Base Case: ${ai_price_targets.get('base_case', 0):.2f}[/]\n"
                            f"[white]Bear Case: ${ai_price_targets.get('bear_case', 0):.2f}[/]\n"
                            f"[white]Time Horizon: 12 months[/]",
                            title="[bold magenta] AI Price Targets[/]",
                            border_style="magenta",
                        )
                    )

        except Exception as e:
            self.console.print(f"[red] Error displaying AI analysis: {e}[/]")

    def print_ml_predictions(self, result):
        """Print ML prediction results with enhanced formatting"""
        try:
            if not result:
                self.console.print("[red] No ML results to display[/]")
                return

            symbol = result.get("symbol", "Unknown")
            current_price = result.get("current_price", 0)
            predictions = result.get("predictions", [])
            accuracy = result.get("model_accuracy", 0)
            model_type = result.get("model_type", "Unknown")

            # Header with accuracy
            self.console.print(
                Panel(
                    f"[bold white]{symbol} ML Predictions[/]\n"
                    f"[white]Model Type: {model_type}[/]\n"
                    f"[green]Model Accuracy: {accuracy:.1f}%[/]\n"
                    f"[cyan]Current Price: ${current_price:.2f}[/]",
                    title="[bold blue] Machine Learning Analysis[/]",
                    border_style="blue",
                )
            )

            if predictions:
                # Create ML predictions table
                table = Table(
                    title=" ML Price Predictions",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold blue",
                )

                table.add_column("Day", style="cyan", no_wrap=True)
                table.add_column("Date", style="white")
                table.add_column("Predicted Price", style="green", justify="right")
                table.add_column("Change", style="yellow", justify="right")
                table.add_column("Return %", style="magenta", justify="right")
                table.add_column("Confidence", style="blue", justify="right")

                for pred in predictions:
                    day = pred.get("day", 0)
                    date = pred.get("date", "Unknown")
                    price = pred.get("predicted_price", 0)
                    predicted_return = pred.get("predicted_return", 0) * 100
                    confidence = pred.get("confidence", 0)

                    # Calculate change from current price
                    change = price - current_price
                    (change / current_price) * 100

                    # Format change with colors
                    change_color = "green" if change >= 0 else "red"
                    return_color = "green" if predicted_return >= 0 else "red"

                    # Confidence color
                    if confidence >= 0.8:
                        conf_color = "green"
                        conf_icon = ""
                    elif confidence >= 0.6:
                        conf_color = "yellow"
                        conf_icon = ""
                    else:
                        conf_color = "red"
                        conf_icon = ""

                    table.add_row(
                        f"Day {day}",
                        date[:10],  # Show only date part
                        f"${price:.2f}",
                        f"[{change_color}]${change:+.2f}[/]",
                        f"[{return_color}]{predicted_return:+.2f}%[/]",
                        f"[{conf_color}]{conf_icon} {confidence:.1%}[/]",
                    )

                self.console.print(table)

            # Model performance summary
            training_samples = result.get("training_samples", 0)
            if training_samples > 0:
                # Accuracy rating
                if accuracy >= 90:
                    rating = " Excellent"
                    rating_color = "bright_green"
                elif accuracy >= 80:
                    rating = "⭐ Very Good"
                    rating_color = "green"
                elif accuracy >= 70:
                    rating = " Good"
                    rating_color = "cyan"
                elif accuracy >= 60:
                    rating = " Fair"
                    rating_color = "yellow"
                else:
                    rating = " Poor"
                    rating_color = "red"

                self.console.print(
                    Panel(
                        f"[white]Training Samples: {training_samples:,}[/]\n"
                        f"[white]Model Accuracy: {accuracy:.1f}%[/]\n"
                        f"[{rating_color}]Performance Rating: {rating}[/]",
                        title="[bold green] Model Performance[/]",
                        border_style="green",
                    )
                )

        except Exception as e:
            self.console.print(f"[red] Error displaying ML results: {e}[/]")

    def print_header(self, title):
        """Print a formatted header"""
        self.console.print(
            Panel(
                f"[bold white]{title}[/]",
                title="[bold blue]ARA AI Analysis[/]",
                border_style="blue",
            )
        )

    def print_ultimate_predictions(self, result):
        """Print ultimate ML prediction results with all features"""
        try:
            if not result:
                self.console.print("[red] No ultimate results to display[/]")
                return

            symbol = result.get("symbol", "Unknown")
            current_price = result.get("current_price", 0)
            predictions = result.get("predictions", [])
            accuracy = result.get("model_accuracy", 0)
            model_type = result.get("model_type", "Unknown")
            feature_count = result.get("feature_count", 0)
            market_status = result.get("market_status", {})
            sector_info = result.get("sector_info", {})
            hf_sentiment = result.get("hf_sentiment", {})

            # Header with comprehensive info
            market_emoji = "" if market_status.get("is_open") else ""
            market_text = "OPEN" if market_status.get("is_open") else "CLOSED"

            self.console.print(
                Panel(
                    f"[bold white]{symbol} ULTIMATE ML Predictions[/]\n"
                    f"[white]Model: {model_type}[/]\n"
                    f"[green]Accuracy: {accuracy:.1f}% | Features: {feature_count} | Models: 8[/]\n"
                    f"[cyan]Current Price: ${current_price:.2f}[/]\n"
                    f"[white]Market Status: {market_emoji} {market_text}[/]",
                    title="[bold blue] ULTIMATE Machine Learning Analysis[/]",
                    border_style="blue",
                )
            )

            # Sector information
            if sector_info.get("sector") != "Unknown":
                sector = sector_info.get("sector", "Unknown")
                industry = sector_info.get("industry", "Unknown")
                market_cap = sector_info.get("market_cap", 0)

                if market_cap > 10_000_000_000:
                    cap_category = "Large Cap"
                elif market_cap > 2_000_000_000:
                    cap_category = "Mid Cap"
                else:
                    cap_category = "Small Cap"

                self.console.print(
                    Panel(
                        f"[white]Sector: [cyan]{sector}[/]\n"
                        f"[white]Industry: [cyan]{industry}[/]\n"
                        f"[white]Market Cap: [green]${market_cap:,.0f}[/] ([yellow]{cap_category}[/])",
                        title="[bold cyan] Company Information[/]",
                        border_style="cyan",
                    )
                )

            # Hugging Face sentiment analysis
            if hf_sentiment and "error" not in hf_sentiment:
                sentiment_label = hf_sentiment.get("label", "NEUTRAL")
                sentiment_score = hf_sentiment.get("score", 0.5)

                if sentiment_label == "POSITIVE":
                    sentiment_color = "green"
                    sentiment_emoji = ""
                elif sentiment_label == "NEGATIVE":
                    sentiment_color = "red"
                    sentiment_emoji = ""
                else:
                    sentiment_color = "yellow"
                    sentiment_emoji = ""

                self.console.print(
                    Panel(
                        f"[{sentiment_color}]{sentiment_emoji} Sentiment: {sentiment_label}[/]\n"
                        f"[white]Confidence: {sentiment_score:.1%}[/]",
                        title="[bold magenta] AI Sentiment Analysis[/]",
                        border_style="magenta",
                    )
                )

            # Financial health analysis
            financial_health = result.get("financial_health", {})
            if financial_health and "error" not in financial_health:
                health_score = financial_health.get("health_score", 60)
                health_grade = financial_health.get("health_grade", "C")
                risk_grade = financial_health.get("risk_grade", "Moderate Risk")

                # Color coding for grades
                if health_grade.startswith("A"):
                    health_color = "green"
                elif health_grade.startswith("B"):
                    health_color = "yellow"
                elif health_grade.startswith("C"):
                    health_color = "orange"
                else:
                    health_color = "red"

                if "Low" in risk_grade:
                    risk_color = "green"
                elif "Moderate" in risk_grade:
                    risk_color = "yellow"
                else:
                    risk_color = "red"

                self.console.print(
                    Panel(
                        f"[white]Financial Health: [{health_color}]{health_grade}[/] ([white]{health_score:.0f}/100[/])\n"
                        f"[white]Risk Assessment: [{risk_color}]{risk_grade}[/]\n"
                        f"[white]Debt/Equity: [cyan]{financial_health.get('debt_to_equity', 0):.2f}[/]\n"
                        f"[white]Current Ratio: [cyan]{financial_health.get('current_ratio', 0):.2f}[/]",
                        title="[bold green] Financial Analysis[/]",
                        border_style="green",
                    )
                )

            if predictions:
                # Create ultimate predictions table
                table = Table(
                    title=" ULTIMATE ML Price Predictions",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold blue",
                )

                table.add_column("Day", style="cyan", no_wrap=True)
                table.add_column("Date", style="white")
                table.add_column("Predicted Price", style="green", justify="right")
                table.add_column("Change", style="yellow", justify="right")
                table.add_column("Return %", style="magenta", justify="right")
                table.add_column("Confidence", style="blue", justify="right")

                for pred in predictions:
                    day = pred.get("day", 0)
                    date = pred.get("date", "Unknown")
                    price = pred.get("predicted_price", 0)
                    predicted_return = pred.get("predicted_return", 0) * 100
                    confidence = pred.get("confidence", 0)

                    # Calculate change from current price
                    change = price - current_price
                    (change / current_price) * 100

                    # Format change with colors
                    change_color = "green" if change >= 0 else "red"
                    return_color = "green" if predicted_return >= 0 else "red"

                    # Enhanced confidence indicators
                    if confidence >= 0.95:
                        conf_color = "bright_green"
                        conf_icon = ""
                    elif confidence >= 0.85:
                        conf_color = "green"
                        conf_icon = ""
                    elif confidence >= 0.75:
                        conf_color = "yellow"
                        conf_icon = ""
                    elif confidence >= 0.65:
                        conf_color = "orange"
                        conf_icon = ""
                    else:
                        conf_color = "red"
                        conf_icon = ""

                    table.add_row(
                        f"Day {day}",
                        date,
                        f"${price:.2f}",
                        f"[{change_color}]${change:+.2f}[/]",
                        f"[{return_color}]{predicted_return:+.2f}%[/]",
                        f"[{conf_color}]{conf_icon} {confidence:.1%}[/]",
                    )

                self.console.print(table)

            # Ultimate model performance summary
            training_samples = result.get("training_samples", 0)

            # Enhanced accuracy rating
            if accuracy >= 98:
                rating = " EXCEPTIONAL"
                rating_color = "bright_green"
            elif accuracy >= 95:
                rating = "⭐ EXCELLENT"
                rating_color = "green"
            elif accuracy >= 90:
                rating = " VERY GOOD"
                rating_color = "cyan"
            elif accuracy >= 85:
                rating = " GOOD"
                rating_color = "yellow"
            else:
                rating = " FAIR"
                rating_color = "orange"

            self.console.print(
                Panel(
                    f"[white]Training Samples: {training_samples:,}[/]\n"
                    f"[white]Model Accuracy: {accuracy:.1f}%[/]\n"
                    f"[white]Feature Engineering: {feature_count} advanced features[/]\n"
                    f"[white]Model Ensemble: 8 ML algorithms[/]\n"
                    f"[{rating_color}]Performance Rating: {rating}[/]",
                    title="[bold green] ULTIMATE Model Performance[/]",
                    border_style="green",
                )
            )

            # Market timing information
            if market_status:
                current_time = market_status.get("current_time", "Unknown")
                next_open = market_status.get("next_open", "Unknown")
                next_close = market_status.get("next_close", "Unknown")

                if market_status.get("is_open"):
                    timing_info = f"[green]Market Open[/]\nNext Close: {next_close}"
                else:
                    timing_info = f"[red]Market Closed[/]\nNext Open: {next_open}"

                self.console.print(
                    Panel(
                        f"[white]Current Time: {current_time}[/]\n{timing_info}",
                        title="[bold yellow]⏰ Market Timing[/]",
                        border_style="yellow",
                    )
                )

        except Exception as e:
            self.console.print(f"[red] Error displaying ultimate results: {e}[/]")

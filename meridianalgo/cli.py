"""
Command-line interface for MeridianAlgo package
"""

import argparse
import sys
from .core import AraAI
from .console import ConsoleManager


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MeridianAlgo - Advanced AI Stock Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ara AAPL                    # Predict AAPL for 5 days
  ara TSLA --days 7           # Predict TSLA for 7 days
  ara MSFT --verbose          # Predict with detailed output
  ara --validate              # Validate previous predictions
  ara --accuracy AAPL         # Show accuracy for AAPL
  ara --system-info           # Show system information
        """,
    )

    parser.add_argument(
        "symbol", nargs="?", help="Stock symbol to analyze (e.g., AAPL, TSLA, MSFT)"
    )

    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=5,
        help="Number of days to predict (default: 5)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cached predictions and generate new ones",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate previous predictions and show accuracy",
    )

    parser.add_argument(
        "--accuracy",
        metavar="SYMBOL",
        help="Show accuracy statistics for a specific symbol",
    )

    parser.add_argument(
        "--system-info", action="store_true", help="Show system and GPU information"
    )

    parser.add_argument(
        "--analyze",
        metavar="SYMBOL",
        help="Perform comprehensive company analysis for a symbol",
    )

    parser.add_argument(
        "--patterns-only",
        action="store_true",
        help="Show only chart pattern analysis (faster)",
    )

    parser.add_argument(
        "--ai-analysis",
        metavar="SYMBOL",
        help="Perform AI-powered company analysis (lightweight)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="ARA AI 2.2.0-Beta - ULTIMATE ML System with Realistic Predictions",
    )

    args = parser.parse_args()

    # Initialize console manager
    console = ConsoleManager(verbose=args.verbose)

    try:
        # Handle system info request
        if args.system_info:
            show_system_info(console)
            return 0

        # Handle validation request
        if args.validate:
            validate_predictions(console, args.verbose)
            return 0

        # Handle accuracy request
        if args.accuracy:
            show_accuracy(console, args.accuracy)
            return 0

        # Handle company analysis request
        if args.analyze:
            analyze_company_cli(console, args.analyze, args.verbose)
            return 0

        # Handle AI analysis request
        if args.ai_analysis:
            ai_analyze_company_cli(console, args.ai_analysis, args.verbose)
            return 0

        # Handle stock prediction
        if args.symbol:
            predict_stock_cli(
                console,
                args.symbol,
                args.days,
                not args.no_cache,
                args.verbose,
                not args.patterns_only,
            )
            return 0

        # No arguments provided, show help
        parser.print_help()
        return 1

    except KeyboardInterrupt:
        console.print_warning("Operation cancelled by user")
        return 1
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        return 1


def predict_stock_cli(console, symbol, days, use_cache, verbose, include_analysis=True):
    """Handle stock prediction via CLI"""
    try:
        console.print_info(f"Analyzing {symbol.upper()} for {days} days...")

        # Initialize Ara AI
        ara = AraAI(verbose=verbose)

        # Make prediction with enhanced analysis
        result = ara.predict(
            symbol.upper(),
            days=days,
            use_cache=use_cache,
            include_analysis=include_analysis,
        )

        if result:
            console.print_prediction_results(result)
        else:
            console.print_error(f"Failed to generate predictions for {symbol.upper()}")

    except Exception as e:
        console.print_error(f"Prediction failed: {e}")


def analyze_company_cli(console, symbol, verbose):
    """Handle comprehensive company analysis via CLI"""
    try:
        console.print_info(f"Performing comprehensive analysis for {symbol.upper()}...")

        # Initialize Ara AI
        ara = AraAI(verbose=verbose)

        # Perform company analysis
        analysis = ara.company_analyzer.analyze_company(symbol.upper())

        if analysis and "error" not in analysis:
            console.print_company_analysis(analysis)
        else:
            error_msg = (
                analysis.get("error", "Unknown error")
                if analysis
                else "Analysis failed"
            )
            console.print_error(f"Company analysis failed: {error_msg}")

    except Exception as e:
        console.print_error(f"Company analysis failed: {e}")


def ai_analyze_company_cli(console, symbol, verbose):
    """Handle AI-powered company analysis via CLI"""
    try:
        console.print_info(
            f"Performing comprehensive AI analysis for {symbol.upper()}..."
        )
        console.print_warning(
            "Note: AI analysis takes longer but provides detailed insights"
        )

        # Initialize Ara AI
        ara = AraAI(verbose=verbose)

        # Perform comprehensive AI analysis
        analysis = ara.analyze_with_ai(symbol.upper())

        if analysis and "error" not in analysis:
            console.print_ai_analysis(analysis)
        else:
            error_msg = (
                analysis.get("error", "Unknown error")
                if analysis
                else "AI analysis failed"
            )
            console.print_error(f"AI analysis failed: {error_msg}")
            console.print_info(
                "Try using regular analysis: python ara.py " + symbol.upper()
            )

    except Exception as e:
        console.print_error(f"AI analysis failed: {e}")
        console.print_info(
            "Try using regular analysis: python ara.py " + symbol.upper()
        )


def validate_predictions(console, verbose):
    """Handle prediction validation via CLI"""
    try:
        console.print_info("Validating previous predictions...")

        # Initialize Ara AI
        ara = AraAI(verbose=verbose)

        # Validate predictions
        result = ara.validate_predictions()

        if result:
            console.print_validation_summary(result)
        else:
            console.print_warning("No predictions available for validation")

    except Exception as e:
        console.print_error(f"Validation failed: {e}")


def show_accuracy(console, symbol):
    """Show accuracy statistics for a symbol"""
    try:
        console.print_info(f"Analyzing accuracy for {symbol.upper()}...")

        # Initialize Ara AI
        ara = AraAI()

        # Get accuracy stats
        result = ara.analyze_accuracy(symbol.upper())

        if result:
            console.print_accuracy_summary(result)
        else:
            console.print_warning(f"No accuracy data available for {symbol.upper()}")

    except Exception as e:
        console.print_error(f"Accuracy analysis failed: {e}")


def show_system_info(console):
    """Show system information"""
    try:
        from . import get_version_info

        # Initialize Ara AI to get system info
        ara = AraAI()
        system_info = ara.get_system_info()

        # Print version info
        version_info = get_version_info()
        console.console.print(
            f"\n[bold green]MeridianAlgo v{version_info['version']}[/]"
        )

        # Print features
        console.console.print("\n[bold blue]Features:[/]")
        for feature in version_info["features"]:
            console.console.print(f"   {feature}")

        # Print GPU info
        gpu_info = system_info.get("gpu_info", {})
        console.print_gpu_info(gpu_info)

        # Print device info
        device = system_info.get("device", "Unknown")
        console.console.print(f"\n[bold cyan]Current Device:[/] {device}")

        # Print model info
        model_info = system_info.get("model_info", {})
        if model_info:
            console.console.print("\n[bold blue]ML Models:[/]")
            for model in model_info.get("models", []):
                console.console.print(f"   {model}")

        # Print cache stats
        cache_stats = system_info.get("cache_stats", {})
        if cache_stats:
            console.console.print("\n[bold blue]Cache Statistics:[/]")
            console.console.print(
                f"   Total Predictions: {cache_stats.get('total_predictions', 0)}"
            )
            console.console.print(f"   Symbols: {cache_stats.get('symbols', 0)}")
            console.console.print(
                f"   File Size: {cache_stats.get('file_size', 0)} bytes"
            )

        # Print accuracy stats
        accuracy_stats = system_info.get("accuracy_stats", {})
        if accuracy_stats:
            console.console.print("\n[bold blue]Overall Accuracy:[/]")
            console.console.print(
                f"   Accuracy Rate: {accuracy_stats.get('accuracy_rate', 0):.1f}%"
            )
            console.console.print(
                f"   Total Predictions: {accuracy_stats.get('total_predictions', 0)}"
            )
            console.console.print(
                f"   Average Error: {accuracy_stats.get('avg_error', 0):.2f}%"
            )

    except Exception as e:
        console.print_error(f"Failed to get system info: {e}")


if __name__ == "__main__":
    sys.exit(main())

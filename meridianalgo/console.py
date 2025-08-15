"""
Console management and rich output formatting
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
import warnings
warnings.filterwarnings('ignore')

class ConsoleManager:
    """Enhanced console manager with rich formatting"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = Console()
    
    def print_system_info(self):
        """Print system initialization information"""
        if not self.verbose:
            return
        
        self.console.print(Panel(
            "[bold green]🚀 Ara AI Stock Analysis Platform[/]\n"
            "[white]Enhanced with ensemble ML models and intelligent caching[/]",
            title="[bold blue]System Initialization[/]",
            border_style="blue"
        ))
    
    def print_gpu_info(self, gpu_info):
        """Print GPU information"""
        if not self.verbose:
            return
        
        if gpu_info['details']:
            gpu_text = "\n".join([f"✅ {detail}" for detail in gpu_info['details']])
        else:
            gpu_text = "❌ No GPU acceleration available"
        
        self.console.print(Panel(
            gpu_text,
            title="[bold cyan]GPU Acceleration[/]",
            border_style="cyan"
        ))
    
    def print_prediction_results(self, result):
        """Print prediction results in a formatted table"""
        try:
            if not result:
                self.console.print("[red]❌ No prediction results to display[/]")
                return
            
            symbol = result.get('symbol', 'Unknown')
            current_price = result.get('current_price', 0)
            predictions = result.get('predictions', [])
            
            # Create main table
            table = Table(
                title=f"📈 {symbol} Stock Predictions",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold blue"
            )
            
            table.add_column("Day", style="cyan", no_wrap=True)
            table.add_column("Date", style="white")
            table.add_column("Predicted Price", style="green", justify="right")
            table.add_column("Change", style="yellow", justify="right")
            table.add_column("Change %", style="magenta", justify="right")
            
            for pred in predictions:
                change_color = "green" if pred.get('change', 0) >= 0 else "red"
                change_pct_color = "green" if pred.get('change_pct', 0) >= 0 else "red"
                
                table.add_row(
                    f"Day {pred.get('day', 1)}",
                    pred.get('date', '').split('T')[0],  # Remove time part
                    f"${pred.get('predicted_price', 0):.2f}",
                    f"[{change_color}]${pred.get('change', 0):+.2f}[/]",
                    f"[{change_pct_color}]{pred.get('change_pct', 0):+.1f}%[/]"
                )
            
            self.console.print(table)
            
            # Print current price info
            self.console.print(f"\n[bold white]📊 Current Price: [green]${current_price:.2f}[/]")
            
            # Print cache info if available
            if result.get('cached'):
                cache_age = result.get('cache_age', 'Unknown')
                self.console.print(f"[yellow]💾 Using cached predictions (Age: {cache_age})[/]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error displaying results: {e}[/]")
    
    def print_accuracy_summary(self, accuracy_stats):
        """Print accuracy summary"""
        try:
            if not accuracy_stats:
                self.console.print("[yellow]⚠️  No accuracy data available[/]")
                return
            
            symbol = accuracy_stats.get('symbol', 'All')
            total = accuracy_stats.get('total_predictions', 0)
            accuracy_rate = accuracy_stats.get('accuracy_rate', 0)
            excellent_rate = accuracy_stats.get('excellent_rate', 0)
            good_rate = accuracy_stats.get('good_rate', 0)
            avg_error = accuracy_stats.get('avg_error', 0)
            
            # Create accuracy table
            table = Table(
                title=f"📊 Accuracy Statistics - {symbol}",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold blue"
            )
            
            table.add_column("Metric", style="white")
            table.add_column("Value", style="green", justify="right")
            
            table.add_row("Total Predictions", str(total))
            table.add_row("Overall Accuracy", f"{accuracy_rate:.1f}%")
            table.add_row("Excellent (<1% error)", f"{excellent_rate:.1f}%")
            table.add_row("Good (<2% error)", f"{good_rate:.1f}%")
            table.add_row("Average Error", f"{avg_error:.2f}%")
            
            # Add recent stats if available
            recent_stats = accuracy_stats.get('recent_stats', {})
            if recent_stats:
                table.add_row("", "")  # Separator
                table.add_row("Recent (30d) Total", str(recent_stats.get('total', 0)))
                table.add_row("Recent (30d) Accuracy", f"{recent_stats.get('accuracy_rate', 0):.1f}%")
                table.add_row("Recent (30d) Avg Error", f"{recent_stats.get('avg_error', 0):.2f}%")
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]❌ Error displaying accuracy: {e}[/]")
    
    def print_validation_summary(self, validation_result):
        """Print validation summary"""
        try:
            if not validation_result:
                self.console.print("[yellow]⚠️  No validation results available[/]")
                return
            
            validated = validation_result.get('validated', 0)
            accuracy_rate = validation_result.get('accuracy_rate', 0)
            excellent_rate = validation_result.get('excellent_rate', 0)
            good_rate = validation_result.get('good_rate', 0)
            avg_error = validation_result.get('avg_error', 0)
            
            self.console.print(Panel(
                f"[green]✅ Validated: {validated} predictions[/]\n"
                f"[cyan]📈 Accuracy Rate: {accuracy_rate:.1f}%[/]\n"
                f"[bright_green]🎯 Excellent (<1%): {excellent_rate:.1f}%[/]\n"
                f"[green]✅ Good (<2%): {good_rate:.1f}%[/]\n"
                f"[white]📉 Average Error: {avg_error:.2f}%[/]",
                title="[bold blue]Validation Summary[/]",
                border_style="blue"
            ))
            
        except Exception as e:
            self.console.print(f"[red]❌ Error displaying validation: {e}[/]")
    
    def print_error(self, message):
        """Print error message"""
        self.console.print(f"[red]❌ {message}[/]")
    
    def print_warning(self, message):
        """Print warning message"""
        self.console.print(f"[yellow]⚠️  {message}[/]")
    
    def print_success(self, message):
        """Print success message"""
        self.console.print(f"[green]✅ {message}[/]")
    
    def print_info(self, message):
        """Print info message"""
        self.console.print(f"[cyan]ℹ️  {message}[/]")
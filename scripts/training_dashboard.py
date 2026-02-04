#!/usr/bin/env python3
"""
Training Dashboard - Monitor training sessions and performance
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def get_training_stats(db_file="training.db"):
    """Get training statistics from database"""
    if not Path(db_file).exists():
        return None

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get total models trained
    cursor.execute("SELECT COUNT(*) FROM model_metadata")
    total_models = cursor.fetchone()[0]

    # Get recent trainings (last 24 hours)
    cursor.execute(
        """
        SELECT COUNT(*) FROM model_metadata 
        WHERE training_date >= datetime('now', '-1 day')
    """
    )
    recent_trainings = cursor.fetchone()[0]

    # Get average accuracy
    cursor.execute("SELECT AVG(accuracy) FROM model_metadata WHERE accuracy > 0")
    avg_accuracy = cursor.fetchone()[0] or 0

    # Get average loss
    cursor.execute("SELECT AVG(loss) FROM model_metadata WHERE loss > 0")
    avg_loss = cursor.fetchone()[0] or 0

    # Get unique symbols
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM model_metadata")
    unique_symbols = cursor.fetchone()[0]

    # Get latest trainings
    cursor.execute(
        """
        SELECT symbol, model_type, training_date, accuracy, loss, epochs
        FROM model_metadata
        ORDER BY training_date DESC
        LIMIT 10
    """
    )
    latest_trainings = cursor.fetchall()

    conn.close()

    return {
        "total_models": total_models,
        "recent_trainings": recent_trainings,
        "avg_accuracy": avg_accuracy,
        "avg_loss": avg_loss,
        "unique_symbols": unique_symbols,
        "latest_trainings": latest_trainings,
    }


def display_dashboard():
    """Display training dashboard"""
    console.clear()

    # Header
    console.print(
        Panel.fit(
            "[bold cyan]ARA AI Training Dashboard[/bold cyan]\n"
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan",
        )
    )

    # Get stats
    stats = get_training_stats()

    if not stats:
        console.print("[yellow]No training data found. Run training first![/yellow]")
        return

    # Summary stats
    summary_table = Table(show_header=False, box=box.ROUNDED, border_style="green")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green bold")

    summary_table.add_row("Total Models Trained", str(stats["total_models"]))
    summary_table.add_row("Trainings (24h)", str(stats["recent_trainings"]))
    summary_table.add_row("Unique Symbols", str(stats["unique_symbols"]))
    summary_table.add_row("Avg Accuracy", f"{stats['avg_accuracy']:.4f}")
    summary_table.add_row("Avg Loss", f"{stats['avg_loss']:.6f}")

    console.print("\n")
    console.print(Panel(summary_table, title="[bold]Training Summary[/bold]", border_style="green"))

    # Latest trainings
    if stats["latest_trainings"]:
        console.print("\n")
        trainings_table = Table(
            title="Latest Training Sessions", box=box.ROUNDED, border_style="blue"
        )
        trainings_table.add_column("Symbol", style="cyan")
        trainings_table.add_column("Type", style="magenta")
        trainings_table.add_column("Date", style="yellow")
        trainings_table.add_column("Accuracy", style="green")
        trainings_table.add_column("Loss", style="red")
        trainings_table.add_column("Epochs", style="blue")

        for training in stats["latest_trainings"]:
            symbol, model_type, date, accuracy, loss, epochs = training
            trainings_table.add_row(
                symbol,
                model_type,
                date[:19] if date else "N/A",
                f"{accuracy:.4f}" if accuracy else "N/A",
                f"{loss:.6f}" if loss else "N/A",
                str(epochs) if epochs else "N/A",
            )

        console.print(trainings_table)

    # Training schedule info
    console.print("\n")
    schedule_info = Table(show_header=False, box=box.ROUNDED, border_style="yellow")
    schedule_info.add_column("Schedule", style="yellow")
    schedule_info.add_column("Details", style="white")

    schedule_info.add_row("Elite Hourly", "24 times per day (every hour) with enhanced performance")
    schedule_info.add_row("Manual", "Run: python scripts/quick_train.py")

    console.print(
        Panel(
            schedule_info,
            title="[bold]Training Schedules[/bold]",
            border_style="yellow",
        )
    )

    console.print("\n")


def main():
    """Main function"""
    try:
        display_dashboard()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()

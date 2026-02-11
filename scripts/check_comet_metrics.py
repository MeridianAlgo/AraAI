import os
from datetime import datetime

from comet_ml import API
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()


def get_latest_metric(metrics, name_pattern):
    relevant = [m for m in metrics if name_pattern == m["metricName"].lower()]
    if not relevant:
        return "N/A"
    # Sort by step or timestamp to get latest
    latest = sorted(
        relevant, key=lambda x: (x.get("step", 0), x.get("timestamp", 0)), reverse=True
    )[0]
    return float(latest["metricValue"])


def check_comet_runs():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        console.print("[bold red]COMET_API_KEY not found in .env[/bold red]")
        return

    api = API(api_key=api_key)
    workspace = "meridianalgo"
    projects = ["ara-ai-stock", "ara-ai-forex"]

    for project in projects:
        console.print(f"\n[bold cyan]Project: {project}[/bold cyan]")
        experiments = api.get_experiments(workspace, project)
        if not experiments:
            console.print(f"[yellow]No experiments found for {project}[/yellow]")
            continue

        # Sort by start time descending
        experiments.sort(key=lambda x: x.get_metadata().get("startTimeMillis", 0), reverse=True)

        table = Table(title=f"Latest 5 Runs for {project}")
        table.add_column("Run Name", style="magenta")
        table.add_column("Date", style="yellow")
        table.add_column("Final Loss", style="green")
        table.add_column("Accuracy", style="blue")
        table.add_column("Dir Acc", style="cyan")
        table.add_column("Status", style="bold")

        for exp in experiments[:5]:
            meta = exp.get_metadata()
            name = meta.get("experimentName", "Unnamed")
            start_time = meta.get("startTimeMillis")
            date_str = (
                datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%d %H:%M")
                if start_time
                else "N/A"
            )

            metrics = exp.get_metrics()

            final_loss = get_latest_metric(metrics, "final_loss")
            accuracy = get_latest_metric(metrics, "accuracy")
            dir_acc = get_latest_metric(metrics, "direction_accuracy")

            # Formatting
            loss_str = f"{final_loss:.6f}" if isinstance(final_loss, float) else "N/A"
            acc_str = f"{accuracy:.2f}%" if isinstance(accuracy, float) else "N/A"
            dir_str = f"{dir_acc:.2f}%" if isinstance(dir_acc, float) else "N/A"
            status = (
                "[green]COMPLETED[/green]"
                if not meta.get("running")
                else "[yellow]RUNNING[/yellow]"
            )

            table.add_row(name, date_str, loss_str, acc_str, dir_str, status)

        console.print(table)


if __name__ == "__main__":
    check_comet_runs()

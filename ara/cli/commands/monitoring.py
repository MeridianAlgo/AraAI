"""
Monitoring and observability CLI commands
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ara.monitoring import get_all_dashboards, export_dashboard_to_file
from ara.utils.monitoring import get_metrics

console = Console()


@click.group()
def monitoring():
    """Monitoring and observability commands"""
    pass


@monitoring.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./dashboards",
    help="Output directory for dashboard files",
)
def export_dashboards(output_dir):
    """Export Grafana dashboards to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Exporting Grafana Dashboards[/bold cyan]\n")

    dashboards = get_all_dashboards()
    dashboard_names = [
        "system_metrics",
        "prediction_accuracy",
        "model_performance",
        "api_performance",
    ]

    for name, dashboard in zip(dashboard_names, dashboards):
        output_file = output_path / f"{name}.json"
        export_dashboard_to_file(dashboard, str(output_file))
        console.print(f"  ✓ Exported [green]{name}.json[/green]")

    console.print(f"\n[bold green]✓ All dashboards exported to:[/bold green] {output_path}\n")

    # Print import instructions
    panel = Panel(
        "[bold]Import to Grafana:[/bold]\n\n"
        "1. Open Grafana UI (http://localhost:3000)\n"
        "2. Go to Dashboards → Import\n"
        "3. Upload the JSON files\n"
        "4. Select Prometheus data source\n"
        "5. Click Import",
        title="Next Steps",
        border_style="cyan",
    )
    console.print(panel)


@monitoring.command()
def show_metrics():
    """Show current system metrics"""
    console.print("\n[bold cyan]Current System Metrics[/bold cyan]\n")

    metrics = get_metrics()

    # Counters table
    if metrics.get("counters"):
        table = Table(title="Counters", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        for name, value in metrics["counters"].items():
            table.add_row(name, str(value))

        console.print(table)
        console.print()

    # Gauges table
    if metrics.get("gauges"):
        table = Table(title="Gauges", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        for name, value in metrics["gauges"].items():
            table.add_row(name, f"{value:.2f}")

        console.print(table)
        console.print()

    # Timings table
    if metrics.get("timings"):
        table = Table(title="Timings (ms)", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")

        for name, timing in metrics["timings"].items():
            table.add_row(
                name,
                str(timing["count"]),
                f"{timing['mean']:.2f}",
                f"{timing['p50']:.2f}",
                f"{timing['p95']:.2f}",
                f"{timing['p99']:.2f}",
            )

        console.print(table)
        console.print()


@monitoring.command()
def health_check():
    """Check system health"""
    import requests

    console.print("\n[bold cyan]System Health Check[/bold cyan]\n")

    try:
        # Try to connect to API
        response = requests.get("http://localhost:8000/health/detailed", timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Overall status
            status_color = "green" if data["status"] == "healthy" else "yellow"
            console.print(
                f"[bold]Status:[/bold] [{status_color}]{data['status'].upper()}[/{status_color}]"
            )
            console.print(f"[bold]Version:[/bold] {data['version']}")
            console.print()

            # Components table
            table = Table(title="Components", show_header=True, header_style="bold magenta")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="dim")

            for component, info in data.get("components", {}).items():
                status = info.get("status", "unknown")
                details = []
                for key, value in info.items():
                    if key != "status":
                        details.append(f"{key}: {value}")

                table.add_row(component, status, ", ".join(details) if details else "-")

            console.print(table)
            console.print()

            # System resources
            if "system" in data:
                sys_info = data["system"]
                table = Table(
                    title="System Resources",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Resource", style="cyan")
                table.add_column("Usage", justify="right", style="yellow")

                table.add_row("CPU", f"{sys_info.get('cpu_percent', 0):.1f}%")
                table.add_row("Memory", f"{sys_info.get('memory_percent', 0):.1f}%")
                table.add_row("Disk", f"{sys_info.get('disk_percent', 0):.1f}%")

                console.print(table)
                console.print()

            # Performance
            if "performance" in data:
                perf = data["performance"]
                table = Table(title="Performance", show_header=True, header_style="bold magenta")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", justify="right", style="green")

                for key, value in perf.items():
                    table.add_row(key.replace("_", " ").title(), f"{value:.2f} ms")

                console.print(table)
                console.print()

        else:
            console.print(f"[bold red]✗ API returned status code {response.status_code}[/bold red]")

    except requests.exceptions.ConnectionError:
        console.print("[bold red]✗ Cannot connect to API[/bold red]")
        console.print("[dim]Make sure the API is running on http://localhost:8000[/dim]\n")
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]\n")


@monitoring.command()
def setup():
    """Show monitoring stack setup instructions"""
    console.print("\n[bold cyan]ARA AI Monitoring Stack Setup[/bold cyan]\n")

    instructions = """
[bold]1. Start the monitoring stack:[/bold]
   cd ara/monitoring
   docker-compose -f docker-compose.monitoring.yml up -d

[bold]2. Access the services:[/bold]
   • Prometheus:   http://localhost:9090
   • Grafana:      http://localhost:3000 (admin/admin)
   • Jaeger:       http://localhost:16686
   • Alertmanager: http://localhost:9093

[bold]3. Export Grafana dashboards:[/bold]
   ara monitoring export-dashboards

[bold]4. Enable distributed tracing (optional):[/bold]
   export OTLP_ENDPOINT=http://localhost:4317
   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

[bold]5. Enable error tracking (optional):[/bold]
   export SENTRY_DSN=your-sentry-dsn
   pip install sentry-sdk

[bold]6. Configure alerts:[/bold]
   Edit ara/monitoring/alertmanager.yml with your email/Slack settings
   Restart: docker-compose restart alertmanager

[bold]7. View metrics:[/bold]
   Open Grafana → Dashboards → ARA AI
    """

    panel = Panel(instructions, title="Setup Instructions", border_style="cyan", padding=(1, 2))
    console.print(panel)

    console.print("\n[dim]For more information, see ara/monitoring/README.md[/dim]\n")


if __name__ == "__main__":
    monitoring()

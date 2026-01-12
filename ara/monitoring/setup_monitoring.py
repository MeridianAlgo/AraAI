#!/usr/bin/env python3
"""
Setup script for ARA AI monitoring stack
Exports Grafana dashboards and provides setup instructions
"""

import json
from pathlib import Path
from ara.monitoring.grafana_dashboards import get_all_dashboards


def export_dashboards():
    """Export all Grafana dashboards to JSON files"""
    output_dir = Path(__file__).parent / "dashboards"
    output_dir.mkdir(exist_ok=True)

    dashboards = get_all_dashboards()
    dashboard_names = [
        "system_metrics",
        "prediction_accuracy",
        "model_performance",
        "api_performance",
    ]

    print("Exporting Grafana dashboards...")
    for name, dashboard in zip(dashboard_names, dashboards):
        output_file = output_dir / f"{name}.json"
        with open(output_file, "w") as f:
            json.dump(dashboard, f, indent=2)
        print(f"  ✓ Exported {name}.json")

    print(f"\nDashboards exported to: {output_dir}")
    return output_dir


def create_grafana_provisioning():
    """Create Grafana provisioning configuration"""
    provisioning_dir = Path(__file__).parent / "grafana-provisioning"
    datasources_dir = provisioning_dir / "datasources"
    dashboards_dir = provisioning_dir / "dashboards"

    # Create directories
    datasources_dir.mkdir(parents=True, exist_ok=True)
    dashboards_dir.mkdir(parents=True, exist_ok=True)

    # Create datasource configuration
    datasource_config = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": "http://prometheus:9090",
                "isDefault": True,
                "editable": True,
            }
        ],
    }

    with open(datasources_dir / "prometheus.yml", "w") as f:
        json.dump(datasource_config, f, indent=2)

    # Create dashboard provider configuration
    dashboard_provider = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "ARA AI Dashboards",
                "orgId": 1,
                "folder": "ARA AI",
                "type": "file",
                "disableDeletion": False,
                "updateIntervalSeconds": 10,
                "allowUiUpdates": True,
                "options": {"path": "/etc/grafana/provisioning/dashboards"},
            }
        ],
    }

    with open(dashboards_dir / "dashboards.yml", "w") as f:
        json.dump(dashboard_provider, f, indent=2)

    print(f"\nGrafana provisioning created in: {provisioning_dir}")
    return provisioning_dir


def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "=" * 70)
    print("ARA AI Monitoring Stack Setup")
    print("=" * 70)

    print("\n1. Start the monitoring stack:")
    print("   cd ara/monitoring")
    print("   docker-compose -f docker-compose.monitoring.yml up -d")

    print("\n2. Access the services:")
    print("   - Prometheus:  http://localhost:9090")
    print("   - Grafana:     http://localhost:3000 (admin/admin)")
    print("   - Jaeger:      http://localhost:16686")
    print("   - Alertmanager: http://localhost:9093")

    print("\n3. Configure ARA AI to export metrics:")
    print("   The API automatically exports metrics at /health/metrics")
    print("   Prometheus will scrape this endpoint every 15 seconds")

    print("\n4. Enable distributed tracing (optional):")
    print("   export OTLP_ENDPOINT=http://localhost:4317")
    print(
        "   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
    )

    print("\n5. Enable error tracking (optional):")
    print("   export SENTRY_DSN=your-sentry-dsn")
    print("   pip install sentry-sdk")

    print("\n6. Import Grafana dashboards:")
    print("   - Dashboards are auto-provisioned from grafana-provisioning/")
    print("   - Or manually import from dashboards/ directory")

    print("\n7. Configure alerts:")
    print("   - Edit alertmanager.yml with your email/Slack settings")
    print("   - Restart alertmanager: docker-compose restart alertmanager")

    print("\n8. View metrics:")
    print("   - Open Grafana at http://localhost:3000")
    print("   - Navigate to Dashboards → ARA AI")
    print("   - Select a dashboard to view")

    print("\n" + "=" * 70)
    print("For more information, see ara/monitoring/README.md")
    print("=" * 70 + "\n")


def main():
    """Main setup function"""
    print("Setting up ARA AI monitoring stack...\n")

    # Export dashboards
    export_dashboards()

    # Create Grafana provisioning
    create_grafana_provisioning()

    # Print instructions
    print_setup_instructions()

    print("Setup complete! Follow the instructions above to start monitoring.")


if __name__ == "__main__":
    main()

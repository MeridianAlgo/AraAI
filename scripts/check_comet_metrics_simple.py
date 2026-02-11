import os
from datetime import datetime

from comet_ml import API
from dotenv import load_dotenv


def check_comet_runs_simple():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    api = API(api_key=api_key)
    workspace = "meridianalgo"
    projects = ["ara-ai-stock", "ara-ai-forex"]

    for project in projects:
        print(f"\nProject: {project}")
        experiments = api.get_experiments(workspace, project)
        if not experiments:
            print(f"No experiments found for {project}")
            continue

        experiments.sort(key=lambda x: x.get_metadata().get("startTimeMillis", 0), reverse=True)

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

            final_loss = "N/A"
            accuracy = "N/A"
            for m in metrics:
                if m["metricName"] == "final_loss":
                    final_loss = m["metricValue"]
                if m["metricName"] == "accuracy":
                    accuracy = m["metricValue"]

            print(f"  {name} | {date_str} | Loss: {final_loss} | Acc: {accuracy}")


if __name__ == "__main__":
    check_comet_runs_simple()

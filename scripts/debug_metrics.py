import os

from comet_ml import API
from dotenv import load_dotenv


def debug_metrics():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    api = API(api_key=api_key)
    workspace = "meridianalgo"
    project = "ara-ai-stock"

    experiments = api.get_experiments(workspace, project)
    experiments.sort(key=lambda x: x.get_metadata().get("startTimeMillis", 0), reverse=True)

    if experiments:
        exp = experiments[0]
        metrics = exp.get_metrics()
        unique_names = sorted(list(set([m["metricName"] for m in metrics])))
        print(f"Metrics for {exp.get_metadata().get('experimentName')}:")
        for name in unique_names:
            print(f" - {name}")


if __name__ == "__main__":
    debug_metrics()

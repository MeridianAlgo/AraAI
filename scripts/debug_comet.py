import os

from comet_ml import API
from dotenv import load_dotenv


def debug_comet():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    api = API(api_key=api_key)
    workspace = "meridianalgo"
    project = "ara-ai-stock"

    experiments = api.get_experiments(workspace, project)
    print(f"Total experiments in {project}: {len(experiments)}")

    if experiments:
        for exp in experiments[:2]:
            meta = exp.get_metadata()
            print(f"Exp: {meta.get('experimentName')}, Status: {meta.get('running')}")
            metrics = exp.get_metrics()
            print(f"Metric names: {[m['metricName'] for m in metrics[:5]]}")


if __name__ == "__main__":
    debug_comet()

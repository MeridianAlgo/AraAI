import os

from comet_ml import API
from dotenv import load_dotenv


def debug_loop():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    api = API(api_key=api_key)
    workspace = "meridianalgo"
    project = "ara-ai-stock"

    experiments = api.get_experiments(workspace, project)
    print(f"Project: {project}, Count: {len(experiments)}")

    for i, exp in enumerate(experiments[:3]):
        print(f"  Exp {i}: {exp.get_metadata().get('experimentName')}")


if __name__ == "__main__":
    debug_loop()

import os

from comet_ml import API
from dotenv import load_dotenv


def list_all():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    api = API(api_key=api_key)

    workspaces = api.get_workspaces()
    print(f"Workspaces: {workspaces}")

    for ws in workspaces:
        projects = api.get_projects(ws)
        print(f"Workspace: {ws}, Projects: {projects}")


if __name__ == "__main__":
    list_all()

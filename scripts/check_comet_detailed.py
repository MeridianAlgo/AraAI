import os
from datetime import datetime

from comet_ml import API
from dotenv import load_dotenv


def check_comet_detailed():
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    api = API(api_key=api_key)
    workspace = "meridianalgo"
    projects = ["meridian-ai-stocks", "meridian-ai-forex", "ara-ai-stock", "ara-ai-forex"]

    for project in projects:
        print(f"\n{'='*20} Project: {project} {'='*20}")
        experiments = api.get_experiments(workspace, project)
        if not experiments:
            print(f"No experiments found for {project}")
            continue

        experiments.sort(key=lambda x: x.get_metadata().get("startTimeMillis", 0), reverse=True)

        for exp in experiments[:3]:
            meta = exp.get_metadata()
            name = meta.get("experimentName", "Unnamed")
            start_time = meta.get("startTimeMillis")
            date_str = (
                datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%d %H:%M")
                if start_time
                else "N/A"
            )

            metrics = exp.get_metrics()

            # Dictionary to store latest values
            latest_metrics = {}
            for m in metrics:
                name_low = m["metricName"].lower()
                val = float(m["metricValue"])
                step = m.get("step")
                if step is None:
                    step = 0
                else:
                    step = int(step)

                if name_low not in latest_metrics or step > latest_metrics[name_low]["step"]:
                    latest_metrics[name_low] = {"val": val, "step": step}

            print(f"\nRun: {name} ({date_str})")
            print(f"  Status: {'Running' if meta.get('running') else 'Finished'}")

            # Key metrics to display
            key_metrics = ["train_loss", "val_loss", "direction_accuracy", "final_loss", "accuracy"]
            for km in key_metrics:
                if km in latest_metrics:
                    val = latest_metrics[km]["val"]
                    if "loss" in km:
                        print(f"  {km:18}: {val:.6f}")
                    else:
                        print(f"  {km:18}: {val:.2f}%")
                else:
                    # Try to find best match if not exact
                    match = [m for m in latest_metrics if km in m]
                    if match:
                        val = latest_metrics[match[0]]["val"]
                        print(f"  {km:18}: {val:.6f} (matched {match[0]})")
                    else:
                        pass  # Metric not found


if __name__ == "__main__":
    check_comet_detailed()

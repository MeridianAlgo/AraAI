import re
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.forex_ml import ForexML
from meridianalgo.unified_ml import UnifiedStockML
from scripts.evaluate_models import evaluate_model


def main():
    db_path = "data/training.db"

    # Evaluate Stock Model
    stock_acc_str = "N/A"
    try:
        stock_model = UnifiedStockML(model_path="models/Meridian.AI_Stocks.pt", model_type="stock")
        if stock_model.ml_system.is_trained():
            stock_acc = evaluate_model(stock_model, "stock", db_path, symbols_to_test=10)
            if stock_acc:
                stock_acc_str = f"{stock_acc['direction_accuracy']:.1f}%"
    except Exception as e:
        print(f"Error evaluating stock model: {e}")

    # Evaluate Forex Model
    forex_acc_str = "N/A"
    try:
        forex_model = ForexML(model_path="models/Meridian.AI_Forex.pt")
        if forex_model.ml_system.is_trained():
            forex_acc = evaluate_model(forex_model, "forex", db_path, symbols_to_test=10)
            if forex_acc:
                forex_acc_str = f"{forex_acc['direction_accuracy']:.1f}%"
    except Exception as e:
        print(f"Error evaluating forex model: {e}")

    print(f"\nFinal Results - Stock: {stock_acc_str}, Forex: {forex_acc_str}")

    # Update README
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text()

        # We need to replace the line starting with "| Validation Accuracy | "
        # We are using regex to replace whatever is currently there.
        new_content = re.sub(
            r"\| Validation Accuracy\s*\|[^|]*\|[^|]*\|",
            f"| Validation Accuracy | >{stock_acc_str} | >{forex_acc_str} |",
            content,
        )

        readme_path.write_text(new_content)
        print("Updated README.md with the latest testing accuracy.")


if __name__ == "__main__":
    main()

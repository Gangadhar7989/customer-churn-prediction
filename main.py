"""
Runs the full pipeline end to end:
  1. Load & clean data
  2. Run EDA and save charts
  3. Train the model and save metrics + charts
  4. Predict churn for a few sample customers
"""

from src.eda import run_eda
from src.train_model import train_and_evaluate
from src.predict import predict_sample


def main():
    print("Step 1/3: Running EDA...")
    run_eda()

    print("\nStep 2/3: Training model...")
    train_and_evaluate()

    print("\nStep 3/3: Sample predictions...")
    print(predict_sample())


if __name__ == "__main__":
    main()

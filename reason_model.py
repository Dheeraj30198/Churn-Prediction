import argparse
from pathlib import Path

from churn_model import load_data, train_reason_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn reason classification model.")
    parser.add_argument("--data-path", type=str, default="Telco_customer_churn.xlsx")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(Path(args.data_path))
    train_reason_model(
        df=df,
        artifact_dir=Path(args.artifact_dir),
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

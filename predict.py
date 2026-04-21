import argparse
from pathlib import Path

import joblib
import pandas as pd
from recommendations import recommend_next_action


def load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def save_tabular(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def align_input_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    missing = [c for c in expected_cols if c not in aligned.columns]
    for col in missing:
        aligned[col] = pd.NA
    return aligned[expected_cols]


def run_inference(
    model_path: Path,
    reason_model_path: Path | None,
    input_path: Path,
    output_path: Path,
    threshold: float,
) -> None:
    model = joblib.load(model_path)
    reason_model = None
    if reason_model_path is not None and reason_model_path.exists():
        reason_model = joblib.load(reason_model_path)

    data = load_tabular(input_path)

    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Loaded churn model does not expose expected input columns.")

    churn_input = align_input_columns(data, list(preprocessor.feature_names_in_))
    probs = model.predict_proba(churn_input)[:, 1]
    labels = (probs >= threshold).astype(int)

    result = data.copy()
    result["churn_probability"] = probs.round(6)
    result["predicted_churn_label"] = labels
    result["predicted_churn_text"] = result["predicted_churn_label"].map({1: "Yes", 0: "No"})
    result["predicted_churn_reason"] = "Not Applicable"

    if reason_model is not None and hasattr(reason_model, "named_steps"):
        reason_preprocessor = reason_model.named_steps.get("preprocessor")
        if reason_preprocessor is not None and hasattr(reason_preprocessor, "feature_names_in_"):
            reason_input = align_input_columns(data, list(reason_preprocessor.feature_names_in_))
            churn_idx = result["predicted_churn_label"] == 1
            if churn_idx.any():
                try:
                    reason_pred = reason_model.predict(reason_input.loc[churn_idx])
                    result.loc[churn_idx, "predicted_churn_reason"] = reason_pred
                except Exception:
                    result.loc[churn_idx, "predicted_churn_reason"] = "Unknown"

    result["recommended_next_action"] = [
        recommend_next_action(
            row=row._asdict(),
            churn_probability=float(prob),
            predicted_reason=str(reason),
        )
        for row, prob, reason in zip(result.itertuples(index=False), probs, result["predicted_churn_reason"])
    ]

    save_tabular(result, output_path)

    print(f"Rows scored: {len(result)}")
    print(f"Threshold: {threshold}")
    print(f"Output saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict churn probability and reason from input data.")
    parser.add_argument("--model-path", type=str, default="artifacts/models/churn_model.joblib")
    parser.add_argument("--reason-model-path", type=str, default="artifacts/models/reason_model.joblib")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="artifacts/predictions/predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=Path(args.model_path),
        reason_model_path=Path(args.reason_model_path) if args.reason_model_path else None,
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        threshold=args.threshold,
    )

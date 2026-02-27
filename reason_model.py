import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLUMNS = {"Churn Label", "Churn Value", "Churn Score", "CLTV"}
REASON_GROUP_MAP = {
    "Competitor had better devices": "Competitor",
    "Competitor made better offer": "Competitor",
    "Competitor offered more data": "Competitor",
    "Competitor offered higher download speeds": "Competitor",
    "Moved": "Relocation",
    "Price too high": "Pricing",
    "Extra data charges": "Pricing",
    "Long distance charges": "Pricing",
    "Product dissatisfaction": "Product/Service",
    "Service dissatisfaction": "Product/Service",
    "Network reliability": "Product/Service",
    "Poor expertise of online support": "Support Experience",
    "Attitude of support person": "Support Experience",
    "Lack of self-service on Website": "Digital Experience",
    "Limited range of services": "Product/Service",
}


def load_data(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    num_cols = x.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def prepare_reason_data(df: pd.DataFrame, grouped: bool) -> tuple[pd.DataFrame, pd.Series]:
    if "Churn Value" not in df.columns or "Churn Reason" not in df.columns:
        raise ValueError("Dataset must contain 'Churn Value' and 'Churn Reason'.")

    data = df.copy()
    for c in data.select_dtypes(include=["object", "string"]).columns:
        data[c] = data[c].astype(str).str.strip()

    if "Total Charges" in data.columns:
        data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")

    churned = data[(data["Churn Value"] == 1) & (data["Churn Reason"].notna())].copy()
    churned = churned[churned["Churn Reason"].astype(str).str.lower() != "nan"]

    y = churned["Churn Reason"]
    if grouped:
        y = y.map(lambda r: REASON_GROUP_MAP.get(r, "Other"))
    drop_cols = {"Churn Reason", "CustomerID"} | LEAKAGE_COLUMNS
    x = churned.drop(columns=[c for c in drop_cols if c in churned.columns], errors="ignore")

    reason_counts = y.value_counts()
    keep_reasons = reason_counts[reason_counts >= 20].index
    keep_mask = y.isin(keep_reasons)
    x = x[keep_mask]
    y = y[keep_mask]

    return x, y


def train_reason_model(args: argparse.Namespace) -> None:
    data = load_data(Path(args.data_path))
    x, y = prepare_reason_data(data, grouped=args.grouped_reasons)
    preprocessor = build_preprocessor(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=25,
                    min_samples_leaf=2,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)

    metrics = {
        "rows_used": int(len(x)),
        "num_reason_classes": int(y.nunique()),
        "grouped_reasons": bool(args.grouped_reasons),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "reason_distribution": y.value_counts().to_dict(),
    }

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Reason model saved to: {model_path}")
    print(f"Reason metrics saved to: {metrics_path}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Reason classes: {metrics['num_reason_classes']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn reason prediction model.")
    parser.add_argument("--data-path", type=str, default="Telco_customer_churn.xlsx")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="artifacts/models/churn_reason_model.joblib")
    parser.add_argument("--metrics-out", type=str, default="artifacts/reports/reason_metrics.json")
    parser.add_argument(
        "--grouped-reasons",
        action="store_true",
        help="Train on grouped reason categories for better accuracy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_reason_model(parse_args())

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")

LEAKAGE_COLUMNS = {
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "CLTV",
}


def load_data(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def run_eda(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    eda_summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_col,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if target_col in df.columns:
        target_counts = df[target_col].value_counts(dropna=False).to_dict()
        eda_summary["target_distribution"] = {str(k): int(v) for k, v in target_counts.items()}

    missing_df = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "missing_count"})
    )
    missing_df["missing_pct"] = (missing_df["missing_count"] / len(df) * 100).round(2)
    missing_df.to_csv(output_dir / "missing_values.csv", index=False)

    with open(output_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        df[target_col].value_counts(dropna=False).plot(kind="bar")
        plt.title("Target Distribution")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "target_distribution.png", dpi=150)
        plt.close()

    top_missing = missing_df.head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_missing, x="missing_pct", y="feature", hue="feature", legend=False)
    plt.title("Top Features by Missing Percentage")
    plt.xlabel("Missing %")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "missingness_top15.png", dpi=150)
    plt.close()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr_cols = numeric_cols[:20]
        corr = df[corr_cols].corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="RdBu_r", center=0)
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "numeric_correlation_heatmap.png", dpi=150)
        plt.close()

    if target_col in df.columns:
        for col in ["Contract", "Internet Service", "Payment Method", "Senior Citizen"]:
            if col in df.columns:
                churn_rate = (
                    df.groupby(col)[target_col]
                    .mean(numeric_only=True)
                    .sort_values(ascending=False)
                    .reset_index()
                )
                plt.figure(figsize=(10, 4))
                sns.barplot(data=churn_rate, x=col, y=target_col, hue=col, legend=False)
                plt.title(f"Average Churn by {col}")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                safe_name = col.lower().replace(" ", "_")
                plt.savefig(output_dir / f"churn_rate_by_{safe_name}.png", dpi=150)
                plt.close()


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    data = df.copy()
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype(str).str.strip()

    if "Total Charges" in data.columns:
        data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")

    if target_col == "Churn Label":
        y = data[target_col].map({"Yes": 1, "No": 0})
        if y.isna().any():
            raise ValueError("Churn Label must contain only 'Yes' and 'No'.")
    else:
        y = pd.to_numeric(data[target_col], errors="raise")

    drop_cols = {"CustomerID", target_col} | LEAKAGE_COLUMNS
    x = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")

    return x, y.astype(int)


def prepare_reason_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_cols = {"Churn Value", "Churn Reason"}
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns for reason model: {missing_cols}")

    data = df.copy()
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype("string").str.strip()

    if "Total Charges" in data.columns:
        data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")

    churn_numeric = pd.to_numeric(data["Churn Value"], errors="coerce")
    churned = data[churn_numeric == 1].copy()

    reasons = churned["Churn Reason"].astype("string").str.strip()
    valid_reason = reasons.notna() & (reasons != "") & (reasons.str.lower() != "nan")
    churned = churned[valid_reason].copy()

    if churned.empty:
        raise ValueError("No valid churned rows with non-empty 'Churn Reason' found.")

    y = churned["Churn Reason"].astype(str)
    drop_cols = {"CustomerID", "Churn Reason"} | LEAKAGE_COLUMNS
    x = churned.drop(columns=[col for col in drop_cols if col in churned.columns], errors="ignore")

    if y.nunique() < 2:
        raise ValueError("Reason model requires at least 2 reason classes.")

    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def build_candidates(preprocessor: ColumnTransformer, random_state: int) -> dict[str, object]:
    return {
        "lr": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "lr_balanced": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
            ]
        ),
        "rf_balanced": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lr_smote": ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "rf_smote": ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("model", RandomForestClassifier(n_estimators=250, random_state=random_state, n_jobs=-1)),
            ]
        ),
    }


def evaluate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def evaluate_reason_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def tune_model(base_model: object, model_name: str, random_state: int, n_iter: int) -> object:
    param_distributions = {}
    if "lr" in model_name:
        param_distributions = {
            "model__C": loguniform(1e-3, 1e2),
        }
    elif "rf" in model_name:
        param_distributions = {
            "model__n_estimators": randint(200, 900),
            "model__max_depth": randint(4, 30),
            "model__min_samples_split": randint(2, 25),
            "model__min_samples_leaf": randint(1, 10),
        }
    if "smote" in model_name:
        param_distributions["smote__k_neighbors"] = randint(3, 9)

    if not param_distributions:
        return base_model

    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    return tuner


def export_feature_importance(model: object, output_path: Path, top_n: int = 25) -> None:
    if not hasattr(model, "named_steps"):
        return

    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    importance = None
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = abs(estimator.coef_).ravel()

    if importance is None:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    importance_df.to_csv(output_path, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", legend=False)
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()


def train_reason_model(
    df: pd.DataFrame,
    artifact_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    model_dir = artifact_dir / "models"
    reports_dir = artifact_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    x, y = prepare_reason_features(df)

    reason_counts = y.value_counts()
    keep_classes = reason_counts[reason_counts >= 2].index
    keep_mask = y.isin(keep_classes)
    x = x[keep_mask]
    y = y[keep_mask]

    if y.nunique() < 2:
        raise ValueError("Reason model requires at least 2 classes with enough samples for split.")

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    preprocessor = build_preprocessor(x)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=25,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)
    metrics = evaluate_reason_model(pipeline, x_test, y_test)
    metrics["rows_used"] = int(len(x))
    metrics["num_reason_classes"] = int(y.nunique())
    metrics["reason_distribution"] = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}

    model_path = model_dir / "reason_model.joblib"
    metrics_path = reports_dir / "reason_metrics.json"

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Reason model saved to: {model_path}")
    print(f"Reason metrics saved to: {metrics_path}")
    print(f"Reason model accuracy: {metrics['accuracy']:.4f}")
    print(f"Reason model F1-macro: {metrics['f1_macro']:.4f}")

    return metrics


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    artifact_dir = Path(args.artifact_dir)
    model_dir = artifact_dir / "models"
    eda_dir = artifact_dir / "eda"
    reports_dir = artifact_dir / "reports"

    for directory in [artifact_dir, model_dir, eda_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_data(data_path)
    if not args.skip_eda:
        run_eda(df, args.target_col, eda_dir)

    x, y = prepare_features(df, args.target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(x)
    candidates = build_candidates(preprocessor, args.random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    cv_results: dict[str, dict] = {}

    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=args.cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )
        cv_results[name] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "f1_mean": float(scores["test_f1"].mean()),
            "precision_mean": float(scores["test_precision"].mean()),
            "recall_mean": float(scores["test_recall"].mean()),
        }
        print(
            f"{name} | ROC-AUC: {cv_results[name]['roc_auc_mean']:.4f} | "
            f"F1: {cv_results[name]['f1_mean']:.4f} | "
            f"Recall: {cv_results[name]['recall_mean']:.4f}"
        )

    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    best_model = candidates[best_name]
    best_model.fit(x_train, y_train)
    best_cv_score = cv_results[best_name]["roc_auc_mean"]

    final_model = best_model
    tuned = False
    best_params = {}

    if not args.skip_tuning:
        tuner = tune_model(best_model, best_name, args.random_state, args.tune_iters)
        if isinstance(tuner, RandomizedSearchCV):
            tuner.fit(x_train, y_train)
            print(f"Tuned {best_name} best CV ROC-AUC: {tuner.best_score_:.4f}")
            if tuner.best_score_ >= best_cv_score:
                final_model = tuner.best_estimator_
                tuned = True
                best_params = tuner.best_params_

    final_metrics = evaluate_model(final_model, x_test, y_test)
    final_metrics["best_model"] = best_name
    final_metrics["model_tuned"] = tuned
    final_metrics["best_params"] = best_params
    final_metrics["cv_results"] = cv_results
    final_metrics["class_distribution"] = y.value_counts().to_dict()

    model_path = model_dir / "churn_model.joblib"
    metrics_path = reports_dir / "metrics.json"
    cv_path = reports_dir / "cv_results.json"
    feature_path = reports_dir / "feature_importance.csv"

    joblib.dump(final_model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)
    export_feature_importance(final_model, feature_path)

    print(f"Best model: {best_name}")
    print(f"Model tuned: {tuned}")
    print(f"Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"Test F1: {final_metrics['f1_score']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"EDA output directory: {eda_dir}")

    if args.train_reason_model:
        try:
            train_reason_model(
                df=df,
                artifact_dir=artifact_dir,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        except Exception as exc:
            print(f"Reason model training skipped due to error: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn prediction workflow with EDA + model improvement.")
    parser.add_argument("--data-path", type=str, default="Telco_customer_churn.xlsx")
    parser.add_argument("--target-col", type=str, default="Churn Value")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune-iters", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--train-reason-model", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

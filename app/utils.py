import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from app.config import (
        CHURN_OUTPUT_COL,
        LOW_RISK_MAX,
        MEDIUM_RISK_MAX,
        PROBABILITY_COL,
        REASON_COL,
        RISK_COL,
    )
except ImportError:
    from config import (  # noqa: F401
        CHURN_OUTPUT_COL,
        LOW_RISK_MAX,
        MEDIUM_RISK_MAX,
        PROBABILITY_COL,
        REASON_COL,
        RISK_COL,
    )


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_models(churn_model_path: Path, reason_model_path: Path):
    if not churn_model_path.exists():
        raise FileNotFoundError(f"Churn model not found: {churn_model_path}")
    if not reason_model_path.exists():
        raise FileNotFoundError(f"Reason model not found: {reason_model_path}")
    churn_model = joblib.load(churn_model_path)
    reason_model = joblib.load(reason_model_path)
    return churn_model, reason_model


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def get_expected_columns(model) -> list[str]:
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError("Model does not expose expected input columns.")


def align_input_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in expected_cols:
        if col not in aligned.columns:
            aligned[col] = pd.NA
    aligned = aligned[expected_cols]
    return aligned


def to_risk_level(probabilities: pd.Series) -> pd.Series:
    bins = [-np.inf, LOW_RISK_MAX, MEDIUM_RISK_MAX, np.inf]
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    return pd.cut(probabilities, bins=bins, labels=labels, include_lowest=True)


def predict_dataframe(
    df: pd.DataFrame,
    churn_model,
    reason_model,
    threshold: float,
) -> pd.DataFrame:
    churn_expected = get_expected_columns(churn_model)
    churn_input = align_input_columns(df, churn_expected)

    # Ensure numeric features are numeric and convert invalid values to NaN for imputation
    numeric_columns = ["Count", "Zip Code", "Latitude", "Longitude", "Tenure Months", "Monthly Charges", "Total Charges"]
    for col in numeric_columns:
        if col in churn_input.columns:
            churn_input[col] = pd.to_numeric(churn_input[col], errors="coerce")

    probabilities = churn_model.predict_proba(churn_input)[:, 1]
    churn_pred = (probabilities >= threshold).astype(int)

    output = df.copy()
    output[CHURN_OUTPUT_COL] = churn_pred
    output[PROBABILITY_COL] = probabilities
    output[RISK_COL] = to_risk_level(output[PROBABILITY_COL]).astype(str)
    output[REASON_COL] = "Not Applicable"

    reason_expected = get_expected_columns(reason_model)
    reason_input = align_input_columns(df, reason_expected)
    churn_mask = output[CHURN_OUTPUT_COL] == 1

    if churn_mask.any():
        try:
            reason_pred = reason_model.predict(reason_input.loc[churn_mask])
            output.loc[churn_mask, REASON_COL] = pd.Series(reason_pred, index=output.index[churn_mask]).astype(str)
        except Exception:
            output.loc[churn_mask, REASON_COL] = "Unknown"

    output[PROBABILITY_COL] = output[PROBABILITY_COL].round(6)
    return output


def infer_ground_truth(df: pd.DataFrame) -> pd.Series | None:
    if "Churn Value" in df.columns:
        values = pd.to_numeric(df["Churn Value"], errors="coerce")
        valid = values.isin([0, 1])
        if valid.any():
            return values.where(valid).dropna().astype(int).reindex(df.index)
    if "Churn Label" in df.columns:
        mapped = df["Churn Label"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        if mapped.notna().any():
            return mapped.astype("float").dropna().astype(int).reindex(df.index)
    return None


def compute_roc_auc_if_available(y_true: pd.Series | None, probs: pd.Series) -> float | None:
    if y_true is None:
        return None
    valid = y_true.notna()
    y_valid = y_true[valid]
    p_valid = probs[valid]
    if y_valid.nunique() < 2:
        return None
    return float(roc_auc_score(y_valid, p_valid))


def get_top_reason(df: pd.DataFrame) -> str:
    reasons = df[REASON_COL]
    reasons = reasons[reasons != "Not Applicable"]
    if reasons.empty:
        return "Not Applicable"
    return str(reasons.value_counts().idxmax())


def extract_feature_importance(churn_model, top_n: int = 20) -> pd.DataFrame:
    if not hasattr(churn_model, "named_steps"):
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = churn_model.named_steps.get("preprocessor")
    estimator = churn_model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = None
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)

    if importance is None:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)
    return importance_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

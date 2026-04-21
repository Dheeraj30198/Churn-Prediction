import sys
sys.path.insert(0, '..')

from pathlib import Path

import streamlit as st

from config import (
    CHURN_MODEL_PATH,
    CHURN_OUTPUT_COL,
    METRICS_PATH,
    PROBABILITY_COL,
    REASON_COL,
    REASON_MODEL_PATH,
    RISK_COL,
)
from utils import (
    compute_roc_auc_if_available,
    dataframe_to_csv_bytes,
    extract_feature_importance,
    get_top_reason,
    infer_ground_truth,
    load_csv,
    load_json,
    load_models,
    predict_dataframe,
)
from visualization import (
    plot_churn_distribution,
    plot_confusion,
    plot_feature_importance,
    plot_reason_distribution,
)


st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("Churn Prediction Dashboard")


@st.cache_resource
def get_models(churn_model_path: Path, reason_model_path: Path):
    return load_models(churn_model_path, reason_model_path)


metrics = load_json(METRICS_PATH)

with st.sidebar:
    st.header("Inference Settings")
    threshold = st.slider("Churn Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.caption(f"Churn model: {CHURN_MODEL_PATH}")
    st.caption(f"Reason model: {REASON_MODEL_PATH}")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to run churn and churn-reason predictions.")
    st.stop()

try:
    churn_model, reason_model = get_models(CHURN_MODEL_PATH, REASON_MODEL_PATH)
except Exception as exc:
    st.error(str(exc))
    st.stop()

try:
    raw_df = load_csv(uploaded_file)
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

pred_df = predict_dataframe(raw_df, churn_model, reason_model, threshold)
y_true = infer_ground_truth(raw_df)
roc_auc_value = compute_roc_auc_if_available(y_true, pred_df[PROBABILITY_COL])

total_customers = int(len(pred_df))
predicted_churn_count = int(pred_df[CHURN_OUTPUT_COL].sum())
churn_rate = (predicted_churn_count / total_customers * 100) if total_customers > 0 else 0.0
top_reason = get_top_reason(pred_df)

kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Customers", f"{total_customers}")
kpi_cols[1].metric("Predicted Churn Count", f"{predicted_churn_count}")
kpi_cols[2].metric("Churn Rate %", f"{churn_rate:.2f}%")
kpi_cols[3].metric("Top Churn Reason", top_reason)
kpi_cols[4].metric("ROC-AUC", f"{roc_auc_value:.4f}" if roc_auc_value is not None else "N/A")

if metrics:
    best_model = metrics.get("best_model", "N/A")
    st.caption(f"Training best churn model: {best_model}")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.pyplot(plot_churn_distribution(pred_df), use_container_width=True)
with chart_col2:
    if y_true is not None and y_true.notna().any():
        valid = y_true.notna()
        st.pyplot(plot_confusion(y_true[valid].astype(int), pred_df.loc[valid, CHURN_OUTPUT_COL].astype(int)), use_container_width=True)
    else:
        st.info("Confusion matrix unavailable because ground truth churn labels are missing.")

chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    st.pyplot(plot_reason_distribution(pred_df), use_container_width=True)
with chart_col4:
    importance_df = extract_feature_importance(churn_model, top_n=20)
    st.pyplot(plot_feature_importance(importance_df), use_container_width=True)

st.subheader("Prediction Output")
st.dataframe(pred_df, use_container_width=True)

download_cols = [CHURN_OUTPUT_COL, PROBABILITY_COL, RISK_COL, REASON_COL]
csv_bytes = dataframe_to_csv_bytes(pred_df)
st.download_button(
    label="Download full prediction CSV",
    data=csv_bytes,
    file_name="predictions_with_reason.csv",
    mime="text/csv",
)

if all(col in pred_df.columns for col in download_cols):
    st.caption("Download includes Churn, Probability, Risk Level, and Predicted Reason columns.")

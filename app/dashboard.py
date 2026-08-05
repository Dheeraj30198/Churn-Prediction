import sys
from pathlib import Path

# Add both the repo root AND the app/ directory to sys.path
# so all imports resolve correctly on Streamlit Cloud
_repo_root = Path(__file__).resolve().parent.parent
_app_dir   = Path(__file__).resolve().parent
for _p in [str(_repo_root), str(_app_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    plot_probability_histogram,
    plot_risk_donut,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #0EA5E9 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 20px 60px rgba(124, 58, 237, 0.35);
}
.hero-banner h1 {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0.5rem 0 0 0;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.kpi-card {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.2);
}
.kpi-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #A78BFA;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E2E8F0;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(124, 58, 237, 0.4);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Upload area */
.upload-section {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 2px dashed rgba(124, 58, 237, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

/* Info box */
.info-box {
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #7DD3FC;
}

/* Risk badges */
.risk-high   { color: #F87171; font-weight: 600; }
.risk-medium { color: #FBBF24; font-weight: 600; }
.risk-low    { color: #34D399; font-weight: 600; }

/* Footer */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    color: #475569;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>📉 Churn Prediction Dashboard</h1>
    <p>AI-powered customer churn analysis · Upload your data · Get instant predictions</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Churn Probability Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Customers with churn probability above this threshold are classified as churners."
    )
    st.markdown("---")
    st.markdown("### 📁 Model Info")
    st.caption(f"🔵 Churn model: `{CHURN_MODEL_PATH.name}`")
    st.caption(f"🟣 Reason model: `{REASON_MODEL_PATH.name}`")
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
1. Download the **sample CSV** below  
2. Upload it using the uploader  
3. Adjust the threshold if needed  
4. Explore predictions & charts  
5. Download results as CSV  
    """)
    st.markdown("---")
    # Sample file download
    sample_path = Path(__file__).resolve().parent.parent / "sample_input_6_customers.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Sample CSV",
                data=f.read(),
                file_name="sample_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Model & Metrics Loading ────────────────────────────────────────────────────
@st.cache_resource
def get_models(churn_model_path: Path, reason_model_path: Path):
    return load_models(churn_model_path, reason_model_path)


metrics = load_json(METRICS_PATH)


# ── File Upload ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📤 Upload Customer Data</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Upload a <strong>CSV file</strong> with customer data to predict churn probability, risk level, and churn reason.
    Don't have a file? Download the <strong>sample CSV</strong> from the sidebar!
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    # Show a nice empty state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #475569;">
            <div style="font-size:4rem;">☁️</div>
            <div style="font-size:1.1rem; margin-top:0.5rem;">Waiting for data...</div>
            <div style="font-size:0.85rem; margin-top:0.3rem;">Upload a CSV file to get started</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ── Load Models ────────────────────────────────────────────────────────────────
try:
    churn_model, reason_model = get_models(CHURN_MODEL_PATH, REASON_MODEL_PATH)
except Exception as exc:
    st.error(f"❌ Failed to load models: {exc}")
    st.stop()

try:
    raw_df = load_csv(uploaded_file)
except Exception as exc:
    st.error(f"❌ Failed to read CSV: {exc}")
    st.stop()


# ── Run Predictions ────────────────────────────────────────────────────────────
with st.spinner("🔮 Running predictions..."):
    pred_df = predict_dataframe(raw_df, churn_model, reason_model, threshold)
    y_true = infer_ground_truth(raw_df)
    roc_auc_value = compute_roc_auc_if_available(y_true, pred_df[PROBABILITY_COL])

total_customers       = int(len(pred_df))
predicted_churn_count = int(pred_df[CHURN_OUTPUT_COL].sum())
churn_rate            = (predicted_churn_count / total_customers * 100) if total_customers > 0 else 0.0
top_reason            = get_top_reason(pred_df)
high_risk_count       = int((pred_df[RISK_COL] == "High Risk").sum())


# ── KPI Cards ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)

kpi_cols = st.columns(5)

kpi_data = [
    ("Total Customers", f"{total_customers:,}", "records uploaded"),
    ("Predicted Churn", f"{predicted_churn_count:,}", f"at threshold {threshold:.0%}"),
    ("Churn Rate", f"{churn_rate:.1f}%", "of all customers"),
    ("High Risk", f"{high_risk_count:,}", "customers > 70% prob"),
    ("ROC-AUC", f"{roc_auc_value:.4f}" if roc_auc_value is not None else "N/A", "model accuracy score"),
]

for col, (label, value, sub) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

if metrics:
    best_model = metrics.get("best_model", "N/A")
    st.caption(f"🏆 Best training model: **{best_model}**")


# ── Charts Row 1 ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Churn Analysis</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.plotly_chart(plot_churn_distribution(pred_df), use_container_width=True)

with chart_col2:
    st.plotly_chart(plot_risk_donut(pred_df), use_container_width=True)


# ── Charts Row 2 ───────────────────────────────────────────────────────────────
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.plotly_chart(plot_probability_histogram(pred_df), use_container_width=True)

with chart_col4:
    if y_true is not None and y_true.notna().any():
        valid = y_true.notna()
        st.plotly_chart(
            plot_confusion(
                y_true[valid].astype(int),
                pred_df.loc[valid, CHURN_OUTPUT_COL].astype(int)
            ),
            use_container_width=True
        )
    else:
        st.info("ℹ️ Confusion matrix unavailable — ground truth labels not found in uploaded CSV.")


# ── Charts Row 3 ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Deep Dive</div>', unsafe_allow_html=True)

chart_col5, chart_col6 = st.columns(2)

with chart_col5:
    st.plotly_chart(plot_reason_distribution(pred_df), use_container_width=True)

with chart_col6:
    importance_df = extract_feature_importance(churn_model, top_n=15)
    st.plotly_chart(plot_feature_importance(importance_df), use_container_width=True)


# ── Prediction Table ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)

display_cols = [CHURN_OUTPUT_COL, PROBABILITY_COL, RISK_COL, REASON_COL]
available_display = [c for c in display_cols if c in pred_df.columns]
other_cols = [c for c in pred_df.columns if c not in available_display]
pred_display = pred_df[available_display + other_cols]

st.dataframe(pred_display, use_container_width=True, height=350)

col_dl1, col_dl2 = st.columns([1, 3])
with col_dl1:
    csv_bytes = dataframe_to_csv_bytes(pred_df)
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit · 
    <a href="https://github.com/Dheeraj30198/Churn-Prediction" style="color:#7C3AED; text-decoration:none;">
        GitHub Repo
    </a>
</div>
""", unsafe_allow_html=True)

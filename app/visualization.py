import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

try:
    from app.config import CHURN_OUTPUT_COL, PROBABILITY_COL, REASON_COL, RISK_COL
except ImportError:
    from config import CHURN_OUTPUT_COL, PROBABILITY_COL, REASON_COL, RISK_COL

# ── Shared colour palette ──────────────────────────────────────────────────────
PURPLE   = "#7C3AED"
INDIGO   = "#4F46E5"
CYAN     = "#0EA5E9"
GREEN    = "#10B981"
AMBER    = "#F59E0B"
RED      = "#EF4444"
SLATE    = "#1E293B"

RISK_COLORS = {
    "High Risk":   RED,
    "Medium Risk": AMBER,
    "Low Risk":    GREEN,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#CBD5E1"),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def _apply_layout(fig, title: str) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color="#E2E8F0")), **PLOTLY_LAYOUT)
    return fig


# ── 1. Churn Distribution Bar ──────────────────────────────────────────────────
def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[CHURN_OUTPUT_COL].value_counts().sort_index()
    labels = {0: "No Churn", 1: "Churn"}

    fig = go.Figure(go.Bar(
        x=[labels.get(i, str(i)) for i in counts.index],
        y=counts.values,
        marker=dict(
            color=[GREEN, RED],
            line=dict(width=0),
        ),
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#E2E8F0"),
    ))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", showgrid=True)
    fig.update_xaxes(showgrid=False)
    return _apply_layout(fig, "📊 Predicted Churn Distribution")


# ── 2. Risk Level Donut ────────────────────────────────────────────────────────
def plot_risk_donut(df: pd.DataFrame) -> go.Figure:
    risk_counts = df[RISK_COL].value_counts()
    labels = risk_counts.index.tolist()
    values = risk_counts.values.tolist()
    colors = [RISK_COLORS.get(l, PURPLE) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0F0F1A", width=2)),
        textfont=dict(color="#E2E8F0"),
        hovertemplate="%{label}: %{value} customers<extra></extra>",
    ))
    fig.update_layout(
        annotations=[dict(
            text="Risk<br>Level",
            x=0.5, y=0.5,
            font=dict(size=14, color="#94A3B8"),
            showarrow=False,
        )],
        **PLOTLY_LAYOUT,
        title=dict(text="🎯 Customer Risk Distribution", font=dict(size=15, color="#E2E8F0")),
    )
    return fig


# ── 3. Probability Histogram ───────────────────────────────────────────────────
def plot_probability_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x=PROBABILITY_COL,
        nbins=30,
        color_discrete_sequence=[PURPLE],
        labels={PROBABILITY_COL: "Churn Probability"},
    )
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.add_vline(
        x=0.5, line_dash="dash", line_color=AMBER,
        annotation_text="Threshold 0.5",
        annotation_font_color=AMBER,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_xaxes(showgrid=False)
    return _apply_layout(fig, "📉 Churn Probability Distribution")


# ── 4. Confusion Matrix Heatmap ────────────────────────────────────────────────
def plot_confusion(y_true: pd.Series, y_pred: pd.Series) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["No Churn", "Churn"]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        colorscale=[[0, "#1A1A2E"], [1, PURPLE]],
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _apply_layout(fig, "🔲 Confusion Matrix")


# ── 5. Churn Reason Bar ────────────────────────────────────────────────────────
def plot_reason_distribution(df: pd.DataFrame) -> go.Figure:
    reasons = df[df[REASON_COL] != "Not Applicable"][REASON_COL]

    if reasons.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No churn reasons to display",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#64748B", size=14),
        )
        return _apply_layout(fig, "💬 Churn Reason Breakdown")

    reason_counts = reasons.value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]

    fig = px.bar(
        reason_counts,
        x="count", y="reason",
        orientation="h",
        color="count",
        color_continuous_scale=[[0, INDIGO], [1, CYAN]],
        labels={"count": "Customers", "reason": ""},
        text="count",
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#E2E8F0"), marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=False)
    return _apply_layout(fig, "💬 Churn Reason Breakdown")


# ── 6. Feature Importance ──────────────────────────────────────────────────────
def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    if importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Feature importance unavailable",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#64748B", size=14),
        )
        return _apply_layout(fig, "⚡ Top Feature Importances")

    df_sorted = importance_df.sort_values("importance")
    fig = px.bar(
        df_sorted,
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=[[0, PURPLE], [1, CYAN]],
        labels={"importance": "Importance Score", "feature": ""},
    )
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=False)
    return _apply_layout(fig, "⚡ Top Feature Importances")

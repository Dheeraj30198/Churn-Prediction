import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from app.config import CHURN_OUTPUT_COL, REASON_COL


sns.set_theme(style="whitegrid")


def plot_churn_distribution(df: pd.DataFrame):
    counts = df[CHURN_OUTPUT_COL].value_counts().sort_index()
    labels = {0: "No", 1: "Yes"}

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=[labels.get(i, str(i)) for i in counts.index], y=counts.values, ax=ax)
    ax.set_title("Predicted Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_confusion(y_true: pd.Series, y_pred: pd.Series):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"], rotation=0)
    fig.tight_layout()
    return fig


def plot_reason_distribution(df: pd.DataFrame):
    reasons = df[df[REASON_COL] != "Not Applicable"][REASON_COL]
    if reasons.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No predicted churn reasons", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    reason_counts = reasons.value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=reason_counts, x="reason", y="count", ax=ax)
    ax.set_title("Predicted Churn Reason Distribution")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    if importance_df.empty:
        ax.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
        ax.set_title("Churn Model Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig

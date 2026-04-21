from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"

CHURN_MODEL_PATH = MODEL_DIR / "churn_model.joblib"
REASON_MODEL_PATH = MODEL_DIR / "reason_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

CHURN_OUTPUT_COL = "Churn"
PROBABILITY_COL = "Probability"
RISK_COL = "Risk Level"
REASON_COL = "Predicted Reason"

LOW_RISK_MAX = 0.3
MEDIUM_RISK_MAX = 0.7

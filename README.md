# Churn Prediction (Subscription Services / Netflix-style)

This project uses the Telco churn dataset as a proxy to build a churn prediction pipeline for subscription-based businesses.

## 1) Environment setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run churn training workflow (EDA + modeling + tuning)

```powershell
.\.venv\Scripts\python churn_model.py --data-path Telco_customer_churn.xlsx --target-col "Churn Value" --artifact-dir artifacts
```

Outputs:
- `artifacts/eda/` (EDA charts + summary + missing-values table)
- `artifacts/models/churn_model.joblib`
- `artifacts/reports/metrics.json`
- `artifacts/reports/cv_results.json`
- `artifacts/reports/feature_importance.csv`
- `artifacts/reports/feature_importance.png`

## 3) Train churn-reason model

```powershell
.\.venv\Scripts\python reason_model.py --data-path Telco_customer_churn.xlsx --artifact-dir artifacts
```

Outputs:
- `artifacts/models/reason_model.joblib`
- `artifacts/reports/reason_metrics.json`

You can also train both models in one run:

```powershell
.\.venv\Scripts\python churn_model.py --data-path Telco_customer_churn.xlsx --target-col "Churn Value" --artifact-dir artifacts --train-reason-model
```

## 4) Predict on real-world input

```powershell
.\.venv\Scripts\python predict.py --input-path sample_real_world_input.csv --output-path artifacts/predictions/sample_predictions.csv --threshold 0.5
```

Output columns added:
- `churn_probability`
- `predicted_churn_label` (0/1)
- `predicted_churn_text` (No/Yes)
- `predicted_churn_reason`
- `recommended_next_action`

## 5) Launch dashboard

```powershell
streamlit run app/dashboard.py
```

Dashboard includes:
- CSV upload scoring
- Churn probability and risk segmentation
- Conditional churn reason prediction
- KPI cards and charts
- Downloadable prediction CSV

## 6) Optional flags

```powershell
# Skip EDA
.\.venv\Scripts\python churn_model.py --skip-eda

# Skip tuning
.\.venv\Scripts\python churn_model.py --skip-tuning

# Increase tuning budget
.\.venv\Scripts\python churn_model.py --tune-iters 40
```

## 7) Guided step-by-step run

- See `STEP_BY_STEP.md` for phase-wise commands.
- Use `run_pipeline_step_by_step.ps1` to run all phases sequentially.

# Churn Prediction (Subscription Services / Netflix-style)

This project uses the Telco churn dataset as a proxy to build a churn prediction pipeline for subscription-based businesses.

## 1) Environment setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run full workflow (EDA + modeling + tuning)

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

## 3) What is implemented right now

- EDA:
  - Target distribution
  - Missingness report
  - Numeric correlation heatmap
  - Churn-rate analysis by key categories
- Model building:
  - Logistic Regression, Random Forest
  - Class-weighted variants
  - SMOTE-based variants for class imbalance
- Model evaluation:
  - Cross-validation (`ROC-AUC`, `F1`, `Precision`, `Recall`)
  - Test metrics (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`)
  - Confusion matrix and classification report
- Model improvement:
  - Randomized hyperparameter tuning on best model
- Explainability:
  - Feature importance export (CSV + plot)

## 4) Optional flags

```powershell
# Skip EDA
.\.venv\Scripts\python churn_model.py --skip-eda

# Skip tuning
.\.venv\Scripts\python churn_model.py --skip-tuning

# Increase tuning budget
.\.venv\Scripts\python churn_model.py --tune-iters 40
```

## 5) Guided step-by-step run

- See `STEP_BY_STEP.md` for phase-wise commands.
- Use `run_pipeline_step_by_step.ps1` to run all phases sequentially.

## 6) Predict on real-world input

Input file must contain customer feature columns used by the model.

Example with the provided sample file:

```powershell
.\.venv\Scripts\python predict.py --input-path sample_real_world_input.csv --output-path artifacts/predictions/sample_predictions.csv --threshold 0.5
```

Output columns added:
- `churn_probability`
- `predicted_churn_label` (0/1)
- `predicted_churn_text` (No/Yes)
- `predicted_churn_reason` (if reason model is available)
- `recommended_next_action`

## 7) Train churn-reason model

Use grouped reason categories for better practical performance:

```powershell
.\.venv\Scripts\python reason_model.py --data-path Telco_customer_churn.xlsx --grouped-reasons
```

Generated:
- `artifacts/models/churn_reason_model.joblib`
- `artifacts/reports/reason_metrics.json`

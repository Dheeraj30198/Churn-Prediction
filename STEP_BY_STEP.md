# Step-by-Step Execution

## Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## Step 1: EDA

```powershell
.\.venv\Scripts\python -c "from pathlib import Path; from churn_model import load_data, run_eda; df=load_data(Path('Telco_customer_churn.xlsx')); run_eda(df, 'Churn Value', Path('artifacts/eda'))"
```

## Step 2: Model Building + Evaluation + Class Imbalance + Feature Importance

```powershell
.\.venv\Scripts\python churn_model.py --data-path Telco_customer_churn.xlsx --target-col "Churn Value" --artifact-dir artifacts --skip-eda --skip-tuning --cv-folds 3
```

## Step 3: Improve Model (Hyperparameter Tuning)

```powershell
.\.venv\Scripts\python churn_model.py --data-path Telco_customer_churn.xlsx --target-col "Churn Value" --artifact-dir artifacts --skip-eda --cv-folds 3 --tune-iters 10
```

## One-command step runner

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline_step_by_step.ps1
```
